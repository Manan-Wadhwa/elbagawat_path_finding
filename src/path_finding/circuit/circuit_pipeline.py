"""
circuit_pipeline.py – El-Bagawat Electrical Circuit Model (Phase 7)
==================================================================
Models pedestrian path finding as electrical current flowing through
a terrain-informed resistor network.
"""

# %%
# CELL 0  Imports, paths, and output directories
import sys
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp
from scipy.sparse.linalg import lgmres
import cv2
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# Suppress warnings
warnings.filterwarnings("ignore")

# Make shared.py importable (it lives in the parent directory)
shared_dir = str(Path(__file__).resolve().parents[1])
if shared_dir not in sys.path:
    sys.path.insert(0, shared_dir)
import shared

# Output directories
FIGS = shared.OUT / "figures" / "circuit"
FIGS.mkdir(parents=True, exist_ok=True)
# %%
# CELL 1  Local helper functions for circuit solver
def build_conductance_matrix(cost_raster, connectivity=8):
    """
    Constructs a sparse conductance (Laplacian) matrix for the grid.
    
    Parameters:
        cost_raster: 2D numpy array of travel costs.
        connectivity: 4 or 8 neighbor connections.
        
    Returns:
        L: scipy.sparse.csr_matrix representing the Laplacian.
    """
    h, w = cost_raster.shape
    n_pixels = h * w
    
    if connectivity == 8:
        offsets = [
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2))
        ]
    else:  # connectivity == 4
        offsets = [
            (0, 1, 1.0),
            (1, 0, 1.0)
        ]
        
    rows = []
    cols = []
    conductances = []
    
    for dr, dc, dist in offsets:
        r_start = 0
        r_end = h - dr
        c_start = max(0, -dc)
        c_end = min(w, w - dc)
        
        if r_start >= r_end or c_start >= c_end:
            continue
            
        r, c = np.meshgrid(np.arange(r_start, r_end), np.arange(c_start, c_end), indexing='ij')
        r = r.ravel()
        c = c.ravel()
        
        tr = r + dr
        tc = c + dc
        
        idx1 = r * w + c
        idx2 = tr * w + tc
        
        cost1 = cost_raster[r, c]
        cost2 = cost_raster[tr, tc]
        
        # R = 0.5 * (cost1 + cost2) * dist
        # G = 1 / R
        avg_cost = 0.5 * (cost1 + cost2)
        avg_cost = np.maximum(avg_cost, 1e-6)
        r_val = avg_cost * dist
        g_val = 1.0 / r_val
        
        # Enforce perfect insulator for buildings/obstacles (cost >= 1e8)
        g_val[(cost1 >= 1e8) | (cost2 >= 1e8)] = 0.0
        
        rows.append(idx1)
        cols.append(idx2)
        conductances.append(g_val)
        
        rows.append(idx2)
        cols.append(idx1)
        conductances.append(g_val)
        
    if len(rows) > 0:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        conductances = np.concatenate(conductances)
        
        # Filter out zero conductances
        valid_edges = conductances > 1e-12
        rows = rows[valid_edges]
        cols = cols[valid_edges]
        conductances = conductances[valid_edges]
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        conductances = np.array([], dtype=np.float64)
        
    diag_conductance = np.bincount(rows, weights=conductances, minlength=n_pixels)
    
    # Handle disconnected nodes to avoid singular system
    disconnected = (diag_conductance < 1e-12)
    diag_conductance[disconnected] = 1.0
    
    diag_indices = np.arange(n_pixels)
    all_rows = np.concatenate([rows, diag_indices])
    all_cols = np.concatenate([cols, diag_indices])
    all_vals = np.concatenate([-conductances, diag_conductance])
    
    L = sp.coo_matrix((all_vals, (all_rows, all_cols)), shape=(n_pixels, n_pixels)).tocsr()
    return L


def snap_pixels(pixels, building_mask):
    """Snaps pixels inside buildings to the nearest non-building pixel."""
    h, w = building_mask.shape
    snapped_pixels = []
    for r, c in pixels:
        if building_mask[r, c] == 1:
            snapped = False
            for radius in range(1, 15):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and building_mask[nr, nc] == 0:
                            r, c = nr, nc
                            snapped = True
                            break
                    if snapped:
                        break
                if snapped:
                    break
        snapped_pixels.append((r, c))
    return snapped_pixels


def solve_linear_system(L, I, method='lgmres'):
    """Solves a sparse linear system LV = I with a fallback to direct spsolve."""
    if method == 'lgmres':
        try:
            V, info = lgmres(L, I, tol=1e-5, maxiter=300)
            if info != 0:
                from scipy.sparse.linalg import spsolve
                V = spsolve(L, I)
        except Exception:
            from scipy.sparse.linalg import spsolve
            V = spsolve(L, I)
    else:
        from scipy.sparse.linalg import spsolve
        V = spsolve(L, I)
    return V


def compute_current_density_from_potentials(V, cost_raster, connectivity=8):
    """Computes node current densities from node potentials."""
    h, w = cost_raster.shape
    n_pixels = h * w
    current_density = np.zeros(n_pixels, dtype=np.float64)
    
    if connectivity == 8:
        offsets = [
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2))
        ]
    else:
        offsets = [
            (0, 1, 1.0),
            (1, 0, 1.0)
        ]
        
    for dr, dc, dist in offsets:
        r_start = 0
        r_end = h - dr
        c_start = max(0, -dc)
        c_end = min(w, w - dc)
        
        if r_start >= r_end or c_start >= c_end:
            continue
            
        r, c = np.meshgrid(np.arange(r_start, r_end), np.arange(c_start, c_end), indexing='ij')
        r = r.ravel()
        c = c.ravel()
        
        tr = r + dr
        tc = c + dc
        
        idx1 = r * w + c
        idx2 = tr * w + tc
        
        cost1 = cost_raster[r, c]
        cost2 = cost_raster[tr, tc]
        avg_cost = 0.5 * (cost1 + cost2)
        avg_cost = np.maximum(avg_cost, 1e-6)
        g_val = 1.0 / (avg_cost * dist)
        g_val[(cost1 >= 1e8) | (cost2 >= 1e8)] = 0.0
        
        v1 = V[idx1]
        v2 = V[idx2]
        current = g_val * (v1 - v2)
        abs_current = np.abs(current)
        
        np.add.at(current_density, idx1, 0.5 * abs_current)
        np.add.at(current_density, idx2, 0.5 * abs_current)
        
    return current_density.reshape((h, w))


def compute_supernode_current(cost_raster, entrance_pixels, shape, sink_idx=None):
    """
    Computes current density where all doors are connected to a single reference/sink door.
    
    Parameters:
        cost_raster: High-res cost surface.
        entrance_pixels: High-res door coordinates.
        shape: Original shape (H, W).
        sink_idx: Index of the door to use as the ground/sink.
    """
    h, w = cost_raster.shape
    max_pixels = 250000
    
    # Downsample
    total_pixels = h * w
    if total_pixels > max_pixels:
        scale = np.sqrt(max_pixels / total_pixels)
        h_down = int(round(h * scale))
        w_down = int(round(w * scale))
    else:
        h_down, w_down = h, w
        
    h_down = max(1, h_down)
    w_down = max(1, w_down)
    
    if h_down != h or w_down != w:
        cost_down = cv2.resize(cost_raster, (w_down, h_down), interpolation=cv2.INTER_NEAREST)
    else:
        cost_down = cost_raster.copy()
        
    # Scale entrance pixels
    entrance_pixels_down = []
    for r, c in entrance_pixels:
        r_d = int(round(r * (h_down / h)))
        c_d = int(round(c * (w_down / w)))
        r_d = np.clip(r_d, 0, h_down - 1)
        c_d = np.clip(c_d, 0, w_down - 1)
        entrance_pixels_down.append((r_d, c_d))
        
    # Snap
    building_mask_down = (cost_down > 1e6).astype(np.uint8)
    entrance_pixels_down = snap_pixels(entrance_pixels_down, building_mask_down)
    
    # Set default sink_idx to the door closest to centroid if not provided
    if sink_idx is None:
        r_coords = np.array([p[0] for p in entrance_pixels])
        c_coords = np.array([p[1] for p in entrance_pixels])
        r_mean = np.mean(r_coords)
        c_mean = np.mean(c_coords)
        dists = np.hypot(r_coords - r_mean, c_coords - c_mean)
        sink_idx = np.argmin(dists)
        
    # Build conductance matrix L
    L = build_conductance_matrix(cost_down, connectivity=8)
    
    # Setup I vector
    n_nodes = h_down * w_down
    I = np.zeros(n_nodes, dtype=np.float64)
    
    sink_pixel = entrance_pixels_down[sink_idx]
    sink_flat = sink_pixel[0] * w_down + sink_pixel[1]
    
    for idx, (r, c) in enumerate(entrance_pixels_down):
        flat_idx = r * w_down + c
        if flat_idx != sink_flat:
            I[flat_idx] += 1.0
            
    total_injected = np.sum(I)
    I[sink_flat] = -total_injected
    
    # Apply Dirichlet boundary condition at sink
    L_solve = L.copy()
    start_idx = L_solve.indptr[sink_flat]
    end_idx = L_solve.indptr[sink_flat + 1]
    L_solve.data[start_idx:end_idx] = 0.0
    diag_idx = np.where(L_solve.indices[start_idx:end_idx] == sink_flat)[0]
    if len(diag_idx) > 0:
        L_solve.data[start_idx + diag_idx[0]] = 1.0
        
    I_solve = I.copy()
    I_solve[sink_flat] = 0.0
    
    # Solve potentials
    V = solve_linear_system(L_solve, I_solve, method='lgmres')
    
    # Compute density
    density_down = compute_current_density_from_potentials(V, cost_down, connectivity=8)
    
    # Upsample
    density_upsampled = cv2.resize(density_down, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    if density_upsampled.max() > 0:
        density_upsampled = density_upsampled / density_upsampled.max()
        
    return density_upsampled


def compute_pairwise_current(cost_raster, entrance_pixels, max_pairs=50):
    """
    Computes pairwise current density between selected pairs of doors.
    """
    h, w = cost_raster.shape
    max_pixels = 250000
    
    # Downsample
    total_pixels = h * w
    if total_pixels > max_pixels:
        scale = np.sqrt(max_pixels / total_pixels)
        h_down = int(round(h * scale))
        w_down = int(round(w * scale))
    else:
        h_down, w_down = h, w
        
    h_down = max(1, h_down)
    w_down = max(1, w_down)
    
    if h_down != h or w_down != w:
        cost_down = cv2.resize(cost_raster, (w_down, h_down), interpolation=cv2.INTER_NEAREST)
    else:
        cost_down = cost_raster.copy()
        
    # Scale entrance pixels
    entrance_pixels_down = []
    for r, c in entrance_pixels:
        r_d = int(round(r * (h_down / h)))
        c_d = int(round(c * (w_down / w)))
        r_d = np.clip(r_d, 0, h_down - 1)
        c_d = np.clip(c_d, 0, w_down - 1)
        entrance_pixels_down.append((r_d, c_d))
        
    # Snap
    building_mask_down = (cost_down > 1e6).astype(np.uint8)
    entrance_pixels_down = snap_pixels(entrance_pixels_down, building_mask_down)
    
    # Generate unique pairs of doors
    pairs = []
    for i in range(len(entrance_pixels_down)):
        for j in range(i + 1, len(entrance_pixels_down)):
            p1 = entrance_pixels_down[i]
            p2 = entrance_pixels_down[j]
            if p1 != p2:
                pairs.append((p1, p2))
                
    if len(pairs) > max_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_pairs)
        
    # Build conductance matrix L
    L = build_conductance_matrix(cost_down, connectivity=8)
    
    accumulated_density = np.zeros((h_down, w_down), dtype=np.float64)
    
    print(f"Solving pairwise current for {len(pairs)} pairs...")
    for idx, (p1, p2) in enumerate(pairs):
        n_nodes = h_down * w_down
        I = np.zeros(n_nodes, dtype=np.float64)
        
        flat_p1 = p1[0] * w_down + p1[1]
        flat_p2 = p2[0] * w_down + p2[1]
        
        I[flat_p1] = 1.0
        I[flat_p2] = -1.0
        
        # Apply Dirichlet boundary condition at p2
        L_solve = L.copy()
        start_idx = L_solve.indptr[flat_p2]
        end_idx = L_solve.indptr[flat_p2 + 1]
        L_solve.data[start_idx:end_idx] = 0.0
        diag_idx = np.where(L_solve.indices[start_idx:end_idx] == flat_p2)[0]
        if len(diag_idx) > 0:
            L_solve.data[start_idx + diag_idx[0]] = 1.0
            
        I_solve = I.copy()
        I_solve[flat_p2] = 0.0
        
        # Solve potentials
        V = solve_linear_system(L_solve, I_solve, method='lgmres')
        
        # Compute density
        density_down = compute_current_density_from_potentials(V, cost_down, connectivity=8)
        accumulated_density += density_down
        
        if (idx + 1) % 10 == 0 or idx == len(pairs) - 1:
            print(f"  Processed {idx + 1}/{len(pairs)} pairs")
            
    # Upsample
    density_upsampled = cv2.resize(accumulated_density, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    if density_upsampled.max() > 0:
        density_upsampled = density_upsampled / density_upsampled.max()
        
    return density_upsampled
# %%
# CELL 2  Load data (footprints, directions, DXF labels)
fp = shared.load_footprints()
df = shared.load_excel_directions()
dxf_labels = shared.load_dxf_labels()

# %%
# CELL 3  Affine alignment and label-footprint matching
H_rough = shared.estimate_rough_affine(dxf_labels, fp)
crosswalk = shared.bipartite_label_match(dxf_labels, H_rough, fp)
fp, n_labelled, n_with_dir, n_no_dir, n_unlabelled = \
    shared.attribute_footprints(fp, crosswalk, df)
    
# %%
# CELL 4  Place doors and load DEM
doors_gdf, doors_pts = shared.place_doors(fp)
dem = shared.load_dem()
hs = shared.hillshade(dem["disp"])
doors_pts = shared.sample_dem_at_doors(doors_pts)

# %%
# CELL 5  Compute Tobler cost surface & building obstruction mask
print("Computing terrain-informed Tobler cost surface ...")
fp_dem_crs = fp.to_crs(dem["crs"]) if fp.crs != dem["crs"] else fp.copy()

dy, dx = np.gradient(dem["disp"], dem["res"], dem["res"])
slope_mag = np.sqrt(dx**2 + dy**2)

tobler_kmh = 6.0 * np.exp(-3.5 * np.abs(slope_mag + 0.05))
tobler_ms = tobler_kmh / 3.6
cost_tobler = 1.0 / np.maximum(tobler_ms, 1e-6)

shapes = [(geom, 1) for geom in fp_dem_crs.geometry if geom is not None]
building_mask = rio_rasterize(
    shapes,
    out_shape=dem["arr"].shape,
    transform=dem["transform"],
    fill=0,
    dtype=np.uint8,
)
cost_obs = np.where(building_mask == 1, 1e9, 0.0)
cost_surface = 0.5 * cost_tobler + cost_obs

# Save cost surface GeoTIFF
profile_cost = dem["profile"].copy()
profile_cost.update(dtype="float32", count=1)
with rasterio.open(shared.OUT / "cost_surface_tobler.tif", "w", **profile_cost) as dst:
    dst.write(cost_surface.astype(np.float32), 1)
print("  Cost surface saved -> outputs/cost_surface_tobler.tif")

# %%
# CELL 6  Project door points to raster pixel indices and snap
door_coords = [(g.x, g.y) for g in doors_pts.geometry]
rows_px, cols_px = rowcol(
    dem["transform"],
    [p[0] for p in door_coords],
    [p[1] for p in door_coords],
)

h, w = dem["arr"].shape
entrance_pixels = []
valid_door_rows = []

for i, (r, c) in enumerate(zip(rows_px, cols_px)):
    if not (0 <= r < h and 0 <= c < w):
        continue
    if building_mask[r, c] == 1:
        snapped = False
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and building_mask[nr, nc] == 0:
                    r, c = nr, nc
                    snapped = True
                    break
            if snapped:
                break
    entrance_pixels.append((r, c))
    valid_door_rows.append(doors_pts.iloc[i])
    
print(f"  Projected {len(entrance_pixels)} doors to raster pixels.")

# Find sink_idx for Chapel 180 (Main Church)
sink_idx = None
for idx, row in enumerate(valid_door_rows):
    if str(row["chapel_id"]) == "180":
        sink_idx = idx
        break
    
# %%
# CELL 7  Solve Supernode Current Density
print("Solving supernode current density (all-to-one, sink at Chapel 180) ...")
supernode_density = compute_supernode_current(
    cost_surface,
    entrance_pixels,
    cost_surface.shape,
    sink_idx=sink_idx
)

# Save supernode current density GeoTIFF
profile_density = dem["profile"].copy()
profile_density.update(dtype="float32", count=1)
with rasterio.open(shared.OUT / "circuit_current_supernode.tif", "w", **profile_density) as dst:
    dst.write(supernode_density.astype(np.float32), 1)
print("  Supernode current density saved -> outputs/circuit_current_supernode.tif")

# %%
# CELL 8  Solve Pairwise Current Density (max_pairs=50)
print("Solving pairwise current density (max_pairs=50) ...")
pairwise_density = compute_pairwise_current(
    cost_surface,
    entrance_pixels,
    max_pairs=50
)

# Save pairwise current density GeoTIFF
with rasterio.open(shared.OUT / "circuit_current_pairwise.tif", "w", **profile_density) as dst:
    dst.write(pairwise_density.astype(np.float32), 1)
print("  Pairwise current density saved -> outputs/circuit_current_pairwise.tif")

# %%
# CELL 9  Plot Supernode Current Density Map
print("Plotting supernode current density map ...")
fp_dem = fp.to_crs(dem["crs"]) if fp.crs != dem["crs"] else fp.copy()
doors_dem = doors_gdf.to_crs(dem["crs"]) if doors_gdf.crs != dem["crs"] else doors_gdf.copy()

fig, ax = plt.subplots(figsize=(18, 15))
ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
          cmap="terrain", alpha=0.6, vmin=dem["e_min"], vmax=dem["e_max"],
          rasterized=True)
ax.imshow(hs, extent=dem["extent"], origin="upper",
          cmap="gray", alpha=0.3, rasterized=True)
          
supernode_masked = np.where(supernode_density > 0.005, supernode_density, np.nan)
im = ax.imshow(supernode_masked, extent=dem["extent"], origin="upper",
               cmap="inferno", alpha=0.85, vmin=0.0, vmax=1.0, rasterized=True)
               
fp_dem.plot(ax=ax, color="none", edgecolor="#333333", linewidth=0.5, alpha=0.7)

for _, row in doors_dem.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color=shared.DIR_CLR.get(row["direction"], "#aaa"),
            linewidth=2.0, zorder=5)
            
shared._draw_chapel_ids(ax, fp_dem, fontsize=4, color="white",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))
                        
plt.colorbar(im, ax=ax, label="Supernode Current Density", shrink=0.5)
ax.set_title("El-Bagawat Resistor Network - Supernode Current Density",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
xmin, ymin, xmax, ymax = fp.total_bounds
ax.set_xlim(xmin - 50, xmax + 50)
ax.set_ylim(ymin - 50, ymax + 50)
plt.tight_layout()

shared.save_fig(FIGS, "17_circuit_supernode_density.png")
plt.close()

# %%
# CELL 10  Plot Pairwise Current Density Map & Save Outputs
print("Plotting pairwise current density map ...")
fig, ax = plt.subplots(figsize=(18, 15))
ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
          cmap="terrain", alpha=0.6, vmin=dem["e_min"], vmax=dem["e_max"],
          rasterized=True)
ax.imshow(hs, extent=dem["extent"], origin="upper",
          cmap="gray", alpha=0.3, rasterized=True)
          
pairwise_masked = np.where(pairwise_density > 0.005, pairwise_density, np.nan)
im = ax.imshow(pairwise_masked, extent=dem["extent"], origin="upper",
               cmap="inferno", alpha=0.85, vmin=0.0, vmax=1.0, rasterized=True)
               
fp_dem.plot(ax=ax, color="none", edgecolor="#333333", linewidth=0.5, alpha=0.7)

for _, row in doors_dem.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color=shared.DIR_CLR.get(row["direction"], "#aaa"),
            linewidth=2.0, zorder=5)
            
shared._draw_chapel_ids(ax, fp_dem, fontsize=4, color="white",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))
                        
plt.colorbar(im, ax=ax, label="Pairwise Current Density", shrink=0.5)
ax.set_title("El-Bagawat Resistor Network - Pairwise Current Density (max_pairs=50)",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
xmin, ymin, xmax, ymax = fp.total_bounds
ax.set_xlim(xmin - 50, xmax + 50)
ax.set_ylim(ymin - 50, ymax + 50)
plt.tight_layout()

shared.save_fig(FIGS, "18_circuit_pairwise_density.png")
plt.close()

# Save door points GeoJSON for reference
gis_out = shared.OUT / "vector_gis"
gis_out.mkdir(exist_ok=True)
doors_pts.to_crs("EPSG:4326").to_file(
    str(shared.OUT / "circuit_door_points.geojson"), driver="GeoJSON"
)
print("  Door points GeoJSON saved -> outputs/circuit_door_points.geojson")
print("=== Pipeline Completed Successfully! ===")
