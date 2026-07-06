"""
space_syntax_pipeline.py – El-Bagawat Space Syntax Analysis (Phase 8)
====================================================================
Computes axial line integration to capture the social/perceptual logic
of movement independent of terrain cost.
"""

# %%
# CELL 0  Imports, paths, and configuration
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
from skimage.morphology import skeletonize
import rasterio
import rasterio.features
import rasterio.transform
import matplotlib.pyplot as plt

# Add the directory of this script to sys.path to import shared.py
script_dir = Path(__file__).resolve().parents[1]
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
import shared


# %%
# CELL 1  Medial axis & axial graph helpers
def generate_free_space_skeleton(footprints_gdf, roi_bounds=None, resolution=0.5):
    """
    Compute the medial axis of the free space (complement of building footprints).
    
    footprints_gdf: GeoDataFrame with building footprint Polygons
    roi_bounds: (minx, miny, maxx, maxy) in working CRS
    resolution: map units per pixel for the analysis grid (default 0.5 m = 1 pixel/0.5m)
    
    Returns: (skeleton_binary, raster_transform)
    """
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds
    
    if roi_bounds is None:
        minx, miny, maxx, maxy = footprints_gdf.total_bounds
        roi_bounds = (minx, miny, maxx, maxy)
        
    minx, miny, maxx, maxy = roi_bounds
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Rasterize building footprints
    shapes = [(geom, 1) for geom in footprints_gdf.geometry if geom is not None]
    footprint_raster = rio_rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=0, dtype=np.uint8)
    
    # Free space = complement, eroded by ~0.5m to represent wall thickness
    free_space = 1 - footprint_raster
    kernel = np.ones((3, 3), np.uint8)  # 1 pixel = 0.5 m at resolution=0.5
    free_space_eroded = cv2.erode(free_space, kernel, iterations=1)
    
    # Skeletonize to medial axis
    skeleton = skeletonize(free_space_eroded.astype(bool))
    
    return skeleton.astype(np.uint8), transform


def skeleton_to_axial_graph(skeleton, raster_transform, crs):
    """
    Convert skeleton binary image to a networkx graph of axial lines.
    Each skeleton pixel = a node; connected pixels = edges.
    
    Returns: networkx.Graph (representing the axial graph)
    """
    G = nx.Graph()
    rows_idx, cols_idx = np.where(skeleton)
    coord_set = set(zip(rows_idx.tolist(), cols_idx.tolist()))
    
    for (r, c) in coord_set:
        neighbors = [
            (r + dr, c + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and (r + dr, c + dc) in coord_set
        ]
        for nb in neighbors:
            G.add_edge((r, c), nb)
            
    return G


def compute_space_syntax_integration(G, raster_transform, crs):
    """
    Compute global integration for each axial line (node in G).
    Integration = 1 / RRA, where RRA = normalized relative asymmetry.
    
    High integration = well-connected to all others = candidate primary street.
    """
    n = G.number_of_nodes()
    if n < 3:
        return gpd.GeoDataFrame(columns=["geometry", "integration_mean", "integration_mean_norm", "integration_u", "integration_v"], crs=crs)
    
    # Compute mean topological depth from each node via BFS
    node_list = list(G.nodes())
    integration_vals = {}
    
    # Diamond baseline D_n (from Hillier & Hanson formula)
    # For large n, D_n ≈ 2*(n+2)/3 * log2((n+2)/3) / (n-1)
    # Use simplified normalization
    D_n = 2.0 / (n - 2) if n > 2 else 1.0
    
    for node in node_list:
        lengths = nx.single_source_shortest_path_length(G, node)
        if len(lengths) < 2:
            integration_vals[node] = 0.0
            continue
        total_depth = sum(lengths.values())
        MD = total_depth / (len(lengths) - 1)  # mean depth excluding self
        RA = 2 * (MD - 1) / (n - 2) if n > 2 else 0
        RRA = RA / D_n if D_n > 0 else 0
        integration_vals[node] = 1.0 / RRA if RRA > 0 else 0.0
    
    # Normalize integration to [0, 1]
    int_vals = np.array(list(integration_vals.values()))
    max_val = int_vals.max()
    
    # Build GeoDataFrame of segments
    segments = []
    for (u, v) in G.edges():
        ux, uy = rasterio.transform.xy(raster_transform, u[0], u[1])
        vx, vy = rasterio.transform.xy(raster_transform, v[0], v[1])
        int_u = integration_vals.get(u, 0)
        int_v = integration_vals.get(v, 0)
        
        int_u_norm = int_u / max_val if max_val > 0 else int_u
        int_v_norm = int_v / max_val if max_val > 0 else int_v
        
        segments.append({
            "geometry": LineString([(ux, uy), (vx, vy)]),
            "integration_mean": (int_u + int_v) / 2,
            "integration_mean_norm": (int_u_norm + int_v_norm) / 2,
            "integration_u": int_u,
            "integration_v": int_v,
        })
    
    gdf = gpd.GeoDataFrame(segments, crs=crs)
    return gdf


def rasterize_integration(syntax_gdf, raster_shape, raster_transform):
    """
    Convert the vector axial line integration GeoDataFrame to a raster
    for use in the ensemble confidence computation.
    """
    from rasterio.features import rasterize as rio_rasterize
    import rasterio.enums
    
    if syntax_gdf.empty:
        return np.zeros(raster_shape, dtype=np.float32)
        
    # Normalize integration values to [0, 1] first
    vals = syntax_gdf["integration_mean"].values
    if vals.max() > 0:
        vals_norm = vals / vals.max()
    else:
        vals_norm = vals
    
    shapes = [(geom, float(val))
              for geom, val in zip(syntax_gdf.geometry, vals_norm)]
    
    integration_raster = rio_rasterize(
        shapes,
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0.0,
        dtype=np.float32,
        merge_alg=rasterio.enums.MergeAlg.replace
    )
    
    return integration_raster

# %%
# CELL 2  Load Building Footprints and DEM
print("=== Phase 8: Space Syntax Pipeline ===")
print("Loading datasets...")
footprints = shared.load_footprints()
dem = shared.load_dem()
crs = dem["crs"]

# Reproject footprints if needed to match DEM CRS
if footprints.crs != crs:
    print(f"Reprojecting footprints from {footprints.crs} to {crs}...")
    footprints = footprints.to_crs(crs)

# %%
# CELL 3  Compute the free space skeleton (medial axis)
print("Generating free space skeleton...")
roi_bounds = (
    dem["bounds"].left,
    dem["bounds"].bottom,
    dem["bounds"].right,
    dem["bounds"].top
)
skeleton, raster_transform = generate_free_space_skeleton(
    footprints,
    roi_bounds=roi_bounds,
    resolution=dem["res"]
)
print(f"  Skeleton raster shape: {skeleton.shape}")

# %%
# CELL 4  Convert skeleton to axial graph
print("Converting skeleton to axial graph...")
G = skeleton_to_axial_graph(skeleton, raster_transform, crs)
print(f"  Axial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# %%
# CELL 5  Compute global integration
print("Computing space syntax integration...")
syntax_gdf = compute_space_syntax_integration(G, raster_transform, crs)
print(f"  Constructed {len(syntax_gdf)} axial line segments with integration attributes.")

# %%
# CELL 6  Rasterize integration values
print("Rasterizing integration...")
integration_raster = rasterize_integration(syntax_gdf, skeleton.shape, raster_transform)

# %%
# CELL 7  Save outputs (raster & vector)
# A. Raster integration map
raster_out_path = shared.OUT / "space_syntax_integration.tif"
print(f"Saving integration raster to {raster_out_path}...")
out_profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': 0.0,
    'width': skeleton.shape[1],
    'height': skeleton.shape[0],
    'count': 1,
    'crs': crs,
    'transform': raster_transform
}
with rasterio.open(raster_out_path, "w", **out_profile) as dst:
    dst.write(integration_raster, 1)
    
# B. Vector GIS axial lines
vector_out_path = shared.OUT / "vector_gis" / "space_syntax_axial_lines.geojson"
print(f"Saving vector axial lines to {vector_out_path}...")
vector_out_path.parent.mkdir(parents=True, exist_ok=True)
syntax_gdf.to_file(str(vector_out_path), driver="GeoJSON")

# %%
# CELL 8  Visualization map
print("Generating visualization map...")
hs = shared.hillshade(dem["disp"])

fig, ax = plt.subplots(figsize=(14, 12))

# Underlay DEM hillshade
ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
          cmap="terrain", alpha=0.6,
          vmin=dem["e_min"], vmax=dem["e_max"], rasterized=True)
ax.imshow(hs, extent=dem["extent"], origin="upper",
          cmap="gray", alpha=0.35, rasterized=True)
          
# Overlay footprints
footprints.plot(ax=ax, color="none", edgecolor="#555555", linewidth=0.5, alpha=0.7)

# Plot axial lines colored by integration
if not syntax_gdf.empty:
    syntax_gdf.plot(ax=ax, column="integration_mean_norm", cmap="plasma", linewidth=1.5,
                    legend=True, legend_kwds={'label': 'Normalized Global Integration', 'shrink': 0.6})
                    
ax.set_title("El-Bagawat Space Syntax Analysis - Phase 8\nGlobal Axial Line Integration (Medial Axis)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
xmin, ymin, xmax, ymax = footprints.total_bounds
ax.set_xlim(xmin - 50, xmax + 50)
ax.set_ylim(ymin - 50, ymax + 50)
                    
# Save the figure using the shared.save_fig helper
figs_dir = shared.OUT / "figures" / "space_syntax"
shared.save_fig(figs_dir, "18_space_syntax_integration.png", dpi=150)
plt.close()

print("Phase 8 Space Syntax Pipeline completed successfully!")
