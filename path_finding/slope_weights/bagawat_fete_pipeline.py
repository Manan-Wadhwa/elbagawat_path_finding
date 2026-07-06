"""
bagawat_fete_pipeline.py  –  El-Bagawat FETE raster-based pedestrian network
==============================================================================
Run as a Jupyter notebook (#%% cells) or as a plain Python script.
Each #%% marker is a separate notebook cell.

FETE = From-Everywhere-To-Everywhere Dijkstra
  Instead of a vector k-NN graph, paths are computed on a raster cost surface
  using Tobler's Hiking Function (slope-weighted travel speed).
  Path density is accumulated across all door-to-door pairs, then
  skeletonised and vectorised into a GIS-compatible network.

Algorithm:
  1–7   Shared data loading (footprints, Excel, DXF, doors, DEM)  ← via shared.py
  8     Build Tobler cost surface + building obstruction mask
  9     FETE multi-source Dijkstra (door→door path accumulation)
  10    Skeletonise + vectorise density map → networkx graph
  11    Edge betweenness centrality + GIS export
  12    Betweenness visualisation
  13    Full composite map
  14    Node statistics
  15    Shortest paths from Chapel 180
  16    Ultra high-resolution zoomable master map
"""

# %%
# CELL 0  Imports, paths, and figure directory
import sys
from pathlib import Path

# Make shared.py importable (it lives two levels up from this file)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import rowcol, xy as rasterio_xy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

# All figures for this pipeline go here
FIGS = shared.OUT / "figures" / "fete"
FIGS.mkdir(parents=True, exist_ok=True)

shared.verify_paths()


# %%
# CELL 1  Load shapefile
fp = shared.load_footprints()
shared.plot_raw_footprints(fp, FIGS)


# %%
# CELL 2  Load Excel, normalise directions, audit outliers
df         = shared.load_excel_directions()
dir_counts = shared.print_direction_audit(df)
shared.plot_direction_audit(df, dir_counts, FIGS)


# %%
# CELL 3  Extract DXF text labels (raw DXF coordinate space)
dxf_labels = shared.load_dxf_labels()
shared.plot_dxf_labels(dxf_labels, FIGS)


# %%
# CELL 4  Bipartite label→polygon matching (Hungarian algorithm)
H_rough   = shared.estimate_rough_affine(dxf_labels, fp)
crosswalk = shared.bipartite_label_match(dxf_labels, H_rough, fp)

print(f"Rough affine H:\n{np.round(H_rough, 3)}")
print(f"Bipartite matches: {len(crosswalk)} / {len(fp)} footprints")
print(crosswalk.head(10).to_string(index=False))
print("Dist stats (m):"); print(crosswalk["dist_m"].describe())

fp, n_labelled, n_with_dir, n_no_dir, n_unlabelled = \
    shared.attribute_footprints(fp, crosswalk, df)

crosswalk.to_csv(shared.OUT / "crosswalk.csv", index=False)
print("→ crosswalk.csv")

shared.plot_bipartite_matching(fp, dxf_labels, crosswalk, H_rough, FIGS)
shared.plot_attribution_map(fp, n_with_dir, n_no_dir, n_unlabelled, FIGS)


# %%
# CELL 5  Approach-5 door placement
doors_gdf, doors_pts = shared.place_doors(fp)

gis_out = shared.OUT / "vector_gis"
gis_out.mkdir(exist_ok=True)
doors_gdf.to_crs("EPSG:4326").to_file(
    str(gis_out / "doors_native_approach5.geojson"), driver="GeoJSON")
doors_pts.to_crs("EPSG:4326").to_file(
    str(gis_out / "door_points_approach5.geojson"), driver="GeoJSON")
print("→ doors_native_approach5.geojson  door_points_approach5.geojson")

shared.plot_doors_all(fp, doors_gdf, FIGS)
shared.plot_doors_zoom(fp, doors_gdf, FIGS)


# %%
# CELL 6  Load DEM + generate hillshade
dem = shared.load_dem()
hs  = shared.plot_dem_hillshade(dem, FIGS)
fp_dem, doors_dem = shared.plot_dem_with_doors(dem, hs, fp, doors_gdf, FIGS)


# %%
# CELL 7  Sample door elevations from DEM
doors_pts = shared.sample_dem_at_doors(doors_pts)
shared.plot_door_elevation_histogram(doors_pts, FIGS)


# %%
# CELL 8  Build Tobler cost surface + building obstruction mask
print("Computing terrain-informed Tobler cost surface …")

# Reproject footprints to DEM CRS for rasterization
fp_dem_crs = fp.to_crs(dem["crs"]) if fp.crs != dem["crs"] else fp.copy()

# Slope from DEM (central differences)
dy, dx      = np.gradient(dem["disp"], dem["res"], dem["res"])
slope_mag   = np.sqrt(dx**2 + dy**2)

# Tobler's Hiking Function: speed (km/h) → cost (seconds per metre)
tobler_kmh  = 6.0 * np.exp(-3.5 * np.abs(slope_mag + 0.05))
tobler_ms   = tobler_kmh / 3.6
cost_tobler = 1.0 / np.maximum(tobler_ms, 1e-6)

# Building obstruction mask — rasterize footprints as hard no-go zones
shapes = [(geom, 1) for geom in fp_dem_crs.geometry if geom is not None]
building_mask = rio_rasterize(
    shapes,
    out_shape=dem["arr"].shape,
    transform=dem["transform"],
    fill=0,
    dtype=np.uint8,
)
cost_obs = np.where(building_mask == 1, 1e9, 0.0)

# Final combined cost surface (terrain 50%, obstruction 100%)
cost_surface = 0.5 * cost_tobler + cost_obs

# Persist to outputs/ for inspection
profile_cost = dem["profile"].copy()
profile_cost.update(dtype="float32", count=1)
with rasterio.open(shared.OUT / "cost_surface_tobler.tif", "w", **profile_cost) as dst:
    dst.write(cost_surface.astype(np.float32), 1)
print("  Cost surface saved → outputs/cost_surface_tobler.tif")


# %%
# CELL 9  FETE multi-source Dijkstra (accumulate path density)
print("Running FETE multi-source Dijkstra …")

# Project door points onto raster pixel indices
door_coords = [(g.x, g.y) for g in doors_pts.geometry]
rows_px, cols_px = rowcol(
    dem["transform"],
    [p[0] for p in door_coords],
    [p[1] for p in door_coords],
)

h, w = dem["arr"].shape
entrance_pixels = []
valid_door_rows  = []

for i, (r, c) in enumerate(zip(rows_px, cols_px)):
    if not (0 <= r < h and 0 <= c < w):
        continue
    # If the door pixel landed inside a building, snap it outward
    if building_mask[r, c] == 1:
        snapped = False
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and building_mask[nr, nc] == 0:
                    r, c    = nr, nc
                    snapped = True
                    break
            if snapped:
                break
    entrance_pixels.append((r, c))
    valid_door_rows.append(doors_pts.iloc[i])

n_entrances = len(entrance_pixels)
print(f"  Projected {n_entrances}/{len(doors_pts)} door points to valid raster pixels.")

# Accumulate path density: for every source door run Dijkstra to all other doors
density = np.zeros(dem["arr"].shape, dtype=np.float32)
paths_computed = 0

for src_idx, src_pixel in enumerate(entrance_pixels):
    mcp = MCP_Geometric(cost_surface, fully_connected=True)
    mcp.find_costs([src_pixel])

    for tgt_idx in range(src_idx + 1, n_entrances):
        try:
            path = mcp.traceback(entrance_pixels[tgt_idx])
            for pr, pc in path:
                density[pr, pc] += 1
            paths_computed += 1
        except Exception:
            pass

    if (src_idx + 1) % 20 == 0 or src_idx == n_entrances - 1:
        print(f"    {src_idx + 1}/{n_entrances} sources  ({paths_computed} paths)")

# Normalise density to [0, 1]
density_norm = density / density.max() if density.max() > 0 else density

# Save density map
profile_density = dem["profile"].copy()
profile_density.update(dtype="float32", count=1)
with rasterio.open(shared.OUT / "fete_density.tif", "w", **profile_density) as dst:
    dst.write(density_norm.astype(np.float32), 1)
print("  FETE density map saved → outputs/fete_density.tif")


# %%
# CELL 10  Skeletonise density map + build vectorised network graph
print("Skeletonising and vectorising the FETE density map …")

# Threshold → binary corridor mask → 1-pixel skeleton
binary = (density_norm > 0.05).astype(np.uint8)
skel   = skeletonize(binary.astype(bool))

# Build a pixel-level graph from the skeleton
skel_G     = nx.Graph()
skel_px    = set(zip(*np.where(skel)))   # set of (row, col) tuples

for r, c in skel_px:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0):
                continue
            nb = (r + dr, c + dc)
            if nb in skel_px:
                dist_m = np.sqrt(dr**2 + dc**2) * dem["res"]
                skel_G.add_edge((r, c), nb, weight=dist_m, dist_m=dist_m)

# Prune short dead-end spurs (< 5 m)
MIN_SPUR_M = 5.0
changed = True
while changed:
    changed = False
    for node in list(skel_G.nodes()):
        if skel_G.degree(node) != 1:
            continue
        chain_len, curr, prev, chain = 0.0, node, None, [node]
        while skel_G.degree(curr) == 1:
            nbs = [n for n in skel_G.neighbors(curr) if n != prev]
            if not nbs:
                break
            chain_len += skel_G[curr][nbs[0]]["weight"]
            prev, curr = curr, nbs[0]
            chain.append(curr)
            if chain_len > MIN_SPUR_M:
                break
        if chain_len <= MIN_SPUR_M:
            skel_G.remove_nodes_from(chain[:-1])
            changed = True

print(f"  Pruned skeleton: {skel_G.number_of_nodes()} nodes, "
      f"{skel_G.number_of_edges()} edges")

# Convert skeleton pixel coords to UTM
skel_nodes  = list(skel_G.nodes())
skel_coords = np.array([rasterio_xy(dem["transform"], r, c)
                         for (r, c) in skel_nodes])
skel_tree   = cKDTree(skel_coords)

# Build final UTM graph
G = nx.Graph()

for i, node in enumerate(skel_nodes):
    ux, uy = skel_coords[i]
    G.add_node(node, x=ux, y=uy, is_door=False,
               chapel_id=None, direction=None, elev=0.0)

for u, v, data in skel_G.edges(data=True):
    ux_u, uy_u = G.nodes[u]["x"], G.nodes[u]["y"]
    ux_v, uy_v = G.nodes[v]["x"], G.nodes[v]["y"]
    dist_m = float(np.hypot(ux_u - ux_v, uy_u - uy_v))
    G.add_edge(u, v, weight=dist_m, dist_m=dist_m, dh_m=0.0)

# Attach door nodes to their nearest skeleton node
door_node_ids = []
for _, row in doors_pts.iterrows():
    door_id = f"door_{row['chapel_id']}"
    dx, dy  = row.geometry.x, row.geometry.y
    elev    = float(row.get("elevation_m", 0.0) or 0.0)
    if np.isnan(elev):
        elev = 0.0

    G.add_node(door_id, x=dx, y=dy, is_door=True,
               chapel_id=row["chapel_id"], direction=row["direction"], elev=elev)
    door_node_ids.append(door_id)

    dist, skel_idx    = skel_tree.query([dx, dy])
    nearest_skel_node = skel_nodes[skel_idx]
    G.add_edge(door_id, nearest_skel_node, weight=dist, dist_m=dist, dh_m=0.0)

# Sample DEM elevation at every non-door node
with rasterio.open(str(shared.DEM_P)) as dem_src:
    node_list   = list(G.nodes())
    node_coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_list]
    elevs       = np.array([v[0] for v in dem_src.sample(node_coords)],
                            dtype=np.float32)
    if dem_src.nodata is not None:
        elevs[elevs == dem_src.nodata] = 0.0
    for i, n in enumerate(node_list):
        if not G.nodes[n]["is_door"] or G.nodes[n]["elev"] == 0.0:
            G.nodes[n]["elev"] = float(elevs[i])

# Update dh_m on every edge now that elevations are set
for u, v in G.edges():
    G[u][v]["dh_m"] = float(abs(G.nodes[u]["elev"] - G.nodes[v]["elev"]))

print(f"  Final graph G: {G.number_of_nodes()} nodes "
      f"({len(door_node_ids)} doors), {G.number_of_edges()} edges")


# %%
# CELL 11  Edge betweenness centrality + GIS export
print("Computing edge betweenness centrality on FETE network …")
edge_bc = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
nx.set_edge_attributes(G, edge_bc, "betweenness")
bc_arr = np.array(list(edge_bc.values()))
print(f"  BC range: {bc_arr.min():.6f} – {bc_arr.max():.6f}")

path_rows = [
    {"geometry":   LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                               (G.nodes[v]["x"], G.nodes[v]["y"])]),
     "betweenness": bc,
     "dist_m":      G[u][v]["dist_m"],
     "dh_m":        G[u][v]["dh_m"],
     "u_chapel":    str(G.nodes[u].get("chapel_id") or ""),
     "v_chapel":    str(G.nodes[v].get("chapel_id") or "")}
    for (u, v), bc in edge_bc.items()
]
paths_gdf = gpd.GeoDataFrame(path_rows, geometry="geometry", crs=fp.crs)

paths_gdf.to_crs("EPSG:4326").to_file(
    str(gis_out / "path_network_fete.geojson"), driver="GeoJSON")
paths_gdf.to_file(str(gis_out / "path_network_fete.shp"))

fp_export = fp.copy()
for col in ["chapel_id", "direction", "raw_dir"]:
    fp_export[col] = fp_export[col].astype(str).replace({"None": "", "nan": ""})
fp_export.to_crs("EPSG:4326").to_file(
    str(gis_out / "footprints_attributed.geojson"), driver="GeoJSON")
fp_export.to_file(str(gis_out / "footprints_attributed.shp"))
doors_gdf.to_file(str(gis_out / "doors_native_approach5.shp"))
doors_pts.to_file(str(gis_out / "door_points_approach5.shp"))
print(f"GIS export complete → {gis_out}")


# %%
# CELL 12  Path network coloured by betweenness
cmap_bc = plt.cm.plasma
norm_bc = mcolors.Normalize(vmin=bc_arr.min(), vmax=bc_arr.max())

door_x = [G.nodes[n]["x"] for n in door_node_ids]
door_y = [G.nodes[n]["y"] for n in door_node_ids]

fig, ax = plt.subplots(figsize=(18, 15))
fp_dem.plot(ax=ax, color="#e8f4fb", edgecolor="#6baed6", linewidth=0.5, alpha=0.8)
for _, row in paths_gdf.iterrows():
    xs, ys = row.geometry.xy
    lw = 0.4 + 3.5 * row["betweenness"] / max(bc_arr.max(), 1e-9)
    ax.plot(xs, ys, color=cmap_bc(norm_bc(row["betweenness"])),
            linewidth=lw, alpha=0.80, zorder=3)
ax.scatter(door_x, door_y, s=12, c="#ff6b6b", zorder=6,
           edgecolors="#900000", linewidths=0.3)
shared._draw_chapel_ids(ax, fp_dem, fontsize=3.5, color="#222")
sm4 = plt.cm.ScalarMappable(cmap=cmap_bc, norm=norm_bc)
plt.colorbar(sm4, ax=ax, label="Edge Betweenness Centrality", shrink=0.5)
ax.set_title("FETE Pedestrian Corridors — Betweenness Centrality",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.tight_layout()
shared.save_fig(FIGS, "11_path_network_betweenness.png",
                extra_formats=(".svg", ".pdf"))
plt.show()


# %%
# CELL 13  Full composite map: DEM + footprints + doors + FETE paths
paths_dem = (paths_gdf.to_crs(dem["crs"])
             if paths_gdf.crs != dem["crs"] else paths_gdf.copy())

fig, ax = plt.subplots(figsize=(18, 14))
ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
          cmap="terrain", alpha=0.75, vmin=dem["e_min"], vmax=dem["e_max"],
          rasterized=True)
ax.imshow(hs, extent=dem["extent"], origin="upper",
          cmap="gray", alpha=0.30, rasterized=True)
fp_dem.plot(ax=ax, color="none", edgecolor="#1a6ea8", linewidth=0.7, alpha=0.8)
for _, row in paths_dem.iterrows():
    xs, ys = row.geometry.xy
    lw = 0.4 + 4.0 * row["betweenness"] / max(bc_arr.max(), 1e-9)
    ax.plot(xs, ys, color=cmap_bc(norm_bc(row["betweenness"])),
            linewidth=lw, alpha=0.85, zorder=4)
for _, row in doors_dem.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color=shared.DIR_CLR.get(row["direction"], "#aaa"),
            linewidth=2.5, zorder=5)
shared._draw_red_dot_no_dir(ax, fp_dem, s=25)
sm5 = plt.cm.ScalarMappable(cmap="terrain",
      norm=mcolors.Normalize(vmin=dem["e_min"], vmax=dem["e_max"]))
plt.colorbar(sm5, ax=ax, label="Elevation (m)", shrink=0.5)
sm6 = plt.cm.ScalarMappable(cmap=cmap_bc, norm=norm_bc)
plt.colorbar(sm6, ax=ax, label="Path Betweenness", shrink=0.5, location="left")
ax.legend(
    handles=shared.direction_legend_patches() + [
        mpatches.Patch(color="red", label="No direction")],
    loc="lower right", fontsize=9, framealpha=0.9)
ax.set_title("Full Composite — DEM + Footprints + Doors + FETE Path Network",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.tight_layout()
shared.save_fig(FIGS, "12_composite_full.png", extra_formats=(".svg", ".pdf"))
plt.show()


# %%
# CELL 14  Node statistics: degree vs betweenness scatter
node_bc  = nx.betweenness_centrality(G, weight="weight", normalized=True)
deg_dict = dict(G.degree())

node_stats = [
    {"chapel_id":   str(G.nodes[n].get("chapel_id") or ""),
     "direction":   str(G.nodes[n].get("direction")  or ""),
     "is_door":     G.nodes[n]["is_door"],
     "degree":      deg_dict[n],
     "betweenness": node_bc[n],
     "elevation_m": G.nodes[n].get("elev", np.nan)}
    for n in G.nodes() if G.nodes[n]["is_door"]
]
node_df = pd.DataFrame(node_stats).sort_values("betweenness", ascending=False)
node_df.to_csv(shared.OUT / "node_statistics_fete.csv", index=False)
print("Top 20 door nodes by betweenness:")
print(node_df.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(node_df["degree"], node_df["betweenness"],
                c=node_df["elevation_m"], cmap="viridis",
                s=40, alpha=0.7, edgecolors="white", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="Elevation (m)")
ax.set_xlabel("Node Degree"); ax.set_ylabel("Betweenness Centrality")
ax.set_title("Door Nodes — Degree vs Betweenness (elevation coloured)", fontsize=13)
for _, r in node_df.head(10).iterrows():
    if r["chapel_id"] and r["chapel_id"] != "None":
        ax.annotate(str(r["chapel_id"]), (r["degree"], r["betweenness"]), fontsize=7)
plt.tight_layout()
shared.save_fig(FIGS, "13_node_degree_betweenness.png")
plt.show()


# %%
# CELL 15  Shortest paths from Chapel 180 (Main Church)
TARGET    = "180"
target_id = f"door_{TARGET}"

if target_id in G.nodes:
    sp_len   = nx.single_source_dijkstra_path_length(G, target_id, weight="weight")
    sp_paths = nx.single_source_dijkstra_path(G, target_id, weight="weight")

    door_targets = [n for n in door_node_ids if n in sp_len]
    print(f"Chapel {TARGET}: {len(door_targets)} reachable chapel doors")

    drawn, sp_rows = set(), []
    for dst in door_targets:
        for a, b in zip(sp_paths[dst][:-1], sp_paths[dst][1:]):
            key = (min(str(a), str(b)), max(str(a), str(b)))
            if key not in drawn:
                drawn.add(key)
                sp_rows.append({
                    "geometry":  LineString([(G.nodes[a]["x"], G.nodes[a]["y"]),
                                             (G.nodes[b]["x"], G.nodes[b]["y"])]),
                    "path_cost": sp_len.get(dst, np.nan),
                })

    sp_gdf = gpd.GeoDataFrame(sp_rows, geometry="geometry", crs=fp.crs)
    sp_dem = (sp_gdf.to_crs(dem["crs"])
              if sp_gdf.crs != dem["crs"] else sp_gdf)

    fig, ax = plt.subplots(figsize=(18, 15))
    fp_dem.plot(ax=ax, color="#e8f4fb", edgecolor="#6baed6",
                linewidth=0.7, alpha=0.9)
    sp_dem.plot(ax=ax, color="#e63946", linewidth=1.5, alpha=0.8,
                label=f"FETE Paths from Chapel {TARGET}")
    ax.scatter([G.nodes[target_id]["x"]], [G.nodes[target_id]["y"]],
               s=200, c="gold", zorder=10, edgecolors="black",
               linewidths=1.5, label=f"Chapel {TARGET}")
    shared._draw_chapel_ids(ax, fp_dem, fontsize=3.5, color="#111")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(f"FETE Shortest Paths from Chapel {TARGET} (Main Church)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    shared.save_fig(FIGS, f"14_shortest_paths_from_{TARGET}.png",
                    extra_formats=(".svg",))
    plt.show()
else:
    print(f"Chapel {TARGET} door not found in the graph.")


# %%
# CELL 16  Ultra high-resolution master zoomable map (300 DPI)
print("Generating ultra high-resolution zoomable master map …")

fig, ax = plt.subplots(figsize=(28, 24))
ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
          cmap="terrain", alpha=0.75, vmin=dem["e_min"], vmax=dem["e_max"],
          rasterized=True)
ax.imshow(hs, extent=dem["extent"], origin="upper",
          cmap="gray", alpha=0.30, rasterized=True)
fp_dem.plot(ax=ax, color="#f8fbfe", edgecolor="#08457e", linewidth=0.8, alpha=0.85)
for _, row in paths_dem.iterrows():
    xs, ys = row.geometry.xy
    lw = 0.5 + 5.0 * row["betweenness"] / max(bc_arr.max(), 1e-9)
    ax.plot(xs, ys, color=cmap_bc(norm_bc(row["betweenness"])),
            linewidth=lw, alpha=0.88, zorder=4)
for _, row in doors_dem.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color=shared.DIR_CLR.get(row["direction"], "#aaa"),
            linewidth=3.0, zorder=5)
shared._draw_chapel_ids(ax, fp_dem, fontsize=5.5, color="#0b1d3a",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  alpha=0.65, ec="none"))
sm_elev = plt.cm.ScalarMappable(
    cmap="terrain",
    norm=mcolors.Normalize(vmin=dem["e_min"], vmax=dem["e_max"]))
plt.colorbar(sm_elev, ax=ax, label="DEM Elevation (m)", shrink=0.45, pad=0.01)
sm_bc = plt.cm.ScalarMappable(cmap=cmap_bc, norm=norm_bc)
plt.colorbar(sm_bc, ax=ax, label="FETE Corridor Betweenness",
             shrink=0.45, location="left", pad=0.01)
ax.legend(handles=shared.direction_legend_patches(),
          loc="lower right", fontsize=12, framealpha=0.95)
ax.set_title(
    "El-Bagawat Necropolis — Master High-Resolution FETE Pedestrian Network",
    fontsize=20, fontweight="bold", pad=15)
ax.set_xlabel("Easting (m)", fontsize=14)
ax.set_ylabel("Northing (m)", fontsize=14)
plt.tight_layout()
shared.save_fig(FIGS, "15_master_highres_zoomable_map.png", dpi=300,
                extra_formats=(".svg", ".pdf"))
plt.show()

# Zoom detail maps
zoom_regions = [
    ("16a_zoom_main_church_cluster",
     [254130, 254240, 2821130, 2821240],
     "Main Church Cluster (Chapel 180 & Surroundings)"),
    ("16b_zoom_north_necropolis",
     [254140, 254260, 2821240, 2821350],
     "North Necropolis Cluster"),
    ("16c_zoom_central_corridors",
     [254110, 254220, 2821180, 2821290],
     "Central Necropolis Navigation Corridors"),
]

print("Generating digital zoom detail maps …")
for fname, (xmin, xmax, ymin, ymax), ztitle in zoom_regions:
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
              cmap="terrain", alpha=0.75,
              vmin=dem["e_min"], vmax=dem["e_max"], rasterized=True)
    ax.imshow(hs, extent=dem["extent"], origin="upper",
              cmap="gray", alpha=0.30, rasterized=True)
    fp_dem.plot(ax=ax, color="#f8fbfe", edgecolor="#08457e",
                linewidth=1.2, alpha=0.88)
    for _, row in paths_dem.iterrows():
        xs, ys = row.geometry.xy
        if not (min(xs) > xmax or max(xs) < xmin or
                min(ys) > ymax or max(ys) < ymin):
            lw = 1.0 + 6.0 * row["betweenness"] / max(bc_arr.max(), 1e-9)
            ax.plot(xs, ys, color=cmap_bc(norm_bc(row["betweenness"])),
                    linewidth=lw, alpha=0.90, zorder=4)
    for _, row in doors_dem.iterrows():
        xs, ys = row.geometry.xy
        if xmin <= xs[0] <= xmax and ymin <= ys[0] <= ymax:
            ax.plot(xs, ys, color=shared.DIR_CLR.get(row["direction"], "#aaa"),
                    linewidth=3.5, zorder=5)
    for _, row in fp_dem.iterrows():
        cid = row["chapel_id"]
        if cid is not None and str(cid) not in ("None", "nan", ""):
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                ax.text(cx, cy, str(cid),
                        ha="center", va="center", fontsize=8,
                        color="#0b1d3a", fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  alpha=0.75, ec="#08457e", lw=0.5))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(f"Digital Zoom: {ztitle}", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    shared.save_fig(FIGS, f"{fname}.png", dpi=300, extra_formats=(".svg",))
    plt.show()