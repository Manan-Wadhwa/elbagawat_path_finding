"""
bagawat_pipeline.py  –  El-Bagawat k-NN obstacle-free pedestrian network
=========================================================================
Run as a Jupyter notebook (#%% cells) or as a plain Python script.
Each #%% marker is a separate notebook cell.

Algorithm:
  1–9   Shared data loading (footprints, Excel, DXF, doors, DEM)  ← via shared.py
  10    Build obstacle-free k-NN proximity graph
  11    Visualise graph (edges coloured by distance)
  12    Edge betweenness centrality + GIS export
  13    Path network betweenness visualisation
  14    Full composite map (DEM + footprints + doors + paths)
  15    Node statistics (degree vs betweenness scatter)
  16    Shortest paths from Chapel 180 (Main Church)
  17    Ultra high-resolution zoomable master map
  18    Interactive HTML explorer (self-contained Leaflet app)
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
import json

# All figures for this pipeline go here
FIGS = shared.OUT / "figures" / "bagawat_pipeline"
FIGS.mkdir(parents=True, exist_ok=True)

shared.verify_paths()


# %%
# CELL 1  Load shapefile
fp = shared.load_footprints()
shared.plot_raw_footprints(fp, FIGS)


# %%
# CELL 2  Load Excel, normalise directions, audit outliers
df          = shared.load_excel_directions()
dir_counts  = shared.print_direction_audit(df)
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
print("-> crosswalk.csv")

shared.plot_bipartite_matching(fp, dxf_labels, crosswalk, H_rough, FIGS)
shared.plot_attribution_map(fp, n_with_dir, n_no_dir, n_unlabelled, FIGS)


# %%
# CELL 5  Approach-5 door placement
doors_gdf, doors_pts = shared.place_doors(fp)

# Export to vector_gis/
gis_out = shared.OUT / "vector_gis"
gis_out.mkdir(exist_ok=True)
doors_gdf.to_crs("EPSG:4326").to_file(str(gis_out / "doors_native_approach5.geojson"), driver="GeoJSON")
doors_pts.to_crs("EPSG:4326").to_file(str(gis_out / "door_points_approach5.geojson"),  driver="GeoJSON")
print("-> doors_native_approach5.geojson  door_points_approach5.geojson")

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
# CELL 8  Build obstacle-free k-NN pedestrian navigation graph
print("Building obstacle-free outdoor pedestrian navigation network ...")

b_union = (fp.geometry.union_all()
           if hasattr(fp.geometry, "union_all") else fp.geometry.unary_union)

# Door nodes
door_pts_list = [
    (r.geometry.x, r.geometry.y, r["chapel_id"], r["direction"],
     r.get("elevation_m", 0.0) or 0.0)
    for _, r in doors_pts.iterrows()
]

# Waypoints: corner-offset points that lie outside buildings
waypoints = []
for poly in fp.geometry:
    if poly is None or poly.is_empty:
        continue
    c = np.array([poly.centroid.x, poly.centroid.y])
    for pt in poly.simplify(0.5).exterior.coords:
        v    = np.array(pt)
        out  = v - c
        norm = np.linalg.norm(out)
        if norm > 0:
            wp = v + (out / norm) * 1.5
            if b_union.distance(Point(wp)) >= 0.1:
                waypoints.append(tuple(wp))

# Thin waypoints (min 3.5 m spacing)
arr  = np.array(waypoints)
keep = [0]
for i in range(1, len(arr)):
    if np.min(np.hypot(arr[keep, 0] - arr[i, 0],
                       arr[keep, 1] - arr[i, 1])) >= 3.5:
        keep.append(i)
wp_pts = [(arr[i, 0], arr[i, 1], None, None, 0.0) for i in keep]

all_pts = door_pts_list + wp_pts
coords  = np.array([(p[0], p[1]) for p in all_pts])
kd      = cKDTree(coords)

G = nx.Graph()
for i, p in enumerate(all_pts):
    G.add_node(i, x=p[0], y=p[1],
               is_door=(i < len(door_pts_list)),
               chapel_id=p[2], direction=p[3], elev=p[4])

SLOPE_W = 0.10
# Primary pass: connect all pairs within 35 m that don't cross buildings
for u, v in kd.query_pairs(r=35.0):
    ls = LineString([coords[u], coords[v]])
    if b_union.intersection(ls).length <= 0.25:
        d  = float(np.hypot(coords[u][0]-coords[v][0],
                            coords[u][1]-coords[v][1]))
        dh = abs(G.nodes[u].get("elev", 0.0) - G.nodes[v].get("elev", 0.0))
        G.add_edge(u, v, weight=d*(1 + SLOPE_W*dh), dist_m=d, dh_m=float(dh))

# Secondary pass: ensure every door is connected (relax threshold to 4 m)
for i in range(len(door_pts_list)):
    if len(G[i]) == 0:
        _, idxs = kd.query(coords[i], k=200)
        for j in idxs[1:]:
            ls = LineString([coords[i], coords[j]])
            if b_union.intersection(ls).length <= 4.0:
                d  = float(np.hypot(coords[i][0]-coords[j][0],
                                    coords[i][1]-coords[j][1]))
                dh = abs(G.nodes[i].get("elev", 0.0) - G.nodes[j].get("elev", 0.0))
                G.add_edge(i, j, weight=d*(1 + SLOPE_W*dh),
                           dist_m=d, dh_m=float(dh))
                break

door_nodes     = [n for n in G.nodes() if G.nodes[n]["is_door"]]
connected_doors = sum(1 for n in door_nodes if len(G[n]) > 0)
print(f"Graph:  {G.number_of_nodes()} nodes  "
      f"({len(door_pts_list)} doors + {len(wp_pts)} waypoints),  "
      f"{G.number_of_edges()} edges")
print(f"Connected doors: {connected_doors} / {len(door_nodes)}")


# %%
# CELL 9  Visualise proximity graph (edges coloured by distance)
edge_dists = [G[u][v]["dist_m"] for u, v in G.edges()]
w_max = max(edge_dists) if edge_dists else 1.0

door_x = [G.nodes[n]["x"] for n in door_nodes]
door_y = [G.nodes[n]["y"] for n in door_nodes]

fig, ax = plt.subplots(figsize=(18, 15))
fp_dem.plot(ax=ax, color="#e8f4fb", edgecolor="#6baed6", linewidth=0.7, alpha=0.9)
for u, v in G.edges():
    xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
    xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
    ax.plot([xu, xv], [yu, yv],
            color=plt.cm.viridis(1.0 - G[u][v]["dist_m"] / w_max),
            linewidth=0.8, alpha=0.6, zorder=3)
ax.scatter(door_x, door_y, s=12, c="#ff6b6b", zorder=5, linewidths=0)
for n in door_nodes:
    cid = G.nodes[n].get("chapel_id")
    if cid:
        ax.text(G.nodes[n]["x"], G.nodes[n]["y"], cid,
                ha="center", va="bottom", fontsize=3.5, color="#111")
sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=mcolors.Normalize(0, w_max))
plt.colorbar(sm, ax=ax, label="Edge distance (m)", shrink=0.5)
ax.set_title(f"Obstacle-Free Navigation Network — {G.number_of_nodes()} nodes, "
             f"{G.number_of_edges()} edges",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.tight_layout()
shared.save_fig(FIGS, "10_proximity_graph.png", extra_formats=(".svg",))
plt.show()


# %%
# CELL 10  Edge betweenness centrality + GIS export
print("Computing edge betweenness centrality ...")
edge_bc = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
nx.set_edge_attributes(G, edge_bc, "betweenness")
bc_arr = np.array(list(edge_bc.values()))
print(f"  BC range: {bc_arr.min():.6f} - {bc_arr.max():.6f}")

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

# GIS export
paths_gdf.to_crs("EPSG:4326").to_file(
    str(gis_out / "path_network_obstacle_free.geojson"), driver="GeoJSON")
paths_gdf.to_file(str(gis_out / "path_network_obstacle_free.shp"))

fp_export = fp.copy()
for col in ["chapel_id", "direction", "raw_dir"]:
    fp_export[col] = fp_export[col].astype(str).replace({"None": "", "nan": ""})
fp_export.to_crs("EPSG:4326").to_file(
    str(gis_out / "footprints_attributed.geojson"), driver="GeoJSON")
fp_export.to_file(str(gis_out / "footprints_attributed.shp"))
doors_gdf.to_file(str(gis_out / "doors_native_approach5.shp"))
doors_pts.to_file(str(gis_out / "door_points_approach5.shp"))
print(f"GIS export complete -> {gis_out}")


# %%
# CELL 11  Path network coloured by betweenness
cmap_bc = plt.cm.plasma
norm_bc = mcolors.Normalize(vmin=bc_arr.min(), vmax=bc_arr.max())

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
ax.set_title("Obstacle-Free Pedestrian Corridors — Betweenness Centrality",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.tight_layout()
shared.save_fig(FIGS, "11_path_network_betweenness.png",
                extra_formats=(".svg", ".pdf"))
plt.show()


# %%
# CELL 12  Full composite map: DEM + footprints + doors + paths
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
ax.set_title("Full Composite — DEM + Footprints + Doors + Obstacle-Free Path Network",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.tight_layout()
shared.save_fig(FIGS, "12_composite_full.png", extra_formats=(".svg", ".pdf"))
plt.show()


# %%
# CELL 13  Node statistics: degree vs betweenness scatter
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
node_df.to_csv(shared.OUT / "node_statistics.csv", index=False)
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
# CELL 14  Shortest paths from Chapel 180 (Main Church)
TARGET     = "180"
src_nodes  = [n for n in G.nodes() if G.nodes[n].get("chapel_id") == TARGET]

if src_nodes:
    src      = src_nodes[0]
    sp_len   = nx.single_source_dijkstra_path_length(G, src, weight="weight")
    sp_paths = nx.single_source_dijkstra_path(G, src, weight="weight")

    door_targets = [n for n in G.nodes()
                    if G.nodes[n]["is_door"] and n in sp_len]
    print(f"Chapel {TARGET}: {len(door_targets)} reachable chapel doors")

    # Build GeoDataFrame of path segments (deduplicated)
    drawn, sp_rows = set(), []
    for dst in door_targets:
        for a, b in zip(sp_paths[dst][:-1], sp_paths[dst][1:]):
            key = (min(a, b), max(a, b))
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
                label=f"Obstacle-Free Paths from {TARGET}")
    ax.scatter([G.nodes[src]["x"]], [G.nodes[src]["y"]],
               s=200, c="gold", zorder=10, edgecolors="black",
               linewidths=1.5, label=f"Chapel {TARGET}")
    shared._draw_chapel_ids(ax, fp_dem, fontsize=3.5, color="#111")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(f"Obstacle-Free Shortest Paths from Chapel {TARGET} (Main Church)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    shared.save_fig(FIGS, f"14_shortest_paths_from_{TARGET}.png",
                    extra_formats=(".svg",))
    plt.show()
else:
    print(f"Chapel {TARGET} not found in the graph.")


# %%
# CELL 15  Ultra high-resolution master zoomable map (300 DPI)
print("Generating ultra high-resolution zoomable master map ...")

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
plt.colorbar(sm_bc, ax=ax, label="Pedestrian Corridor Betweenness",
             shrink=0.45, location="left", pad=0.01)
ax.legend(handles=shared.direction_legend_patches(),
          loc="lower right", fontsize=12, framealpha=0.95)
ax.set_title(
    "El-Bagawat Necropolis — Master High-Resolution Obstacle-Free Pedestrian Network",
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

print("Generating digital zoom detail maps ...")
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