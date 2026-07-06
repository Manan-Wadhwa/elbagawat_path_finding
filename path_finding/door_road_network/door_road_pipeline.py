# %%
# CELL 0  Imports, paths, and configuration
import sys
import os
import warnings
from pathlib import Path

# Add parent directory to sys.path to import shared.py
shared_dir = str(Path(__file__).resolve().parents[1])
if shared_dir not in sys.path:
    sys.path.insert(0, shared_dir)
import shared

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN

# Outputs directories
FIGS = shared.OUT / "figures" / "door_road_network"
FIGS.mkdir(parents=True, exist_ok=True)
GIS_OUT = shared.OUT / "vector_gis"
GIS_OUT.mkdir(parents=True, exist_ok=True)

shared.verify_paths()

# %%
# CELL 1  Load shapefile footprints and generate doors
print("Loading building footprints and database...")
fp = shared.load_footprints()
df = shared.load_excel_directions()
dxf_labels = shared.load_dxf_labels()

print("Resolving affine alignment and placing doors...")
H_rough = shared.estimate_rough_affine(dxf_labels, fp)
crosswalk = shared.bipartite_label_match(dxf_labels, H_rough, fp)
fp_attributed, _, _, _, _ = shared.attribute_footprints(fp, crosswalk, df)
doors_gdf, doors_pts = shared.place_doors(fp_attributed)

# %%
# CELL 2  Project road coordinates in front of doors
print("Projecting street points 3.5 meters in front of door entrances...")
road_points = []
for idx, row in doors_gdf.iterrows():
    poly = fp_attributed.loc[row["shp_idx"]].geometry
    dc = row["direction"]
    mx, my, nx_, ny_, el = shared.best_wall(poly, dc)
    
    # normal vector (nx_, ny_) points outward from the wall
    door_pt = row["door_pt"]
    rx = door_pt.x + nx_ * 3.5
    ry = door_pt.y + ny_ * 3.5
    
    road_points.append({
        "chapel_id": row["chapel_id"],
        "direction": dc,
        "x": rx,
        "y": ry,
        "door_x": door_pt.x,
        "door_y": door_pt.y
    })
    
df_road = pd.DataFrame(road_points)
print(f"  Projected {len(df_road)} points in front of doors.")

# %%
# CELL 3  Cluster E-W and N-S corridors using DBSCAN
print("Clustering door alignments into street corridors...")

def split_by_gap_and_length(group, coordinate_name="proj", max_gap=12.0, max_length=35.0):
    sorted_group = group.sort_values(by=coordinate_name)
    subgroups = []
    current_subgroup = []
    start_val = None
    for idx, row in sorted_group.iterrows():
        val = row[coordinate_name]
        if not current_subgroup:
            current_subgroup.append(row)
            start_val = val
        else:
            prev_val = current_subgroup[-1][coordinate_name]
            # Split if gap is too large OR if the segment length exceeds max_length
            if (val - prev_val > max_gap) or (val - start_val > max_length):
                subgroups.append(pd.DataFrame(current_subgroup))
                current_subgroup = [row]
                start_val = val
            else:
                current_subgroup.append(row)
    if current_subgroup:
        subgroups.append(pd.DataFrame(current_subgroup))
    return subgroups

roads_list = []

# East-West Streets (doors facing North or South)
ew_mask = df_road["direction"].isin(["N", "S"])
df_ew = df_road[ew_mask].copy()
if len(df_ew) > 0:
    # 2D DBSCAN to find spatial street clusters
    coords_ew = df_ew[["x", "y"]].values
    db_ew = DBSCAN(eps=10.0, min_samples=1).fit(coords_ew)
    df_ew["cluster"] = db_ew.labels_
    
    for cid, group in df_ew.groupby("cluster"):
        pts = group[["x", "y"]].values
        mean = pts.mean(axis=0)
        
        # Determine principal axis direction vector using SVD
        if len(group) > 1 and np.std(pts, axis=0).max() > 0.1:
            centered = pts - mean
            uu, dd, vv = np.linalg.svd(centered)
            v = vv[0]  # Unit direction vector
        else:
            v = np.array([1.0, 0.0])  # Default horizontal
            
        # Project points onto direction vector to get 1D coordinates along the street
        group["proj"] = (pts - mean).dot(v)
        
        # Split by gap and maximum length
        subgroups = split_by_gap_and_length(group, "proj", max_gap=12.0, max_length=35.0)
        
        for sub_idx, sub_group in enumerate(subgroups):
            sub_pts = sub_group[["x", "y"]].values
            sub_mean = sub_pts.mean(axis=0)
            
            if len(sub_group) > 1 and np.std(sub_pts, axis=0).max() > 0.1:
                sub_centered = sub_pts - sub_mean
                uu, dd, vv = np.linalg.svd(sub_centered)
                sub_v = vv[0]
                proj_vals = sub_centered.dot(sub_v)
                min_p = proj_vals.min()
                max_p = proj_vals.max()
                
                # Apply 1.0m end extensions
                p1 = sub_mean + (min_p - 1.0) * sub_v
                p2 = sub_mean + (max_p + 1.0) * sub_v
            else:
                # Single door or zero variance: draw horizontal segment
                p1 = sub_mean - 2.0 * np.array([1.0, 0.0])
                p2 = sub_mean + 2.0 * np.array([1.0, 0.0])
                
            geom = LineString([p1, p2])
            
            # Check if double-sided or includes 116-120
            has_n = "N" in sub_group["direction"].values
            has_s = "S" in sub_group["direction"].values
            has_116_120 = any(str(cid_val) in ["116", "117", "118", "119", "120"] for cid_val in sub_group["chapel_id"].values)
            is_double = (has_n and has_s) or has_116_120
            
            roads_list.append({
                "geometry": geom,
                "orientation": "E-W",
                "is_double": is_double,
                "cluster_id": f"EW_{cid}_{sub_idx}",
                "avg_coord": avg_y if 'avg_y' in locals() else sub_mean[1],
                "min_val": min_x if 'min_x' in locals() else min(p1[0], p2[0]),
                "max_val": max_x if 'max_x' in locals() else max(p1[0], p2[0])
            })

# North-South Streets (doors facing East or West)
ns_mask = df_road["direction"].isin(["E", "W"])
df_ns = df_road[ns_mask].copy()
if len(df_ns) > 0:
    # 2D DBSCAN to find spatial street clusters
    coords_ns = df_ns[["x", "y"]].values
    db_ns = DBSCAN(eps=10.0, min_samples=1).fit(coords_ns)
    df_ns["cluster"] = db_ns.labels_
    
    for cid, group in df_ns.groupby("cluster"):
        pts = group[["x", "y"]].values
        mean = pts.mean(axis=0)
        
        # Determine principal axis direction vector using SVD
        if len(group) > 1 and np.std(pts, axis=0).max() > 0.1:
            centered = pts - mean
            uu, dd, vv = np.linalg.svd(centered)
            v = vv[0]
        else:
            v = np.array([0.0, 1.0])  # Default vertical
            
        # Project points onto direction vector
        group["proj"] = (pts - mean).dot(v)
        
        # Split by gap and maximum length
        subgroups = split_by_gap_and_length(group, "proj", max_gap=12.0, max_length=35.0)
        
        for sub_idx, sub_group in enumerate(subgroups):
            sub_pts = sub_group[["x", "y"]].values
            sub_mean = sub_pts.mean(axis=0)
            
            if len(sub_group) > 1 and np.std(sub_pts, axis=0).max() > 0.1:
                sub_centered = sub_pts - sub_mean
                uu, dd, vv = np.linalg.svd(sub_centered)
                sub_v = vv[0]
                proj_vals = sub_centered.dot(sub_v)
                min_p = proj_vals.min()
                max_p = proj_vals.max()
                
                # Apply 1.0m end extensions
                p1 = sub_mean + (min_p - 1.0) * sub_v
                p2 = sub_mean + (max_p + 1.0) * sub_v
            else:
                # Single door or zero variance: draw vertical segment
                p1 = sub_mean - 2.0 * np.array([0.0, 1.0])
                p2 = sub_mean + 2.0 * np.array([0.0, 1.0])
                
            geom = LineString([p1, p2])
            
            # Check if double-sided or includes 116-120
            has_e = "E" in sub_group["direction"].values
            has_w = "W" in sub_group["direction"].values
            has_116_120 = any(str(cid_val) in ["116", "117", "118", "119", "120"] for cid_val in sub_group["chapel_id"].values)
            is_double = (has_e and has_w) or has_116_120
            
            roads_list.append({
                "geometry": geom,
                "orientation": "N-S",
                "is_double": is_double,
                "cluster_id": f"NS_{cid}_{sub_idx}",
                "avg_coord": avg_x if 'avg_x' in locals() else sub_mean[0],
                "min_val": min_y if 'min_y' in locals() else min(p1[1], p2[1]),
                "max_val": max_y if 'max_val' in locals() else max(p1[1], p2[1])
            })

print(f"  Created {len(roads_list)} primary street segments.")

# %%
# CELL 4  Solve Intersections and Connection Pathways
print("Computing street intersections and connecting close endpoints...")
intersections_list = []
connections_list = []

# Detect direct intersections
for i in range(len(roads_list)):
    for j in range(i+1, len(roads_list)):
        r1 = roads_list[i]["geometry"]
        r2 = roads_list[j]["geometry"]
        if r1.intersects(r2):
            pt = r1.intersection(r2)
            if isinstance(pt, Point):
                intersections_list.append({
                    "geometry": pt,
                    "type": "intersection",
                    "roads": f"{roads_list[i]['cluster_id']}-{roads_list[j]['cluster_id']}"
                })

# Connect close endpoints to nearby streets (threshold = 12.0m)
for i, r1 in enumerate(roads_list):
    geom1 = r1["geometry"]
    coords1 = list(geom1.coords)
    endpoints1 = [Point(coords1[0]), Point(coords1[-1])]
    
    for j, r2 in enumerate(roads_list):
        if i == j:
            continue
        geom2 = r2["geometry"]
        
        for ep in endpoints1:
            dist = ep.distance(geom2)
            if 0.1 < dist < 12.0:
                closest_pt = geom2.interpolate(geom2.project(ep))
                conn_geom = LineString([ep, closest_pt])
                connections_list.append({
                    "geometry": conn_geom,
                    "u_road": r1["cluster_id"],
                    "v_road": r2["cluster_id"],
                    "dist_m": round(dist, 3)
                })
                # Add nodes at both ends of the connection
                intersections_list.append({
                    "geometry": ep,
                    "type": "connection_endpoint",
                    "roads": r1["cluster_id"]
                })
                intersections_list.append({
                    "geometry": closest_pt,
                    "type": "connection_junction",
                    "roads": r2["cluster_id"]
                })

# %%
# CELL 5  Save GIS outputs
roads_gdf = gpd.GeoDataFrame(roads_list, geometry="geometry", crs=fp.crs)
roads_gdf.to_crs("EPSG:4326").to_file(str(GIS_OUT / "door_roads_network.geojson"), driver="GeoJSON")

if len(connections_list) > 0:
    conn_gdf = gpd.GeoDataFrame(connections_list, geometry="geometry", crs=fp.crs)
    conn_gdf.to_crs("EPSG:4326").to_file(str(GIS_OUT / "door_roads_connections.geojson"), driver="GeoJSON")
else:
    conn_gdf = gpd.GeoDataFrame(columns=["geometry"])

if len(intersections_list) > 0:
    raw_nodes_gdf = gpd.GeoDataFrame(intersections_list, geometry="geometry", crs=fp.crs)
    coords_nodes = np.array([[pt.x, pt.y] for pt in raw_nodes_gdf.geometry])
    
    # 2D DBSCAN with eps=12.0m: grouping nearby connection points
    db_nodes = DBSCAN(eps=12.0, min_samples=1).fit(coords_nodes)
    raw_nodes_gdf["cluster"] = db_nodes.labels_
    
    hubs_list = []
    for cid, group in raw_nodes_gdf.groupby("cluster"):
        centroid = Point(group.geometry.x.mean(), group.geometry.y.mean())
        weight = len(group)
        is_major_intersection = any(t == "intersection" for t in group["type"].values)
        
        # Only keep as a node if weight >= 2 or if it's a direct crossing intersection
        # This filters out isolated endpoints and makes the map way cleaner!
        if weight >= 2 or is_major_intersection:
            hubs_list.append({
                "geometry": centroid,
                "weight": weight,
                "cluster_id": f"HUB_{cid}",
                "is_major": weight >= 3 or is_major_intersection
            })
            
    if len(hubs_list) > 0:
        intersections_gdf = gpd.GeoDataFrame(hubs_list, geometry="geometry", crs=fp.crs)
        intersections_gdf.to_crs("EPSG:4326").to_file(str(GIS_OUT / "door_roads_intersections.geojson"), driver="GeoJSON")
    else:
        intersections_gdf = gpd.GeoDataFrame(columns=["geometry", "weight"])
else:
    intersections_gdf = gpd.GeoDataFrame(columns=["geometry"])

print(f"Exported road layers to {GIS_OUT}:")
print("  -> door_roads_network.geojson")
print("  -> door_roads_connections.geojson")
print("  -> door_roads_intersections.geojson")

# %%
# CELL 6  Plot and Save Visualization
print("Generating door-based road network map...")
dem = shared.load_dem()
hs = shared.hillshade(dem["disp"])

fig, ax = plt.subplots(figsize=(16, 14))

# Underlay DEM hillshade (zorder=0)
ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
          cmap="terrain", alpha=0.6, vmin=dem["e_min"], vmax=dem["e_max"], rasterized=True, zorder=0)
ax.imshow(hs, extent=dem["extent"], origin="upper",
          cmap="gray", alpha=0.3, rasterized=True, zorder=1)

# Plot congregation hubs (zorder=2, translucent, scaled back)
if len(intersections_gdf) > 0 and not intersections_gdf.empty:
    sizes = np.minimum(350, 60 + 35 * intersections_gdf["weight"])
    intersections_gdf.plot(ax=ax, color="#8e44ad", markersize=sizes, alpha=0.35, edgecolor="none", label="Congregation Hub", zorder=2)

# Plot building footprints (zorder=3)
fp_attributed.plot(ax=ax, color="#f5f5f5", edgecolor="#999999", linewidth=0.5, alpha=0.85, zorder=3)

# Plot single-sided street segments (blue, zorder=4)
single_roads = roads_gdf[~roads_gdf["is_double"]]
if len(single_roads) > 0:
    single_roads.plot(ax=ax, color="#3498db", linewidth=2.0, label="Single-Sided Road (One-Way)", zorder=4)

# Plot double-sided street segments (orange/thicker, zorder=4)
double_roads = roads_gdf[roads_gdf["is_double"]]
if len(double_roads) > 0:
    double_roads.plot(ax=ax, color="#e67e22", linewidth=5.0, label="Double-Sided Road (Two-Way)", zorder=4)

# Plot connection links (dashed blue, zorder=4)
if len(conn_gdf) > 0:
    conn_gdf.plot(ax=ax, color="#2980b9", linestyle="--", linewidth=1.5, label="Connection Pathway", zorder=4)

# Plot door segments (zorder=5)
for _, row in doors_gdf.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color=shared.DIR_CLR.get(row["direction"], "#aaa"), linewidth=2.0, zorder=5)

# Plot projected door road points (zorder=5)
ax.scatter(df_road["x"], df_road["y"], color="red", s=10, alpha=0.6, label="Door Road Points", zorder=5)

# Draw chapel IDs (zorder=6)
shared._draw_chapel_ids(ax, fp_attributed, fontsize=4.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0))

# Zoom axes to building footprints with 50m buffer
xmin, ymin, xmax, ymax = fp.total_bounds
ax.set_xlim(xmin - 50, xmax + 50)
ax.set_ylim(ymin - 50, ymax + 50)

ax.set_title("El-Bagawat — Door-Based Procedural Road Network Concept", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Easting (m)", fontsize=12)
ax.set_ylabel("Northing (m)", fontsize=12)

# Custom Legend
legend_patches = [
    mpatches.Patch(color="#f5f5f5", edgecolor="#999999", label="Building Footprint"),
    plt.Line2D([0], [0], color="#3498db", lw=2, label="Single-Sided Road"),
    plt.Line2D([0], [0], color="#e67e22", lw=5, label="Double-Sided Road (Two-Way)"),
    plt.Line2D([0], [0], color="#2980b9", linestyle="--", lw=1.5, label="Connection Pathway"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8e44ad', markeredgecolor='none', alpha=0.5, markersize=10, label="Congregation Hub (Plaza Zone)"),
    plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='red', markersize=8, label="Projected Door Point")
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=10, framealpha=0.95)

plt.tight_layout()
shared.save_fig(FIGS, "21_door_roads_visualization.png", dpi=150)
plt.close()

print("=== PIPELINE RUN COMPLETE ===")
