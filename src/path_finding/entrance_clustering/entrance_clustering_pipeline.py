"""
entrance_clustering_pipeline.py - Approach 5: Entrance Clustering & Access Hub Network
=====================================================================================
Processes El-Bagawat doors, projects them outward, clusters them using DBSCAN,
and connects doors to hubs, and hubs to each other using a Gabriel graph.
"""

# %%
# CELL 0: Imports, path resolution, and configuration
import sys
from pathlib import Path

# Make shared.py importable (it lives in the parent directory)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from shapely.geometry import Point, LineString
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN

# Verify paths
shared.verify_paths()

# %%
# CELL 1: Gabriel Graph Helper Functions
def compute_gabriel_graph(points):
    """
    Compute Gabriel graph edges for a set of points.
    points: np.ndarray of shape (N, 2)
    Returns: list of tuples (u, v) representing edges
    """
    n = len(points)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]
        
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                edges.add((min(u, v), max(u, v)))
                
    gabriel_edges = []
    for u, v in edges:
        p_u = points[u]
        p_v = points[v]
        mid = (p_u + p_v) / 2.0
        r = np.linalg.norm(p_u - p_v) / 2.0
        
        is_gabriel = True
        for w in range(n):
            if w == u or w == v:
                continue
            dist_to_mid = np.linalg.norm(points[w] - mid)
            if dist_to_mid < r - 1e-5:
                is_gabriel = False
                break
        if is_gabriel:
            gabriel_edges.append((u, v))
            
    return list(gabriel_edges)

# %%
# CELL 2: Entrance to Node Network Constructor
def compute_entrance_to_node_network(doors_gdf, eps=12.0, fp=None):
    """
    DBSCAN cluster door centroids (projected outward).
    Centroid of each cluster is a street junction hub.
    Connect doors to hubs, and hubs to each other (via Gabriel or k-NN graph).
    """
    if fp is None:
        fp = shared.load_footprints()
        
    # 1. Project doors outward along their normal vectors by 3.0m to find where they lead
    projected_coords = []
    for idx, row in doors_gdf.iterrows():
        poly = fp.loc[row["shp_idx"]].geometry
        dc = row["direction"]
        mx, my, nx_, ny_, el = shared.best_wall(poly, dc)
        
        # row["door_pt"] contains the door midpoint (0.5m outward from wall)
        # We project 3.0m further along the normal vector (nx_, ny_)
        door_pt = row["door_pt"]
        proj_x = door_pt.x + nx_ * 3.0
        proj_y = door_pt.y + ny_ * 3.0
        projected_coords.append((proj_x, proj_y))
        
    doors_gdf_updated = doors_gdf.copy()
    doors_gdf_updated["proj_x"] = [p[0] for p in projected_coords]
    doors_gdf_updated["proj_y"] = [p[1] for p in projected_coords]
    
    # 2. DBSCAN cluster these projected points
    coords_arr = np.array(projected_coords)
    db = DBSCAN(eps=eps, min_samples=1).fit(coords_arr)
    doors_gdf_updated["cluster_id"] = db.labels_
    
    # 3. Centroid of each cluster is a street junction hub
    hubs_dict = {}
    for cid in set(db.labels_):
        cluster_rows = doors_gdf_updated[doors_gdf_updated["cluster_id"] == cid]
        avg_x = cluster_rows["proj_x"].mean()
        avg_y = cluster_rows["proj_y"].mean()
        hubs_dict[cid] = (avg_x, avg_y)
        
    # Build GeoDataFrame of hubs
    hubs_rows = []
    for cid, (hx, hy) in hubs_dict.items():
        cluster_rows = doors_gdf_updated[doors_gdf_updated["cluster_id"] == cid]
        chapels = cluster_rows["chapel_id"].dropna().unique().tolist()
        hubs_rows.append({
            "cluster_id": cid,
            "geometry": Point(hx, hy),
            "x": hx,
            "y": hy,
            "size": len(cluster_rows),
            "chapels": ", ".join(map(str, chapels))
        })
    hubs_gdf = gpd.GeoDataFrame(hubs_rows, geometry="geometry", crs=fp.crs)
    
    # 4. Build a graph G
    G = nx.Graph()
    
    # Add hub nodes
    for _, row in hubs_gdf.iterrows():
        cid = row["cluster_id"]
        G.add_node(f"hub_{cid}", 
                   x=row["x"], 
                   y=row["y"], 
                   type="hub", 
                   cluster_id=cid,
                   size=row["size"],
                   chapels=row["chapels"])
                   
    # Add door nodes and connect to their hubs
    for i, row in doors_gdf_updated.iterrows():
        door_pt = row["door_pt"]
        cid = row["cluster_id"]
        door_id = f"door_{i}"
        
        # Add door node
        G.add_node(door_id, 
                   x=door_pt.x, 
                   y=door_pt.y, 
                   type="door", 
                   chapel_id=row["chapel_id"],
                   direction=row["direction"],
                   cluster_id=cid)
                   
        # Connect door to its hub
        hub_coords = hubs_dict[cid]
        dist = float(np.hypot(door_pt.x - hub_coords[0], door_pt.y - hub_coords[1]))
        G.add_edge(door_id, f"hub_{cid}", 
                   edge_type="door_to_hub", 
                   weight=dist, 
                   dist_m=dist,
                   cluster_id=cid)
                   
    # 5. Connect hubs to each other using Gabriel graph
    hub_ids = list(hubs_dict.keys())
    hub_pts = np.array([hubs_dict[cid] for cid in hub_ids])
    
    if len(hub_pts) > 1:
        gabriel_edges = compute_gabriel_graph(hub_pts)
        
        # Footprint union for intersection check
        b_union = (fp.geometry.union_all()
                   if hasattr(fp.geometry, "union_all") else fp.geometry.unary_union)
                   
        for u_idx, v_idx in gabriel_edges:
            u_cid = hub_ids[u_idx]
            v_cid = hub_ids[v_idx]
            pt_u = hub_pts[u_idx]
            pt_v = hub_pts[v_idx]
            
            dist = float(np.hypot(pt_u[0] - pt_v[0], pt_u[1] - pt_v[1]))
            ls = LineString([pt_u, pt_v])
            
            intersects = b_union.intersects(ls)
            inter_len = b_union.intersection(ls).length if intersects else 0.0
            
            G.add_edge(f"hub_{u_cid}", f"hub_{v_cid}", 
                       edge_type="hub_to_hub", 
                       weight=dist, 
                       dist_m=dist,
                       intersects_building=intersects,
                       intersection_len_m=inter_len)
                       
    return G, hubs_gdf, doors_gdf_updated

# %%
# CELL 3  Load data (footprints, directions, DXF labels)
print("=== STARTING ENTRANCE CLUSTERING PIPELINE ===")
fp = shared.load_footprints()
df = shared.load_excel_directions()
dxf_labels = shared.load_dxf_labels()

# %%
# CELL 4  Hungarian matching & door placement
H_rough = shared.estimate_rough_affine(dxf_labels, fp)
crosswalk = shared.bipartite_label_match(dxf_labels, H_rough, fp)
fp_attr, _, _, _, _ = shared.attribute_footprints(fp, crosswalk, df)
doors_gdf, _ = shared.place_doors(fp_attr)

# %%
# CELL 5  Compute network using Entrance Clustering (Approach 5)
print("Clustering projected door locations (eps=12.0)...")
G, hubs_gdf, doors_gdf_updated = compute_entrance_to_node_network(doors_gdf, eps=12.0, fp=fp_attr)

# %%
# CELL 6  Generate edge GeoDataFrame for GIS export
edge_rows = []
for u, v, data in G.edges(data=True):
    u_data = G.nodes[u]
    v_data = G.nodes[v]
    geom = LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])
    
    row = {
        "geometry": geom,
        "u": u,
        "v": v,
        "edge_type": data.get("edge_type"),
        "dist_m": round(data.get("dist_m", 0.0), 3),
        "cluster_id": data.get("cluster_id"),
        "intersects_building": data.get("intersects_building", False),
        "intersection_len_m": round(data.get("intersection_len_m", 0.0), 3),
    }
    edge_rows.append(row)
    
network_edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=fp.crs)

# %%
# CELL 7  Export GIS layers (GeoJSONs)
gis_out = shared.OUT / "vector_gis"
gis_out.mkdir(parents=True, exist_ok=True)

network_edges_gdf.to_crs("EPSG:4326").to_file(
    str(gis_out / "entrance_clustering_network.geojson"), driver="GeoJSON"
)
hubs_gdf.to_crs("EPSG:4326").to_file(
    str(gis_out / "entrance_clustering_hubs.geojson"), driver="GeoJSON"
)
print(f"Exported GIS layers to {gis_out}:")
print("  -> entrance_clustering_network.geojson")
print("  -> entrance_clustering_hubs.geojson")

# %%
# CELL 8  Plot visualization
print("Plotting entrance clustering network...")
fig, ax = plt.subplots(figsize=(16, 14))

# Plot building footprints
fp_attr.plot(ax=ax, color="#eef3f7", edgecolor="#b0c4de", linewidth=0.5, alpha=0.9)

# Plot door segments
for _, row in doors_gdf_updated.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color="#333333", linewidth=2.0, zorder=5)
    
# Plot door midpoints
ax.scatter([pt.x for pt in doors_gdf_updated["door_pt"]], 
           [pt.y for pt in doors_gdf_updated["door_pt"]], 
           color="#e74c3c", s=15, zorder=6)
           
# Plot door projection lines
for idx, row in doors_gdf_updated.iterrows():
    ax.plot([row["door_pt"].x, row["proj_x"]], 
            [row["door_pt"].y, row["proj_y"]], 
            color="#95a5a6", linestyle=":", linewidth=0.8, zorder=3)
            
# Plot door-to-hub edges
d2h_edges = network_edges_gdf[network_edges_gdf["edge_type"] == "door_to_hub"]
for _, row in d2h_edges.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color="#3498db", linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)
    
# Plot hub-to-hub edges
h2h_edges = network_edges_gdf[network_edges_gdf["edge_type"] == "hub_to_hub"]
for _, row in h2h_edges.iterrows():
    xs, ys = row.geometry.xy
    if row["intersects_building"]:
        ax.plot(xs, ys, color="#e67e22", linestyle="-.", linewidth=1.2, alpha=0.8, zorder=4)
    else:
        ax.plot(xs, ys, color="#2ecc71", linestyle="-", linewidth=2.0, alpha=0.9, zorder=4)
        
# Plot hubs
hubs_gdf.plot(ax=ax, color="#9b59b6", markersize=hubs_gdf["size"] * 15 + 30, 
              edgecolor="black", linewidth=1.0, zorder=7)
              
# Annotate hubs
for _, row in hubs_gdf.iterrows():
    ax.text(row["geometry"].x, row["geometry"].y, f"H{row['cluster_id']}",
            fontsize=7, fontweight="bold", color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="#8e44ad", ec="black", lw=0.5, alpha=0.85),
            zorder=8)
            
ax.set_title("El-Bagawat — Entrance Clustering & Access Hub Network (Approach 5)", 
             fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Easting (m)", fontsize=12)
ax.set_ylabel("Northing (m)", fontsize=12)

# Legend
legend_elements = [
    mpatches.Patch(color="#eef3f7", edgecolor="#b0c4de", label="Building Footprint"),
    plt.Line2D([0], [0], color="#333333", lw=2, label="Door Segment"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label="Door Midpoint"),
    plt.Line2D([0], [0], color="#95a5a6", linestyle=":", lw=1, label="Door Projection vector (3m)"),
    plt.Line2D([0], [0], color="#3498db", linestyle="--", lw=1, label="Door-to-Hub Edge"),
    plt.Line2D([0], [0], color="#2ecc71", linestyle="-", lw=2, label="Hub-to-Hub Edge (Obstacle-Free)"),
    plt.Line2D([0], [0], color="#e67e22", linestyle="-.", lw=1.5, label="Hub-to-Hub Edge (Crosses Building)"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', markeredgecolor='black', markersize=12, label="Access Hub")
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=10, framealpha=0.9)

# Save figure
figs_dir = shared.OUT / "figures" / "entrance_clustering"
shared.save_fig(figs_dir, "20_entrance_clustering_network.png", dpi=150)
plt.close()

print("=== PIPELINE RUN COMPLETE ===")
