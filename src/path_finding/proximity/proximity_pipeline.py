"""
proximity_pipeline.py  -  El-Bagawat Proximity Graphs Pipeline (Phase 9)
=======================================================================
Implements Gabriel Graphs, Beta-skeletons, barrier-crossing edge filtration,
Steiner tree approximation, and edge density rasterization.
"""

# %%
# CELL 0  Imports, paths, and configuration
import sys
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import rowcol
import networkx as nx
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import distance_matrix

# Add parent directory to sys.path to import shared.py
shared_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(shared_dir))
import shared

# %%
# CELL 1  Proximity Graph helper functions
def gabriel_graph(entrance_points_array, crs):
    """
    Compute the Gabriel Graph for the given entrance points.
    An edge exists between pi and pj if the closed disk with diameter pipj contains no other points.
    
    Parameters:
    -----------
    entrance_points_array : np.ndarray
        Array of shape (N, 2) containing coordinate points.
    crs : CRS or str
        The coordinate reference system to assign to the graph.
        
    Returns:
    --------
    G : nx.Graph
        The Gabriel Graph with nodes 0 to N-1 and edge weights as Euclidean distances.
    """
    n = len(entrance_points_array)
    G = nx.Graph()
    G.graph['crs'] = crs
    
    for i in range(n):
        G.add_node(i, x=float(entrance_points_array[i, 0]), y=float(entrance_points_array[i, 1]))
        
    if n < 2:
        return G
        
    # Calculate all pairwise distances
    dist = distance_matrix(entrance_points_array, entrance_points_array)
    
    for i in range(n):
        for j in range(i + 1, n):
            pi = entrance_points_array[i]
            pj = entrance_points_array[j]
            mid = (pi + pj) / 2.0
            r = dist[i, j] / 2.0
            
            # Extract all other points to test
            other_pts = np.delete(entrance_points_array, [i, j], axis=0)
            if len(other_pts) == 0:
                G.add_edge(i, j, weight=float(dist[i, j]))
                continue
                
            dists_to_mid = np.linalg.norm(other_pts - mid, axis=1)
            # If no other point lies inside the circle (r - epsilon)
            if not np.any(dists_to_mid < r - 1e-9):
                G.add_edge(i, j, weight=float(dist[i, j]))
                
    return G


def beta_skeleton(entrance_points_array, beta=1.5):
    """
    Compute the beta-skeleton (lune-based) for the given entrance points.
    For beta >= 1, the neighborhood is the intersection of two closed disks 
    of radius beta*d(pi, pj)/2 centered at:
    c1 = (1 - beta/2)*pi + (beta/2)*pj
    c2 = (beta/2)*pi + (1 - beta/2)*pj
    
    Parameters:
    -----------
    entrance_points_array : np.ndarray
        Array of shape (N, 2) containing coordinate points.
    beta : float
        The beta parameter (must be >= 1.0).
        
    Returns:
    --------
    G : nx.Graph
        The beta-skeleton Graph with nodes 0 to N-1 and edge weights as Euclidean distances.
    """
    if beta < 1.0:
        raise ValueError("Beta parameter must be >= 1.0 for the standard lune-based skeleton.")
        
    n = len(entrance_points_array)
    G = nx.Graph()
    
    for i in range(n):
        G.add_node(i, x=float(entrance_points_array[i, 0]), y=float(entrance_points_array[i, 1]))
        
    if n < 2:
        return G
        
    dist = distance_matrix(entrance_points_array, entrance_points_array)
    
    for i in range(n):
        for j in range(i + 1, n):
            pi = entrance_points_array[i]
            pj = entrance_points_array[j]
            d = dist[i, j]
            
            c1 = (1.0 - beta / 2.0) * pi + (beta / 2.0) * pj
            c2 = (beta / 2.0) * pi + (1.0 - beta / 2.0) * pj
            R = (beta / 2.0) * d
            
            other_pts = np.delete(entrance_points_array, [i, j], axis=0)
            if len(other_pts) == 0:
                G.add_edge(i, j, weight=float(d))
                continue
                
            d1 = np.linalg.norm(other_pts - c1, axis=1)
            d2 = np.linalg.norm(other_pts - c2, axis=1)
            
            # Inside the intersection of both disks
            in_neighborhood = (d1 < R - 1e-9) & (d2 < R - 1e-9)
            if not np.any(in_neighborhood):
                G.add_edge(i, j, weight=float(d))
                
    return G


def remove_barrier_crossing_edges(proximity_graph, cost_raster, raster_transform, entrance_points_array, barrier_percentile=95):
    """
    Remove edges from proximity_graph that cross high-cost barrier areas.
    The barrier threshold is determined by a percentile of the cost_raster.
    
    Parameters:
    -----------
    proximity_graph : nx.Graph
        The input proximity graph.
    cost_raster : np.ndarray
        2D raster array representing walking costs.
    raster_transform : Affine
        Affine transform mapping pixel coordinates to map coordinates.
    entrance_points_array : np.ndarray
        Array of shape (N, 2) containing node coordinates.
    barrier_percentile : float
        Percentile of the cost surface to define as a barrier.
        
    Returns:
    --------
    G : nx.Graph
        Filtered graph with barrier-crossing edges removed.
    """
    G = proximity_graph.copy()
    
    # Calculate threshold based on percentile of valid costs (excluding nan/inf)
    valid_costs = cost_raster[np.isfinite(cost_raster)]
    if len(valid_costs) == 0:
        print("Warning: Cost raster contains no valid finite values. No edges removed.")
        return G
        
    threshold = np.percentile(valid_costs, barrier_percentile)
    print(f"  Barrier cost threshold (percentile {barrier_percentile}%): {threshold:.4f}")
    
    h, w = cost_raster.shape
    edges_to_remove = []
    
    for u, v in G.edges():
        p1 = entrance_points_array[u]
        p2 = entrance_points_array[v]
        
        dist = np.linalg.norm(p2 - p1)
        # Sample points along the segment, spaced by 0.2 meters or at least 10 points
        num_samples = max(10, int(np.ceil(dist / 0.2)))
        t = np.linspace(0, 1, num_samples)
        sampled_pts = p1 + t[:, np.newaxis] * (p2 - p1)
        
        # Convert UTM coordinates to row/col indices
        rows, cols = rowcol(raster_transform, sampled_pts[:, 0], sampled_pts[:, 1])
        
        crosses_barrier = False
        for r, c in zip(rows, cols):
            if 0 <= r < h and 0 <= c < w:
                val = cost_raster[r, c]
                if np.isnan(val) or val > threshold:
                    crosses_barrier = True
                    break
            else:
                # Treat out-of-bounds as barrier
                crosses_barrier = True
                break
                
        if crosses_barrier:
            edges_to_remove.append((u, v))
            
    G.remove_edges_from(edges_to_remove)
    print(f"  Filtered: removed {len(edges_to_remove)} edges crossing barrier pixels. Edges left: {G.number_of_edges()}")
    return G


def steiner_tree_approximation(proximity_graph, entrance_indices, entrance_points_array):
    """
    Approximate the Steiner tree spanning all terminals (entrance_indices).
    Uses NetworkX metric closure approximation.
    
    Parameters:
    -----------
    proximity_graph : nx.Graph
        The filtered proximity graph.
    entrance_indices : list-like
        List of node indices representing terminals.
    entrance_points_array : np.ndarray
        Array of shape (N, 2) containing node coordinates.
        
    Returns:
    --------
    tree : nx.Graph
        The approximated Steiner tree with node attributes x and y.
    """
    from networkx.algorithms.approximation import steiner_tree as nx_steiner_tree
    
    G = proximity_graph.copy()
    
    # Assign coordinates as node attributes
    for idx in G.nodes():
        G.nodes[idx]['x'] = float(entrance_points_array[idx, 0])
        G.nodes[idx]['y'] = float(entrance_points_array[idx, 1])
        
    terminals = list(entrance_indices)
    
    if G.number_of_edges() == 0:
        print("Warning: Graph has 0 edges. Cannot compute Steiner tree.")
        return G
        
    if not nx.is_connected(G):
        print("Warning: Graph is disconnected. Computing Steiner tree on the largest connected component...")
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        # Choose component containing the most terminals
        best_comp = max(components, key=lambda c: len(c.intersection(terminals)))
        subG = G.subgraph(best_comp).copy()
        comp_terminals = [t for t in terminals if t in best_comp]
        print(f"  Largest component has {len(best_comp)} nodes, including {len(comp_terminals)} terminals.")
        
        if len(comp_terminals) <= 1:
            print("Warning: Component has <= 1 terminal. Returning empty graph.")
            return nx.Graph()
            
        try:
            tree = nx_steiner_tree(subG, comp_terminals, weight='weight')
        except Exception as e:
            print(f"  nx_steiner_tree failed ({e}). Falling back to minimum spanning tree of that component.")
            tree = nx.minimum_spanning_tree(subG, weight='weight')
    else:
        try:
            tree = nx_steiner_tree(G, terminals, weight='weight')
        except Exception as e:
            print(f"  nx_steiner_tree failed ({e}). Falling back to minimum spanning tree.")
            tree = nx.minimum_spanning_tree(G, weight='weight')
            
    # Guarantee coordinate attributes are in the returned tree nodes
    for node in tree.nodes():
        tree.nodes[node]['x'] = float(entrance_points_array[node, 0])
        tree.nodes[node]['y'] = float(entrance_points_array[node, 1])
        
    return tree


def proximity_graph_to_edge_density(gabriel_graph, beta_graphs, entrance_points_array, raster_shape, raster_transform):
    """
    Accumulate edges from Gabriel graph and beta-skeletons into an edge density raster.
    
    Parameters:
    -----------
    gabriel_graph : nx.Graph
        Gabriel graph.
    beta_graphs : list or nx.Graph
        List of beta-skeleton graphs.
    entrance_points_array : np.ndarray
        Array of shape (N, 2) containing node coordinates.
    raster_shape : tuple
        (height, width) of the output raster.
    raster_transform : Affine
        Transform of the output raster.
        
    Returns:
    --------
    density : np.ndarray
        2D float32 array containing the count of overlapping edges in each pixel.
    """
    from rasterio.enums import MergeAlg
    
    all_graphs = [gabriel_graph]
    if isinstance(beta_graphs, list):
        all_graphs.extend(beta_graphs)
    elif isinstance(beta_graphs, dict):
        all_graphs.extend(beta_graphs.values())
    elif isinstance(beta_graphs, nx.Graph):
        all_graphs.append(beta_graphs)
        
    geoms = []
    for G in all_graphs:
        for u, v in G.edges():
            p1 = entrance_points_array[u]
            p2 = entrance_points_array[v]
            line = LineString([p1, p2])
            geoms.append((line, 1.0))
            
    density = np.zeros(raster_shape, dtype=np.float32)
    if geoms:
        # Use rasterize with MergeAlg.add to sum up overlapping line segments
        density = rio_rasterize(
            geoms,
            out_shape=raster_shape,
            transform=raster_transform,
            fill=0,
            merge_alg=MergeAlg.add,
            dtype=np.float32
        )
        
    return density


# %%
# CELL 2  Load building footprints and Excel data
print("=== STARTING PHASE 9: PROXIMITY GRAPH PIPELINE ===")
shared.verify_paths()

print("\n[Step 1] Loading footprints and database...")
fp = shared.load_footprints()
df = shared.load_excel_directions()

# %%
# CELL 3  Match text labels to footprints (Hungarian matching)
print("\n[Step 2] Aligning DXF labels with building footprints...")
dxf_labels = shared.load_dxf_labels()
H_rough = shared.estimate_rough_affine(dxf_labels, fp)
crosswalk = shared.bipartite_label_match(dxf_labels, H_rough, fp)
fp_attributed, n_labelled, n_with_dir, n_no_dir, n_unlabelled = shared.attribute_footprints(fp, crosswalk, df)

# %%
# CELL 4  Place doors (Approach-5)
print("\n[Step 3] Placing doors...")
doors_gdf, doors_pts = shared.place_doors(fp_attributed)

print(f"Placed {len(doors_pts)} door points.")
entrance_points_array = np.array([[geom.x, geom.y] for geom in doors_pts.geometry])

# %%
# CELL 5  Load or generate Cost Surface
print("\n[Step 4] Resolving Cost Surface...")
cost_surface_path = shared.OUT / "cost_surface_tobler.tif"

if not cost_surface_path.exists():
    print("  Cost surface raster not found. Generating from DEM...")
    dem = shared.load_dem()
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
    cost_transform = dem["transform"]
    cost_shape = cost_surface.shape
    
    # Save generated cost surface
    profile_cost = dem["profile"].copy()
    profile_cost.update(dtype="float32", count=1)
    with rasterio.open(cost_surface_path, "w", **profile_cost) as dst:
        dst.write(cost_surface.astype(np.float32), 1)
    cost_profile = profile_cost
else:
    print(f"  Loading cost surface from {cost_surface_path}...")
    with rasterio.open(str(cost_surface_path)) as src:
        cost_surface = src.read(1)
        cost_transform = src.transform
        cost_shape = cost_surface.shape
        cost_profile = src.profile
        
# %%
# CELL 6  Compute Gabriel graph and Beta-skeleton
print("\n[Step 5] Computing Proximity Graphs...")
print("  Calculating Gabriel Graph...")
gabriel = gabriel_graph(entrance_points_array, fp.crs)
print(f"    -> Nodes: {gabriel.number_of_nodes()}, Edges: {gabriel.number_of_edges()}")

print("  Calculating Beta-Skeleton (beta = 1.5)...")
beta_1_5 = beta_skeleton(entrance_points_array, beta=1.5)
print(f"    -> Nodes: {beta_1_5.number_of_nodes()}, Edges: {beta_1_5.number_of_edges()}")

# %%
# CELL 7  Filter out barrier-crossing edges (buildings)
print("\n[Step 6] Filtering out building barrier crossings (percentile=95)...")
print("  Filtering Gabriel Graph...")
gabriel_clean = remove_barrier_crossing_edges(
    gabriel, cost_surface, cost_transform, entrance_points_array, barrier_percentile=95
)

print("  Filtering Beta-Skeleton...")
beta_clean = remove_barrier_crossing_edges(
    beta_1_5, cost_surface, cost_transform, entrance_points_array, barrier_percentile=95
)

# %%
# CELL 8  Steiner Tree Approximation
print("\n[Step 7] Approximating Steiner Tree spanning all doors...")
entrance_indices = list(range(len(entrance_points_array)))
steiner_tree = steiner_tree_approximation(
    gabriel_clean, entrance_indices, entrance_points_array
)
print(f"  Steiner Tree -> Nodes: {steiner_tree.number_of_nodes()}, Edges: {steiner_tree.number_of_edges()}")

# %%
# CELL 9  Export Edge Density
print("\n[Step 8] Exporting Edge Density Raster...")
edge_density = proximity_graph_to_edge_density(
    gabriel_clean, [beta_clean], entrance_points_array, cost_shape, cost_transform
)

density_out_path = shared.OUT / "proximity_edge_density.tif"
profile_density = cost_profile.copy()
profile_density.update(dtype="float32", count=1, nodata=0.0)
with rasterio.open(str(density_out_path), "w", **profile_density) as dst:
    dst.write(edge_density.astype(np.float32), 1)
print(f"  Saved edge density to: {density_out_path}")

# %%
# CELL 10  Export Steiner Tree to GeoJSON
print("\n[Step 9] Exporting Steiner Tree vector network...")
steiner_edges = []
for u, v, data in steiner_tree.edges(data=True):
    p1 = Point(steiner_tree.nodes[u]['x'], steiner_tree.nodes[u]['y'])
    p2 = Point(steiner_tree.nodes[v]['x'], steiner_tree.nodes[v]['y'])
    line = LineString([p1, p2])
    steiner_edges.append({
        "geometry": line,
        "node_u": int(u),
        "node_v": int(v),
        "weight": float(data.get("weight", 0.0))
    })
    
if steiner_edges:
    gdf_steiner = gpd.GeoDataFrame(steiner_edges, crs=fp.crs)
    gdf_steiner_wgs84 = gdf_steiner.to_crs("EPSG:4326")
    
    vector_out_dir = shared.OUT / "vector_gis"
    vector_out_dir.mkdir(exist_ok=True)
    geojson_out_path = vector_out_dir / "proximity_steiner_tree.geojson"
    
    gdf_steiner_wgs84.to_file(str(geojson_out_path), driver="GeoJSON")
    print(f"  Saved Steiner Tree GeoJSON to: {geojson_out_path}")
else:
    print("  Warning: Steiner Tree has no edges. Skipped GeoJSON export.")
    
# %%
# CELL 11  Visualise and save the network plots
print("\n[Step 10] Plotting and saving visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Left Panel: Gabriel Graph
axes[0].set_title("Clean Gabriel Graph", fontsize=16, fontweight="bold")
fp.plot(ax=axes[0], color="#e8f4fb", edgecolor="#6baed6", linewidth=0.5, alpha=0.8)
for u, v in gabriel_clean.edges():
    p1 = entrance_points_array[u]
    p2 = entrance_points_array[v]
    axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], color="indigo", linewidth=0.8, alpha=0.6)
axes[0].scatter(entrance_points_array[:, 0], entrance_points_array[:, 1], s=10, c="red", label="Doors", zorder=5)
axes[0].set_xlabel("Easting (m)")
axes[0].set_ylabel("Northing (m)")
axes[0].legend(loc="lower left")
axes[0].set_aspect('equal')

# Right Panel: Steiner Tree
axes[1].set_title("Steiner Tree Approximation", fontsize=16, fontweight="bold")
fp.plot(ax=axes[1], color="#e8f4fb", edgecolor="#6baed6", linewidth=0.5, alpha=0.8)
for u, v in steiner_tree.edges():
    p1 = [steiner_tree.nodes[u]['x'], steiner_tree.nodes[u]['y']]
    p2 = [steiner_tree.nodes[v]['x'], steiner_tree.nodes[v]['y']]
    axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], color="darkgreen", linewidth=1.2, alpha=0.8)
axes[1].scatter(entrance_points_array[:, 0], entrance_points_array[:, 1], s=10, c="red", label="Doors", zorder=5)
axes[1].set_xlabel("Easting (m)")
axes[1].set_ylabel("Northing (m)")
axes[1].legend(loc="lower left")
axes[1].set_aspect('equal')

plt.suptitle("Phase 9 Proximity Graphs — Gabriel Graph & Steiner Tree", fontsize=20, fontweight="bold")
plt.tight_layout()

# Save network figure
fig_out_dir = shared.OUT / "figures" / "proximity"
fig_out_dir.mkdir(parents=True, exist_ok=True)
fig_out_path = fig_out_dir / "19_proximity_graphs.png"

plt.savefig(str(fig_out_path), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved comparison plot to: {fig_out_path}")

print("\n=== PHASE 9 PROXIMITY GRAPH PIPELINE COMPLETED SUCCESSFULLY ===")
