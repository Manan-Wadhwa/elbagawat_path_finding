"""
run_subsampled_space_syntax.py
Runs Space Syntax Analysis (Phase 8) on a subsampled bounding box matching the user's screenshot.
"""

import sys
import os
import time
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

# Add parent directory to sys.path to import shared
script_dir = Path(__file__).resolve().parents[1]
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
import shared


def generate_free_space_skeleton(footprints_gdf, roi_bounds, resolution=0.5):
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds
    
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
    kernel = np.ones((3, 3), np.uint8)
    free_space_eroded = cv2.erode(free_space, kernel, iterations=1)
    
    # Skeletonize to medial axis
    skeleton = skeletonize(free_space_eroded.astype(bool))
    
    return skeleton.astype(np.uint8), transform


def skeleton_to_axial_graph(skeleton, raster_transform, crs):
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
    n = G.number_of_nodes()
    if n < 3:
        return gpd.GeoDataFrame(columns=["geometry", "integration_mean", "integration_mean_norm", "integration_u", "integration_v"], crs=crs)
    
    node_list = list(G.nodes())
    integration_vals = {}
    
    D_n = 2.0 / (n - 2) if n > 2 else 1.0
    
    for i, node in enumerate(node_list):
        if i % 200 == 0:
            print(f"  BFS loop progress: {i}/{n} nodes...")
            sys.stdout.flush()
        lengths = nx.single_source_shortest_path_length(G, node)
        if len(lengths) < 2:
            integration_vals[node] = 0.0
            continue
        total_depth = sum(lengths.values())
        MD = total_depth / (len(lengths) - 1)
        RA = 2 * (MD - 1) / (n - 2) if n > 2 else 0
        RRA = RA / D_n if D_n > 0 else 0
        integration_vals[node] = 1.0 / RRA if RRA > 0 else 0.0
    
    int_vals = np.array(list(integration_vals.values()))
    max_val = int_vals.max()
    
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
    from rasterio.features import rasterize as rio_rasterize
    import rasterio.enums
    
    if syntax_gdf.empty:
        return np.zeros(raster_shape, dtype=np.float32)
        
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


def main():
    print("=== Subsampled Space Syntax Analysis ===")
    
    # 1. Load Data
    footprints = shared.load_footprints()
    dem = shared.load_dem()
    crs = dem["crs"]
    
    if footprints.crs != crs:
        print(f"Reprojecting footprints to {crs}...")
        footprints = footprints.to_crs(crs)
        
    # 2. Define Bounding Box matching screenshot (from selected chapels)
    # min_x = 254130.36, max_x = 254382.28
    # min_y = 2820877.56, max_y = 2821040.56
    # Let's apply a 20m buffer:
    roi_bounds = (254110.36, 2820857.56, 254402.28, 2821060.56)
    print(f"Subsampled Region of Interest bounds: {roi_bounds}")
    
    # 3. Generate Skeleton
    print("Generating skeleton on subsampled grid...")
    t0 = time.time()
    skeleton, raster_transform = generate_free_space_skeleton(
        footprints,
        roi_bounds=roi_bounds,
        resolution=dem["res"]
    )
    print(f"Skeleton shape: {skeleton.shape} in {time.time() - t0:.2f}s")
    
    # 4. Convert to Graph
    print("Converting skeleton to axial graph...")
    t0 = time.time()
    G = skeleton_to_axial_graph(skeleton, raster_transform, crs)
    print(f"Axial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges in {time.time() - t0:.2f}s")
    
    if G.number_of_nodes() == 0:
        print("Error: No skeleton pixels found in the selected region.")
        return
        
    # 5. Compute Integration
    print("Computing space syntax integration (BFS loop)...")
    t0 = time.time()
    syntax_gdf = compute_space_syntax_integration(G, raster_transform, crs)
    print(f"Computed integration for {len(syntax_gdf)} segments in {time.time() - t0:.2f}s")
    
    # 6. Rasterize
    print("Rasterizing integration...")
    integration_raster = rasterize_integration(syntax_gdf, skeleton.shape, raster_transform)
    
    # 7. Save outputs
    # Create output paths with suffix
    raster_out_path = shared.OUT / "space_syntax_integration_subsampled.tif"
    vector_out_path = shared.OUT / "vector_gis" / "space_syntax_axial_lines_subsampled.geojson"
    
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
        
    print(f"Saving vector axial lines to {vector_out_path}...")
    vector_out_path.parent.mkdir(parents=True, exist_ok=True)
    syntax_gdf.to_file(str(vector_out_path), driver="GeoJSON")
    
    # 8. Plot Visualization Map zoomed to the selected area
    print("Generating visualization map...")
    hs = shared.hillshade(dem["disp"])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Underlay DEM hillshade
    ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
              cmap="terrain", alpha=0.6,
              vmin=dem["e_min"], vmax=dem["e_max"], rasterized=True)
    ax.imshow(hs, extent=dem["extent"], origin="upper",
              cmap="gray", alpha=0.35, rasterized=True)
              
    # Overlay building footprints
    footprints.plot(ax=ax, color="none", edgecolor="#555555", linewidth=0.8, alpha=0.7)
    
    # Plot axial lines colored by integration
    if not syntax_gdf.empty:
        syntax_gdf.plot(ax=ax, column="integration_mean_norm", cmap="plasma", linewidth=2.0,
                        legend=True, legend_kwds={'label': 'Normalized Integration (Subsampled)', 'shrink': 0.6})
    
    ax.set_title("El-Bagawat Subsampled Space Syntax Analysis\nGlobal Axial Line Integration (Medial Axis)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    # Zoom exactly to ROI bounds
    ax.set_xlim(roi_bounds[0], roi_bounds[2])
    ax.set_ylim(roi_bounds[1], roi_bounds[3])
    
    # Save the figure
    figs_dir = shared.OUT / "figures" / "space_syntax"
    shared.save_fig(figs_dir, "18_space_syntax_integration_subsampled.png", dpi=150)
    plt.close()
    
    print("Subsampled Space Syntax Pipeline completed successfully!")


if __name__ == "__main__":
    main()
