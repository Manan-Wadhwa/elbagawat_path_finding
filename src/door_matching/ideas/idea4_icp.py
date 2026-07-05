import ezdxf
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import KDTree
import cv2
import warnings
import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

warnings.filterwarnings("ignore")

# Import necessary functions from pipeline
from pipeline import get_dxf_labels, compute_affine, bipartite_label_match, transform_pt

SHP_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp")
DXF_WORKING = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "Site_CAD_Working_converted.dxf")
OUT_GEOJSON = os.path.join(BASE_DIR, "annotator", "doors_idea4.geojson")

def extract_dxf_walls_by_layer(dxf_path, layer_name='BUILDINGS'):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    lines = []
    for entity in msp.query('LWPOLYLINE'):
        if entity.dxf.layer.upper() == layer_name.upper() or layer_name is None:
            pts = entity.get_points('xy')
            for i in range(len(pts) - 1):
                lines.append(((pts[i][0], pts[i][1]), (pts[i+1][0], pts[i+1][1])))
            if entity.closed:
                lines.append(((pts[-1][0], pts[-1][1]), (pts[0][0], pts[0][1])))
    return lines

def extract_shp_lines(polygon):
    lines = []
    if polygon.geom_type == 'Polygon':
        coords = list(polygon.exterior.coords)
        for i in range(len(coords) - 1):
            lines.append((coords[i], coords[i+1]))
    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                lines.append((coords[i], coords[i+1]))
    return lines

def sample_points_from_lines(lines, step=0.5):
    pts = []
    for p1, p2 in lines:
        length = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if length == 0:
            pts.append(p1)
            continue
        num_steps = max(2, int(length / step))
        for i in range(num_steps):
            t = i / (num_steps - 1)
            pts.append((p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1])))
    if len(pts) == 0:
        return np.empty((0, 2))
    return np.array(pts)

def icp(source_pts, target_pts, max_iters=50, tolerance=1e-5):
    if len(source_pts) == 0 or len(target_pts) == 0:
        return np.eye(3)
        
    src = np.copy(source_pts)
    tree = KDTree(target_pts)
    
    prev_error = float('inf')
    H_total = np.eye(3)
    
    for i in range(max_iters):
        distances, indices = tree.query(src)
        error = np.mean(distances)
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
        
        closest_pts = target_pts[indices]
        
        M, inliers = cv2.estimateAffinePartial2D(src, closest_pts)
        if M is None:
            break
            
        H = np.vstack((M, [0, 0, 1]))
        H_total = H @ H_total
        
        src_hom = np.hstack((src, np.ones((src.shape[0], 1))))
        src = (M @ src_hom.T).T
        
    return H_total

def transform_lines(lines, H):
    new_lines = []
    for p1, p2 in lines:
        p1_new = transform_pt(p1, H)
        p2_new = transform_pt(p2, H)
        new_lines.append((p1_new, p2_new))
    return new_lines

def find_gaps_in_lines(lines, gap_min=0.5, gap_max=3.5, snap_tol=0.2):
    if not lines: return []
    G = nx.Graph()
    def snap(pt, nodes, tol):
        for n in nodes:
            if np.hypot(pt[0]-n[0], pt[1]-n[1]) < tol:
                return n
        return pt
        
    for p1, p2 in lines:
        n1 = snap(p1, list(G.nodes), snap_tol)
        n2 = snap(p2, list(G.nodes), snap_tol)
        G.add_edge(n1, n2)
        
    loose_ends = [n for n in G.nodes if G.degree(n) == 1]
    gaps = []
    for i in range(len(loose_ends)):
        for j in range(i+1, len(loose_ends)):
            p1, p2 = loose_ends[i], loose_ends[j]
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if gap_min <= dist <= gap_max:
                gaps.append((p1, p2))
    return gaps

def main():
    print("Loading datasets...")
    footprints = gpd.read_file(SHP_PATH)
    footprints['ID'] = footprints['ID'].astype(str)
    
    dxf_labels = get_dxf_labels(DXF_WORKING)
    H_global = compute_affine(dxf_labels, footprints)
    
    crosswalk = bipartite_label_match(dxf_labels, H_global, footprints)
    
    print("Extracting DXF lines from BUILDINGS layer...")
    all_dxf_lines = extract_dxf_walls_by_layer(DXF_WORKING, layer_name='BUILDINGS')
    print("Transforming lines to UTM...")
    all_dxf_lines_utm = transform_lines(all_dxf_lines, H_global)
    
    doors = []
    
    print("Processing each building...")
    for idx, row in crosswalk.iterrows():
        chapel_id = row['chapel_id']
        footprint_id = row['footprint_id']
        
        if chapel_id not in dxf_labels:
            continue
            
        px, py = dxf_labels[chapel_id]
        px_u, py_u = transform_pt((px, py), H_global)
        
        # Extract DXF lines within 20m of the UTM label
        local_lines_utm = []
        for p1, p2 in all_dxf_lines_utm:
            d1 = np.hypot(p1[0]-px_u, p1[1]-py_u)
            d2 = np.hypot(p2[0]-px_u, p2[1]-py_u)
            if d1 <= 20.0 or d2 <= 20.0:
                local_lines_utm.append((p1, p2))
                
        if not local_lines_utm:
            continue
            
        # Get SHP lines
        fp = footprints[footprints['ID'] == footprint_id]
        if fp.empty:
            continue
            
        polygon = fp.iloc[0].geometry
        shp_lines = extract_shp_lines(polygon)
        
        # Sample points
        src_pts = sample_points_from_lines(local_lines_utm, step=0.2)
        tgt_pts = sample_points_from_lines(shp_lines, step=0.2)
        
        if len(src_pts) == 0 or len(tgt_pts) == 0:
            continue
            
        # Run ICP
        H_local = icp(src_pts, tgt_pts, max_iters=50, tolerance=1e-5)
        
        # Apply local transform to UTM lines
        aligned_lines_utm = transform_lines(local_lines_utm, H_local)
        
        # Find gaps
        gaps = find_gaps_in_lines(aligned_lines_utm, gap_min=0.5, gap_max=3.5, snap_tol=0.2)
        
        for p1, p2 in gaps:
            doors.append({
                "chapel_id": chapel_id,
                "footprint_id": footprint_id,
                "geometry": LineString([p1, p2])
            })
            
    print(f"Found {len(doors)} doors.")
    
    if doors:
        gdf_doors = gpd.GeoDataFrame(doors, crs=footprints.crs)
        os.makedirs(os.path.dirname(OUT_GEOJSON), exist_ok=True)
        gdf_doors.to_crs("EPSG:4326").to_file(OUT_GEOJSON, driver="GeoJSON")
        print(f"Saved to {OUT_GEOJSON}")
    else:
        print("No doors found.")

if __name__ == "__main__":
    main()
