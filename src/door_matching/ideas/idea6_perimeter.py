import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ezdxf
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

SHP_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp")
DXF_WORKING = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "Site_CAD_Working_converted.dxf")
OUT_GEOJSON = os.path.join(BASE_DIR, "annotator", "doors_idea6.geojson")

def extract_dxf_walls(dxf_path):
    print("Extracting DXF Walls...")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    lines = []
    for entity in msp.query('LWPOLYLINE'):
        pts = entity.get_points('xy')
        for i in range(len(pts) - 1):
            lines.append(((pts[i][0], pts[i][1]), (pts[i+1][0], pts[i+1][1])))
        if entity.closed:
            lines.append(((pts[-1][0], pts[-1][1]), (pts[0][0], pts[0][1])))
    return lines

def get_dxf_labels(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    labels = {}
    for e in doc.modelspace().query('TEXT MTEXT'):
        text = e.dxf.text.strip()
        if text.isdigit():
            labels[text] = (e.dxf.insert.x, e.dxf.insert.y)
    return labels

def compute_affine(dxf_labels, footprints):
    bootstrap_ids = ['23', '24', '25', '26', '175', '210']
    dxf_pts, utm_pts = [], []
    for b_id in bootstrap_ids:
        if b_id in dxf_labels:
            px, py = dxf_labels[b_id]
            fp = footprints[footprints['ID'].astype(str) == str(b_id)]
            if not fp.empty:
                cx, cy = fp.iloc[0].geometry.centroid.coords[0]
                dxf_pts.append((px, py))
                utm_pts.append((cx, cy))
    
    M_init, _ = cv2.estimateAffinePartial2D(np.array(dxf_pts), np.array(utm_pts))
    H_init = np.vstack([M_init, [0, 0, 1]])
    
    def transform_pt_local(px, py, H):
        pt = np.array([[px, py, 1.0]], dtype=np.float64)
        m = (H @ pt.T).T
        return m[0, :2] / m[0, 2]
    
    all_dxf, all_utm = [], []
    for lbl, (px, py) in dxf_labels.items():
        rough_utm = transform_pt_local(px, py, H_init)
        rough_pt = Point(rough_utm[0], rough_utm[1])
        dists = footprints.geometry.centroid.distance(rough_pt)
        min_idx = dists.idxmin()
        if dists.min() < 15.0:
            exact_utm = footprints.loc[min_idx].geometry.centroid.coords[0]
            all_dxf.append((px, py))
            all_utm.append(exact_utm)
            
    M_final, _ = cv2.estimateAffinePartial2D(np.array(all_dxf), np.array(all_utm))
    H_final = np.vstack([M_final, [0, 0, 1]])
    return H_final

def transform_pt(pt, H):
    arr = np.array([[pt[0], pt[1], 1.0]], dtype=np.float64)
    m = (H @ arr.T).T
    return (m[0, 0]/m[0, 2], m[0, 1]/m[0, 2])

def get_coverage_ratio(segment, dxf_lines, rtree_idx):
    if segment.length == 0:
        return 1.0 # Ignore 0-length segments
    
    # Create flat buffer of 2 meters
    # cap_style=2 is flat, cap_style=3 is square. We use flat to not extend past the segment
    buffer = segment.buffer(2.0, cap_style=2)
    
    possible_matches_index = list(rtree_idx.query(buffer))
    
    intervals = []
    for idx in possible_matches_index:
        wall = dxf_lines[idx]
        intersected = wall.intersection(buffer)
        if intersected.is_empty:
            continue
            
        geom_list = [intersected] if isinstance(intersected, LineString) else (intersected.geoms if hasattr(intersected, 'geoms') else [])
        
        for g in geom_list:
            if not isinstance(g, LineString):
                continue
            if g.is_empty:
                continue
            d1 = segment.project(Point(g.coords[0]))
            d2 = segment.project(Point(g.coords[-1]))
            start_d = min(d1, d2)
            end_d = max(d1, d2)
            intervals.append((start_d, end_d))
            
    if not intervals:
        return 0.0
        
    intervals.sort()
    merged = []
    curr_start, curr_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= curr_end:
            curr_end = max(curr_end, end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = start, end
    merged.append((curr_start, curr_end))
    
    covered_length = sum(end - start for start, end in merged)
    return covered_length / segment.length

def main():
    print("Loading Footprints...")
    footprints = gpd.read_file(SHP_PATH)
    footprints['ID'] = footprints['ID'].astype(str)
    
    print("Loading DXF Labels and Computing Transform...")
    dxf_labels = get_dxf_labels(DXF_WORKING)
    H = compute_affine(dxf_labels, footprints)
    
    print("Extracting and Transforming DXF Walls...")
    lines = extract_dxf_walls(DXF_WORKING)
    
    utm_lines = []
    for p1, p2 in lines:
        p1_u = transform_pt(p1, H)
        p2_u = transform_pt(p2, H)
        utm_lines.append(LineString([p1_u, p2_u]))
        
    # Build RTree for fast spatial queries
    rtree_idx = STRtree(utm_lines)
    
    print("Processing Building Perimeters for Doors...")
    doors = []
    
    for idx, row in footprints.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        # Handle MultiPolygons if any
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        
        for poly in polys:
            ext_coords = list(poly.exterior.coords)
            
            best_segment = None
            min_coverage = float('inf')
            
            for i in range(len(ext_coords) - 1):
                p1 = ext_coords[i]
                p2 = ext_coords[i+1]
                segment = LineString([p1, p2])
                
                if segment.length < 0.1:
                    continue # Ignore very tiny segments
                    
                cov_ratio = get_coverage_ratio(segment, utm_lines, rtree_idx)
                
                if cov_ratio < min_coverage:
                    min_coverage = cov_ratio
                    best_segment = segment
                    
            if best_segment is not None:
                # Create 1m door geometry in the center of the best segment
                mid_dist = best_segment.length / 2.0
                start_dist = max(0.0, mid_dist - 0.5)
                end_dist = min(best_segment.length, mid_dist + 0.5)
                
                door_geom = LineString([
                    best_segment.interpolate(start_dist),
                    best_segment.interpolate(end_dist)
                ])
                doors.append({
                    'building_id': row['ID'],
                    'min_coverage': min_coverage,
                    'geometry': door_geom
                })
                
    print(f"Generated {len(doors)} doors.")
    
    if doors:
        doors_gdf = gpd.GeoDataFrame(doors, crs=footprints.crs)
        os.makedirs(os.path.dirname(OUT_GEOJSON), exist_ok=True)
        # Convert to EPSG:4326 for web compatibility
        doors_gdf.to_crs("EPSG:4326").to_file(OUT_GEOJSON, driver="GeoJSON")
        print(f"Saved to {OUT_GEOJSON}")
    else:
        print("No doors generated.")

if __name__ == "__main__":
    main()
