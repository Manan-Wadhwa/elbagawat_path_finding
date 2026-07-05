import ezdxf
import cv2
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline import SHP_PATH, DXF_WORKING, extract_dxf_walls, get_dxf_labels, compute_affine, transform_pt, bipartite_label_match

def rasterize_and_subtract():
    print("Loading data for Idea 5...")
    footprints = gpd.read_file(SHP_PATH)
    footprints['ID'] = footprints['ID'].astype(str)
    
    dxf_labels = get_dxf_labels(DXF_WORKING)
    H = compute_affine(dxf_labels, footprints)
    lines = extract_dxf_walls(DXF_WORKING)
    
    crosswalk = bipartite_label_match(dxf_labels, H, footprints)
    
    # Pre-transform walls to UTM using global H
    lines_utm = []
    for p1, p2 in lines:
        lines_utm.append((transform_pt(p1, H), transform_pt(p2, H)))

    # Get local offsets
    dxf_labels_utm = {}
    for text, (px, py) in dxf_labels.items():
        dxf_labels_utm[text] = transform_pt((px, py), H)
        
    offsets = {}
    for idx, row in crosswalk.iterrows():
        lbl_id = row['chapel_id']
        fp_id = row['footprint_id']
        if lbl_id in dxf_labels_utm:
            pt_lbl = dxf_labels_utm[lbl_id]
            fp = footprints[footprints['ID'] == fp_id]
            if not fp.empty:
                pt_fp = fp.iloc[0].geometry.centroid
                offsets[lbl_id] = (pt_fp.x - pt_lbl[0], pt_fp.y - pt_lbl[1], fp.iloc[0].geometry)

    doors = []
    
    # Raster resolution (meters per pixel)
    res = 0.05 
    
    print("Processing local rasterization for each building...")
    for lbl_id, (ox, oy, poly) in offsets.items():
        local_lines = []
        bounds = poly.bounds 
        pad = 5.0
        minx, miny, maxx, maxy = bounds[0]-pad, bounds[1]-pad, bounds[2]+pad, bounds[3]+pad
        
        for p1, p2 in lines_utm:
            sp1 = (p1[0] + ox, p1[1] + oy)
            sp2 = (p2[0] + ox, p2[1] + oy)
            
            if (minx <= sp1[0] <= maxx and miny <= sp1[1] <= maxy) or (minx <= sp2[0] <= maxx and miny <= sp2[1] <= maxy):
                local_lines.append((sp1, sp2))
                
        if not local_lines: continue
        
        w = int((maxx - minx) / res)
        h = int((maxy - miny) / res)
        
        raster_boundary = np.zeros((h, w), dtype=np.uint8)
        raster_walls = np.zeros((h, w), dtype=np.uint8)
        
        def to_pix(pt):
            return int((pt[0] - minx) / res), int((maxy - pt[1]) / res)
            
        if poly.geom_type == 'Polygon':
            coords = list(poly.exterior.coords)
            pts = np.array([to_pix(pt) for pt in coords], np.int32)
            # Draw thick boundary to ensure coverage
            cv2.polylines(raster_boundary, [pts], isClosed=True, color=255, thickness=20) 
            
        for p1, p2 in local_lines:
            # Draw walls thick as well
            cv2.line(raster_walls, to_pix(p1), to_pix(p2), color=255, thickness=20) 
            
        # Uncovered boundary: boundary AND NOT walls
        uncovered = cv2.bitwise_and(raster_boundary, cv2.bitwise_not(raster_walls))
        
        contours, _ = cv2.findContours(uncovered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        def to_utm(pt):
            px, py = pt[0], pt[1]
            return minx + px * res, maxy - py * res
            
        for cnt in contours:
            if len(cnt) >= 2:
                pts = [to_utm(pt[0]) for pt in cnt]
                
                max_d = 0
                farthest_pair = (pts[0], pts[-1])
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        d = np.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
                        if d > max_d:
                            max_d = d
                            farthest_pair = (pts[i], pts[j])
                            
                # Check for reasonable door gap size
                if 0.5 <= max_d <= 5.0:
                    doors.append({"geom": LineString([farthest_pair[0], farthest_pair[1]]), "type": "idea5"})

    out_path = os.path.join(BASE_DIR, "annotator", "doors_idea5.geojson")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if doors:
        gpd.GeoDataFrame(doors, geometry=[d['geom'] for d in doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"Idea 5 Complete. Found {len(doors)} doors. Saved to {out_path}")

if __name__ == "__main__":
    rasterize_and_subtract()
