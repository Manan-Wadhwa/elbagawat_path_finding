import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ezdxf
import networkx as nx
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import cv2

SHP_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp")
DXF_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "Site_CAD_Working_converted.dxf")

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
    
    def transform_pt(px, py, H):
        pt = np.array([[px, py, 1.0]], dtype=np.float64)
        m = (H @ pt.T).T
        return m[0, :2] / m[0, 2]
    
    all_dxf, all_utm = [], []
    for lbl, (px, py) in dxf_labels.items():
        rough_utm = transform_pt(px, py, H_init)
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

def extract_filtered_walls(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    lines = []
    # ONLY EXTRACT FROM 'BUILDINGS' LAYER (Idea 2)
    for entity in msp.query('LWPOLYLINE[layer=="BUILDINGS"]'):
        pts = entity.get_points('xy')
        for i in range(len(pts) - 1):
            lines.append(((pts[i][0], pts[i][1]), (pts[i+1][0], pts[i+1][1])))
        if entity.closed:
            lines.append(((pts[-1][0], pts[-1][1]), (pts[0][0], pts[0][1])))
    return lines

if __name__ == "__main__":
    print("Running Idea 2: Layer-Filtered Gaps...")
    footprints = gpd.read_file(SHP_PATH)
    dxf_labels = get_dxf_labels(DXF_PATH)
    H = compute_affine(dxf_labels, footprints)
    
    lines = extract_filtered_walls(DXF_PATH)
    
    # Transform to UTM
    lines_utm = []
    points = {}
    for p1, p2 in lines:
        p1_u = transform_pt(p1, H)
        p2_u = transform_pt(p2, H)
        lines_utm.append((p1_u, p2_u))
        
        p1_round = (round(p1_u[0], 2), round(p1_u[1], 2))
        p2_round = (round(p2_u[0], 2), round(p2_u[1], 2))
        points[p1_round] = points.get(p1_round, 0) + 1
        points[p2_round] = points.get(p2_round, 0) + 1
        
    loose_ends = [p for p, count in points.items() if count == 1]
    
    doors = []
    buffered_mask = footprints.boundary.buffer(3.5).unary_union
    
    for i in range(len(loose_ends)):
        for j in range(i+1, len(loose_ends)):
            p1, p2 = loose_ends[i], loose_ends[j]
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if dist <= 3.5:
                ls = LineString([p1, p2])
                if ls.intersects(buffered_mask):
                    doors.append({"geom": ls, "type": "idea2"})
                    
    if len(doors) > 0:
        gdf = gpd.GeoDataFrame(doors, geometry=[d['geom'] for d in doors], crs=footprints.crs)
        out_path = os.path.join(BASE_DIR, "annotator", "doors_idea2.geojson")
        gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
        print(f"Exported {len(doors)} Layer-Filtered doors to {out_path}")
    else:
        print("No doors found after filtering.")
