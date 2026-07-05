import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import geopandas as gpd
import ezdxf
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
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

footprints = gpd.read_file(SHP_PATH)
dxf_labels = get_dxf_labels(DXF_PATH)
H = compute_affine(dxf_labels, footprints)

doc = ezdxf.readfile(DXF_PATH)
msp = doc.modelspace()
arcs = [e for e in msp if e.dxftype() == 'ARC' and e.dxf.radius < 500]

fp_bounds = footprints.boundary.unary_union

doors = []
for arc in arcs:
    cx, cy = arc.dxf.center.x, arc.dxf.center.y
    r = arc.dxf.radius
    
    utm_pt = transform_pt((cx, cy), H)
    pt = Point(utm_pt[0], utm_pt[1])
    
    # Snap to nearest footprint boundary
    p1, p2 = nearest_points(fp_bounds, pt)
    
    # Create a line segment representing the door (parallel to boundary is hard, so we just make a simple cross or line)
    # The simplest is a 1m line centered at p1
    door_line = LineString([(p1.x - 0.5, p1.y - 0.5), (p1.x + 0.5, p1.y + 0.5)])
    doors.append({"geom": door_line, "type": "arc"})

if len(doors) > 0:
    gdf = gpd.GeoDataFrame(doors, geometry=[d['geom'] for d in doors], crs=footprints.crs)
    out_path = os.path.join(BASE_DIR, "annotator", "doors_idea1.geojson")
    gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"Exported {len(doors)} ARC doors to {out_path}")
