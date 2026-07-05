import fitz
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import ezdxf
import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shapely.ops import nearest_points

PDF_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "bagawat print.pdf")
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

def get_dxf_bbox(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')
    # Use text labels as a proxy for the map extent since lines can have weird bounding boxes
    for e in msp.query('TEXT MTEXT'):
        x, y = e.dxf.insert.x, e.dxf.insert.y
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    return (min_x, min_y, max_x, max_y)

def transform_pt(pt, H):
    arr = np.array([[pt[0], pt[1], 1.0]], dtype=np.float64)
    m = (H @ arr.T).T
    return (m[0, 0]/m[0, 2], m[0, 1]/m[0, 2])

if __name__ == "__main__":
    print("Running Idea 10: Automatic Georeferencing via BBox Mapping...")
    
    # 1. Run Idea 8 logic to extract pixel coordinates of blue marks
    import run_idea8
    img = run_idea8.render_pdf(PDF_PATH, dpi=600)
    pixel_doors = run_idea8.extract_blue_marks(img)
    
    # 2. Get BBox of dark pixels in the image (proxy for CAD layout extent)
    print("Computing Image BBox...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # Dark pixels become white
    # Find bounding box of all white pixels
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    img_min_x, img_min_y, img_max_x, img_max_y = x, y, x+w, y+h
    
    # 3. Get BBox of DXF labels
    print("Computing DXF BBox...")
    dxf_min_x, dxf_min_y, dxf_max_x, dxf_max_y = get_dxf_bbox(DXF_PATH)
    
    # 4. Compute pixel -> DXF mapping (assume no rotation, just scale and translation)
    # Note: DXF Y-axis usually goes UP. Image Y-axis goes DOWN.
    scale_x = (dxf_max_x - dxf_min_x) / (img_max_x - img_min_x)
    scale_y = (dxf_max_y - dxf_min_y) / (img_max_y - img_min_y)
    
    def pixel_to_dxf(px, py):
        # Map pixel to percentage of bbox
        pct_x = (px - img_min_x) / (img_max_x - img_min_x)
        pct_y = (py - img_min_y) / (img_max_y - img_min_y)
        # Map percentage to DXF (invert Y axis)
        dx = dxf_min_x + (pct_x * (dxf_max_x - dxf_min_x))
        dy = dxf_max_y - (pct_y * (dxf_max_y - dxf_min_y))
        return (dx, dy)
        
    # 5. Load UTM transform
    print("Computing DXF -> UTM transform...")
    footprints = gpd.read_file(SHP_PATH)
    dxf_labels = get_dxf_labels(DXF_PATH)
    H = compute_affine(dxf_labels, footprints)
    
    fp_bounds = footprints.boundary.unary_union
    
    doors = []
    for p1_px, p2_px in pixel_doors:
        # Convert P1
        dxf_p1 = pixel_to_dxf(p1_px[0], p1_px[1])
        utm_p1 = transform_pt(dxf_p1, H)
        
        # Convert P2
        dxf_p2 = pixel_to_dxf(p2_px[0], p2_px[1])
        utm_p2 = transform_pt(dxf_p2, H)
        
        # We snap the center of the line to the nearest building edge
        cx = (utm_p1[0] + utm_p2[0]) / 2
        cy = (utm_p1[1] + utm_p2[1]) / 2
        
        nearest_pt, _ = nearest_points(fp_bounds, Point(cx, cy))
        
        # Re-center the vector on the snapped point
        dx = utm_p2[0] - utm_p1[0]
        dy = utm_p2[1] - utm_p1[1]
        
        # Normalize length to 1m
        length = np.hypot(dx, dy)
        if length > 0:
            dx /= length
            dy /= length
            
        sp1 = (nearest_pt.x - dx*0.5, nearest_pt.y - dy*0.5)
        sp2 = (nearest_pt.x + dx*0.5, nearest_pt.y + dy*0.5)
        
        doors.append({"geom": LineString([sp1, sp2]), "type": "idea10"})
        
    # 6. Save outputs
    out_path10 = os.path.join(BASE_DIR, "annotator", "doors_idea10.geojson")
    out_path8  = os.path.join(BASE_DIR, "annotator", "doors_idea8.geojson")
    
    gdf = gpd.GeoDataFrame(doors, geometry=[d['geom'] for d in doors], crs=footprints.crs)
    gdf.to_crs("EPSG:4326").to_file(out_path10, driver="GeoJSON")
    gdf.to_crs("EPSG:4326").to_file(out_path8, driver="GeoJSON") # Save as Idea 8 as well since it's the georeferenced version
    
    print(f"Exported {len(doors)} Georeferenced CV Map doors!")
