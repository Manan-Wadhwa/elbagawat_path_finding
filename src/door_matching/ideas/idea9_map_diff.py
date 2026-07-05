import fitz
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline import SHP_PATH, DXF_WORKING, extract_dxf_walls, get_dxf_labels, compute_affine, transform_pt
from run_idea8 import render_pdf, PDF_PATH

def run_idea9():
    print("Running Idea 9: Map vs DXF Diff")
    
    # 1. Load DXF and get UTM walls
    footprints = gpd.read_file(SHP_PATH)
    footprints['ID'] = footprints['ID'].astype(str)
    
    dxf_labels = get_dxf_labels(DXF_WORKING)
    H_global = compute_affine(dxf_labels, footprints)
    lines = extract_dxf_walls(DXF_WORKING)
    
    lines_utm = []
    for p1, p2 in lines:
        lines_utm.append((transform_pt(p1, H_global), transform_pt(p2, H_global)))
        
    # 2. Render DXF to image
    res = 0.2 # 20cm per pixel
    all_x = [p[0] for line in lines_utm for p in line]
    all_y = [p[1] for line in lines_utm for p in line]
    minx, maxx = min(all_x) - 10, max(all_x) + 10
    miny, maxy = min(all_y) - 10, max(all_y) + 10
    
    w = int((maxx - minx) / res)
    h = int((maxy - miny) / res)
    
    # Draw black lines on white background to match PDF map
    dxf_img = np.full((h, w), 255, dtype=np.uint8)
    
    def to_pix(pt):
        return int((pt[0] - minx) / res), int((maxy - pt[1]) / res)
        
    print("Rasterizing DXF...")
    for p1, p2 in lines_utm:
        cv2.line(dxf_img, to_pix(p1), to_pix(p2), 0, 2)
        
    # 3. Render PDF
    # We use 400 DPI as a balance between detail and memory limit for feature matching
    pdf_img = render_pdf(PDF_PATH, dpi=400) 
    pdf_gray = cv2.cvtColor(pdf_img, cv2.COLOR_BGR2GRAY)
    
    # 4. Feature matching
    print("Extracting SIFT features...")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(pdf_gray, None)
    kp2, des2 = sift.detectAndCompute(dxf_img, None)
    
    print(f"Found {len(kp1)} features in PDF and {len(kp2)} features in DXF render.")
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    if des1 is not None and des2 is not None and len(des1) > 2 and len(des2) > 2:
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for match_res in matches:
            if len(match_res) == 2:
                m, n = match_res
                if m.distance < 0.7 * n.distance:
                    good.append(m)
    else:
        good = []
            
    print(f"Found {len(good)} good matches.")
    
    if len(good) < 10:
        print("Not enough matches to align!")
        return
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H_align, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Homography found:\n", H_align)
    
    # 5. Extract blue annotations from PDF
    hsv = cv2.cvtColor(pdf_img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((5,5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    # 6. Warp blue mask to DXF image space
    blue_warped = cv2.warpPerspective(blue_mask, H_align, (w, h))
    
    # 7. Subtract DXF walls (where dxf_img is black) from blue_warped
    walls_mask = (dxf_img < 128).astype(np.uint8) * 255
    # Expand walls a bit to be safe
    walls_mask = cv2.dilate(walls_mask, kernel, iterations=2)
    
    # Subtract walls from blue annotations
    blue_subtracted = cv2.bitwise_and(blue_warped, cv2.bitwise_not(walls_mask))
    
    # Save raw output
    raw_out = os.path.join(BASE_DIR, "annotator", "idea9_raw_subtracted.png")
    os.makedirs(os.path.dirname(raw_out), exist_ok=True)
    cv2.imwrite(raw_out, blue_subtracted)
    print(f"Saved raw image to {raw_out}")
    
    # 8. Find contours and convert to GeoJSON
    contours, _ = cv2.findContours(blue_subtracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doors = []
    
    def to_utm(px, py):
        return minx + px * res, maxy - py * res
        
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 1000:
            if len(cnt) >= 2:
                [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                cx, cy = float(x[0]), float(y[0])
                dx, dy = float(vx[0]), float(vy[0])
                
                # Try to fit line length using contour points
                pts = [pt[0] for pt in cnt]
                max_d = 0
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        d = np.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
                        if d > max_d:
                            max_d = d
                
                length_m = max_d * res
                if 0.5 <= length_m <= 5.0:
                    px1, py1 = cx - dx * max_d/2, cy - dy * max_d/2
                    px2, py2 = cx + dx * max_d/2, cy + dy * max_d/2
                    
                    u1 = to_utm(px1, py1)
                    u2 = to_utm(px2, py2)
                    doors.append({"geom": LineString([u1, u2]), "type": "idea9"})
                    
    out_path = os.path.join(BASE_DIR, "annotator", "doors_idea9.geojson")
    if doors:
        gpd.GeoDataFrame(doors, geometry=[d['geom'] for d in doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"Idea 9 Complete. Found {len(doors)} doors. Saved to {out_path}")

if __name__ == "__main__":
    run_idea9()
