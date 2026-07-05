import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fitz
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

PDF_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "bagawat print.pdf")
OUT_GEOJSON = os.path.join(BASE_DIR, "annotator", "doors_idea8.geojson")

def render_pdf(pdf_path, dpi=800):
    print(f"Opening PDF... rendering at {dpi} DPI")
    doc = fitz.open(pdf_path)
    page = doc[0]
    # DPI 800 is a good compromise between 1200 (Out of Memory) and 300 (too blurry)
    # 800 DPI = ~33k x 21k pixels = ~2GB RAM
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    print(f"Rendered pixmap: {pix.width}x{pix.height}")
    
    # Convert to OpenCV numpy array
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def extract_blue_marks(img):
    print("Converting to HSV and thresholding blue...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color (the annotations in anot.png are a distinct light blue)
    # Blue is around Hue 100-130
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    print("Finding contours...")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    doors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50 and area < 5000: # filter out noise and massive blobs
            if len(cnt) >= 2: # need at least 2 points to fit a line
                [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                cx, cy = float(x[0]), float(y[0])
                dx, dy = float(vx[0]), float(vy[0])
                # Create a 20-pixel long line representing the stroke
                p1 = (cx - dx * 10, cy - dy * 10)
                p2 = (cx + dx * 10, cy + dy * 10)
                doors.append((p1, p2))
    
    print(f"Found {len(doors)} potential blue marks (lines) in pixel coordinates.")
    return doors

if __name__ == "__main__":
    try:
        img = render_pdf(PDF_PATH, dpi=600) # Start with 600 DPI to avoid memory crash
        pixel_doors = extract_blue_marks(img)
        
        # We don't have a homography matrix yet to map pixels to UTM.
        # But for now, we just output dummy points to satisfy the UI, 
        # or we skip saving if we can't map them.
        
        print("Idea 8 (CV Extraction) Complete. Needs Georeferencing mapping to save to GeoJSON.")
    except Exception as e:
        print(f"Error during Idea 8: {e}")
