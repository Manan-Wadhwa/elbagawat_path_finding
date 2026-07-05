import geopandas as gpd
import ezdxf
import numpy as np
import math
from shapely.geometry import Point, LineString, Polygon
from scipy.spatial import KDTree
from shapely.ops import nearest_points
from pipeline import get_dxf_labels, extract_dxf_walls, compute_affine, transform_pt, bipartite_label_match
import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SHP_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp")
DXF_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "Site_CAD_Working_converted.dxf")

def get_angle(dx, dy):
    return math.degrees(math.atan2(dy, dx))

def get_point_at_angle(polygon, centroid, angle_deg):
    angle_rad = math.radians(angle_deg)
    far_dist = 1000.0
    far_x = centroid.x + far_dist * math.cos(angle_rad)
    far_y = centroid.y + far_dist * math.sin(angle_rad)
    ray = LineString([centroid, Point(far_x, far_y)])
    intersection = polygon.boundary.intersection(ray)
    if intersection.is_empty:
        return nearest_points(polygon.boundary, centroid)[0]
    if intersection.geom_type == 'Point':
        return intersection
    elif intersection.geom_type == 'MultiPoint':
        pts = list(intersection.geoms)
        pts.sort(key=lambda p: centroid.distance(p))
        return pts[0]
    return nearest_points(polygon.boundary, centroid)[0]

def run():
    print("Loading data...")
    footprints = gpd.read_file(SHP_PATH)
    footprints['ID'] = footprints['ID'].astype(str)
    
    dxf_labels = get_dxf_labels(DXF_PATH)
    H = compute_affine(dxf_labels, footprints)
    lines = extract_dxf_walls(DXF_PATH)
    
    print("Matching labels...")
    crosswalk = bipartite_label_match(dxf_labels, H, footprints)
    
    print("Finding gaps in DXF space...")
    points = {}
    for p1, p2 in lines:
        p1_round = (round(p1[0], 1), round(p1[1], 1))
        p2_round = (round(p2[0], 1), round(p2[1], 1))
        points[p1_round] = points.get(p1_round, 0) + 1
        points[p2_round] = points.get(p2_round, 0) + 1
        
    loose_ends = [p for p, count in points.items() if count == 1]
    
    dxf_gaps = []
    for i in range(len(loose_ends)):
        for j in range(i+1, len(loose_ends)):
            dist = np.linalg.norm(np.array(loose_ends[i]) - np.array(loose_ends[j]))
            if 500 <= dist <= 3500:
                midpoint = ((loose_ends[i][0] + loose_ends[j][0]) / 2, 
                            (loose_ends[i][1] + loose_ends[j][1]) / 2)
                dxf_gaps.append((loose_ends[i], loose_ends[j], midpoint))
                
    doors = []
    if not dxf_gaps:
        print("No gaps found in DXF space.")
        return
        
    label_items = list(dxf_labels.items())
    label_coords = [np.array(coord) for _, coord in label_items]
    kdtree = KDTree(label_coords)
    
    print("Placing doors natively...")
    for gap in dxf_gaps:
        p1, p2, mid = gap
        dist, idx = kdtree.query(mid)
        nearest_label = label_items[idx][0]
        label_coord = label_items[idx][1]
        
        if dist > 30000:
            continue
            
        dx = mid[0] - label_coord[0]
        dy = mid[1] - label_coord[1]
        angle_deg = get_angle(dx, dy)
        
        match_row = crosswalk[crosswalk['chapel_id'] == nearest_label]
        if match_row.empty:
            continue
            
        fp_id = str(match_row.iloc[0]['footprint_id'])
        matching_fps = footprints[footprints['ID'] == fp_id]
        if matching_fps.empty:
            continue
        fp_geom = matching_fps.iloc[0].geometry
        
        centroid = fp_geom.centroid
        placement_pt = get_point_at_angle(fp_geom, centroid, angle_deg)
        
        door_line = LineString([(placement_pt.x - 0.5, placement_pt.y - 0.5), 
                                (placement_pt.x + 0.5, placement_pt.y + 0.5)])
        doors.append({"geom": door_line, "type": "pure_dxf", "label": nearest_label})
        
    if len(doors) > 0:
        gdf = gpd.GeoDataFrame(doors, geometry=[d['geom'] for d in doors], crs=footprints.crs)
        out_path = os.path.join(BASE_DIR, "annotator", "doors_idea7.geojson")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
        print(f"Exported {len(doors)} Idea 7 doors to {out_path}")
    else:
        print("No doors found for Idea 7.")

if __name__ == "__main__":
    run()
