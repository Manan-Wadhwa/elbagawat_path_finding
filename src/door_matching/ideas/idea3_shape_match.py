import geopandas as gpd
import ezdxf
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint
import cv2
import networkx as nx
from pipeline import get_dxf_labels, compute_affine, extract_dxf_walls, transform_pt
import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SHP_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp")
DXF_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "Site_CAD_Working_converted.dxf")

def get_hu_moments(polygon):
    if polygon.is_empty:
        return np.zeros(7)
    coords = np.array(polygon.exterior.coords)
    contour = coords.reshape((-1, 1, 2)).astype(np.float32)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments

def run():
    print("Loading data...")
    footprints = gpd.read_file(SHP_PATH)
    dxf_labels = get_dxf_labels(DXF_PATH)
    H = compute_affine(dxf_labels, footprints)
    
    lines = extract_dxf_walls(DXF_PATH)
    
    print("Grouping DXF lines into clusters by proximity...")
    G = nx.Graph()
    for i, l in enumerate(lines):
        G.add_node(i, line=l)
    
    endpoints = []
    for i, l in enumerate(lines):
        endpoints.append((i, np.array(l[0]), np.array(l[1])))
    
    snap_tol = 500.0
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            if (np.linalg.norm(endpoints[i][1] - endpoints[j][1]) < snap_tol or
                np.linalg.norm(endpoints[i][1] - endpoints[j][2]) < snap_tol or
                np.linalg.norm(endpoints[i][2] - endpoints[j][1]) < snap_tol or
                np.linalg.norm(endpoints[i][2] - endpoints[j][2]) < snap_tol):
                G.add_edge(i, j)
                
    clusters = list(nx.connected_components(G))
    doors = []
    
    print("Computing Hu Moments and matching...")
    for cluster in clusters:
        cluster_lines = [lines[i] for i in cluster]
        
        utm_lines = []
        pts_utm = []
        for p1, p2 in cluster_lines:
            p1_u = transform_pt(p1, H)
            p2_u = transform_pt(p2, H)
            utm_lines.append((p1_u, p2_u))
            pts_utm.extend([p1_u, p2_u])
            
        if len(pts_utm) < 3:
            continue
            
        cluster_hull = MultiPoint(pts_utm).convex_hull
        if not isinstance(cluster_hull, Polygon):
            continue
            
        hu_cluster = get_hu_moments(cluster_hull)
        
        cluster_centroid = cluster_hull.centroid
        distances = footprints.geometry.distance(cluster_centroid)
        nearby_indices = distances.nsmallest(5).index
        
        best_match = None
        best_diff = float('inf')
        
        for idx in nearby_indices:
            fp_geom = footprints.loc[idx].geometry
            fp_hull = fp_geom.convex_hull
            if not isinstance(fp_hull, Polygon):
                continue
            hu_fp = get_hu_moments(fp_hull)
            
            diff = np.linalg.norm(hu_cluster - hu_fp)
            if diff < best_diff:
                best_diff = diff
                best_match = fp_geom
                
        if best_match is not None and best_diff < 5.0:
            points = {}
            for p1_u, p2_u in utm_lines:
                p1_round = (round(p1_u[0], 2), round(p1_u[1], 2))
                p2_round = (round(p2_u[0], 2), round(p2_u[1], 2))
                points[p1_round] = points.get(p1_round, 0) + 1
                points[p2_round] = points.get(p2_round, 0) + 1
                
            loose_ends = [p for p, count in points.items() if count == 1]
            
            for i in range(len(loose_ends)):
                for j in range(i+1, len(loose_ends)):
                    dist = np.linalg.norm(np.array(loose_ends[i]) - np.array(loose_ends[j]))
                    if 0.5 <= dist <= 3.5:
                        door_line = LineString([loose_ends[i], loose_ends[j]])
                        doors.append({"geom": door_line, "type": "shape_match", "diff": best_diff})

    if len(doors) > 0:
        gdf = gpd.GeoDataFrame(doors, geometry=[d['geom'] for d in doors], crs=footprints.crs)
        out_path = os.path.join(BASE_DIR, "annotator", "doors_idea3.geojson")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
        print(f"Exported {len(doors)} Idea 3 doors to {out_path}")
    else:
        print("No doors found for Idea 3.")

if __name__ == "__main__":
    run()
