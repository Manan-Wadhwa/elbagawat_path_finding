# %%
# CELL 0  Imports, helper functions, and configuration
import ezdxf
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.pyplot as plt
import warnings
import csv
from methods import get_greedy_doors, get_collinear_doors, get_boolean_doors, get_og_doors, get_native_doors, get_local_doors, get_hybrid_doors
warnings.filterwarnings("ignore")

# Paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHP_PATH = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp")
DXF_WORKING = os.path.join(BASE_DIR, "data", "BaseSiteCAD", "BaseSiteCAD", "Site_CAD_Working_converted.dxf")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
OUT_GPKG = os.path.join(OUT_DIR, "ElBagawat_Master.gpkg")
OUT_CSV = os.path.join(OUT_DIR, "ElBagawat_Crosswalk.csv")
OUT_3D = os.path.join(OUT_DIR, "3D_Blueprint.csv")

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

def extract_dxf_entrances(lines, gap_min=500.0, gap_max=3500.0, snap_tol=250.0):
    print("Detecting Topological Gaps...")
    if not lines: return []
    G = nx.Graph()
    def snap(pt, nodes, tol):
        for n in nodes:
            if np.hypot(pt[0]-n[0], pt[1]-n[1]) < tol:
                return n
        return pt
    for p1, p2 in lines:
        n1 = snap(p1, G.nodes, snap_tol)
        n2 = snap(p2, G.nodes, snap_tol)
        G.add_edge(n1, n2)
    loose_ends = [n for n in G.nodes if G.degree(n) == 1]
    entrances = []
    for i in range(len(loose_ends)):
        for j in range(i+1, len(loose_ends)):
            p1, p2 = loose_ends[i], loose_ends[j]
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if gap_min <= dist <= gap_max:
                entrances.append({"p1": p1, "p2": p2, "dist": dist})
    print(f"Found {len(entrances)} raw gaps.")
    return entrances

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

def process_entrances(lines, H, footprints, crosswalk, dxf_labels):
    print("Generating door vectors using 3 distinct methods...")
    
    # Transform lines to UTM
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
    
    loose_end_vectors = {}
    for p1_u, p2_u in lines_utm:
        p1_round = (round(p1_u[0], 2), round(p1_u[1], 2))
        p2_round = (round(p2_u[0], 2), round(p2_u[1], 2))
        if p1_round in loose_ends:
            loose_end_vectors[p1_round] = np.array([p1_u[0]-p2_u[0], p1_u[1]-p2_u[1]])
        if p2_round in loose_ends:
            loose_end_vectors[p2_round] = np.array([p2_u[0]-p1_u[0], p2_u[1]-p1_u[1]])
            
    buffered_mask = footprints.boundary.buffer(3.5).unary_union
    
    og_doors = get_og_doors(loose_ends, 3.5, buffered_mask)
    greedy_doors = get_greedy_doors(loose_ends, 3.5, buffered_mask)
    collinear_doors = get_collinear_doors(loose_ends, loose_end_vectors, 3.5, buffered_mask)
    
    wall_geoms = [LineString([p1, p2]) for p1, p2 in lines_utm]
    walls_gdf = gpd.GeoDataFrame(geometry=wall_geoms, crs=footprints.crs)
    boolean_doors = get_boolean_doors(walls_gdf, footprints)
    
    dxf_labels_utm = {}
    for text, (px, py) in dxf_labels.items():
        dxf_labels_utm[text] = transform_pt((px, py), H)
        
    native_doors = get_native_doors(footprints)
    local_doors = get_local_doors(lines_utm, loose_ends, crosswalk, dxf_labels_utm, footprints)
    hybrid_doors = get_hybrid_doors(greedy_doors, footprints)
    
    print(f"Native: {len(native_doors)}, Local: {len(local_doors)}, Hybrid: {len(hybrid_doors)}")
    return lines_utm, og_doors, greedy_doors, collinear_doors, boolean_doors, native_doors, local_doors, hybrid_doors, walls_gdf

def generate_3d_blueprint(lines, valid_entrances, H, footprints):
    print("Generating 3D Model Blueprint...")
    data = []
    # Lines
    for p1, p2 in lines:
        p1_u = transform_pt(p1, H)
        p2_u = transform_pt(p2, H)
        data.append({"Type": "Wall", "StartX": p1_u[0], "StartY": p1_u[1], "EndX": p2_u[0], "EndY": p2_u[1], "Width": 0})
        
    for ent in valid_entrances:
        p1_u = ent['p1_utm']
        p2_u = ent['p2_utm']
        data.append({"Type": "Door", "StartX": p1_u[0], "StartY": p1_u[1], "EndX": p2_u[0], "EndY": p2_u[1], "Width": ent['width_m']})
        
    df = pd.DataFrame(data)
    df.to_csv(OUT_3D, index=False)
    print(f"Blueprint saved to {OUT_3D}")

def bipartite_label_match(dxf_labels, H, footprints_gdf, max_dist_m=100.0):
    label_coords = []
    label_texts = []
    for text, (px, py) in dxf_labels.items():
        crs_pt = transform_pt((px, py), H)
        label_coords.append(crs_pt)
        label_texts.append(text)
    label_coords = np.array(label_coords)
    footprint_centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in footprints_gdf.geometry])
    footprint_ids = footprints_gdf['ID'].values
    
    cost_matrix = np.zeros((len(label_coords), len(footprint_centroids)))
    for i in range(len(label_coords)):
        for j in range(len(footprint_centroids)):
            dist = np.hypot(label_coords[i][0] - footprint_centroids[j][0],
                            label_coords[i][1] - footprint_centroids[j][1])
            cost_matrix[i, j] = dist if dist < max_dist_m else 1e6
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    results = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 1e6:
            results.append({
                "chapel_id": label_texts[i],
                "footprint_id": str(footprint_ids[j]),
                "dist_m": cost_matrix[i, j],
                "match_method": "bipartite"
            })
    return pd.DataFrame(results)

# %%
# CELL 1  Load shapefile & DXF labels
footprints = gpd.read_file(SHP_PATH)
footprints['ID'] = footprints['ID'].astype(str)

dxf_labels = get_dxf_labels(DXF_WORKING)
H = compute_affine(dxf_labels, footprints)

lines = extract_dxf_walls(DXF_WORKING)

# %%
# CELL 2  Affine transformation & bipartite label matching
crosswalk = bipartite_label_match(dxf_labels, H, footprints)
crosswalk.to_csv(OUT_CSV, index=False)

master_polygons = footprints.merge(crosswalk[['footprint_id', 'chapel_id', 'match_method']], 
                                   left_on='ID', right_on='footprint_id', how='left')
for col in master_polygons.columns:
    if master_polygons[col].dtype == object: master_polygons[col] = master_polygons[col].astype(str)
        
master_polygons.to_file(OUT_GPKG, layer='buildings', driver='GPKG')

# %%
# CELL 3  Process entrances & save vector outputs
print("Saving Data...")
lines_utm, og_doors, greedy_doors, collinear_doors, boolean_doors, native_doors, local_doors, hybrid_doors, walls_gdf = process_entrances(lines, H, footprints, crosswalk, dxf_labels)

walls_gdf.to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "walls.geojson"), driver="GeoJSON")

# Save the 4 door versions
if len(og_doors) > 0: gpd.GeoDataFrame(og_doors, geometry=[d['geom'] for d in og_doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "doors_og.geojson"), driver="GeoJSON")
gpd.GeoDataFrame(greedy_doors, geometry=[d['geom'] for d in greedy_doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "doors_greedy.geojson"), driver="GeoJSON")
gpd.GeoDataFrame(collinear_doors, geometry=[d['geom'] for d in collinear_doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "doors_collinear.geojson"), driver="GeoJSON")
gpd.GeoDataFrame(boolean_doors, geometry=[d['geom'] for d in boolean_doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "doors_boolean.geojson"), driver="GeoJSON")

if len(native_doors) > 0: gpd.GeoDataFrame(native_doors, geometry=[d['geom'] for d in native_doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "doors_native.geojson"), driver="GeoJSON")
if len(local_doors) > 0: gpd.GeoDataFrame(local_doors, geometry=[d['geom'] for d in local_doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "doors_local.geojson"), driver="GeoJSON")
if len(hybrid_doors) > 0: gpd.GeoDataFrame(hybrid_doors, geometry=[d['geom'] for d in hybrid_doors], crs=footprints.crs).to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "doors_hybrid.geojson"), driver="GeoJSON")

# And buildings for the hover effect
footprints.to_crs("EPSG:4326").to_file(os.path.join(OUT_DIR, "vector_gis", "buildings.geojson"), driver="GeoJSON")
print("Pipeline Complete.")
