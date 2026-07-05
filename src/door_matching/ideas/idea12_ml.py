import os
# Dynamic workspace path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def main():
    base_dir = BASE_DIR
    buildings_path = os.path.join(base_dir, os.path.join("data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp"))
    doors_gt_path = os.path.join(base_dir, os.path.join("annotator", "doors_idea1.geojson")
    out_path = os.path.join(base_dir, os.path.join("annotator", "doors_idea12.geojson")
    
    print("Loading data...")
    buildings = gpd.read_file(buildings_path)
    
    print("Calculating geometric features...")
    buildings['area'] = buildings.geometry.area
    
    bounds = buildings.bounds
    width = bounds['maxx'] - bounds['minx']
    height = bounds['maxy'] - bounds['miny']
    buildings['aspect_ratio'] = np.where(height == 0, 0, width / height)
    
    def get_orientation(geom):
        rect = geom.minimum_rotated_rectangle
        try:
            coords = list(rect.exterior.coords)
            max_len = 0
            angle = 0
            for i in range(len(coords)-1):
                p1 = coords[i]
                p2 = coords[i+1]
                length = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
                if length > max_len:
                    max_len = length
                    angle = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
            return angle % 180
        except:
            return 0
            
    buildings['orientation'] = buildings.geometry.apply(get_orientation)
    
    distances = np.zeros(len(buildings))
    for idx, row in buildings.iterrows():
        dists = buildings.geometry.distance(row.geometry)
        dists.loc[idx] = np.inf
        distances[idx] = dists.min()
    buildings['dist_to_nn'] = distances
    
    print("Loading Ground Truth labels from Idea 1...")
    doors = gpd.read_file(doors_gt_path)
    buildings['door_label'] = np.nan
    buildings['centroid'] = buildings.geometry.centroid
    
    for idx, door in doors.iterrows():
        dc = door.geometry.centroid
        dists = buildings['centroid'].distance(dc)
        b_idx = dists.idxmin()
        
        bc = buildings.loc[b_idx, 'centroid']
        dx = dc.x - bc.x
        dy = dc.y - bc.y
        
        if abs(dx) > abs(dy):
            label = 2 if dx > 0 else 3 # East / West
        else:
            label = 0 if dy > 0 else 1 # North / South
            
        buildings.at[b_idx, 'door_label'] = label
        
    features = ['area', 'aspect_ratio', 'orientation', 'dist_to_nn']
    
    train_df = buildings.dropna(subset=['door_label'])
    test_df = buildings[buildings['door_label'].isna()]
    
    if len(train_df) == 0:
        print("No training data available. Check doors_idea1.geojson.")
        return
        
    print(f"Training RandomForest on {len(train_df)} annotated buildings...")
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(train_df[features], train_df['door_label'])
    
    print(f"Predicting for {len(test_df)} unannotated buildings...")
    preds = clf.predict(test_df[features])
    
    buildings.loc[test_df.index, 'predicted_label'] = preds
    buildings.loc[train_df.index, 'predicted_label'] = train_df['door_label']
    
    print("Generating door geometries...")
    out_doors = []
    
    for idx, row in buildings.iterrows():
        label = row['predicted_label']
        geom = row.geometry
        cx, cy = row['centroid'].x, row['centroid'].y
        minx, miny, maxx, maxy = geom.bounds
        
        L = max(maxx-minx, maxy-miny) + 100
        if label == 0:
            ray = LineString([(cx, cy), (cx, cy + L)])
        elif label == 1:
            ray = LineString([(cx, cy), (cx, cy - L)])
        elif label == 2:
            ray = LineString([(cx, cy), (cx + L, cy)])
        else:
            ray = LineString([(cx, cy), (cx - L, cy)])
            
        intersection = geom.boundary.intersection(ray)
        
        if intersection.is_empty:
             px, py = cx, cy
             if label == 0: py = maxy
             elif label == 1: py = miny
             elif label == 2: px = maxx
             elif label == 3: px = minx
        elif intersection.geom_type == 'Point':
             px, py = intersection.x, intersection.y
        elif intersection.geom_type == 'MultiPoint':
             pts = list(intersection.geoms)
             pts.sort(key=lambda p: Point(cx, cy).distance(p), reverse=True)
             px, py = pts[0].x, pts[0].y
        else:
             px, py = intersection.centroid.x, intersection.centroid.y
             
        if label in [0, 1]:
             door_geom = LineString([(px - 0.5, py), (px + 0.5, py)])
        else:
             door_geom = LineString([(px, py - 0.5), (px, py + 0.5)])
             
        out_doors.append({
            'geometry': door_geom,
            'building_id': idx,
            'label': int(label),
            'is_predicted': pd.isna(row['door_label'])
        })
        
    doors_gdf = gpd.GeoDataFrame(out_doors, crs=buildings.crs)
    # create output dir if needed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doors_gdf.to_file(out_path, driver='GeoJSON')
    print(f"Saved {len(doors_gdf)} doors to {out_path}")

if __name__ == "__main__":
    main()
