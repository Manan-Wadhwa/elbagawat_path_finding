import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import numpy as np
import warnings
warnings.filterwarnings("ignore")

fp = gpd.read_file(ros.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp"))
master = gpd.read_file('ElBagawat_Master.gpkg', layer='entrances')

print(f"Original entrances: {len(master)}")

# 1. Boundary Masking
boundaries = fp.boundary
buffered_boundaries = boundaries.buffer(1.5) # 1.5 meter buffer around exterior walls
boundary_union = buffered_boundaries.unary_union

valid_entrances = master[master.geometry.intersects(boundary_union)]
print(f"After boundary masking: {len(valid_entrances)}")

# 2. Clustering
coords = np.array([[geom.x, geom.y] for geom in valid_entrances.geometry])
if len(coords) > 0:
    clustering = DBSCAN(eps=2.0, min_samples=1).fit(coords) # cluster points within 2 meters
    valid_entrances['cluster'] = clustering.labels_
    
    # Take centroid of each cluster
    merged = valid_entrances.dissolve(by='cluster')
    merged.geometry = merged.geometry.centroid
    print(f"After spatial clustering: {len(merged)}")
