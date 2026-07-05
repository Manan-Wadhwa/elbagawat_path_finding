import os
import geopandas as gpd
import pandas as pd

shp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "BaseSiteCAD", "130_BuildingFootprintsVectorData", "BuildingTracesCurrent", "Buildings_Mask.shp")
gdf = gpd.read_file(shp_path)
print(gdf.columns)
print(gdf.head())
