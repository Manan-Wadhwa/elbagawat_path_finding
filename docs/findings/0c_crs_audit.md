# Phase 0: Step 0c — CRS Audit and Alignment Decisions

## Metadata
- **Date**: 2026-06-24
- **Working Coordinate Reference System (CRS)**: **`EPSG:32636`** (WGS 84 / UTM zone 36N)
- **Target Coordinate Units**: Meters

---

## 1. Inventory CRS Audit Results
I performed a CRS audit on all spatial files located in the project's data subdirectories:

| Layer ID | Relative File Path | Detected CRS | Alignment Action |
|---|---|---|---|
| **`footprints`** | `100_Data/130_BuildingFootprintsVectorData/BuildingTracesCurrent/Buildings_Mask.shp` | **`EPSG:32636`** | None (Authoritative Baseline) |
| **`roi_full`** | `100_Data/110_GISRegionOfInterest/Bagawat_ROI.shp` | **`EPSG:4326`** | **Reproject** to `EPSG:32636` |
| **`roi_small`** | `100_Data/110_GISRegionOfInterest/BagawatROI_Smaller.shp` | **`EPSG:4326`** | **Reproject** to `EPSG:32636` |
| **`dem`** | `100_Data/150_DigitalElevationModel/Generated_DEMs/Current_DEM/Bagawat-DEM-NewImageryOnly-0.4m-DEM.tif` | **`EPSG:32636`** | None (Matches Working CRS) |
| **`wv2_p001`** | `100_Data/140_SAR_Imagery/DigitalGlobe_2018/MONO/058239078010_01/058239078010_01_P001_MUL/18JUN14090738-M2AS_R1C1-058239078010_01_P001.TIF` | **`EPSG:32636`** | None (Matches Working CRS) |

---

## 2. CRS Alignment Decisions
1. **Authoritative Working CRS**: `EPSG:32636` (WGS 84 / UTM zone 36N) will serve as the project-wide coordinate reference system. This matches the metric projection used by the footprint traces, DEM, and WV-2 tiles.
2. **Reprojection Plan**:
   - The boundary vector files, [Bagawat_ROI.shp](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/100_Data/110_GISRegionOfInterest/Bagawat_ROI.shp) and [BagawatROI_Smaller.shp](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/100_Data/110_GISRegionOfInterest/BagawatROI_Smaller.shp), are currently in geographic coordinates (`EPSG:4326`). They must be reprojected to `EPSG:32636` using `geopandas` (`gdf.to_crs(epsg=32636)`) prior to any spatial masking or clipping.
   - The primary DEM and satellite raster imagery are natively aligned to `EPSG:32636`, so they do not require resampling for coordinate alignment.
