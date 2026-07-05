# Phase 0: Step 0b — Read the QGIS GCP Files and Selection Analysis

## Metadata
- **Date**: 2026-06-24
- **Target Files**:
  - `100_Data/130_BuildingFootprintsVectorData/BuildingTracesCurrent/Buildings_Mask.shp.points` (23 points)
  - `100_Data/130_BuildingFootprintsVectorData/BuildingTracesCurrent/Buildings_Mask.shp.points1.points` (29 points)
  - `100_Data/130_BuildingFootprintsVectorData/BuildingTracesCurrent/Buildings_Mask.shp.points2.points` (32 points)

---

## 1. File Inventory and Coordinate Validation
A comparative analysis was run on the three GCP files in the active footprints directory.
- All three files share the same coordinate system (`PROJCRS["WGS 84 / UTM zone 36N", ...]` - `EPSG:32636`).
- Encoding was identified as `latin-1` due to the degree symbol (`°`) in the CRS header comments.
- The columns in each file are: `['mapX', 'mapY', 'sourceX', 'sourceY', 'enable', 'dX', 'dY', 'residual']`.
  - `sourceX` and `sourceY` correspond to the coordinates of the source raster.
  - `mapX` and `mapY` correspond to the coordinates of the target map.
  - `enable` is the binary flag (1.0 = enabled) indicating whether to use the point in fitting. All points in all three files are enabled (100% active).

---

## 2. Subset Relationship and Coordinate Agreement
I tested the files for subset relationships and coordinate differences:
- `points` (23 points) is a strict subset of `points1` (29 points).
- `points1` (29 points) is a strict subset of `points2` (32 points).
- The maximum coordinate difference for matching points across all three files is **exactly 0.0**.
  - `points` vs `points1` (23 matches): Max source pixel difference = `0.000000`
  - `points1` vs `points2` (29 matches): Max source pixel difference = `0.000000`
  - `points` vs `points2` (23 matches): Max source pixel difference = `0.000000`

---

## 3. Recommended File Selection
Because there is perfect coordinate agreement on all overlapping points, we select the set with the highest density of control points to minimize registration error and spatial distortion.

- **Authoritative Selection**: **`Buildings_Mask.shp.points2.points`**
- **Point Count**: 32 enabled points
- **Spatial Envelope**:
  - `mapX`: `[254087.22, 254269.70]` (UTM meters)
  - `mapY`: `[2820828.09, 2821252.91]` (UTM meters)
  - `sourceX`: `[254084.93, 254268.59]`
  - `sourceY`: `[2820825.56, 2821249.19]`

---

## 4. Evaluation of GCP Transformation Quality
An affine transformation was fitted to the selected 32 GCPs from `source` space to `map` space:
- **Projective Transform (Homography) RMSE**: **`0.978 meters`**
- **Affine Transform RMSE**: **`0.997 meters`**
- **Affine Matrix Coefficients**:
  - `mapX = 0.99693213 * sourceX - 0.00597522 * sourceY + 17637.4366`
  - `mapY = 0.00157438 * sourceX + 1.00401030 * sourceY - 11710.6325`

The sub-meter RMSE confirms that the GCP points are highly consistent and represent a minor coordinate shift/shear adjustment between the source layer and the target UTM zone 36N coordinates.
