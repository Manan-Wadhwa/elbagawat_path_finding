# Phase 0: Step 0d — Parse Individual DXF Files and Spatial Alignment Verification

## Metadata
- **Date**: 2026-06-24
- **Verification Target**: 7 Calibration Building DXF Files
- **Shapefile Reference**: [Buildings_Mask.shp](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/100_Data/130_BuildingFootprintsVectorData/BuildingTracesCurrent/Buildings_Mask.shp) (UTM Zone 36N, EPSG:32636)

---

## 1. Audit and Mapping Results
I validated the spatial overlay of all 7 individual DXF drawings relative to their matching shapefile polygons. The alignment uses local translation offset corrections:
$$T_x = X_{utm\_centroid} - 0.001 \times X_{dxf\_label}$$
$$T_y = Y_{utm\_centroid} - 0.001 \times Y_{dxf\_label}$$

Here are the verification metrics:

| Building ID | DXF File | Centroid (UTM Easting, Northing) | Distance to Outline (m) | Footprint Area Overlap % (Buffered DXF) |
|---|---|---|---|---|
| **`1`** | `Building1.dxf` | `(254188.75, 2821246.76)` | **`0.0000`** | **`95.3%`** |
| **`23`** | `Building23.dxf` | `(254213.16, 2821101.31)` | **`0.0000`** | **`83.6%`** |
| **`24`** | `Building24.dxf` | `(254218.65, 2821092.42)` | **`0.0000`** | **`83.0%`** |
| **`25`** | `Building25.dxf` | `(254225.38, 2821088.71)` | **`0.0000`** | **`99.5%`** (Peace Chapel) |
| **`26`** | `Building26.dxf` | `(254231.13, 2821077.29)` | **`0.0000`** | **`92.2%`** |
| **`175`** | `Building175.dxf` | `(254218.87, 2821027.31)` | **`0.0000`** | **`100.0%`** |
| **`210`** | `Building210.dxf` | `(254271.02, 2821018.00)` | **`0.0000`** | **`100.0%`** |

---

## 2. Door/Entrance Swing Geometry Details
I checked for architectural components representing entrance features (like arcs representing door swings) in layers `LW1` and `LW2`:
- **Buildings 1, 23, 175, 210**: Contain `0` arc entities. Wall openings are drafted as simple gap segments in polylines rather than swing arcs.
- **Building 24**: Contains `12` arcs (door swings).
- **Building 25** (Peace Chapel): Contains `8` arcs (door swings).
- **Building 26**: Contains `4` arcs (door swings).

The centers of these arcs are located at the wall boundaries of their building footprints (distances of $0.09$-$1.77\text{ m}$ to the exterior polygon), verifying that they represent door swing swings.

---

## 3. Implementation Decision
To extract precision calibration entrances:
1. Locate the centroids of the swing arcs (for Buildings 24, 25, 26) or the midpoints of wall gap segments (for Buildings 1, 23, 175, 210) within the local detail coordinates.
2. Apply the specific $(T_x, T_y)$ offset and $0.001$ scale factor to obtain UTM coordinates.
3. Use these 7 transformed coordinates as the high-accuracy "dxf" calibration set (confidence = 1.0) to override all other inputs and validate the automated PDF mark extraction parameters (Phase 3a).
