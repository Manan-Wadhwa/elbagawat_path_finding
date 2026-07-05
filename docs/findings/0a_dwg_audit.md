# Phase 0: Step 0a — Inspect the DWG and DXF Layer Audit

## Metadata
- **Date**: 2026-06-24
- **Target File**: [Site_CAD_Working_converted.dxf](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/100_Data/120_SiteReport/BaseSiteCAD/Site_CAD_Working_converted.dxf) (converted from `SITE CAD WORKING.dwg`)
- **GIS Footprints**: [Buildings_Mask.shp](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/100_Data/130_BuildingFootprintsVectorData/BuildingTracesCurrent/Buildings_Mask.shp) (UTM Zone 36N, EPSG:32636)

---

## 1. Layer and Entity Counts
The CAD file structure includes 7 layers with the following model space entity counts:
- **`0`**: 1 `IMAGE` (points to raster scan overlay)
- **`BUILDINGS`**: 832 geometries (`LWPOLYLINE` outlines, arcs, circles)
- **`NUMBERING`**: 275 `TEXT` entities (numerical chapel identifiers)
- **`LW1`**: 195 geometries (originally off/hidden, contains lightweight outlines)
- **`LW2`**: 208 geometries (originally off/hidden, contains lightweight outlines)
- **`ABOVE`**: 111 geometries (originally off/hidden, contains upper/roof outlines)
- **`Defpoints`**: Standard non-plotting guide points

---

## 2. Coordinate System Segmentation (Space A vs. Space B)
The drawing contains two disjoint coordinate spaces:
1. **Space A (Main Map)**: X in `[-40000, 194000]`, Y in `[-53000, 394000]`. This space contains the main site-wide survey map representing the 263 buildings (`BUILDINGS` layer) and text labels 1–263 (`NUMBERING` layer).
2. **Space B (Detail Sheets)**: X in `[240000, 310000]`, Y in `[-2000, 22000]`. This space contains detailed floor plans (Level 1, Level 2, and Above) for the 7 calibration buildings, positioned near detail labels `1`, `23`, `24`, `25` (Peace), `26`, `175`, `210`.

---

## 3. Coordinate Reconciliation and Scale Mapping
An affine transformation was calculated mapping Space A text label coordinates to the shapefile building centroids:
- **Scale Factor**: Exact scale is **`0.001`**, confirming that the CAD drawing units are in **millimeters** and the shapefile units are in **meters**.
- **Rotation**: Negligible (less than 0.001 degrees).
- **Residuals**: The median distance error is **`1.03 meters`**, which represents hand-placement positioning of text labels inside the building outlines rather than geometric distortion.
- **Transformation Formula (DXF -> UTM)**:
  - `shp_x = 0.000998898903 * dxf_x + 0.000014609945 * dxf_y + 254120.7365`
  - `shp_y = -0.000011192744 * dxf_x + 0.000992123572 * dxf_y + 2820859.1231`

---

## 4. Evaluation of Path-Related Layers
The layers `LW1` (Level 1 detail) and `LW2` (Level 2 detail) do not contain path or road geometry:
- They are located entirely in Space B and represent room configurations and architectural structures for individual buildings.
- A buffer intersection check shows that only $11$-$12\%$ of their entities intersect the main buildings layer, confirming they represent decoupled detail plans rather than a path network.
- **Conclusion**: There is **no path/road geometry** in the CAD file layer structure. Path reconstruction must be performed synthetically using down-stream modeling (FETE, circuits, spectral).
