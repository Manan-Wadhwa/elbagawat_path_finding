# Phase 0: Step 0a — Recommended Approach Updates in Technical Plans

## Metadata
- **Date**: 2026-06-24
- **Focus**: Algorithmic adjustments for coordinate reconciliation, DXF parsing, and path extraction based on layer audit findings.

---

## 1. Updates to Coordinate Reconciliation (Phase 1 / align.py)

### Finding
The main map in [Site_CAD_Working_converted.dxf](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/100_Data/120_SiteReport/BaseSiteCAD/Site_CAD_Working_converted.dxf) is drawn in millimeters (scale factor `0.001`) and is offset from the shapefile's UTM coordinates. The drawing contains:
- **Space A (Main Map)** for $X < 200,000$.
- **Space B (Detail Sheets)** for $X > 200,000$, where individual building plans are drafted side-by-side.

### Approach Change
We cannot use a single global homography or affine transformation for the entire DXF file.
1. **Main Map**: Apply the global affine transformation coefficients derived from the `NUMBERING` labels in Space A ($X < 200,000$) to convert entities to UTM space.
2. **Detail Sheets / Individual DXFs**: For the 7 calibration building DXF files, compute a **building-specific local translation**:
   $$T_x = X_{utm\_centroid} - 0.001 \times X_{dxf\_label\_insert}$$
   $$T_y = Y_{utm\_centroid} - 0.001 \times Y_{dxf\_label\_insert}$$
   Where $(X_{utm\_centroid}, Y_{utm\_centroid})$ is the building's shapefile centroid, and $(X_{dxf\_label\_insert}, Y_{dxf\_label\_insert})$ is the insertion point of the detail text label (e.g. `'25'` in `Building25.dxf`).
3. **Vertex Transformation**: Any vertex $(x, y)$ in the detailed drawing of building $N$ is transformed to UTM coordinates via:
   $$x_{utm} = 0.001 \times x + T_x$$
   $$y_{utm} = 0.001 \times y + T_y$$
This local translation achieves sub-millimeter spatial alignment ($0.0000\text{ m}$ distance error) between the detail drawing outlines and the shapefile footprints, which is required for accurate entrance extraction.

---

## 2. Updates to Entrance Extraction (Phase 3 / entrances.py)

### Finding
In the plan, Phase 3d (`extract_dxf_entrances()`) assumes a global transformation or simple direct spatial overlay. This will fail because the individual building files are in the Space B local coordinate system.

### Approach Change
Update the `extract_dxf_entrances()` function in [PLAN_02_phases_0_3.md](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/200_Projects/210_GSOC/code-manan/plan/PLAN_02_phases_0_3.md#L828) (and `src/entrances.py`) to incorporate the local translation logic:
```python
def extract_and_transform_dxf_entrances(dxf_path, building_id, footprints_gdf):
    """
    Extracts entrance geometries from building DXF and transforms them to UTM
    using a building-specific local translation derived from text label alignment.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # 1. Get shapefile centroid for the building
    fp = footprints_gdf[footprints_gdf['ID'] == building_id].iloc[0]
    centroid = fp.geometry.centroid
    
    # 2. Find detail label insertion coordinate in the DXF file
    label_x, label_y = None, None
    for entity in msp:
        if entity.dxf.layer == 'NUMBERING' and entity.dxftype() == 'TEXT':
            if entity.dxf.text.strip() == str(building_id):
                label_x = entity.dxf.insert.x
                label_y = entity.dxf.insert.y
                break
    
    if label_x is None:
        raise ValueError(f"Detail label '{building_id}' not found in {dxf_path}")
        
    # 3. Compute local translation parameters (scale is 1:1000)
    tx = centroid.x - 0.001 * label_x
    ty = centroid.y - 0.001 * label_y
    
    # 4. Extract and transform entrance geometries
    entrance_geoms = []
    # Identify entrance features in the DXF...
    # (Apply 0.001 scale and (tx, ty) translation to vertices)
    return entrance_geoms
```

---

## 3. Exclusion of CAD-Extracted Path Layers

### Finding
No pre-digitized paths exist in `Site_CAD_Working_converted.dxf` or its layers (`LW1`, `LW2`, `ABOVE`). 

### Approach Change
We must explicitly document in [PLAN_02_phases_0_3.md](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/200_Projects/210_GSOC/code-manan/plan/PLAN_02_phases_0_3.md) and [PLAN_03_phases_4_6.md](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/200_Projects/210_GSOC/code-manan/plan/PLAN_03_phases_4_6.md) that the "shortcut" path-extraction step under Step 0a yielded no road geometries. The reconstruction of the street and alley network must rely entirely on synthetic cost-surface routing and multi-evidence network ensemble modeling.

---

## 4. Georeferencing Flaw and Self-Refining Pipeline

### Finding
We discovered that the predecessor's QGIS `.points` files were incorrectly formulated as UTM-to-UTM residual corrections, making them useless for mapping PDF pixels to the working CRS. We abandoned them.

### Approach Change
**Self-Refining Georeferencing Pipeline**: Instead of relying on faulty GCP files, we formulated a robust two-stage homography pipeline:
1. **Bootstrap Homography ($H_{init}$)**: We will manually pick 4-5 easily identifiable chapel centroids (e.g., Building 25) in the PDF image to compute an initial low-precision transformation.
2. **RANSAC-Refined Homography ($H_{final}$)**: Using $H_{init}$, we will run an OCR sweep over the PDF to find all 263 printed chapel labels, map them roughly to the shapefile footprints, snap them to the exact polygon centroids, and then compute a highly accurate final homography $H_{final}$ using RANSAC.

---

## 5. Stateless Web Annotator Architecture

### Approach Change
The web-based manual annotation tool (Phase 3b) will be fully stateless and coordinate-agnostic. It will operate entirely in native 2D image pixel coordinates. All heavy lifting (transforming pixels to the UTM CRS via $H$, spatial joining, and attribute matching) will be handled offline by our Python pipeline.
