import os
"""Run cells 1-8 of Task1_FullSite.ipynb including Approach 1 (Dijkstra)."""
import json, sys, traceback, os
import matplotlib
matplotlib.use("Agg")  # headless - no display

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Task1_FullSite.ipynb")
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

os.chdir(CODE_DIR)
globs = {}

TARGET_CELLS = [1, 2, 3, 4, 5, 6, 7, 8]  # up through Approach 1 Dijkstra

for cell_idx in TARGET_CELLS:
    c = nb["cells"][cell_idx]
    src = "".join(c["source"])
    ct = c["cell_type"]
    if ct != "code":
        print(f"Cell {cell_idx}: markdown, skipping")
        continue
    print(f"\n--- Executing Cell {cell_idx} ---")
    try:
        exec(compile(src, f"<cell{cell_idx}>", "exec"), globs)
        print(f"  Cell {cell_idx}: OK")
    except Exception:
        print(f"  Cell {cell_idx}: ERROR")
        traceback.print_exc()
        sys.exit(1)

# Report on Approach 1 outputs
import pathlib
out = pathlib.Path(CODE_DIR) / "output"
m1_shp = out / "m1_dijkstra" / "m1_dijkstra_paths.shp"
m1_png = out / "m1_dijkstra" / "m1_dijkstra_plot.png"
m1_hm  = out / "m1_dijkstra" / "m1_dijkstra_heatmap.tif"
print(f"\n=== Approach 1 Dijkstra outputs ===")
print(f"  Shapefile:  {'EXISTS' if m1_shp.exists() else 'MISSING'}  ({m1_shp})")
print(f"  Plot:       {'EXISTS' if m1_png.exists() else 'MISSING'}")
print(f"  Heatmap:    {'EXISTS' if m1_hm.exists() else 'MISSING'}")

if m1_shp.exists():
    import geopandas as gpd
    gdf = gpd.read_file(m1_shp)
    print(f"  Paths found: {len(gdf)}")
    if not gdf.empty:
        print(f"  Total path length: {gdf.geometry.length.sum():.1f} m")
        print(f"  Mean path length:  {gdf.geometry.length.mean():.1f} m")
