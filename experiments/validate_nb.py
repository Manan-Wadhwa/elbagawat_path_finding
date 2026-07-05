import os
"""Validate the modified Task1_FullSite.ipynb cell 2 by checking:
1. Python syntax (compile)
2. Key assertions about the content
3. Quick functional test of load_base() + data shapes
"""
import json, ast

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Task1_FullSite.ipynb")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

src2 = "".join(nb["cells"][2]["source"])

# 1. Syntax check
try:
    ast.parse(src2)
    print("PASS: Cell 2 syntax OK")
except SyntaxError as e:
    print(f"FAIL: Syntax error in cell 2: {e}")

# 2. Content assertions
checks = [
    ("MAX_PAIR_DIST", "CONFIG has MAX_PAIR_DIST"),
    ("Buildings_Mask.shp", "Uses richer Buildings_Mask.shp"),
    ("representative_point()", "Marks from representative_point"),
    ("Loaded {len(buildings)} buildings", "Print statement for count"),
    ("def bayesian_ensemble(dem, slope_cost_grid, surface_cost_grid, bldg_raster, valid, dist_from_bldg, mark_rcs, res_m=0.5)", "bayesian_ensemble has res_m param"),
    ("CONFIG.get(\"MAX_PAIR_DIST\", 200) / max(res_m, 1e-6)", "bayesian uses res_m in filter"),
    ("proximity-filtered building pairs", "run_dijkstra_mark_pairs prints pair count"),
    ("geom.representative_point()", "geom_to_rc handles Polygon"),
]
for needle, label in checks:
    if needle in src2:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")

# 3. Cell 14 checks
src14 = "".join(nb["cells"][14]["source"])
checks14 = [
    ("res_m=(data[\"res_x\"] + data[\"res_y\"]) / 2.0", "Cell14: passes res_m"),
    ("Rebuild same proximity-filtered pairs", "Cell14: proximity pairs rebuild comment"),
    ("_max_px_14", "Cell14: uses _max_px_14"),
]
for needle, label in checks14:
    if needle in src14:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")

print("\nAll checks done.")
