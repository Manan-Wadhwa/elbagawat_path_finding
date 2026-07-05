import os
"""
Patch pass 4:
 1. Fix LineString crash: skip paths with < 2 points
 2. Replace distance-based pairs with KNN pairs (scipy KDTree, K=K_NEAREST)
    in run_dijkstra_mark_pairs() and bayesian_ensemble()
 3. Reduce PROB_K 20->3 for full-site tractability
 4. Fix cell 14 post-call pairs to use the same KNN logic
"""
import json

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Task1_FullSite.ipynb")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

src2 = "".join(nb["cells"][2]["source"])

# ── 1. Add scipy KDTree import (already has scipy.ndimage, add spatial) ──────
OLD_IMPORT = "from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation"
NEW_IMPORT = ("from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation\n"
              "from scipy.spatial import KDTree")
assert OLD_IMPORT in src2
src2 = src2.replace(OLD_IMPORT, NEW_IMPORT, 1)
print("Added scipy KDTree import")

# ── 2. Reduce PROB_K from 20 to 3 ────────────────────────────────────────────
OLD_PROB = '    "PROB_K": 20,'
NEW_PROB = '    "PROB_K": 3,   # reduced for full-site tractability (was 20 in Task1 subset)'
assert OLD_PROB in src2
src2 = src2.replace(OLD_PROB, NEW_PROB, 1)
print("Reduced PROB_K: 20 -> 3")

# ── 3. Fix run_dijkstra_mark_pairs: KNN pairs + LineString guard ──────────────
OLD_PAIRS_HDR = (
    "def run_dijkstra_mark_pairs(data, composite, mark_nodes, include_prob=True):\n"
    "    mark_rcs = [n[\"rc\"] for n in mark_nodes]\n"
    "    # Proximity-filter: only route between buildings within MAX_PAIR_DIST metres\n"
    "    res = (data[\"res_x\"] + data[\"res_y\"]) / 2.0\n"
    "    max_px = CONFIG.get(\"MAX_PAIR_DIST\", 200) / res\n"
    "    pairs = []\n"
    "    for _i in range(len(mark_rcs)):\n"
    "        for _j in range(_i + 1, len(mark_rcs)):\n"
    "            _dr = mark_rcs[_i][0] - mark_rcs[_j][0]\n"
    "            _dc = mark_rcs[_i][1] - mark_rcs[_j][1]\n"
    "            if (_dr * _dr + _dc * _dc) ** 0.5 <= max_px:\n"
    "                pairs.append((_i, _j))\n"
    "    print(f\"  Routing {len(pairs)} proximity-filtered building pairs (MAX_PAIR_DIST={CONFIG['MAX_PAIR_DIST']}m)\")\n"
)
NEW_PAIRS_HDR = (
    "def run_dijkstra_mark_pairs(data, composite, mark_nodes, include_prob=True):\n"
    "    mark_rcs = [n[\"rc\"] for n in mark_nodes]\n"
    "    # KNN pairs: connect each building to its K nearest neighbours only\n"
    "    k_nn = min(CONFIG.get(\"K_NEAREST\", 4) + 1, len(mark_rcs))\n"
    "    _coords = np.array([[r, c] for r, c in mark_rcs], dtype=np.float32)\n"
    "    _tree = KDTree(_coords)\n"
    "    _, _idxs = _tree.query(_coords, k=k_nn)\n"
    "    _pair_set = set()\n"
    "    for _i, _nbrs in enumerate(_idxs):\n"
    "        for _j in _nbrs[1:]:  # skip self (index 0)\n"
    "            _pair_set.add((min(_i, _j), max(_i, _j)))\n"
    "    pairs = sorted(_pair_set)\n"
    "    print(f\"  Routing {len(pairs)} KNN pairs (K={CONFIG.get('K_NEAREST',4)} neighbours per building)\")\n"
)
assert OLD_PAIRS_HDR in src2, f"run_dijkstra_mark_pairs header not found"
src2 = src2.replace(OLD_PAIRS_HDR, NEW_PAIRS_HDR, 1)
print("Fixed run_dijkstra_mark_pairs: KNN pairs")

# ── 4. Fix LineString guard in run_dijkstra_mark_pairs ────────────────────────
OLD_LINESTRING = (
    "        coords = [rasterio.transform.xy(data[\"dem_tf\"], r, c) for r, c in path]\n"
    "        path_records.append(\n"
    "            {\"source\": f\"M{i}\", \"target\": f\"M{j}\", \"cost\": float(cost_val), \"geometry\": LineString(coords)}\n"
    "        )\n"
)
NEW_LINESTRING = (
    "        coords = [rasterio.transform.xy(data[\"dem_tf\"], r, c) for r, c in path]\n"
    "        if len(coords) < 2:\n"
    "            continue  # skip degenerate single-point paths\n"
    "        path_records.append(\n"
    "            {\"source\": f\"M{i}\", \"target\": f\"M{j}\", \"cost\": float(cost_val), \"geometry\": LineString(coords)}\n"
    "        )\n"
)
assert OLD_LINESTRING in src2, "LineString block not found"
src2 = src2.replace(OLD_LINESTRING, NEW_LINESTRING, 1)
print("Fixed LineString crash guard in run_dijkstra_mark_pairs")

# ── 5. Fix bayesian_ensemble: replace proximity filter with KNN ───────────────
OLD_BAYES_PAIRS = (
    "    # Proximity-filter: same MAX_PAIR_DIST threshold as run_dijkstra_mark_pairs\n"
    "    _max_px_b = CONFIG.get(\"MAX_PAIR_DIST\", 200) / max(res_m, 1e-6)\n"
    "    pairs = []\n"
    "    for _bi in range(len(mark_rcs)):\n"
    "        for _bj in range(_bi + 1, len(mark_rcs)):\n"
    "            _dr = mark_rcs[_bi][0] - mark_rcs[_bj][0]\n"
    "            _dc = mark_rcs[_bi][1] - mark_rcs[_bj][1]\n"
    "            if (_dr * _dr + _dc * _dc) ** 0.5 <= _max_px_b:\n"
    "                pairs.append((_bi, _bj))\n"
    "    print(f\"  Bayesian: {len(pairs)} proximity-filtered pairs\")\n"
)
NEW_BAYES_PAIRS = (
    "    # KNN pairs: same K_NEAREST logic as run_dijkstra_mark_pairs\n"
    "    _k_b = min(CONFIG.get(\"K_NEAREST\", 4) + 1, len(mark_rcs))\n"
    "    _coords_b = np.array([[r, c] for r, c in mark_rcs], dtype=np.float32)\n"
    "    _tree_b = KDTree(_coords_b)\n"
    "    _, _idxs_b = _tree_b.query(_coords_b, k=_k_b)\n"
    "    _pair_set_b = set()\n"
    "    for _bi, _nbrs_b in enumerate(_idxs_b):\n"
    "        for _bj in _nbrs_b[1:]:\n"
    "            _pair_set_b.add((min(_bi, _bj), max(_bi, _bj)))\n"
    "    pairs = sorted(_pair_set_b)\n"
    "    print(f\"  Bayesian: {len(pairs)} KNN pairs\")\n"
)
assert OLD_BAYES_PAIRS in src2, "bayesian pairs block not found"
src2 = src2.replace(OLD_BAYES_PAIRS, NEW_BAYES_PAIRS, 1)
print("Fixed bayesian_ensemble: KNN pairs")

nb["cells"][2]["source"] = [src2]

# ── 6. Fix cell 14: rebuild pairs with same KNN logic ────────────────────────
src14 = "".join(nb["cells"][14]["source"])

OLD_POST = (
    "path_records = []\n"
    "# Rebuild same proximity-filtered pairs that bayesian_ensemble used internally\n"
    "_res_14 = (data[\"res_x\"] + data[\"res_y\"]) / 2.0\n"
    "_max_px_14 = CONFIG.get(\"MAX_PAIR_DIST\", 200) / max(_res_14, 1e-6)\n"
    "pairs = []\n"
    "for _pi in range(len(mark_rcs)):\n"
    "    for _pj in range(_pi + 1, len(mark_rcs)):\n"
    "        _dr = mark_rcs[_pi][0] - mark_rcs[_pj][0]\n"
    "        _dc = mark_rcs[_pi][1] - mark_rcs[_pj][1]\n"
    "        if (_dr * _dr + _dc * _dc) ** 0.5 <= _max_px_14:\n"
    "            pairs.append((_pi, _pj))\n"
    "for (i, j), path in zip(pairs, best_paths):\n"
)
NEW_POST = (
    "path_records = []\n"
    "# Rebuild same KNN pairs that bayesian_ensemble used internally\n"
    "_k_14 = min(CONFIG.get(\"K_NEAREST\", 4) + 1, len(mark_rcs))\n"
    "_coords_14 = np.array([[r, c] for r, c in mark_rcs], dtype=np.float32)\n"
    "_tree_14 = KDTree(_coords_14)\n"
    "_, _idxs_14 = _tree_14.query(_coords_14, k=_k_14)\n"
    "_pair_set_14 = set()\n"
    "for _pi, _nbrs_14 in enumerate(_idxs_14):\n"
    "    for _pj in _nbrs_14[1:]:\n"
    "        _pair_set_14.add((min(_pi, _pj), max(_pi, _pj)))\n"
    "pairs = sorted(_pair_set_14)\n"
    "for (i, j), path in zip(pairs, best_paths):\n"
)
assert OLD_POST in src14, "cell 14 post-pairs not found"
src14 = src14.replace(OLD_POST, NEW_POST, 1)
nb["cells"][14]["source"] = [src14]
print("Fixed cell 14: KNN pairs rebuild")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nNotebook saved!")
