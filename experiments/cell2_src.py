from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.transform import rowcol
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.plot import show
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import heapq
from itertools import product

BASE = Path(__file__).resolve().parent.parent
print(BASE)
DATA = BASE
if not DATA.exists():
    pass
OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "N_CLUSTERS": 4,
    "SAMPLE_MAX": 40000,
    "SLOPE_EXP": 20.0,
    "W_SLOPE": 0.45,
    "W_SURF": 0.55,
    "CORRIDOR_PENALTY": 0.55,
    "CORRIDOR_DECAY": 18.0,
    "SNAP_RADIUS": 15,
    "K_NEAREST": 4,
    "PROB_K": 20,
    "PROB_NOISE": 0.12,
    "RW_AGENTS": 3000,
    "RW_BETA": 3.0,
    "RW_GOAL_ALPHA": 0.4,
    "RW_MAX_STEPS": 2000,
}

WALK_PRESETS = {
    "original": [0.5, 0.9, 1.5, 2.2],
    "compressed": [0.7, 0.9, 1.1, 1.4],
}
WALK_PRESET = "compressed"

def align_to_dem(path, dem_tf, dem_crs, H, W):
    with rasterio.open(path) as src:
        arr = np.zeros((src.count, H, W), dtype=np.float32)
        for i in range(1, src.count + 1):
            reproject(
                rasterio.band(src, i),
                arr[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dem_tf,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )
    return arr


def load_base():
    with rasterio.open(DATA / "Digital Elevation Model/DEM_20250502/Bagawat_DEM.tif") as src:
        dem = src.read(1).astype(np.float32)
        dem_meta = src.meta.copy()
        dem_crs = src.crs
        dem_tf = src.transform
        H, W = src.height, src.width
        res_x, res_y = src.res
        dem_nd = src.nodata

    valid = np.isfinite(dem) if dem_nd is None else (dem != dem_nd)
    ortho = align_to_dem(DATA / "Digital Elevation Model/DEM_20250502/Generated_DEMs/Current_DEM/bagawat-OrthoImage_GeoEyeAndWV2_NotBundled_20250301.tif", dem_tf, dem_crs, H, W)
    sar = ortho.copy()

    buildings = gpd.read_file(DATA / "Geospatial Data/Building Traces/building_masks.shp").to_crs(dem_crs)
    marks = gpd.read_file(DATA / "Geospatial Data/Building Traces/building_masks.shp").to_crs(dem_crs)

    shapes = [(geom, 1) for geom in buildings.geometry if geom is not None]
    bldg_raster = rasterize(
        shapes, out_shape=(H, W), transform=dem_tf, fill=0, dtype=np.uint8
    ).astype(bool)

    return {
        "dem": dem,
        "dem_meta": dem_meta,
        "dem_crs": dem_crs,
        "dem_tf": dem_tf,
        "H": H,
        "W": W,
        "res_x": res_x,
        "res_y": res_y,
        "valid": valid,
        "sar": sar,
        "ortho": ortho,
        "buildings": buildings,
        "marks": marks,
        "bldg_raster": bldg_raster,
    }


def surface_classification(sar, ortho, valid, preset="compressed"):
    intensity = ortho[:3].mean(axis=0) if ortho.shape[0] >= 3 else ortho[0]
    features = np.concatenate([sar.transpose(1, 2, 0), intensity[..., None]], axis=2)
    flat = features.reshape(-1, features.shape[2])
    flat_mask = valid.reshape(-1)

    rng = np.random.default_rng(42)
    idx = np.where(flat_mask)[0]
    sample = rng.choice(idx, size=min(CONFIG["SAMPLE_MAX"], len(idx)), replace=False)
    scaler = StandardScaler()
    km = KMeans(n_clusters=CONFIG["N_CLUSTERS"], random_state=42, n_init=10)
    km.fit(scaler.fit_transform(flat[sample]))

    labels = np.full(flat.shape[0], -1, dtype=np.int16)
    labels[flat_mask] = km.predict(scaler.transform(flat[flat_mask]))
    class_map = labels.reshape(valid.shape)

    means = {
        k: flat[flat_mask][labels[flat_mask] == k].mean()
        for k in range(CONFIG["N_CLUSTERS"])
    }
    sorted_cls = sorted(means, key=means.get, reverse=True)
    walk_costs = WALK_PRESETS[preset]
    walk_cost = {sorted_cls[i]: walk_costs[i] for i in range(CONFIG["N_CLUSTERS"])}

    surface_cost = np.ones_like(valid, dtype=np.float32)
    for cls, c in walk_cost.items():
        surface_cost[class_map == cls] = c
    return class_map, surface_cost


def slope_cost(dem, res_x, res_y, valid):
    dy, dx = np.gradient(dem, res_y, res_x)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    cost = np.exp(slope_deg / CONFIG["SLOPE_EXP"]).astype(np.float32)
    cost[~valid] = 1e6
    return slope_deg, cost


def corridor_bonus(bldg_raster):
    dist_from_bldg = distance_transform_edt(~bldg_raster).astype(np.float32)
    mult = 1.0 + CONFIG["CORRIDOR_PENALTY"] * (
        1.0 - np.exp(-dist_from_bldg / CONFIG["CORRIDOR_DECAY"])
    )
    return mult.astype(np.float32)


def composite_cost(slope_cost_grid, surface_cost_grid, bldg_raster, valid, corridor_mult=None):
    composite = CONFIG["W_SLOPE"] * slope_cost_grid + CONFIG["W_SURF"] * surface_cost_grid
    if corridor_mult is not None:
        composite = composite * corridor_mult
    composite = gaussian_filter(composite.astype(np.float32), sigma=0.8)
    composite[bldg_raster] = 1e6
    composite[~valid] = 1e6
    return composite.astype(np.float32)


def geom_to_rc(geom, dem_tf, H, W):
    pt = geom if geom.geom_type == "Point" else list(geom.geoms)[0]
    r, c = rowcol(dem_tf, pt.x, pt.y)
    return int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))


def snap_to_clear(r, c, cost_grid, radius=15):
    if cost_grid[r, c] < 1e5:
        return r, c
    h, w = cost_grid.shape
    best_d, best_rc = np.inf, (r, c)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and cost_grid[nr, nc] < 1e5:
                d = dr**2 + dc**2
                if d < best_d:
                    best_d, best_rc = d, (nr, nc)
    return best_rc


def mark_nodes_from_marks(data, cost_grid):
    nodes = []
    for idx, row in data["marks"].iterrows():
        raw = geom_to_rc(row.geometry, data["dem_tf"], data["H"], data["W"])
        snapped = snap_to_clear(*raw, cost_grid, radius=CONFIG["SNAP_RADIUS"])
        nodes.append({"rc": snapped, "label": f"M{idx}"})
    return nodes


def building_nodes(data, cost_grid):
    nodes = []
    for idx, row in data["buildings"].iterrows():
        cent = row.geometry.centroid
        raw = geom_to_rc(cent, data["dem_tf"], data["H"], data["W"])
        snapped = snap_to_clear(*raw, cost_grid, radius=CONFIG["SNAP_RADIUS"])
        if cost_grid[snapped] < 1e5:
            nodes.append({"rc": snapped, "label": f"B{idx}"})
    return nodes


def dijkstra(cost_grid, start, end):
    h, w = cost_grid.shape
    dist = np.full((h, w), np.inf)
    prev = {}
    dist[start] = 0.0
    pq = [(0.0, start)]
    MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    DIAG = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    while pq:
        d, cur = heapq.heappop(pq)
        if d > dist[cur]:
            continue
        if cur == end:
            break
        r, c = cur
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if cost_grid[nr, nc] >= 1e5:
                continue
            step = cost_grid[nr, nc] * (np.sqrt(2) if (dr, dc) in DIAG else 1.0)
            nd = d + step
            if nd < dist[(nr, nc)]:
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd, (nr, nc)))
    if dist[end] == np.inf:
        return [], np.inf
    path, cur = [], end
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path, dist[end]


def tobler_step_cost(dem, res_x, res_y, surface_cost_grid, r, c, nr, nc, dr, dc):
    dz = float(dem[nr, nc] - dem[r, c])
    dx_m = dc * res_x
    dy_m = dr * res_y
    dist_m = np.sqrt(dx_m**2 + dy_m**2)
    if dist_m < 1e-9:
        return 1.0
    grade = dz / dist_m
    speed = 6.0 * np.exp(-3.5 * abs(grade + 0.05))
    speed = max(speed, 0.1)
    return (dist_m / speed) * surface_cost_grid[nr, nc]


def astar_tobler(dem, res_x, res_y, surface_cost_grid, corridor_mult, obstacle, start, end):
    h, w = dem.shape
    er, ec = end

    def heuristic(r, c):
        min_cost_pp = 0.01
        return np.sqrt((r - er) ** 2 + (c - ec) ** 2) * min_cost_pp

    dist = np.full((h, w), np.inf)
    prev = {}
    dist[start] = 0.0
    pq = [(heuristic(*start), 0.0, start)]
    MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while pq:
        f, g, cur = heapq.heappop(pq)
        if g > dist[cur]:
            continue
        if cur == end:
            break
        r, c = cur
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if obstacle[nr, nc]:
                continue
            step = tobler_step_cost(dem, res_x, res_y, surface_cost_grid, r, c, nr, nc, dr, dc)
            step *= corridor_mult[nr, nc]
            nd = g + step
            if nd < dist[(nr, nc)]:
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd + heuristic(nr, nc), nd, (nr, nc)))

    if dist[end] == np.inf:
        return [], np.inf
    path, cur = [], end
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path, dist[end]


def random_walk_agents(start, end, cost_grid, obstacle, rng):
    H, W = cost_grid.shape
    MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    freq = np.zeros((H, W), dtype=np.float32)
    successes = 0
    er, ec = end

    for _ in range(CONFIG["RW_AGENTS"]):
        r, c = start
        visited = {(r, c)}
        reached = False
        for _step in range(CONFIG["RW_MAX_STEPS"]):
            if (r, c) == end:
                reached = True
                break
            candidates = []
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if obstacle[nr, nc]:
                    continue
                candidates.append((nr, nc, dr, dc))
            if not candidates:
                break
            costs = np.array([cost_grid[nr, nc] for nr, nc, _, _ in candidates])
            weights_b = np.exp(-CONFIG["RW_BETA"] * costs)
            dists = np.array([np.sqrt((nr - er) ** 2 + (nc - ec) ** 2) for nr, nc, _, _ in candidates])
            inv_dist = 1.0 / (dists + 1e-6)
            weights_g = inv_dist / inv_dist.sum()
            weights = (1 - CONFIG["RW_GOAL_ALPHA"]) * (weights_b / weights_b.sum()) + CONFIG["RW_GOAL_ALPHA"] * weights_g
            weights /= weights.sum()
            idx = rng.choice(len(candidates), p=weights)
            r, c, _, _ = candidates[idx]
            visited.add((r, c))
        if reached:
            successes += 1
            for pr, pc in visited:
                freq[pr, pc] += 1
    freq /= max(successes, 1)
    return freq, successes


def bayesian_ensemble(dem, slope_cost_grid, surface_cost_grid, bldg_raster, valid, dist_from_bldg, mark_rcs):
    SLOPE_WEIGHTS = [0.2, 0.4, 0.6]
    SURF_WEIGHTS = [0.4, 0.6, 0.8]
    CORRIDOR_WEIGHTS = [0.3, 0.5, 0.7]

    combos = []
    for ws, wsurf, wcorr in product(SLOPE_WEIGHTS, SURF_WEIGHTS, CORRIDOR_WEIGHTS):
        total = ws + wsurf
        combos.append({"w_slope": ws / total, "w_surf": wsurf / total, "w_corr": wcorr})

    def score_path(path):
        if not path:
            return 0.0
        dists = [dist_from_bldg[r, c] for r, c in path]
        return np.exp(-np.mean(dists) / 12.0)

    pairs = [(i, j) for i in range(len(mark_rcs)) for j in range(i + 1, len(mark_rcs))]
    combo_scores = np.ones(len(combos), dtype=np.float64)
    combo_paths = [[] for _ in combos]

    for ci, combo in enumerate(combos):
        cm = combo["w_slope"] * slope_cost_grid + combo["w_surf"] * surface_cost_grid
        cm *= 1.0 + combo["w_corr"] * (1.0 - np.exp(-dist_from_bldg / 18.0))
        cm = gaussian_filter(cm.astype(np.float32), sigma=0.8)
        cm[bldg_raster] = 1e6
        cm[~valid] = 1e6
        for i, j in pairs:
            path, _ = dijkstra(cm, mark_rcs[i], mark_rcs[j])
            combo_scores[ci] *= score_path(path)
            combo_paths[ci].append(path)

    posterior = combo_scores / combo_scores.sum() if combo_scores.sum() > 0 else np.ones(len(combos)) / len(combos)
    threshold = np.percentile(posterior, 50)
    survivors = [ci for ci in range(len(combos)) if posterior[ci] >= threshold]
    weights = posterior[survivors]
    weights /= weights.sum()

    frequency = np.zeros_like(dem, dtype=np.float32)
    for ci, w in zip(survivors, weights):
        for path in combo_paths[ci]:
            for r, c in path:
                frequency[r, c] += w

    best_ci = survivors[int(np.argmax(weights))]
    best_paths = combo_paths[best_ci]
    return frequency, best_paths, combos, posterior


def visibility_graph(buildings, dem_tf, dem, surface_cost):
    obstacle_geom = buildings.geometry.union_all() # [g for g in buildings.geometry if g is not None])
    obstacle_geom_buffered = obstacle_geom.buffer(-0.05)

    def extract_corners(geom):
        if geom is None:
            return []
        if geom.geom_type == "MultiPolygon":
            pts = []
            for g in geom.geoms:
                pts += list(g.exterior.coords)
            return pts
        return list(geom.exterior.coords)

    waypoints = []
    for _, row in buildings.iterrows():
        for xy in extract_corners(row.geometry):
            pt = Point(xy)
            if not obstacle_geom_buffered.contains(pt):
                waypoints.append(pt)

    seen = set()
    unique = []
    for pt in waypoints:
        key = (round(pt.x, 1), round(pt.y, 1))
        if key not in seen:
            seen.add(key)
            unique.append(pt)
    waypoints = unique

    def xy_to_rc(x, y):
        r, c = rowcol(dem_tf, x, y)
        return int(r), int(c)

    wp_rc = [xy_to_rc(pt.x, pt.y) for pt in waypoints]

    def sample_line_cost(r0, c0, r1, c1):
        pts = []
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        for _ in range(max(dr, dc) + 1):
            pts.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        costs = [
            surface_cost[pr, pc]
            for pr, pc in pts
            if 0 <= pr < dem.shape[0] and 0 <= pc < dem.shape[1]
        ]
        return np.mean(costs) if costs else 1.0

    def is_visible(pt_a, pt_b):
        line = LineString([pt_a, pt_b])
        return not obstacle_geom.intersects(line)

    adj = {i: [] for i in range(len(waypoints))}
    for i in range(len(waypoints)):
        for j in range(i + 1, len(waypoints)):
            if is_visible(waypoints[i], waypoints[j]):
                r0, c0 = wp_rc[i]
                r1, c1 = wp_rc[j]
                avg_cost = sample_line_cost(r0, c0, r1, c1)
                dx = waypoints[j].x - waypoints[i].x
                dy = waypoints[j].y - waypoints[i].y
                dist_m = np.sqrt(dx**2 + dy**2)
                weight = dist_m * avg_cost
                adj[i].append((j, weight))
                adj[j].append((i, weight))

    return waypoints, adj


def plot_paths(data, path_records, frequency_map, title, out_name, mark_nodes):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    ext = [
        data["dem_tf"][2],
        data["dem_tf"][2] + data["W"] * data["dem_tf"][0],
        data["dem_tf"][5] + data["H"] * data["dem_tf"][4],
        data["dem_tf"][5]
    ]

    for ax, bg in zip(axes, ["ortho", "shade", "prob"]):
        if bg in ("ortho", "prob"):
            with rasterio.open(DATA / "Digital Elevation Model/DEM_20250502/Generated_DEMs/Current_DEM/bagawat-OrthoImage_GeoEyeAndWV2_NotBundled_20250301.tif") as src:
                show(src, ax=ax, alpha=(0.5 if bg == "prob" else 1.0))
        else:
            ls = mcolors.LightSource(azdeg=315, altdeg=45)
            hs = ls.hillshade(data["dem"], vert_exag=2, dx=data["res_x"], dy=data["res_y"])
            ax.imshow(hs, cmap="gray", origin="upper", extent=ext)

        if bg == "prob" and frequency_map is not None and frequency_map.max() > 0:
            freq_norm = frequency_map / frequency_map.max()
            cmap = plt.cm.hot_r
            cmap.set_under(alpha=0)
            ax.imshow(
                freq_norm,
                cmap=cmap,
                alpha=0.85,
                vmin=0.05,
                vmax=1.0,
                origin="upper",
                extent=ext,
            )

        bldg_color = "white" if bg == "prob" else "#FF4444"
        data["buildings"].plot(ax=ax, facecolor="none", edgecolor=bldg_color, linewidth=0.8)

        for rec in path_records:
            gpd.GeoDataFrame([rec], crs=data["dem_crs"]).plot(
                ax=ax, color="#00CFFF", linewidth=2.5, zorder=5
            )

        for n in mark_nodes:
            xy = rasterio.transform.xy(data["dem_tf"], n["rc"][0], n["rc"][1])
            ax.plot(*xy, "*", color="yellow", markersize=18, markeredgecolor="black", zorder=8)
            ax.annotate(
                n["label"],
                xy=xy,
                xytext=(5, 5),
                textcoords="offset points",
                color="yellow",
                fontsize=9,
                weight="bold",
            )

        ax.set_axis_off()

    plt.suptitle(title, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=250, bbox_inches="tight")
    plt.show()


def run_dijkstra_mark_pairs(data, composite, mark_nodes, include_prob=True):
    mark_rcs = [n["rc"] for n in mark_nodes]
    pairs = [(i, j) for i in range(len(mark_rcs)) for j in range(i + 1, len(mark_rcs))]
    path_records = []
    frequency_map = np.zeros_like(data["dem"], dtype=np.float32)
    for i, j in pairs:
        path, cost_val = dijkstra(composite, mark_rcs[i], mark_rcs[j])
        if not path:
            continue
        coords = [rasterio.transform.xy(data["dem_tf"], r, c) for r, c in path]
        path_records.append(
            {"source": f"M{i}", "target": f"M{j}", "cost": float(cost_val), "geometry": LineString(coords)}
        )
        if include_prob:
            pf = np.zeros_like(data["dem"], dtype=np.float32)
            for _ in range(CONFIG["PROB_K"]):
                noise = np.random.uniform(
                    1 - CONFIG["PROB_NOISE"], 1 + CONFIG["PROB_NOISE"], composite.shape
                ).astype(np.float32)
                perturbed = composite * noise
                perturbed[data["bldg_raster"]] = 1e6
                p, _ = dijkstra(perturbed, mark_rcs[i], mark_rcs[j])
                for r, c in p:
                    pf[r, c] += 1
            frequency_map += pf / max(CONFIG["PROB_K"], 1)
        else:
            for r, c in path:
                frequency_map[r, c] += 1
    return path_records, frequency_map, mark_rcs