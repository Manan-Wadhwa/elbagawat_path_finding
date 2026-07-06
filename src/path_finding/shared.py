"""
shared.py  –  El-Bagawat common pipeline utilities
===================================================
Imported by both bagawat_pipeline.py (k-NN graph) and
bagawat_fete_pipeline.py (FETE raster routing).

Covers:
  - Path resolution (BASE, data files, output dirs)
  - Excel direction loading + normalisation
  - DXF label extraction
  - Bipartite label→polygon matching (Hungarian algorithm)
  - Approach-5 door placement (shapefile-native, direction-aware)
  - DEM loading + hillshade
  - Standard colour palettes and legend patches
  - Shared visualisation helpers (save_fig, chapel_labels_on_ax)
"""

import warnings
warnings.filterwarnings("ignore")

import math
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import ezdxf
from pathlib import Path
from shapely.geometry import Point, LineString
from shapely.geometry import box as shapely_box
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Paths  (BASE = project root, 3 levels up from this file)
# ---------------------------------------------------------------------------

BASE  = Path(__file__).resolve().parents[2]
SHP   = str(BASE / "data" / "BaseSiteCAD" / "130_BuildingFootprintsVectorData" / "BuildingTracesCurrent" / "Buildings_Mask.shp")
DXF_P = str(BASE / "data" / "Site_CAD_Working_converted.dxf")
EXCEL = str(BASE / "data" / "2026 El Bagawat Database Draft 1.xlsx")
DEM_P = str(BASE / "data" / "BaseSiteCAD" / "150_DigitalElevationModel" / "Generated_DEMs" / "Current_DEM" / "Bagawat-DEM-NewImageryOnly-0.4m-DEM.tif")
OUT   = BASE / "outputs"
OUT.mkdir(exist_ok=True)


def verify_paths():
    """Print a quick existence check for all key input files."""
    print("Input file check:")
    for label, p_raw in [("SHP", SHP), ("DXF", DXF_P), ("EXCEL", EXCEL), ("DEM", DEM_P)]:
        p = Path(p_raw)
        status = "OK" if p.exists() else "MISSING"
        print(f"  {label:5s}  {status:7s}  ->  {p.name}")
    print(f"  OUT        ->  {OUT}\n")


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# Per-direction colours used on all maps
DIR_CLR = {
    "N":          "#f5a623",
    "S":          "#7ed321",
    "E":          "#9013fe",
    "W":          "#d0021b",
    "S_fallback": "#aaaaaa",
    None:         "#aaaaaa",
}

# Attribution-map colours
ATTR_CLR = {
    "has_dir":    "#3a86ff",
    "no_dir":     "#ffb703",
    "unlabelled": "#cccccc",
}

def direction_legend_patches():
    return [
        mpatches.Patch(color=DIR_CLR["S"], label="South"),
        mpatches.Patch(color=DIR_CLR["N"], label="North"),
        mpatches.Patch(color=DIR_CLR["E"], label="East"),
        mpatches.Patch(color=DIR_CLR["W"], label="West"),
        mpatches.Patch(color=DIR_CLR["S_fallback"], label="Southernmost fallback"),
        mpatches.Patch(color="red",        label="Red dot: no direction"),
    ]


# ---------------------------------------------------------------------------
# Figure saving helper
# ---------------------------------------------------------------------------

def save_fig(figs_dir: Path, name: str, dpi=150, extra_formats=()):
    """
    Save the current matplotlib figure to figs_dir/name.
    extra_formats: additional suffixes to export, e.g. ('.svg', '.pdf')
    """
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    plt.savefig(figs_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    for fmt in extra_formats:
        plt.savefig(figs_dir / f"{stem}{fmt}", bbox_inches="tight")
    print(f"  -> {stem}.png" + ("".join(f" / {stem}{f}" for f in extra_formats)))


# ---------------------------------------------------------------------------
# Step 1 – Load shapefile
# ---------------------------------------------------------------------------

def load_footprints():
    fp = gpd.read_file(str(SHP))
    print(f"Footprints: {len(fp)} polygons,  CRS: {fp.crs}")
    return fp


def plot_raw_footprints(fp, figs_dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    fp.plot(ax=ax, color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.set_title("El-Bagawat Building Footprints (raw, no chapel labels)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    save_fig(figs_dir, "01_footprints_raw.png")
    plt.show()


# ---------------------------------------------------------------------------
# Step 2 – Load & audit Excel database
# ---------------------------------------------------------------------------

CHAPEL_COL = "Chapel Number (according to Fakhry)"
DIR_COL    = "Entrace Direction"   # note: original typo in the spreadsheet


def normalise_dir(s):
    """Normalise a raw direction string to a list of direction codes ('N', 'S', 'E', 'W')."""
    if s is None:
        return []
    if isinstance(s, float) and math.isnan(s):
        return []
    s = str(s).strip()
    if s.lower() in ("nan", "none", ""):
        return []
    
    # Remove parenthetical clauses
    import re
    s_clean = re.sub(r'\(.*?\)', '', s)
    
    DIR_MAP = {
        "NORTH": "N", "N": "N",
        "SOUTH": "S", "S": "S",
        "EAST": "E", "E": "E",
        "WEST": "W", "W": "W"
    }
    
    words = re.findall(r'\b(NORTH|SOUTH|EAST|WEST|N|S|E|W)\b', s_clean.upper())
    seen = set()
    result = []
    for w in words:
        val = DIR_MAP[w]
        if val not in seen:
            seen.add(val)
            result.append(val)
    return result


def load_excel_directions():
    df = pd.read_excel(str(EXCEL), sheet_name="Database Full")
    print(f"Excel: {len(df)} rows x {len(df.columns)} cols")
    df["chapel_id"] = df[CHAPEL_COL].astype(str).str.strip()
    df["raw_dir"]   = df[DIR_COL].astype(str).str.strip()
    df["direction"] = df["raw_dir"].apply(normalise_dir)
    return df


def print_direction_audit(df):
    flat_dirs = []
    for d in df["direction"]:
        if isinstance(d, (list, tuple)):
            flat_dirs.extend(d)
            
    missing_count = df["direction"].apply(lambda d: len(d) == 0 if isinstance(d, (list, tuple)) else True).sum()
    
    counts_dict = pd.Series(flat_dirs).value_counts().to_dict()
    if missing_count > 0:
        counts_dict[None] = missing_count
        
    dir_counts = pd.Series(counts_dict)
    
    missing_dir = df[df["direction"].apply(lambda d: len(d) == 0 if isinstance(d, (list, tuple)) else True)]
    print(f"\nDirection distribution:\n{dir_counts.to_string()}")
    print(f"\nMissing directions: {len(missing_dir)} / {len(df)}")

    print("\n=== Outlier / Anomaly Analysis ===")
    dupes   = df[df.duplicated("chapel_id", keep=False) & (df["chapel_id"] != "nan")]
    non_num = df[~df["chapel_id"].str.match(r"^\d+$") & (df["chapel_id"] != "nan")]
    ambig   = df[df["raw_dir"].str.contains(
                    r"table|secondary|\?|compound|Court|\(", case=False, na=False)]
    print(f"  Duplicate chapel IDs : {dupes['chapel_id'].unique().tolist() or 'none'}")
    print(f"  Non-numeric IDs      : {non_num['chapel_id'].tolist() or 'none'}")
    print(f"  Ambiguous directions : {len(ambig)}")
    for _, r in ambig.iterrows():
        print(f"    Chapel {r['chapel_id']:>4s}  ->  {r['raw_dir']}")
    return dir_counts


def plot_direction_audit(df, dir_counts, figs_dir):
    flat_dirs = []
    for d in df["direction"]:
        if isinstance(d, (list, tuple)):
            flat_dirs.extend(d)
            
    missing_count = df["direction"].apply(lambda d: len(d) == 0 if isinstance(d, (list, tuple)) else True).sum()
    
    counts_dict = pd.Series(flat_dirs).value_counts().to_dict()
    if missing_count > 0:
        counts_dict[None] = missing_count
        
    dir_counts = pd.Series(counts_dict)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    labels_pie  = [str(x) if str(x) != "None" else "Missing" for x in dir_counts.index]
    pie_colors  = ["#e05252", "#5296e0", "#52c452", "#e0a052", "#aaaaaa"][:len(labels_pie)]
    axes[0].pie(dir_counts.values, labels=labels_pie, colors=pie_colors,
                autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11})
    axes[0].set_title("Entrance Direction Distribution (342 Excel records)", fontsize=13)

    raw_counts = df["raw_dir"].value_counts().head(12)
    axes[1].barh(raw_counts.index[::-1], raw_counts.values[::-1], color="#4a9fd4")
    axes[1].set_xlabel("Count")
    axes[1].set_title("Top 12 Raw Direction Strings", fontsize=13)
    axes[1].tick_params(axis="y", labelsize=8)

    plt.suptitle("El-Bagawat Excel Direction Audit", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(figs_dir, "02_excel_audit_directions.png")
    plt.show()


# ---------------------------------------------------------------------------
# Step 3 – Extract DXF text labels
# ---------------------------------------------------------------------------

def load_dxf_labels():
    """Return dict of {chapel_id_str: (dxf_x, dxf_y)} for all numeric text labels."""
    print("Parsing DXF for numeric text labels ...")
    doc = ezdxf.readfile(str(DXF_P))
    msp = doc.modelspace()

    labels = {}
    for entity in msp.query("TEXT MTEXT"):
        try:
            text = entity.dxf.text.strip()
        except Exception:
            try:
                text = entity.plain_mtext().strip()
            except Exception:
                continue
        if text.isdigit():
            labels[text] = (entity.dxf.insert.x, entity.dxf.insert.y)

    print(f"  Numeric labels found: {len(labels)}")
    return labels


def plot_dxf_labels(dxf_labels, figs_dir):
    xs = [v[0] for v in dxf_labels.values()]
    ys = [v[1] for v in dxf_labels.values()]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, ys, s=8, c="tomato", alpha=0.6, zorder=3)
    for k, (lx, ly) in list(dxf_labels.items())[:40]:
        ax.annotate(k, (lx, ly), fontsize=5, ha="center", color="#333333")
    ax.set_title(f"DXF Numeric Text Labels ({len(dxf_labels)} total) - raw DXF space",
                 fontsize=13)
    ax.set_xlabel("DXF X"); ax.set_ylabel("DXF Y")
    plt.tight_layout()
    save_fig(figs_dir, "03_dxf_labels_raw.png")
    plt.show()


# ---------------------------------------------------------------------------
# Step 4 – Bipartite label→polygon matching (Hungarian algorithm)
# ---------------------------------------------------------------------------

def estimate_rough_affine(dxf_lbl, fp_gdf):
    """Estimate a uniform-scale affine from DXF to UTM using centroid stats."""
    src   = np.array(list(dxf_lbl.values()), dtype=np.float64)
    fp_cx = np.array([g.centroid.x for g in fp_gdf.geometry])
    fp_cy = np.array([g.centroid.y for g in fp_gdf.geometry])

    dxf_x_rng = src[:, 0].max() - src[:, 0].min()
    dxf_y_rng = src[:, 1].max() - src[:, 1].min()
    fp_x_rng  = fp_cx.max()     - fp_cx.min()
    fp_y_rng  = fp_cy.max()     - fp_cy.min()

    if dxf_x_rng < 1e-6 or dxf_y_rng < 1e-6:
        raise ValueError("DXF labels appear degenerate (zero spread)")

    scale = ((fp_x_rng / dxf_x_rng) + (fp_y_rng / dxf_y_rng)) / 2.0
    tx    = fp_cx.mean() - scale * src[:, 0].mean()
    ty    = fp_cy.mean() - scale * src[:, 1].mean()

    return np.array([[scale, 0, tx],
                     [0, scale, ty],
                     [0, 0,      1]], dtype=np.float64)


def apply_affine(pt, H):
    arr = np.array([pt[0], pt[1], 1.0])
    m   = H @ arr
    return (m[0] / m[2], m[1] / m[2])


def bipartite_label_match(dxf_lbl, H, fp_gdf, max_dist=120.0):
    """Run the Hungarian algorithm to assign DXF labels to shapefile polygons."""
    keys    = list(dxf_lbl.keys())
    lcoords = np.array([apply_affine(dxf_lbl[k], H) for k in keys])
    fp_c    = np.array([[g.centroid.x, g.centroid.y] for g in fp_gdf.geometry])

    cost = np.full((len(keys), len(fp_c)), 1e8)
    for i, lc in enumerate(lcoords):
        d = np.hypot(lc[0] - fp_c[:, 0], lc[1] - fp_c[:, 1])
        cost[i, d < max_dist] = d[d < max_dist]

    ri, ci = linear_sum_assignment(cost)
    rows = [
        {"chapel_id":    keys[i],
         "shp_idx":      j,
         "dist_m":       round(float(cost[i, j]), 3),
         "label_utm_x":  round(float(lcoords[i, 0]), 2),
         "label_utm_y":  round(float(lcoords[i, 1]), 2)}
        for i, j in zip(ri, ci) if cost[i, j] < 1e7
    ]
    return pd.DataFrame(rows)


def attribute_footprints(fp, crosswalk, df):
    """Join crosswalk + Excel directions back onto the footprint GeoDataFrame."""
    fp = fp.copy()
    fp["chapel_id"] = None
    fp["direction"] = [() for _ in range(len(fp))]
    fp["raw_dir"]   = None

    for _, cw in crosswalk.iterrows():
        idx, cid = cw["shp_idx"], cw["chapel_id"]
        fp.at[idx, "chapel_id"] = cid
        match = df[df["chapel_id"] == cid]
        if not match.empty:
            dirs = match.iloc[0]["direction"]
            fp.at[idx, "direction"] = tuple(dirs) if isinstance(dirs, (list, tuple)) else dirs
            fp.at[idx, "raw_dir"]   = match.iloc[0]["raw_dir"]

    n_labelled   = fp["chapel_id"].notna().sum()
    n_with_dir   = fp["direction"].apply(lambda d: len(d) > 0 if isinstance(d, (list, tuple)) else False).sum()
    n_no_dir     = n_labelled - n_with_dir
    n_unlabelled = fp["chapel_id"].isna().sum()
    print(f"Attribution:  labelled={n_labelled}  with_dir={n_with_dir}"
          f"  no_dir={n_no_dir}  unlabelled={n_unlabelled}")
    return fp, n_labelled, n_with_dir, n_no_dir, n_unlabelled


def plot_bipartite_matching(fp, dxf_labels, crosswalk, H_rough, figs_dir):
    all_utm = [apply_affine(v, H_rough) for v in dxf_labels.values()]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter([g.centroid.x for g in fp.geometry], [g.centroid.y for g in fp.geometry],
               s=12, c="steelblue", label="Shapefile centroids", alpha=0.6, zorder=3)
    ax.scatter([p[0] for p in all_utm], [p[1] for p in all_utm],
               s=8, c="tomato", label="DXF labels (UTM)", alpha=0.6, zorder=4)
    for _, cw in crosswalk.iterrows():
        idx = cw["shp_idx"]
        ax.plot([cw["label_utm_x"], fp.iloc[idx].geometry.centroid.x],
                [cw["label_utm_y"], fp.iloc[idx].geometry.centroid.y],
                color="gray", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=10)
    ax.set_title(f"DXF->UTM Bipartite Matching ({len(crosswalk)} pairs)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    save_fig(figs_dir, "04a_bipartite_matching.png")
    plt.show()


def plot_attribution_map(fp, n_with_dir, n_no_dir, n_unlabelled, figs_dir):
    fig, ax = plt.subplots(figsize=(18, 15))

    for idx, row in fp.iterrows():
        if   row["chapel_id"] is None:   c = ATTR_CLR["unlabelled"]
        elif not row["direction"]:       c = ATTR_CLR["no_dir"]
        else:                            c = ATTR_CLR["has_dir"]
        fp.loc[[idx]].plot(ax=ax, color=c, edgecolor="white", linewidth=0.4, alpha=0.85)

    # Chapel ID text labels
    for _, row in fp.iterrows():
        if row["chapel_id"] is not None:
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            ax.text(cx, cy, str(row["chapel_id"]),
                    ha="center", va="center", fontsize=4.5,
                    color="white", fontweight="bold")

    # Red dots for chapels with no direction
    no_dir_fp = fp[fp["chapel_id"].notna() & fp["direction"].apply(lambda d: len(d) == 0)]
    if len(no_dir_fp):
        ax.scatter(no_dir_fp.geometry.centroid.x, no_dir_fp.geometry.centroid.y,
                   s=35, c="red", zorder=10, edgecolors="darkred", linewidths=0.5)

    patches = [
        mpatches.Patch(color=ATTR_CLR["has_dir"],    label=f"Has direction ({n_with_dir})"),
        mpatches.Patch(color=ATTR_CLR["no_dir"],     label=f"No direction in Excel ({n_no_dir})"),
        mpatches.Patch(color=ATTR_CLR["unlabelled"], label=f"Unlabelled polygon ({n_unlabelled})"),
        mpatches.Patch(color="red",                  label="Labelled but no direction (red dot)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title("El-Bagawat - Chapel Attribution + Missing Direction Flags",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    save_fig(figs_dir, "04b_attribution_map.png")
    plt.show()
    print(f"    ({len(no_dir_fp)} red-dot chapels without a known entrance direction)")


# ---------------------------------------------------------------------------
# Step 5 – Approach-5 door placement (shapefile-native, direction-aware)
# ---------------------------------------------------------------------------

DOOR_HALF = 1.0   # metres — half-length of the door segment
DOOR_OFF  = 0.5   # metres — outward offset from wall midpoint

DIRECTION_VECS = {
    "N": np.array([ 0,  1], dtype=float),
    "S": np.array([ 0, -1], dtype=float),
    "E": np.array([ 1,  0], dtype=float),
    "W": np.array([-1,  0], dtype=float),
}


def best_wall(polygon, dir_code):
    """
    Find the polygon wall whose outward normal best aligns with dir_code.
    Returns (mid_x, mid_y, norm_x, norm_y, edge_len).
    """
    target = DIRECTION_VECS.get(dir_code, DIRECTION_VECS["S"])
    c      = np.array([polygon.centroid.x, polygon.centroid.y])
    coords = np.array(polygon.exterior.coords)

    best_score, best_seg = -np.inf, None
    for a, b in zip(coords[:-1], coords[1:]):
        mid = (a + b) / 2.0
        out = mid - c
        ln  = np.linalg.norm(out)
        if ln < 1e-9:
            continue
        score = float(np.dot(out / ln, target))
        if score > best_score:
            best_score, best_seg = score, (a, b, mid)

    if best_seg is None:
        return c[0], c[1], 0.0, -1.0, 0.0

    a, b, mid = best_seg
    el  = np.linalg.norm(b - a)
    out = mid - c
    ol  = np.linalg.norm(out)
    nx_, ny_ = (out / ol) if ol > 1e-9 else (0.0, -1.0)
    return float(mid[0]), float(mid[1]), float(nx_), float(ny_), float(el)


def southernmost_wall(polygon):
    """
    Fallback for chapels without a known direction.
    Returns (mid_x, mid_y, norm_x, norm_y, edge_len) for the lowest wall.
    """
    c      = np.array([polygon.centroid.x, polygon.centroid.y])
    coords = np.array(polygon.exterior.coords)

    best_y, best_seg = np.inf, None
    for a, b in zip(coords[:-1], coords[1:]):
        my = (a[1] + b[1]) / 2.0
        if my < best_y:
            best_y   = my
            best_seg = (a, b, np.array([(a[0]+b[0])/2, (a[1]+b[1])/2]))

    if best_seg is None:
        return c[0], c[1], 0.0, -1.0, 0.0

    a, b, mid = best_seg
    el  = np.linalg.norm(b - a)
    out = mid - c
    ol  = np.linalg.norm(out)
    nx_, ny_ = (out / ol) if ol > 1e-9 else (0.0, -1.0)
    return float(mid[0]), float(mid[1]), float(nx_), float(ny_), float(el)


def make_door_geometry(mid_x, mid_y, norm_x, norm_y, half, off):
    """Return (LineString door segment, Point door midpoint)."""
    px, py = mid_x + norm_x * off, mid_y + norm_y * off
    ux, uy = -norm_y, norm_x          # tangent perpendicular to normal
    seg = LineString([(px - ux*half, py - uy*half),
                      (px + ux*half, py + uy*half)])
    return seg, Point(px, py)


def place_doors(fp):
    """Run Approach-5 door placement on all footprints. Returns (doors_gdf, doors_pts)."""
    rows = []
    for idx, row in fp.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue
        dc_list = row["direction"]
        if isinstance(dc_list, (list, tuple)) and len(dc_list) > 0:
            for dc in dc_list:
                mx, my, nx_, ny_, el = best_wall(poly, dc)
                src, conf = "direction_attributed", 0.85
                hl = min(DOOR_HALF, el / 3.0)
                ls, pt = make_door_geometry(mx, my, nx_, ny_, hl, DOOR_OFF)
                rows.append({
                    "geometry":   ls,
                    "door_pt":    pt,
                    "chapel_id":  row["chapel_id"],
                    "direction":  dc,
                    "raw_dir":    row["raw_dir"],
                    "source":     src,
                    "confidence": conf,
                    "edge_len_m": round(el, 2),
                    "shp_idx":    idx,
                })
        # If empty list, do NOT place any doors (keep empty, no fallback to south).

    doors_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=fp.crs)
    doors_pts = gpd.GeoDataFrame(
        [{"geometry":  r["door_pt"],
          "chapel_id": r["chapel_id"],
          "direction": r["direction"],
          "source":    r["source"],
          "confidence": r["confidence"]}
         for r in rows],
        geometry="geometry", crs=fp.crs,
    )

    attributed = (doors_gdf["source"] == "direction_attributed").sum()
    fallback   = (doors_gdf["source"] == "southernmost_fallback").sum()
    print(f"Doors placed:  {len(doors_gdf)} total  "
          f"(attributed={attributed}, fallback={fallback})")
    return doors_gdf, doors_pts


def plot_doors_all(fp, doors_gdf, figs_dir):
    patches = direction_legend_patches()
    fig, ax = plt.subplots(figsize=(18, 15))
    fp.plot(ax=ax, color="#e8f4fb", edgecolor="#6baed6", linewidth=0.7, alpha=0.9)
    for _, row in doors_gdf.iterrows():
        xs, ys = row.geometry.xy
        ax.plot(xs, ys, color=DIR_CLR.get(row["direction"], "#888"),
                linewidth=2.0, zorder=4)
    _draw_chapel_ids(ax, fp, fontsize=4, color="#111")
    _draw_red_dot_no_dir(ax, fp, s=35)
    ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Approach 5 - Shapefile-Native Direction-Aware Doors",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    save_fig(figs_dir, "05_approach5_doors_all.png")
    plt.show()


def plot_doors_zoom(fp, doors_gdf, figs_dir, zoom_radius=200):
    cx_m = fp.geometry.centroid.x.mean()
    cy_m = fp.geometry.centroid.y.mean()
    box  = shapely_box(cx_m - zoom_radius, cy_m - zoom_radius,
                       cx_m + zoom_radius, cy_m + zoom_radius)
    fp_z    = fp[fp.geometry.intersects(box)]
    doors_z = doors_gdf[doors_gdf.geometry.intersects(box)]

    patches = direction_legend_patches()
    fig, ax = plt.subplots(figsize=(14, 12))
    fp_z.plot(ax=ax, color="#d6eaf8", edgecolor="#2980b9", linewidth=1.0)
    for _, row in doors_z.iterrows():
        xs, ys = row.geometry.xy
        ax.plot(xs, ys, color=DIR_CLR.get(row["direction"], "#888"),
                linewidth=3.5, zorder=4, solid_capstyle="round")
    _draw_chapel_ids(ax, fp_z, fontsize=8, color="#111")
    _draw_red_dot_no_dir(ax, fp_z, s=80)
    ax.legend(handles=patches, loc="lower left", fontsize=10)
    ax.set_title(f"Detail Zoom - Central Cluster (+/-{zoom_radius} m from centroid)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    save_fig(figs_dir, "06_approach5_zoom.png")
    plt.show()


# ---------------------------------------------------------------------------
# Step 6 – DEM loading + hillshade
# ---------------------------------------------------------------------------

def load_dem():
    """
    Open the DEM file and return a dict with all relevant arrays/metadata.
    Keys: arr, crs, nodata, bounds, res, transform, disp, e_min, e_max, extent
    """
    print("Loading DEM ...")
    with rasterio.open(str(DEM_P)) as src:
        dem = {
            "arr":       src.read(1).astype(np.float32),
            "crs":       src.crs,
            "nodata":    src.nodata,
            "bounds":    src.bounds,
            "res":       abs(src.transform[0]),
            "transform": src.transform,
            "profile":   src.profile.copy(),
        }

    nd = dem["nodata"]
    dem["disp"]   = (np.where(dem["arr"] == nd, np.nan, dem["arr"])
                     if nd is not None else dem["arr"].copy())
    dem["e_min"]  = float(np.nanmin(dem["disp"]))
    dem["e_max"]  = float(np.nanmax(dem["disp"]))
    dem["extent"] = [dem["bounds"].left,  dem["bounds"].right,
                     dem["bounds"].bottom, dem["bounds"].top]

    print(f"  CRS={dem['crs']}  shape={dem['arr'].shape}  "
          f"res={dem['res']:.2f} m  elev={dem['e_min']:.1f}-{dem['e_max']:.1f} m")
    return dem


def hillshade(arr, azimuth=315, altitude=45):
    """Classic hillshade from a 2D elevation array."""
    az_r  = np.radians(360 - azimuth + 90)
    alt_r = np.radians(altitude)
    fill  = np.where(np.isnan(arr), 0, arr)
    dy, dx = np.gradient(fill)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    hs     = (np.sin(alt_r) * np.cos(slope)
              + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect))
    return np.clip(hs, 0, 1)


def plot_dem_hillshade(dem, figs_dir):
    hs = hillshade(dem["disp"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
              cmap="terrain", alpha=0.85,
              vmin=dem["e_min"], vmax=dem["e_max"], rasterized=True)
    ax.imshow(hs, extent=dem["extent"], origin="upper",
              cmap="gray", alpha=0.40, rasterized=True)
    sm = plt.cm.ScalarMappable(
        cmap="terrain",
        norm=mcolors.Normalize(vmin=dem["e_min"], vmax=dem["e_max"]))
    plt.colorbar(sm, ax=ax, label="Elevation (m)", shrink=0.6)
    ax.set_title("El-Bagawat - DEM Hillshade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    save_fig(figs_dir, "07_dem_hillshade.png")
    plt.show()
    return hs   # return so callers can reuse it


def plot_dem_with_doors(dem, hs, fp, doors_gdf, figs_dir):
    """Overlay building footprints and Approach-5 doors on the DEM."""
    fp_dem    = fp.to_crs(dem["crs"])        if fp.crs        != dem["crs"] else fp.copy()
    doors_dem = doors_gdf.to_crs(dem["crs"]) if doors_gdf.crs != dem["crs"] else doors_gdf.copy()

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.imshow(dem["disp"], extent=dem["extent"], origin="upper",
              cmap="terrain", alpha=0.80,
              vmin=dem["e_min"], vmax=dem["e_max"], rasterized=True)
    ax.imshow(hs, extent=dem["extent"], origin="upper",
              cmap="gray", alpha=0.35, rasterized=True)
    fp_dem.plot(ax=ax, color="none", edgecolor="#1a6ea8", linewidth=0.8, alpha=0.9)
    for _, row in doors_dem.iterrows():
        xs, ys = row.geometry.xy
        ax.plot(xs, ys, color=DIR_CLR.get(row["direction"], "#aaa"),
                linewidth=2.0, zorder=5)
    _draw_chapel_ids(ax, fp_dem, fontsize=4, color="white",
                     bbox=dict(boxstyle="round,pad=0.1", fc="#1a6ea8", alpha=0.5, lw=0))
    _draw_red_dot_no_dir(ax, fp_dem, s=30)
    sm2 = plt.cm.ScalarMappable(
        cmap="terrain",
        norm=mcolors.Normalize(vmin=dem["e_min"], vmax=dem["e_max"]))
    plt.colorbar(sm2, ax=ax, label="Elevation (m)", shrink=0.55)
    ax.legend(handles=direction_legend_patches(), loc="lower left",
              fontsize=9, framealpha=0.9)
    ax.set_title("El-Bagawat - DEM + Footprints + Approach-5 Doors",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    save_fig(figs_dir, "08_dem_footprints_doors.png")
    plt.show()
    return fp_dem, doors_dem


# ---------------------------------------------------------------------------
# Step 7 – DEM elevation sampling at door points
# ---------------------------------------------------------------------------

def sample_dem_at_doors(doors_pts):
    """Add an 'elevation_m' column to doors_pts by sampling the DEM."""
    with rasterio.open(str(DEM_P)) as src:
        pts_reproj = (doors_pts.to_crs(src.crs)
                      if doors_pts.crs != src.crs else doors_pts)
        coords = [(g.x, g.y) for g in pts_reproj.geometry]
        vals   = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan

    doors_pts = doors_pts.copy()
    doors_pts["elevation_m"] = vals
    print("Door elevation stats:")
    print(pd.Series(vals).describe().to_string())
    return doors_pts


def plot_door_elevation_histogram(doors_pts, figs_dir):
    elevs = doors_pts["elevation_m"].values
    valid = elevs[~np.isnan(elevs)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid, bins=40, color="#3a86ff", edgecolor="white", linewidth=0.5)
    ax.axvline(np.nanmean(elevs), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.nanmean(elevs):.1f} m")
    ax.set_xlabel("Elevation (m)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Door-Point Elevations (DEM sample)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    save_fig(figs_dir, "09_door_elevation_histogram.png")
    plt.show()


# ---------------------------------------------------------------------------
# Shared visualisation helpers (private)
# ---------------------------------------------------------------------------

def _draw_chapel_ids(ax, fp, fontsize=4, color="white", bbox=None):
    for _, row in fp.iterrows():
        cid = row.get("chapel_id")
        if cid is not None and str(cid) not in ("None", "nan", ""):
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            kw = dict(ha="center", va="center",
                      fontsize=fontsize, color=color, fontweight="bold")
            if bbox:
                kw["bbox"] = bbox
            ax.text(cx, cy, str(cid), **kw)


def _draw_red_dot_no_dir(ax, fp, s=35):
    no_dir = fp[fp["chapel_id"].notna() & fp["direction"].apply(lambda d: len(d) == 0)]
    if len(no_dir):
        ax.scatter(no_dir.geometry.centroid.x, no_dir.geometry.centroid.y,
                   s=s, c="red", zorder=10, edgecolors="darkred", linewidths=0.5)


# ---------------------------------------------------------------------------
# Step 9 – Multiprocessing Worker Helpers (Windows spawn compatibility)
# ---------------------------------------------------------------------------
_worker_cost_surface = None


def _cpu_worker_init(cost_arr):
    """Pool initializer: store cost_surface inside each worker process."""
    global _worker_cost_surface
    _worker_cost_surface = cost_arr


def _cpu_worker(args):
    """
    Run Dijkstra for one source door; return local density + skip count.
    """
    src_idx, entrance_pixels = args
    shape = _worker_cost_surface.shape
    local_density = np.zeros(shape, dtype=np.float64)
    skipped = 0

    mcp = MCP_Geometric(_worker_cost_surface, fully_connected=True)
    mcp.find_costs([entrance_pixels[src_idx]])

    for tgt_idx in range(src_idx + 1, len(entrance_pixels)):
        try:
            path = mcp.traceback(entrance_pixels[tgt_idx])
            for pr, pc in path:
                local_density[pr, pc] += 1.0
        except MemoryError:
            raise
        except Exception:
            skipped += 1

    return local_density, skipped

