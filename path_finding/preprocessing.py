import math, re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import ezdxf
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point
import shared

CHAPEL_COL = "Chapel Number (according to Fakhry)"
DIR_COL    = "Entrace Direction"

DIR_MAP = {"NORTH": "N", "N": "N", "SOUTH": "S", "S": "S", "EAST": "E", "E": "E", "WEST": "W", "W": "W"}
DOOR_HALF = 1.0
DOOR_OFF  = 0.5
DIRECTION_VECS = {"N": np.array([0,1.],float), "S": np.array([0,-1.],float),
                  "E": np.array([1.,0],float), "W": np.array([-1.,0],float)}


def load_footprints():
    fp = gpd.read_file(str(shared.SHP))
    print(f"Footprints: {len(fp)} polygons,  CRS: {fp.crs}")
    return fp


def normalise_dir(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    s = re.sub(r'\(.*?\)', '', str(s).strip()).upper()
    seen, result = set(), []
    for w in re.findall(r'\b(NORTH|SOUTH|EAST|WEST|N|S|E|W)\b', s):
        v = DIR_MAP[w]
        if v not in seen:
            seen.add(v); result.append(v)
    return result


def load_excel_directions():
    df = pd.read_excel(str(shared.EXCEL), sheet_name="Database Full")
    print(f"Excel: {len(df)} rows x {len(df.columns)} cols")
    df["chapel_id"] = df[CHAPEL_COL].astype(str).str.strip()
    df["raw_dir"]   = df[DIR_COL].astype(str).str.strip()
    df["direction"] = df["raw_dir"].apply(normalise_dir)
    return df


def load_dxf_labels():
    print("Parsing DXF for numeric text labels ...")
    msp = ezdxf.readfile(str(shared.DXF_P)).modelspace()
    labels = {}
    for e in msp.query("TEXT MTEXT"):
        try:
            text = e.dxf.text.strip()
        except Exception:
            try: text = e.plain_mtext().strip()
            except Exception: continue
        if text.isdigit():
            labels[text] = (e.dxf.insert.x, e.dxf.insert.y)
    print(f"  Numeric labels found: {len(labels)}")
    return labels


def estimate_rough_affine(dxf_lbl, fp_gdf):
    src   = np.array(list(dxf_lbl.values()), dtype=np.float64)
    fp_cx = np.array([g.centroid.x for g in fp_gdf.geometry])
    fp_cy = np.array([g.centroid.y for g in fp_gdf.geometry])
    scale = ((fp_cx.ptp() / src[:,0].ptp()) + (fp_cy.ptp() / src[:,1].ptp())) / 2.0
    tx    = fp_cx.mean() - scale * src[:,0].mean()
    ty    = fp_cy.mean() - scale * src[:,1].mean()
    return np.array([[scale,0,tx],[0,scale,ty],[0,0,1]], dtype=np.float64)


def bipartite_label_match(dxf_lbl, H, fp_gdf, max_dist=120.0):
    keys   = list(dxf_lbl.keys())
    lcoords = np.array([H @ [*dxf_lbl[k], 1] for k in keys])[:, :2]
    fp_c   = np.array([[g.centroid.x, g.centroid.y] for g in fp_gdf.geometry])
    cost   = np.full((len(keys), len(fp_c)), 1e8)
    for i, lc in enumerate(lcoords):
        d = np.hypot(lc[0] - fp_c[:,0], lc[1] - fp_c[:,1])
        cost[i, d < max_dist] = d[d < max_dist]
    ri, ci = linear_sum_assignment(cost)
    return pd.DataFrame([{"chapel_id": keys[i], "shp_idx": j, "dist_m": round(float(cost[i,j]),3),
                          "label_utm_x": round(float(lcoords[i,0]),2), "label_utm_y": round(float(lcoords[i,1]),2)}
                         for i, j in zip(ri, ci) if cost[i,j] < 1e7])


def attribute_footprints(fp, crosswalk, df):
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
    n_lab = fp["chapel_id"].notna().sum()
    n_dir = fp["direction"].apply(lambda d: len(d) > 0 if isinstance(d, (list, tuple)) else False).sum()
    print(f"Attribution: labelled={n_lab}  with_dir={n_dir}  no_dir={n_lab-n_dir}  unlabelled={fp['chapel_id'].isna().sum()}")
    return fp, n_lab, n_dir, n_lab - n_dir, fp["chapel_id"].isna().sum()


def best_wall(polygon, dir_code):
    target = DIRECTION_VECS.get(dir_code, DIRECTION_VECS["S"])
    c      = np.array([polygon.centroid.x, polygon.centroid.y])
    coords = np.array(polygon.exterior.coords)
    best_score, best_seg = -np.inf, None
    for a, b in zip(coords[:-1], coords[1:]):
        mid = (a + b) / 2.0
        out = mid - c
        ln  = np.linalg.norm(out)
        if ln < 1e-9: continue
        score = float(np.dot(out / ln, target))
        if score > best_score:
            best_score, best_seg = score, (a, b, mid)
    if best_seg is None:
        return c[0], c[1], 0.0, -1.0, 0.0
    a, b, mid = best_seg
    el  = np.linalg.norm(b - a)
    out = mid - c; ol = np.linalg.norm(out)
    nx_, ny_ = (out / ol) if ol > 1e-9 else (0.0, -1.0)
    return float(mid[0]), float(mid[1]), float(nx_), float(ny_), float(el)


def place_doors(fp):
    rows = []
    for idx, row in fp.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty: continue
        for dc in (row["direction"] if isinstance(row["direction"], (list,tuple)) else []):
            mx, my, nx_, ny_, el = best_wall(poly, dc)
            hl  = min(DOOR_HALF, el / 3.0)
            px, py = mx + nx_ * DOOR_OFF, my + ny_ * DOOR_OFF
            ux, uy = -ny_, nx_
            seg = LineString([(px-ux*hl, py-uy*hl), (px+ux*hl, py+uy*hl)])
            rows.append({"geometry": seg, "door_pt": Point(px, py), "chapel_id": row["chapel_id"],
                         "direction": dc, "raw_dir": row["raw_dir"],
                         "source": "direction_attributed", "confidence": 0.85,
                         "edge_len_m": round(el, 2), "shp_idx": idx})
    doors_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=fp.crs)
    doors_pts = gpd.GeoDataFrame(
        [{"geometry": r["door_pt"], "chapel_id": r["chapel_id"], "direction": r["direction"],
          "source": r["source"], "confidence": r["confidence"]} for r in rows],
        geometry="geometry", crs=fp.crs)
    print(f"Doors placed: {len(doors_gdf)}")
    return doors_gdf, doors_pts


if __name__ == "__main__":
    cache = shared.OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    fp = load_footprints()
    df = load_excel_directions()
    dxf_labels = load_dxf_labels()
    H = estimate_rough_affine(dxf_labels, fp)
    crosswalk = bipartite_label_match(dxf_labels, H, fp)
    fp_attr, *_ = attribute_footprints(fp, crosswalk, df)
    doors_gdf, doors_pts = place_doors(fp_attr)

    fp_attr.to_file(str(cache / "attributed_footprints.geojson"), driver="GeoJSON")
    doors_gdf.to_file(str(cache / "doors.geojson"), driver="GeoJSON")
    doors_pts.to_file(str(cache / "door_points.geojson"), driver="GeoJSON")
    crosswalk.to_csv(cache / "crosswalk.csv", index=False)
    shared.verify_paths()
    print("Done.")
