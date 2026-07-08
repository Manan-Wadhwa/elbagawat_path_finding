import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.patches as mpatches
import networkx as nx
from skimage.morphology import skeletonize, erosion
from rasterio.features import rasterize as rio_rasterize
from pathlib import Path
from shapely.geometry import Point, LineString


BASE = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path.cwd()

_env = {}
_env_path = BASE / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            _env[_k.strip()] = _v.strip().strip('"').strip("'")

def _p(key, rel):
    v = _env.get(key)
    p = Path(v) if v else None
    return str(p if p and p.is_absolute() else BASE / (v or rel))

SHP   = _p("SHP_PATH",   "data/BaseSiteCAD/130_BuildingFootprintsVectorData/BuildingTracesCurrent/Buildings_Mask.shp")
DXF_P = _p("DXF_PATH",   "data/Site_CAD_Working_converted.dxf")
EXCEL = _p("EXCEL_PATH", "data/2026 El Bagawat Database Draft 1.xlsx")
DEM_P = _p("DEM_PATH",   "data/BaseSiteCAD/150_DigitalElevationModel/Generated_DEMs/Current_DEM/Bagawat-DEM-NewImageryOnly-0.4m-DEM.tif")
OUT   = BASE / "outputs"
OUT.mkdir(exist_ok=True)


def verify_paths():
    for label, p in [("SHP", SHP), ("DXF", DXF_P), ("EXCEL", EXCEL), ("DEM", DEM_P)]:
        print(f"  {label:5s}  {'OK' if Path(p).exists() else 'MISSING':7s}  ->  {Path(p).name}")
    print(f"  OUT        ->  {OUT}\n")


def get_base_preprocessed_data():
    cache = OUT / "cache"
    paths = {k: cache / v for k, v in {
        "fp":       "attributed_footprints.geojson",
        "doors":    "doors.geojson",
        "pts":      "door_points.geojson",
        "cw":       "crosswalk.csv",
    }.items()}
    if not all(p.exists() for p in paths.values()):
        raise FileNotFoundError("Cache missing. Run `python path_finding/preprocessing.py` first.")

    fp = gpd.read_file(str(paths["fp"]))
    def _parse(d):
        if not d:
            return ()
        if isinstance(d, (list, tuple)):
            return tuple(d)
        d = d.replace("'", "").replace('"', "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        return tuple(x.strip() for x in d.split(",") if x.strip())
    fp["direction"] = fp["direction"].apply(_parse)

    doors_df = gpd.read_file(str(paths["doors"]))
    pts_df = gpd.read_file(str(paths["pts"]))
    
    from shapely import wkt
    if "door_pt" in doors_df.columns:
        doors_df["door_pt"] = doors_df["door_pt"].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
    if "door_pt" in pts_df.columns:
        pts_df["door_pt"] = pts_df["door_pt"].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)

    return fp, doors_df, pts_df, pd.read_csv(paths["cw"])


def generate_free_space_skeleton(footprints_gdf, roi_bounds=None, resolution=0.5):
    from rasterio.transform import from_bounds
    minx, miny, maxx, maxy = roi_bounds or footprints_gdf.total_bounds
    w = int(round((maxx - minx) / resolution))
    h = int(round((maxy - miny) / resolution))
    transform = from_bounds(minx, miny, maxx, maxy, w, h)
    shapes = [(g, 1) for g in footprints_gdf.geometry if g is not None]
    raster = rio_rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
    free = erosion(1 - raster, footprint=np.ones((3, 3), np.uint8))
    return skeletonize(free.astype(bool)).astype(np.uint8), transform


def skeleton_to_axial_graph(skeleton, raster_transform, crs):
    G = nx.Graph()
    coord_set = set(zip(*np.where(skeleton)))
    for (r, c) in coord_set:
        for nb in [(r+dr, c+dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0) and (r+dr,c+dc) in coord_set]:
            G.add_edge((r, c), nb)
    return G


def compute_space_syntax_integration(G, raster_transform, crs):
    import rasterio.transform as rt
    n = G.number_of_nodes()
    if n < 3:
        return gpd.GeoDataFrame(columns=["geometry","integration_mean","integration_mean_norm","integration_u","integration_v"], crs=crs)
    D_n = 2.0 / (n - 2) if n > 2 else 1.0
    vals = {}
    for node in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, node)
        if len(lengths) < 2:
            vals[node] = 0.0
            continue
        MD  = sum(lengths.values()) / (len(lengths) - 1)
        RA  = 2 * (MD - 1) / (n - 2) if n > 2 else 0
        RRA = RA / D_n if D_n > 0 else 0
        vals[node] = 1.0 / RRA if RRA > 0 else 0.0
    mx = max(vals.values()) or 1
    segs = []
    for (u, v) in G.edges():
        ux, uy = rt.xy(raster_transform, u[0], u[1])
        vx, vy = rt.xy(raster_transform, v[0], v[1])
        iu, iv = vals.get(u, 0), vals.get(v, 0)
        segs.append({"geometry": LineString([(ux,uy),(vx,vy)]),
                     "integration_mean": (iu+iv)/2,
                     "integration_mean_norm": (iu/mx+iv/mx)/2,
                     "integration_u": iu, "integration_v": iv})
    return gpd.GeoDataFrame(segs, crs=crs)


def rasterize_integration(syntax_gdf, raster_shape, raster_transform):
    import rasterio.enums
    if syntax_gdf.empty:
        return np.zeros(raster_shape, dtype=np.float32)
    vals = syntax_gdf["integration_mean"].values
    if vals.max() > 0:
        vals = vals / vals.max()
    return rio_rasterize(
        [(g, float(v)) for g, v in zip(syntax_gdf.geometry, vals)],
        out_shape=raster_shape, transform=raster_transform,
        fill=0.0, dtype=np.float32, merge_alg=rasterio.enums.MergeAlg.replace)


DIR_CLR = {"N": "#f5a623", "S": "#7ed321", "E": "#9013fe", "W": "#d0021b", "S_fallback": "#aaaaaa", None: "#aaaaaa"}

def direction_legend_patches():
    return [mpatches.Patch(color=DIR_CLR[k], label=l)
            for k, l in [("S","South"),("N","North"),("E","East"),("W","West"),("S_fallback","Southernmost fallback")]]


def save_fig(figs_dir, name, dpi=150, extra_formats=()):
    import matplotlib.pyplot as plt
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    plt.savefig(figs_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    for fmt in extra_formats:
        plt.savefig(figs_dir / f"{stem}{fmt}", bbox_inches="tight")
    print(f"  -> {stem}.png")


def load_dem():
    print("Loading DEM ...")
    with rasterio.open(str(DEM_P)) as src:
        dem = {"arr": src.read(1).astype(np.float32), "crs": src.crs,
               "nodata": src.nodata, "bounds": src.bounds,
               "res": abs(src.transform[0]), "transform": src.transform, "profile": src.profile.copy()}
    nd = dem["nodata"]
    dem["disp"]   = np.where(dem["arr"] == nd, np.nan, dem["arr"]) if nd is not None else dem["arr"].copy()
    dem["e_min"]  = float(np.nanmin(dem["disp"]))
    dem["e_max"]  = float(np.nanmax(dem["disp"]))
    dem["extent"] = [dem["bounds"].left, dem["bounds"].right, dem["bounds"].bottom, dem["bounds"].top]
    print(f"  CRS={dem['crs']}  shape={dem['arr'].shape}  res={dem['res']:.2f} m  elev={dem['e_min']:.1f}-{dem['e_max']:.1f} m")
    return dem


def hillshade(arr, azimuth=315, altitude=45):
    az_r, alt_r = np.radians(360 - azimuth + 90), np.radians(altitude)
    fill = np.where(np.isnan(arr), 0, arr)
    dy, dx = np.gradient(fill)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    return np.clip(np.sin(alt_r)*np.cos(slope) + np.cos(alt_r)*np.sin(slope)*np.cos(az_r-aspect), 0, 1)


def sample_dem_at_doors(doors_pts):
    with rasterio.open(str(DEM_P)) as src:
        pts = doors_pts.to_crs(src.crs) if doors_pts.crs != src.crs else doors_pts
        vals = np.array([v[0] for v in src.sample([(g.x, g.y) for g in pts.geometry])], dtype=np.float32)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    doors_pts = doors_pts.copy()
    doors_pts["elevation_m"] = vals
    print("Door elevation stats:")
    print(pd.Series(vals).describe().to_string())
    return doors_pts


def _draw_chapel_ids(ax, fp, fontsize=4, color="white", bbox=None):
    for _, row in fp.iterrows():
        cid = row.get("chapel_id")
        if cid is not None and str(cid) not in ("None", "nan", ""):
            kw = dict(ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold")
            if bbox:
                kw["bbox"] = bbox
            ax.text(row.geometry.centroid.x, row.geometry.centroid.y, str(cid), **kw)
