import numpy as np
from shapely.geometry import LineString
import geopandas as gpd

def get_greedy_doors(loose_ends, gap_max, buffered_mask):
    pairs = []
    for i in range(len(loose_ends)):
        for j in range(i+1, len(loose_ends)):
            p1, p2 = loose_ends[i], loose_ends[j]
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if dist <= gap_max:
                pairs.append((dist, p1, p2))
    pairs.sort(key=lambda x: x[0])
    used = set()
    doors = []
    for dist, p1, p2 in pairs:
        if p1 not in used and p2 not in used:
            ls = LineString([p1, p2])
            if ls.intersects(buffered_mask):
                used.add(p1)
                used.add(p2)
                doors.append({"geom": ls, "type": "greedy"})
    return doors

def get_og_doors(loose_ends, gap_max, buffered_mask):
    pairs = []
    for i in range(len(loose_ends)):
        for j in range(i+1, len(loose_ends)):
            p1, p2 = loose_ends[i], loose_ends[j]
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if dist <= gap_max:
                pairs.append((dist, p1, p2))
    doors = []
    for dist, p1, p2 in pairs:
        ls = LineString([p1, p2])
        if ls.intersects(buffered_mask):
            doors.append({"geom": ls, "type": "og"})
    return doors

def get_collinear_doors(loose_ends, loose_end_vectors, gap_max, buffered_mask):
    pairs = []
    for i in range(len(loose_ends)):
        for j in range(i+1, len(loose_ends)):
            p1, p2 = loose_ends[i], loose_ends[j]
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if dist <= gap_max:
                v1 = loose_end_vectors.get(p1)
                v2 = loose_end_vectors.get(p2)
                if v1 is not None and v2 is not None:
                    n1 = v1 / (np.linalg.norm(v1) + 1e-9)
                    n2 = v2 / (np.linalg.norm(v2) + 1e-9)
                    # Parallel or anti-parallel
                    if abs(np.dot(n1, n2)) > 0.866: # within 30 degrees
                        pairs.append((dist, p1, p2))
    pairs.sort(key=lambda x: x[0])
    used = set()
    doors = []
    for dist, p1, p2 in pairs:
        if p1 not in used and p2 not in used:
            ls = LineString([p1, p2])
            if ls.intersects(buffered_mask):
                used.add(p1)
                used.add(p2)
                doors.append({"geom": ls, "type": "collinear"})
    return doors

def get_boolean_doors(cad_lines_gdf, footprints_gdf):
    # Buffer CAD lines by 1.5m to ensure they cover the shapefile boundary
    cad_buffered = cad_lines_gdf.geometry.buffer(1.5).unary_union
    
    doors = []
    for idx, row in footprints_gdf.iterrows():
        boundary = row.geometry.boundary
        # Subtract the CAD lines from the boundary
        diff = boundary.difference(cad_buffered)
        if diff.is_empty:
            continue
        if diff.geom_type == 'MultiLineString':
            for line in diff.geoms:
                if line.length > 0.5 and line.length < 5.0:
                    doors.append({"geom": line, "type": "boolean"})
        elif diff.geom_type == 'LineString':
            if diff.length > 0.5 and diff.length < 5.0:
                doors.append({"geom": diff, "type": "boolean"})
    return doors

def get_native_doors(footprints_gdf):
    doors = []
    for idx, row in footprints_gdf.iterrows():
        poly = row.geometry
        if poly.geom_type != 'Polygon': continue
        coords = list(poly.exterior.coords)
        best_seg = None
        max_south = -float('inf')
        for i in range(len(coords)-1):
            p1, p2 = coords[i], coords[i+1]
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            length = np.hypot(dx, dy)
            if length < 0.1: continue
            cx, cy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            score = -cy # southernmost
            if score > max_south:
                max_south = score
                best_seg = (p1, p2)
        if best_seg:
            p1, p2 = best_seg
            cx, cy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            length = np.hypot(dx, dy)
            if length > 0:
                ux, uy = dx/length, dy/length
                door_len = min(1.0, length/2)
                dp1 = (cx - ux*door_len/2, cy - uy*door_len/2)
                dp2 = (cx + ux*door_len/2, cy + uy*door_len/2)
                doors.append({"geom": LineString([dp1, dp2]), "type": "native"})
    return doors

def get_local_doors(lines_utm, loose_ends, crosswalk, dxf_labels_utm, footprints_gdf):
    doors = []
    offsets = {}
    for idx, row in crosswalk.iterrows():
        lbl_id = row['chapel_id']
        fp_id = row['footprint_id']
        if lbl_id in dxf_labels_utm:
            pt_lbl = dxf_labels_utm[lbl_id]
            fp = footprints_gdf[footprints_gdf['ID'] == fp_id]
            if not fp.empty:
                pt_fp = fp.iloc[0].geometry.centroid
                offsets[lbl_id] = (pt_fp.x - pt_lbl[0], pt_fp.y - pt_lbl[1], pt_lbl, fp.iloc[0].geometry)
                
    used = set()
    for lbl_id, (ox, oy, pt_lbl, poly) in offsets.items():
        local_ends = []
        for p in loose_ends:
            if np.hypot(p[0]-pt_lbl[0], p[1]-pt_lbl[1]) < 25.0:
                local_ends.append(p)
                
        shifted_ends = [(p[0]+ox, p[1]+oy) for p in local_ends]
        
        pairs = []
        for i in range(len(shifted_ends)):
            for j in range(i+1, len(shifted_ends)):
                p1, p2 = shifted_ends[i], shifted_ends[j]
                dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if dist <= 3.5:
                    pairs.append((dist, p1, p2))
        pairs.sort(key=lambda x: x[0])
        
        buffered_mask = poly.boundary.buffer(3.5)
        for dist, p1, p2 in pairs:
            if p1 not in used and p2 not in used:
                ls = LineString([p1, p2])
                if ls.intersects(buffered_mask):
                    used.add(p1)
                    used.add(p2)
                    doors.append({"geom": ls, "type": "local"})
    return doors

def get_hybrid_doors(greedy_doors, footprints_gdf):
    doors = []
    from shapely.geometry import Point, LineString
    from shapely.ops import nearest_points
    fp_bounds = footprints_gdf.boundary.unary_union
    
    for d in greedy_doors:
        geom = d['geom']
        cent = geom.centroid
        p1, p2 = nearest_points(fp_bounds, cent)
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        shifted_geom = LineString([(geom.coords[0][0]+dx, geom.coords[0][1]+dy), 
                                   (geom.coords[1][0]+dx, geom.coords[1][1]+dy)])
        doors.append({"geom": shifted_geom, "type": "hybrid"})
    return doors
