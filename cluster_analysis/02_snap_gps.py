import pandas as pd
import geopandas as gpd
import osmnx as ox
from config import (
    MOBILITY_FILE,
    IBGE_BAIRROS,
    FOCUS_BAIRRO,
    PREPROCESS_START_TIME,
    PREPROCESS_END_TIME,
    CHUNK_SIZE,
    METRIC_CRS,
    NETWORK_TYPE,
    OSM_EDGES_FILE,
    SNAPPED_POINTS_FILE,
)
from utils import load_focus_bairro, make_edge_uid, safe_to_parquet

# Main
print("Loading focus bairro...")

focus_bairro = load_focus_bairro(IBGE_BAIRROS, FOCUS_BAIRRO)
focus_union = focus_bairro.geometry.union_all()

print(f"Downloading/loading OSM network for {FOCUS_BAIRRO}...")

G = ox.graph_from_polygon(focus_union, network_type=NETWORK_TYPE, simplify=True)

G_proj = ox.project_graph(G, to_crs=METRIC_CRS)

nodes, edges = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)
edges = edges[edges.geometry.notna()].copy()
edges_reset = edges.reset_index().copy()
edges_reset["edge_uid"] = [
    make_edge_uid(u, v, key)
    for u, v, key in zip(edges_reset["u"], edges_reset["v"], edges_reset["key"])
]
edges_reset.to_file(OSM_EDGES_FILE, driver="GPKG")

print(f"Saved OSM edges: {OSM_EDGES_FILE}")
print(f"Edges in network: {len(edges_reset):,}")

edge_geom_lookup = (
    edges_reset.set_index(["u", "v", "key"])["geometry"].to_dict()
)

minx, miny, maxx, maxy = focus_bairro.total_bounds

records = []

print("Reading mobility CSV and snapping GPS points to nearest OSM edges...")

for chunk_num, chunk in enumerate(
    pd.read_csv(MOBILITY_FILE, chunksize=CHUNK_SIZE),
    start=1,
):
    print(f"Processing chunk {chunk_num}...")

    needed = ["lat", "lng", "timestamp", "id", "tripId"]
    missing = [col for col in needed if col not in chunk.columns]

    if missing:
        raise ValueError(f"Mobility file missing required columns: {missing}")

    chunk = chunk.dropna(subset=["lat", "lng", "timestamp", "id", "tripId"]).copy()

    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
    chunk = chunk.dropna(subset=["timestamp"]).copy()

    chunk = chunk[
        (chunk["timestamp"] >= PREPROCESS_START_TIME)
        & (chunk["timestamp"] <= PREPROCESS_END_TIME)
    ].copy()

    if chunk.empty:
        continue

    chunk = chunk[
        (chunk["lng"] >= minx)
        & (chunk["lng"] <= maxx)
        & (chunk["lat"] >= miny)
        & (chunk["lat"] <= maxy)
    ].copy()

    if chunk.empty:
        continue

    gdf = gpd.GeoDataFrame(
        chunk,
        geometry=gpd.points_from_xy(chunk["lng"], chunk["lat"]),
        crs="EPSG:4326",
    )

    gdf = gdf[gdf.geometry.within(focus_union)].copy()

    if gdf.empty:
        continue

    gdf = gdf.to_crs(METRIC_CRS)

    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()

    nearest = ox.distance.nearest_edges(
        G_proj,
        X=xs,
        Y=ys,
        return_dist=False,
    )

    nearest_df = pd.DataFrame(list(nearest), columns=["u", "v", "key"])

    gdf["u"] = nearest_df["u"].to_numpy()
    gdf["v"] = nearest_df["v"].to_numpy()
    gdf["key"] = nearest_df["key"].to_numpy()

    gdf["edge_uid"] = [
        make_edge_uid(u, v, key)
        for u, v, key in zip(gdf["u"], gdf["v"], gdf["key"])
    ]

    snapped_points = []

    for point, u_i, v_i, key_i in zip(
        gdf.geometry,
        gdf["u"],
        gdf["v"],
        gdf["key"],
    ):
        edge_geom = edge_geom_lookup.get((u_i, v_i, key_i))

        if edge_geom is None:
            snapped_points.append(point)
        else:
            snapped_point = edge_geom.interpolate(edge_geom.project(point))
            snapped_points.append(snapped_point)

    gdf["raw_x"] = gdf.geometry.x
    gdf["raw_y"] = gdf.geometry.y

    gdf["snap_x"] = [p.x for p in snapped_points]
    gdf["snap_y"] = [p.y for p in snapped_points]

    keep_cols = [
        "timestamp",
        "edge_uid",
        "id",
        "tripId",
        "lat",
        "lng",
        "raw_x",
        "raw_y",
        "snap_x",
        "snap_y",
    ]

    optional_cols = ["lineId", "lineName", "headsign", "direction"]

    for col in optional_cols:
        if col in gdf.columns:
            keep_cols.append(col)

    records.append(gdf[keep_cols].copy())

if not records:
    raise ValueError("No GPS points were snapped for this area/time window.")

snapped = pd.concat(records, ignore_index=True)

print(f"Total snapped rows: {len(snapped):,}")
print(f"Unique buses: {snapped['id'].nunique():,}")
print(f"Unique edges: {snapped['edge_uid'].nunique():,}")

safe_to_parquet(snapped, SNAPPED_POINTS_FILE)