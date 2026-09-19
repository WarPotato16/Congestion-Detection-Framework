import numpy as np
import pandas as pd
import geopandas as gpd
from config import (
    METRIC_CRS,
    SNAPPED_POINTS_FILE,
    ENRICHED_POINTS_FILE,
    PREPARED_GTFS_STOPS,
    PREPARED_GTFS_TERMINALS,
    PREPARED_GTFS_SHAPES,
    PREPARED_GTFS_ROUTES,
    PREPARED_GTFS_ROUTE_SHAPES,
    NEAR_STOP_METERS,
    NEAR_TERMINAL_METERS,
)
from utils import safe_read_table, safe_to_parquet

def build_route_id_map(routes):
    pairs = []

    for _, row in routes.iterrows():
        target_route_id = str(row.get("route_id", "")).strip()

        if not target_route_id:
            continue

        candidate_cols = [
            "route_id",
            "route_id_original",
            "route_short_name",
            "route_long_name",
        ]

        for col in candidate_cols:
            if col not in routes.columns:
                continue

            value = row.get(col)

            if pd.isna(value):
                continue

            key = str(value).strip()

            if not key:
                continue

            pairs.append((key, target_route_id))

    if not pairs:
        return {}

    map_df = pd.DataFrame(pairs, columns=["key", "route_id"]).drop_duplicates()

    counts = map_df.groupby("key")["route_id"].nunique()

    ambiguous_keys = set(counts[counts > 1].index)
    unique_keys = set(counts[counts == 1].index)

    if ambiguous_keys:
        print(
            f"Warning: {len(ambiguous_keys)} route labels map to multiple GTFS routes. "
            "Those ambiguous labels will not be used automatically."
        )

    clean_map = (
        map_df[map_df["key"].isin(unique_keys)]
        .drop_duplicates("key")
        .set_index("key")["route_id"]
        .to_dict()
    )

    return clean_map

def print_route_label_candidates(routes, labels_to_check):
    print("\nRoute label candidate diagnostic:")

    candidate_cols = [
        "route_id",
        "route_id_original",
        "route_short_name",
        "route_long_name",
    ]

    for label in labels_to_check:
        label = str(label).strip()

        matches = []

        for _, row in routes.iterrows():
            for col in candidate_cols:
                if col not in routes.columns:
                    continue

                value = row.get(col)

                if pd.isna(value):
                    continue

                if str(value).strip() == label:
                    matches.append(row)

        if not matches:
            print(f"\n{label}: no GTFS candidates found")
            continue

        match_df = pd.DataFrame(matches).drop_duplicates()

        print(f"\n{label}: {len(match_df)} candidate row(s)")

        show_cols = [
            col for col in [
                "feed_id",
                "route_id",
                "route_id_original",
                "route_short_name",
                "route_long_name",
            ]
            if col in match_df.columns
        ]

        print(match_df[show_cols].drop_duplicates().to_string(index=False))

def add_nearest_stops(points_gdf, stops_gdf):
    print("Adding nearest GTFS stop...")

    if stops_gdf.empty:
        points_gdf["nearest_stop_id"] = np.nan
        points_gdf["nearest_stop_name"] = np.nan
        points_gdf["distance_to_stop_m"] = np.nan
        points_gdf["is_near_stop"] = False
        return points_gdf

    cols = ["stop_id", "geometry"]

    if "stop_name" in stops_gdf.columns:
        cols.append("stop_name")

    points_gdf = ensure_point_row_id(points_gdf)

    nearest = gpd.sjoin_nearest(
        points_gdf,
        stops_gdf[cols],
        how="left",
        distance_col="distance_to_stop_m",
    )

    nearest = (
        nearest.sort_values(["point_row_id", "distance_to_stop_m"])
        .drop_duplicates("point_row_id")
        .sort_values("point_row_id")
    )

    rename_map = {
        "stop_id": "nearest_stop_id",
        "stop_name": "nearest_stop_name",
    }

    nearest = nearest.rename(columns=rename_map)

    if "index_right" in nearest.columns:
        nearest = nearest.drop(columns=["index_right"])

    nearest["is_near_stop"] = nearest["distance_to_stop_m"] <= NEAR_STOP_METERS

    return nearest


def add_nearest_terminals(points_gdf, terminals_gdf):
    print("Adding nearest GTFS terminal...")

    if terminals_gdf.empty:
        points_gdf["nearest_terminal_stop_id"] = np.nan
        points_gdf["nearest_terminal_name"] = np.nan
        points_gdf["distance_to_terminal_m"] = np.nan
        points_gdf["is_near_terminal"] = False
        return points_gdf

    cols = ["stop_id", "geometry"]

    if "stop_name" in terminals_gdf.columns:
        cols.append("stop_name")

    points_gdf = ensure_point_row_id(points_gdf)

    nearest = gpd.sjoin_nearest(
        points_gdf,
        terminals_gdf[cols],
        how="left",
        distance_col="distance_to_terminal_m",
    )

    nearest = (
        nearest.sort_values(["point_row_id", "distance_to_terminal_m"])
        .drop_duplicates("point_row_id")
        .sort_values("point_row_id")
    )

    nearest = nearest.rename(
        columns={
            "stop_id": "nearest_terminal_stop_id",
            "stop_name": "nearest_terminal_name",
        }
    )

    if "index_right" in nearest.columns:
        nearest = nearest.drop(columns=["index_right"])

    nearest["is_near_terminal"] = (
        nearest["distance_to_terminal_m"] <= NEAR_TERMINAL_METERS
    )

    return nearest


def add_route_id(points_gdf):
    print("Adding GTFS route_id from telemetry lineId...")

    routes = pd.read_csv(PREPARED_GTFS_ROUTES, dtype=str)
    print_route_label_candidates(
        routes,
        labels_to_check=[
            "45", "48", "53", "OC2", "36", "35", "39A", "OC1",
            "46", "38A", "44", "40", "OC3", "34A", "37",
        ],
    )
    route_map = build_route_id_map(routes)

    points_gdf["gtfs_route_id"] = pd.Series(pd.NA, index=points_gdf.index, dtype="string")

    if "lineId" in points_gdf.columns:
        points_gdf["lineId_str"] = points_gdf["lineId"].astype(str).str.strip()
        points_gdf["gtfs_route_id"] = points_gdf["lineId_str"].map(route_map)
    
    if "lineName" in points_gdf.columns:
        missing = points_gdf["gtfs_route_id"].isna()
        points_gdf["lineName_str"] = points_gdf["lineName"].astype(str).str.strip()
        points_gdf.loc[missing, "gtfs_route_id"] = points_gdf.loc[missing, "lineName_str"].map(route_map)
    
    points_gdf["gtfs_feed_id"] = pd.Series(pd.NA, index=points_gdf.index, dtype="string")
    matched_mask = points_gdf["gtfs_route_id"].notna()

    points_gdf.loc[matched_mask, "gtfs_feed_id"] = (points_gdf.loc[matched_mask, "gtfs_route_id"].astype(str).str.split("::", n=1).str[0])
    matched = points_gdf["gtfs_route_id"].notna().sum()

    print("\nTop unmatched telemetry lineIds: ")
    if "lineId" in points_gdf.columns:
        unmatched = points_gdf[points_gdf["gtfs_route_id"].isna()].copy()
        print(unmatched["lineId"].astype(str).value_counts().head(30))
    print("\nTop matched telemetry lineIds:")

    if "lineId" in points_gdf.columns:
        matched_points = points_gdf[points_gdf["gtfs_route_id"].notna()].copy()
        print(matched_points["lineId"].astype(str).value_counts().head(30))

    print(f"Matched GTFS route_id for {matched:,} / {len(points_gdf):,} points.")

    if "gtfs_feed_id" in points_gdf.columns:
        print("Matched points by GTFS feed:")
        print(points_gdf["gtfs_feed_id"].value_counts(dropna=False))
    
    return points_gdf

def ensure_point_row_id(points_gdf):
    points_gdf = points_gdf.copy()

    if "point_row_id" not in points_gdf.columns:
        points_gdf = points_gdf.reset_index(drop=True)
        points_gdf["point_row_id"] = np.arange(len(points_gdf))

    return points_gdf

def add_distance_along_shape(points_gdf):
    print("Adding distance along GTFS shape where possible...")

    if not PREPARED_GTFS_SHAPES.exists() or not PREPARED_GTFS_ROUTE_SHAPES.exists():
        points_gdf["shape_id"] = pd.Series(pd.NA, index=points_gdf.index, dtype="string")
        points_gdf["distance_along_shape_m"] = np.nan
        return points_gdf

    shapes_gdf = gpd.read_file(PREPARED_GTFS_SHAPES).to_crs(METRIC_CRS)

    if shapes_gdf.empty:
        points_gdf["shape_id"] = pd.Series(pd.NA, index=points_gdf.index, dtype="string")
        points_gdf["distance_along_shape_m"] = np.nan
        return points_gdf

    route_shapes = pd.read_csv(PREPARED_GTFS_ROUTE_SHAPES, dtype=str)

    if route_shapes.empty:
        points_gdf["shape_id"] = pd.Series(pd.NA, index=points_gdf.index, dtype="string")
        points_gdf["distance_along_shape_m"] = np.nan
        return points_gdf

    shape_geom_lookup = {
        str(row["shape_id"]): geom
        for row, geom in zip(shapes_gdf.to_dict("records"), shapes_gdf.geometry)
    }

    shape_lookup = {}

    for _, row in route_shapes.iterrows():
        route_id = str(row.get("route_id", "")).strip()
        direction_id = str(row.get("direction_id", "")).strip()
        shape_id = str(row.get("shape_id", "")).strip()

        if route_id and shape_id in shape_geom_lookup:
            shape_lookup[(route_id, direction_id)] = shape_id

    points_gdf["shape_id"] = pd.Series(pd.NA, index=points_gdf.index, dtype="string")
    points_gdf["distance_along_shape_m"] = np.nan

    if "direction" in points_gdf.columns:
        points_gdf["direction_str"] = points_gdf["direction"].astype(str).str.strip()
    else:
        points_gdf["direction_str"] = "unknown"

    if "gtfs_route_id" not in points_gdf.columns:
        return points_gdf

    for (route_id, direction_str), idx in points_gdf.groupby(
        ["gtfs_route_id", "direction_str"]
    ).groups.items():
        if pd.isna(route_id):
            continue

        route_id = str(route_id)
        direction_str = str(direction_str)

        shape_id = shape_lookup.get((route_id, direction_str))

        if shape_id is None:
            candidate_shape_ids = [
                sid for (rid, _dir), sid in shape_lookup.items() if rid == route_id
            ]

            if candidate_shape_ids:
                shape_id = candidate_shape_ids[0]

        if shape_id is None:
            continue

        geom = shape_geom_lookup.get(shape_id)

        if geom is None:
            continue

        points_gdf.loc[idx, "shape_id"] = shape_id

        distances = [geom.project(point) for point in points_gdf.loc[idx, "geometry"]]
        points_gdf.loc[idx, "distance_along_shape_m"] = distances

    matched = points_gdf["distance_along_shape_m"].notna().sum()

    print(f"Distance along shape matched for {matched:,} / {len(points_gdf):,} points.")

    return points_gdf


# Main
points = safe_read_table(SNAPPED_POINTS_FILE)

points_gdf = gpd.GeoDataFrame(
    points,
    geometry=gpd.points_from_xy(points["snap_x"], points["snap_y"]),
    crs=METRIC_CRS,
)

stops_gdf = gpd.read_file(PREPARED_GTFS_STOPS).to_crs(METRIC_CRS)
terminals_gdf = gpd.read_file(PREPARED_GTFS_TERMINALS).to_crs(METRIC_CRS)

points_gdf = add_nearest_stops(points_gdf, stops_gdf)
points_gdf = add_nearest_terminals(points_gdf, terminals_gdf)
points_gdf = add_route_id(points_gdf)
points_gdf = add_distance_along_shape(points_gdf)

# Drop geometry before parquet to avoid awkward object columns.
# Coordinates are already stored as snap_x/snap_y.
if "geometry" in points_gdf.columns:
    points_gdf = points_gdf.drop(columns=["geometry"])

safe_to_parquet(pd.DataFrame(points_gdf), ENRICHED_POINTS_FILE)