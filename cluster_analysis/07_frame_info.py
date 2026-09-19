import argparse
import sys
import pandas as pd
import geopandas as gpd
from config import (
    FOCUS_BAIRRO,
    IBGE_BAIRROS,
    METRIC_CRS,
    SLOW_SCORE_POINTS_FILE,
    PREPARED_GTFS_STOPS,
    PREPARED_GTFS_TERMINALS,
    CLUSTER_FRAMES_FILE,
    CLUSTER_SLOW_SCORE_THRESHOLD,
)
from utils import load_focus_bairro, safe_read_table, count_true, safe_nunique, filter_gdf_to_bairro, get_row_value

CLUSTER_TYPE_COLUMNS = [
    "event_type",
    "cluster_type",
    "event_class",
    "classification",
    "class_label",
]

def get_cluster_type(row):
    for col in CLUSTER_TYPE_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])

    return "unclassified"

def find_nearest_time(target_time, available_times):
    available = pd.Series(pd.to_datetime(sorted(available_times)))

    if available.empty:
        return None, None

    diffs = (available - target_time).abs()
    idx = diffs.idxmin()

    return available.loc[idx], diffs.loc[idx]

def print_frame_summary(
    t,
    points_df,
    clusters_df,
    stops_count,
    terminals_count,
    requested_time=None,
    time_diff=None,
):
    total_points = len(points_df)
    active_buses = safe_nunique(points_df, "id")
    active_edges = safe_nunique(points_df, "edge_uid")

    if "slow_score" in points_df.columns:
        fallback_slow = points_df["slow_score"] >= CLUSTER_SLOW_SCORE_THRESHOLD
    else:
        fallback_slow = None

    raw_slow_points = count_true(
        points_df,
        "raw_slow_candidate",
        fallback=fallback_slow,
    )

    stop_dwell_points = count_true(points_df, "stop_dwell_candidate")
    terminal_dwell_points = count_true(points_df, "terminal_dwell_candidate")
    road_slow_points = count_true(points_df, "road_slow_candidate")

    cluster_count = len(clusters_df)

    if cluster_count > 0 and "num_buses" in clusters_df.columns:
        largest_cluster_size = int(clusters_df["num_buses"].max())
    else:
        largest_cluster_size = 0

    print()
    print("=" * 54)
    print("FRAME SUMMARY")
    print("=" * 54)

    if requested_time is not None:
        print(f"Requested time: {requested_time}")

    print(f"Matched frame:  {t}")

    if time_diff is not None:
        print(f"Time offset:    {time_diff}")

    print()
    print("Point counts")
    print("-" * 54)
    print(f"GPS points:              {total_points}")
    print(f"Active buses:            {active_buses}")
    print(f"Assigned OSM edges:      {active_edges}")
    print()
    print(f"Raw slow points:         {raw_slow_points}")
    print(f"Road-candidate points:   {road_slow_points}")
    print(f"Stop-dwell points:       {stop_dwell_points}")
    print(f"Terminal-dwell points:   {terminal_dwell_points}")
    print()
    print("Cluster counts")
    print("-" * 54)
    print(f"Slow clusters:           {cluster_count}")
    print(f"Largest cluster:         {largest_cluster_size} buses")
    print()
    print("Map context")
    print("-" * 54)
    print(f"GTFS stops in bairro:    {stops_count}")
    print(f"GTFS terminals in bairro:{terminals_count}")

    if cluster_count > 0:
        print()
        print("Clusters in this frame")
        print("-" * 54)

        display_cols = [
            "event_id",
            "event_type",
            "cluster_type",
            "num_buses",
            "num_lines",
            "mean_slow_score",
            "max_slow_score",
        ]

        available_cols = [
            col for col in display_cols
            if col in clusters_df.columns
        ]

        if available_cols:
            clusters_out = clusters_df[available_cols].copy()

            if "event_type" not in clusters_out.columns:
                clusters_out["event_type"] = clusters_df.apply(
                    get_cluster_type,
                    axis=1,
                )

            sort_cols = [
                col for col in ["num_buses", "num_lines", "mean_slow_score"]
                if col in clusters_out.columns
            ]

            if sort_cols:
                clusters_out = clusters_out.sort_values(
                    by=sort_cols,
                    ascending=[False] * len(sort_cols),
                )

            print(clusters_out.to_string(index=False))
        else:
            print(clusters_df.to_string(index=False))

    print("=" * 54)
    print()

# Main
parser = argparse.ArgumentParser(
    description="Print frame summary information for a given date and time."
)

parser.add_argument(
    "datetime_parts",
    nargs="+",
    help='Date/time, for example: 2026-03-11 07:45:00',
)

parser.add_argument(
    "--exact",
    action="store_true",
    help="Require exact time_bin match instead of using nearest frame.",
)

args = parser.parse_args()

requested_time_string = " ".join(args.datetime_parts)
requested_time = pd.to_datetime(requested_time_string)

print("Loading points...")
points = safe_read_table(SLOW_SCORE_POINTS_FILE)
points["time_bin"] = pd.to_datetime(points["time_bin"])

print("Loading clusters...")
if CLUSTER_FRAMES_FILE.exists():
    clusters = pd.read_csv(CLUSTER_FRAMES_FILE)
    clusters["time_bin"] = pd.to_datetime(clusters["time_bin"])
else:
    clusters = pd.DataFrame(columns=["time_bin"])

print("Loading stops and terminals...")
focus_bairro = load_focus_bairro(IBGE_BAIRROS, FOCUS_BAIRRO)
focus_bairro_metric = focus_bairro.to_crs(METRIC_CRS)

stops = gpd.read_file(PREPARED_GTFS_STOPS).to_crs(METRIC_CRS)
terminals = gpd.read_file(PREPARED_GTFS_TERMINALS).to_crs(METRIC_CRS)

stops_bairro = filter_gdf_to_bairro(stops, focus_bairro_metric)
terminals_bairro = filter_gdf_to_bairro(terminals, focus_bairro_metric)

available_times = set(points["time_bin"].dropna().unique())

if not clusters.empty:
    available_times.update(clusters["time_bin"].dropna().unique())

if args.exact:
    matched_time = requested_time
    time_diff = pd.Timedelta(0)

    if matched_time not in available_times:
        print()
        print(f"No exact frame found for {requested_time}.")
        print("Try without --exact to use the nearest available frame.")
        sys.exit()
else:
    matched_time, time_diff = find_nearest_time(
        requested_time,
        available_times,
    )

    if matched_time is None:
        print("No available frames found.")
        sys.exit()

points_df = points[points["time_bin"] == matched_time].copy()

if not clusters.empty:
    clusters_df = clusters[clusters["time_bin"] == matched_time].copy()
else:
    clusters_df = pd.DataFrame()

print_frame_summary(
    matched_time,
    points_df,
    clusters_df,
    len(stops_bairro),
    len(terminals_bairro),
    requested_time=requested_time,
    time_diff=time_diff,
)