import numpy as np
import pandas as pd
import sys

from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

from config import (
    SLOW_SCORE_POINTS_FILE,
    CLUSTER_FRAMES_FILE,
    CLUSTER_POINTS_FILE,
    TRACKED_EVENTS_FILE,
    DBSCAN_EPS_METERS,
    DBSCAN_MIN_SAMPLES,
    CLUSTER_HULL_BUFFER_METERS,
    FRAME_FREQ,
    TRACK_MAX_CENTROID_DISTANCE_M,
    ROAD_MIN_UNIQUE_ROUTES,
    ROAD_MIN_LINE_ENTROPY,
    ROAD_MAX_DWELL_FRACTION,
    ROAD_MAX_TERMINAL_FRACTION,
    ROAD_MIN_UNIQUE_BUSES,
    DWELL_EVENT_MIN_DWELL_FRACTION,
    TERMINAL_EVENT_MIN_TERMINAL_FRACTION,
)

from utils import (
    safe_read_table,
    normalized_entropy,
    largest_category_share,
)


def classify_cluster_row(row):
    if row["dwell_fraction"] >= DWELL_EVENT_MIN_DWELL_FRACTION:
        return "stop_or_terminal_dwell"

    if row["terminal_fraction"] >= TERMINAL_EVENT_MIN_TERMINAL_FRACTION:
        return "stop_or_terminal_dwell"

    if (
        row["num_lines"] >= ROAD_MIN_UNIQUE_ROUTES
        and row["line_entropy"] >= ROAD_MIN_LINE_ENTROPY
        and row["dwell_fraction"] <= ROAD_MAX_DWELL_FRACTION
        and row["terminal_fraction"] <= ROAD_MAX_TERMINAL_FRACTION
        and row["num_buses"] >= ROAD_MIN_UNIQUE_BUSES
    ):
        return "road_network_candidate"

    return "mixed_or_uncertain"


def detect_clusters_for_frame(time_bin, frame_df):
    cluster_rows = []
    cluster_point_rows = []

    candidate = frame_df[frame_df["raw_slow_candidate"] == True].copy()

    if len(candidate) < DBSCAN_MIN_SAMPLES:
        return cluster_rows, cluster_point_rows

    coords = candidate[["snap_x", "snap_y"]].to_numpy()

    clustering = DBSCAN(
        eps=DBSCAN_EPS_METERS,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit(coords)

    candidate["cluster_id"] = clustering.labels_

    cluster_labels = [
        label for label in sorted(candidate["cluster_id"].unique()) if label != -1
    ]

    for label in cluster_labels:
        cluster_df = candidate[candidate["cluster_id"] == label].copy()

        if len(cluster_df) < DBSCAN_MIN_SAMPLES:
            continue

        multipoint = MultiPoint(
            list(zip(cluster_df["snap_x"], cluster_df["snap_y"]))
        )

        hull = multipoint.convex_hull.buffer(CLUSTER_HULL_BUFFER_METERS)
        centroid = hull.centroid

        if "gtfs_route_id" in cluster_df.columns and cluster_df["gtfs_route_id"].notna().any():
            line_values = cluster_df["gtfs_route_id"]
        elif "lineId" in cluster_df.columns:
            line_values = cluster_df["lineId"]
        else:
            line_values = pd.Series(["unknown"] * len(cluster_df))

        line_entropy = normalized_entropy(line_values)
        largest_line_share = largest_category_share(line_values)

        if "edge_uid" in cluster_df.columns:
            dominant_edge = cluster_df["edge_uid"].astype(str).value_counts().idxmax()
        else:
            dominant_edge = "unknown"

        clean_line_values = line_values.dropna().astype(str)

        if not clean_line_values.empty:
            dominant_line = clean_line_values.value_counts().idxmax()
        else:
            dominant_line = "unknown"

        dwell_fraction = float(
            cluster_df["stop_or_terminal_dwell_candidate"].fillna(False).mean()
        )

        terminal_fraction = float(
            cluster_df["terminal_dwell_candidate"].fillna(False).mean()
        )

        near_terminal_fraction = float(
            cluster_df["is_near_terminal"].fillna(False).mean()
        )

        cluster_row = {
            "time_bin": time_bin,
            "frame_cluster_id": int(label),
            "num_points": len(cluster_df),
            "num_buses": cluster_df["id"].nunique(),
            "num_lines": line_values.dropna().astype(str).nunique(),
            "line_entropy": line_entropy,
            "largest_line_share": largest_line_share,
            "mean_slow_score": cluster_df["slow_score"].mean(),
            "max_slow_score": cluster_df["slow_score"].max(),
            "mean_rolling_speed_kmh": cluster_df["rolling_speed_kmh"].mean(),
            "dwell_fraction": dwell_fraction,
            "terminal_fraction": terminal_fraction,\
            "near_terminal_fraction": near_terminal_fraction,
            "centroid_x": centroid.x,
            "centroid_y": centroid.y,
            "dominant_edge": dominant_edge,
            "dominant_line": dominant_line,
            "hull_wkt": hull.wkt,
        }

        cluster_row["cluster_type"] = classify_cluster_row(cluster_row)

        cluster_rows.append(cluster_row)

        point_keep_cols = [
            "timestamp",
            "time_bin",
            "id",
            "tripId",
            "edge_uid",
            "snap_x",
            "snap_y",
            "slow_score",
            "rolling_speed_kmh",
            "raw_slow_candidate",
            "stop_dwell_candidate",
            "terminal_dwell_candidate",
            "stop_or_terminal_dwell_candidate",
            "is_dwell_candidate",
            "is_near_stop",
            "is_near_terminal",
            "road_slow_candidate",
        ]

        point_keep_cols = [
            col for col in point_keep_cols
            if col in cluster_df.columns
        ]

        for optional_col in ["lineId", "lineName", "gtfs_route_id"]:
            if optional_col in cluster_df.columns:
                point_keep_cols.append(optional_col)

        cluster_points = cluster_df[point_keep_cols].copy()
        cluster_points["frame_cluster_id"] = int(label)

        cluster_point_rows.append(cluster_points)

    return cluster_rows, cluster_point_rows

def assign_event_ids(cluster_frames):
    if cluster_frames.empty:
        cluster_frames["event_id"] = []
        return cluster_frames

    cluster_frames = cluster_frames.copy()
    cluster_frames["time_bin"] = pd.to_datetime(cluster_frames["time_bin"])
    cluster_frames = cluster_frames.sort_values(["time_bin", "frame_cluster_id"]).copy()
    cluster_frames["event_id"] = -1

    frame_delta = pd.Timedelta(FRAME_FREQ)
    max_time_gap = frame_delta * 2

    next_event_id = 0
    active_events = {}

    for idx, row in cluster_frames.iterrows():
        t = row["time_bin"]
        cx = row["centroid_x"]
        cy = row["centroid_y"]

        best_event_id = None
        best_distance = None

        for event_id, event_info in active_events.items():
            dt = t - event_info["last_time"]

            if dt < pd.Timedelta(0) or dt > max_time_gap:
                continue

            dx = cx - event_info["last_centroid_x"]
            dy = cy - event_info["last_centroid_y"]
            dist = float(np.sqrt(dx * dx + dy * dy))

            if dist <= TRACK_MAX_CENTROID_DISTANCE_M:
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_event_id = event_id

        if best_event_id is None:
            best_event_id = next_event_id
            next_event_id += 1

        cluster_frames.loc[idx, "event_id"] = best_event_id

        active_events[best_event_id] = {
            "last_time": t,
            "last_centroid_x": cx,
            "last_centroid_y": cy,
        }

    return cluster_frames

# Main
print("Loading slow-score points...")

points = safe_read_table(SLOW_SCORE_POINTS_FILE)
points["time_bin"] = pd.to_datetime(points["time_bin"])

cluster_rows = []
cluster_point_tables = []

print("Detecting slow clusters by frame...")

for time_bin, frame_df in points.groupby("time_bin"):
    rows, point_tables = detect_clusters_for_frame(time_bin, frame_df)

    cluster_rows.extend(rows)
    cluster_point_tables.extend(point_tables)

cluster_frames = pd.DataFrame(cluster_rows)

if cluster_frames.empty:
    print("No clusters found.")

    cluster_frames.to_csv(CLUSTER_FRAMES_FILE, index=False)
    pd.DataFrame().to_csv(CLUSTER_POINTS_FILE, index=False)
    pd.DataFrame().to_csv(TRACKED_EVENTS_FILE, index=False)

    sys.exit(0)

print(f"Detected frame-clusters: {len(cluster_frames):,}")

cluster_frames = assign_event_ids(cluster_frames)
cluster_frames.to_csv(CLUSTER_FRAMES_FILE, index=False)

print(f"Saved cluster frames: {CLUSTER_FRAMES_FILE}")

if cluster_point_tables:
    cluster_points = pd.concat(cluster_point_tables, ignore_index=True)

    event_lookup = cluster_frames[
        ["time_bin", "frame_cluster_id", "event_id"]
    ].copy()

    event_lookup["time_bin"] = pd.to_datetime(event_lookup["time_bin"])
    cluster_points["time_bin"] = pd.to_datetime(cluster_points["time_bin"])

    cluster_points = cluster_points.merge(
        event_lookup,
        on=["time_bin", "frame_cluster_id"],
        how="left",
    )

    cluster_points.to_csv(CLUSTER_POINTS_FILE, index=False)

    print(f"Saved cluster points: {CLUSTER_POINTS_FILE}")

tracked = (
    cluster_frames.groupby("event_id")
    .agg(
        start_time=("time_bin", "min"),
        end_time=("time_bin", "max"),
        frames=("time_bin", "nunique"),
        max_buses=("num_buses", "max"),
        max_lines=("num_lines", "max"),
        max_points=("num_points", "max"),
        mean_line_entropy=("line_entropy", "mean"),
        max_line_entropy=("line_entropy", "max"),
        mean_slow_score=("mean_slow_score", "mean"),
        peak_slow_score=("max_slow_score", "max"),
        mean_dwell_fraction=("dwell_fraction", "mean"),
        mean_terminal_fraction=("terminal_fraction", "mean"),
        first_centroid_x=("centroid_x", "first"),
        first_centroid_y=("centroid_y", "first"),
        last_centroid_x=("centroid_x", "last"),
        last_centroid_y=("centroid_y", "last"),
    )
    .reset_index()
)

tracked["duration_minutes"] = (
    pd.to_datetime(tracked["end_time"]) - pd.to_datetime(tracked["start_time"])
).dt.total_seconds() / 60

tracked["displacement_m"] = np.sqrt(
    (tracked["last_centroid_x"] - tracked["first_centroid_x"]) ** 2
    + (tracked["last_centroid_y"] - tracked["first_centroid_y"]) ** 2
)

tracked["propagation_speed_m_per_min"] = np.where(
    tracked["duration_minutes"] > 0,
    tracked["displacement_m"] / tracked["duration_minutes"],
    0,
)

tracked.to_csv(TRACKED_EVENTS_FILE, index=False)

print(f"Saved tracked events: {TRACKED_EVENTS_FILE}")

print("Cluster type counts:")
print(cluster_frames["cluster_type"].value_counts())