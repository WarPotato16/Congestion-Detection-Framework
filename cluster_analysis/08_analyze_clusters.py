import pandas as pd
import sys

from config import (
    CLUSTER_FRAMES_FILE,
    TRACKED_EVENTS_FILE,
    EVENT_SUMMARY_FILE,
    ROAD_MIN_DURATION_MINUTES,
    ROAD_MIN_UNIQUE_ROUTES,
    ROAD_MIN_LINE_ENTROPY,
    ROAD_MIN_UNIQUE_BUSES,
    ROAD_MAX_DWELL_FRACTION,
    ROAD_MAX_TERMINAL_FRACTION,
    DWELL_EVENT_MIN_DWELL_FRACTION,
    TERMINAL_EVENT_MIN_TERMINAL_FRACTION,
    STATIONARY_MAX_SPEED_M_PER_MIN,
    STATIONARY_MAX_DISPLACEMENT_M,
    RUBBER_MIN_DURATION_MINUTES,
    RUBBER_MIN_DISPLACEMENT_M,
    RUBBER_MAX_SPEED_M_PER_MIN,
    RUBBER_MIN_MAX_BUSES,
    RUBBER_MIN_MAX_LINES,
    FRAGMENT_MAX_DURATION_MINUTES,
    FRAGMENT_MIN_SPEED_M_PER_MIN
)

def classify_event(row):
    if (
        row["mean_dwell_fraction"] >= DWELL_EVENT_MIN_DWELL_FRACTION
        or row["mean_terminal_fraction"] >= TERMINAL_EVENT_MIN_TERMINAL_FRACTION
    ):
        return "stop_or_terminal_dwell"

    if (
        row["duration_minutes"] >= ROAD_MIN_DURATION_MINUTES
        and row["max_lines"] >= ROAD_MIN_UNIQUE_ROUTES
        and row["max_line_entropy"] >= ROAD_MIN_LINE_ENTROPY
        and row["max_buses"] >= ROAD_MIN_UNIQUE_BUSES
        and row["mean_dwell_fraction"] <= ROAD_MAX_DWELL_FRACTION
        and row["mean_terminal_fraction"] <= ROAD_MAX_TERMINAL_FRACTION
    ):
        return "road_network_congestion_candidate"

    return "mixed_or_uncertain"

def classify_movement(row):
    if (row["duration_minutes"] <= FRAGMENT_MAX_DURATION_MINUTES
        and row["propagation_speed_m_per_min"] >= FRAGMENT_MIN_SPEED_M_PER_MIN):
        return "short_fast_fragment_or_tracking_jump"
    
    if (
        row["duration_minutes"] >= RUBBER_MIN_DURATION_MINUTES
        and row["displacement_m"] >= RUBBER_MIN_DISPLACEMENT_M
        and row["propagation_speed_m_per_min"] <= RUBBER_MAX_SPEED_M_PER_MIN
        and row["max_buses"] >= RUBBER_MIN_MAX_BUSES
        and row["max_lines"] >= RUBBER_MIN_MAX_LINES
        ):
            return "rubber_banding_candidate"

    if (row["propagation_speed_m_per_min"] <= STATIONARY_MAX_SPEED_M_PER_MIN
        and row["displacement_m"] <= STATIONARY_MAX_DISPLACEMENT_M
        ):
            return "stationary_bottleneck_candidate"
    
    return "uncertain_movement"

# Main
print("Loading cluster frames and tracked events...")

cluster_frames = pd.read_csv(CLUSTER_FRAMES_FILE)
tracked = pd.read_csv(TRACKED_EVENTS_FILE)

if cluster_frames.empty or tracked.empty:
    print("No cluster events to analyze.")
    sys.exit(0)

cluster_frames["time_bin"] = pd.to_datetime(cluster_frames["time_bin"])

extra = (
    cluster_frames.groupby("event_id")
    .agg(
        total_frame_clusters=("frame_cluster_id", "count"),
        total_cluster_points=("num_points", "sum"),
        mean_largest_line_share=("largest_line_share", "mean"),
        min_largest_line_share=("largest_line_share", "min"),
        dominant_cluster_type=("cluster_type", lambda s: s.value_counts().idxmax()),
        road_candidate_frames=(
            "cluster_type",
            lambda s: int((s == "road_network_candidate").sum()),
        ),
        stop_or_terminal_dwell_frames=(
            "cluster_type",
            lambda s: int((s == "stop_or_terminal_dwell").sum()),
        ),
    )
    .reset_index()
)

summary = tracked.merge(extra, on="event_id", how="left")

summary["event_type"] = summary.apply(classify_event, axis=1)
summary["movement_type"] = summary.apply(classify_movement, axis=1)

summary = summary.sort_values(
    ["event_type", "duration_minutes", "max_buses", "peak_slow_score"],
    ascending=[True, False, False, False],
)

summary.to_csv(EVENT_SUMMARY_FILE, index=False)

print(f"Saved event summary: {EVENT_SUMMARY_FILE}")

print("\nEvent type counts:")
print(summary["event_type"].value_counts())

print("\nTop road-network candidates:")

road = summary[
    summary["event_type"] == "road_network_congestion_candidate"
].copy()

if road.empty:
    print("No road-network candidates found under current thresholds.")

else:
    cols = [
        "event_id",
        "start_time",
        "end_time",
        "duration_minutes",
        "max_buses",
        "max_lines",
        "max_line_entropy",
        "mean_slow_score",
        "peak_slow_score",
        "mean_dwell_fraction",
        "mean_terminal_fraction",
        "displacement_m",
        "propagation_speed_m_per_min",
    ]

    print(road[cols].head(20).to_string(index=False))