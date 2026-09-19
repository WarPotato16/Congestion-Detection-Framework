import numpy as np
import pandas as pd
from config import (
    ENRICHED_POINTS_FILE,
    SLOW_SCORE_POINTS_FILE,
    FRAME_FREQ,
    ROLLING_SPEED_WINDOW,
    ROLLING_SPEED_METHOD,
    FREE_FLOW_QUANTILE,
    STOPPED_THRESHOLD_KMH,
    MAX_SPEED_KMH,
    MIN_TIME_DIFF_SECONDS,
    MAX_TIME_DIFF_SECONDS,
    CLUSTER_SLOW_SCORE_THRESHOLD,
)
from utils import haversine, safe_read_table, safe_to_parquet

# Main
print("Loading enriched snapped points...")

points = safe_read_table(ENRICHED_POINTS_FILE)

print(f"Rows loaded: {len(points):,}")

points["timestamp"] = pd.to_datetime(points["timestamp"], errors="coerce")

points = points.dropna(subset=["timestamp", "lat", "lng", "id", "tripId"]).copy()
points = points.sort_values(["id", "tripId", "timestamp"]).copy()

group_cols = ["id", "tripId"]

print("Computing instantaneous speed...")

points["prev_timestamp"] = points.groupby(group_cols)["timestamp"].shift(1)
points["prev_lat"] = points.groupby(group_cols)["lat"].shift(1)
points["prev_lng"] = points.groupby(group_cols)["lng"].shift(1)

points["time_diff_seconds"] = (
    points["timestamp"] - points["prev_timestamp"]
).dt.total_seconds()

points["distance_km"] = haversine(
    points["prev_lat"],
    points["prev_lng"],
    points["lat"],
    points["lng"],
)

points["speed_kmh"] = points["distance_km"] / (
    points["time_diff_seconds"] / 3600
)

valid_speed = (
    points["speed_kmh"].notna()
    & points["time_diff_seconds"].between(
        MIN_TIME_DIFF_SECONDS,
        MAX_TIME_DIFF_SECONDS,
    )
    & (points["speed_kmh"] >= 0)
    & (points["speed_kmh"] <= MAX_SPEED_KMH)
)

points.loc[~valid_speed, "speed_kmh"] = np.nan

print("Computing rolling speed...")

if ROLLING_SPEED_METHOD == "median":
    points["rolling_speed_kmh"] = points.groupby(group_cols)["speed_kmh"].transform(
        lambda s: s.rolling(
            window=ROLLING_SPEED_WINDOW,
            min_periods=1,
        ).median()
    )

else:
    points["rolling_speed_kmh"] = points.groupby(group_cols)["speed_kmh"].transform(
        lambda s: s.rolling(
            window=ROLLING_SPEED_WINDOW,
            min_periods=1,
        ).mean()
    )

points["rolling_speed_kmh"] = points.groupby(group_cols)[
    "rolling_speed_kmh"
].transform(lambda s: s.bfill().ffill())

valid_rolling_speeds = points.loc[
    points["rolling_speed_kmh"].notna()
    & points["rolling_speed_kmh"].between(0, MAX_SPEED_KMH),
    "rolling_speed_kmh",
]

moving_rolling_speeds = valid_rolling_speeds[
    valid_rolling_speeds > STOPPED_THRESHOLD_KMH
]

if moving_rolling_speeds.empty:
    raise ValueError("No valid moving rolling speeds found.")

free_flow_speed = moving_rolling_speeds.quantile(FREE_FLOW_QUANTILE)

if free_flow_speed <= 0:
    raise ValueError("Free-flow speed is zero.")

print(f"Estimated rolling free-flow speed: {free_flow_speed:.2f} km/h")

points["slow_score"] = 1 - (points["rolling_speed_kmh"] / free_flow_speed)
points["slow_score"] = points["slow_score"].clip(lower=0, upper=1)
points["slow_score"] = points["slow_score"].fillna(0)

points["time_bin"] = points["timestamp"].dt.floor(FRAME_FREQ)

if "is_near_stop" not in points.columns:
    points["is_near_stop"] = False

if "is_near_terminal" not in points.columns:
    points["is_near_terminal"] = False

if "distance_to_stop_m" not in points.columns:
    points["distance_to_stop_m"] = np.nan

if "distance_to_terminal_m" not in points.columns:
    points["distance_to_terminal_m"] = np.nan

points["raw_slow_candidate"] = (points["slow_score"] >= CLUSTER_SLOW_SCORE_THRESHOLD)
points["stop_dwell_candidate"] = (points["raw_slow_candidate"] & (points["rolling_speed_kmh"] <= STOPPED_THRESHOLD_KMH) & points["is_near_stop"].fillna(False) & ~points["is_near_terminal"].fillna(False))
points["terminal_dwell_candidate"] = (points["raw_slow_candidate"] & (points["rolling_speed_kmh"] <= STOPPED_THRESHOLD_KMH) & points["is_near_terminal"].fillna(False))
points["stop_or_terminal_dwell_candidate"] = (points["stop_dwell_candidate"] | points["terminal_dwell_candidate"])
points["is_dwell_candidate"] = points["stop_or_terminal_dwell_candidate"]

# Remove only points that look like actual terminal dwell/layover
points["road_slow_candidate"] = (points["raw_slow_candidate"] & ~points["stop_or_terminal_dwell_candidate"])

print("Instantaneous speed summary:")
print(points["speed_kmh"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

print("Rolling speed summary:")
print(
    points["rolling_speed_kmh"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
    )
)

print("Slow-score summary:")
print(points["slow_score"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

print("Candidate counts:")

candidate_cols = [
    "raw_slow_candidate",
    "stop_dwell_candidate",
    "terminal_dwell_candidate",
    "road_slow_candidate",
    "is_near_stop",
    "is_near_terminal"]

existing_candidate_cols = [
    col for col in candidate_cols
    if col in points.columns
]

print(points[existing_candidate_cols].sum())

safe_to_parquet(points, SLOW_SCORE_POINTS_FILE)
