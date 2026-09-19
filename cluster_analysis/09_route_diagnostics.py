import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    SLOW_SCORE_POINTS_FILE,
    CLUSTER_POINTS_FILE,
    EVENT_SUMMARY_FILE,
    CLUSTER_SLOW_SCORE_THRESHOLD,
)

from utils import safe_read_table


GTFS_ROUTES_FILE = Path("cluster_analysis/processed_data/gtfs/gtfs_routes.csv")


def get_route_col(df):
    if "gtfs_route_id" in df.columns and df["gtfs_route_id"].notna().any():
        return "gtfs_route_id"
    if "lineId" in df.columns:
        return "lineId"
    if "lineName" in df.columns:
        return "lineName"
    raise ValueError("No route/line column found.")


def add_route_short_names(diagnostics, route_col, routes_file=GTFS_ROUTES_FILE):
    """
    Add route_short_name labels using the prepared GTFS routes lookup.

    The diagnostics are still grouped by the full GTFS route_id, which avoids
    accidentally merging routes from different feeds that might share a short
    name. The short name is used for display in tables and plots.
    """
    diagnostics = diagnostics.copy()

    if not routes_file.exists():
        print(f"Warning: GTFS routes file not found: {routes_file}")
        print("Using compact route_id labels instead of route_short_name.")
        diagnostics["route_short_name"] = diagnostics[route_col].apply(shorten_route_id)
        return diagnostics

    routes = pd.read_csv(routes_file)

    required_cols = {"route_id", "route_short_name"}
    if not required_cols.issubset(routes.columns):
        print(f"Warning: {routes_file} is missing route_id or route_short_name.")
        print(f"Available columns: {list(routes.columns)}")
        diagnostics["route_short_name"] = diagnostics[route_col].apply(shorten_route_id)
        return diagnostics

    route_lookup = routes[["route_id", "route_short_name"]].copy()
    route_lookup["route_id"] = route_lookup["route_id"].astype(str).str.strip()
    route_lookup["route_short_name"] = route_lookup["route_short_name"].astype(str).str.strip()
    route_lookup = route_lookup.drop_duplicates(subset=["route_id"])

    diagnostics[route_col] = diagnostics[route_col].astype(str).str.strip()

    diagnostics = diagnostics.merge(
        route_lookup,
        left_on=route_col,
        right_on="route_id",
        how="left",
    )

    diagnostics["route_short_name"] = diagnostics["route_short_name"].fillna(
        diagnostics[route_col].apply(shorten_route_id)
    )

    diagnostics = diagnostics.drop(columns=["route_id"])
    return diagnostics


def shorten_route_id(route_id):
    """Fallback label if route_short_name is unavailable."""
    route_id = str(route_id)
    if "::" in route_id:
        return route_id.split("::", 1)[1]
    return route_id


print("Loading slow-score points...")
points = safe_read_table(SLOW_SCORE_POINTS_FILE)

print("Loading cluster points...")
cluster_points = pd.read_csv(CLUSTER_POINTS_FILE)

print("Loading event summary...")
events = pd.read_csv(EVENT_SUMMARY_FILE)

route_col = get_route_col(points)

points = points[
    points[route_col].notna()
    & (points[route_col].astype(str).str.strip() != "")
].copy()

cluster_points = cluster_points[
    cluster_points[route_col].notna()
    & (cluster_points[route_col].astype(str).str.strip() != "")
].copy()

points[route_col] = points[route_col].astype(str).str.strip()
cluster_points[route_col] = cluster_points[route_col].astype(str).str.strip()

# Baseline: all slow points by route
all_slow = points[
    points["slow_score"] >= CLUSTER_SLOW_SCORE_THRESHOLD
].copy()

baseline = (
    all_slow.groupby(route_col)
    .agg(
        all_slow_points=("slow_score", "count"),
        mean_slow_score_all=("slow_score", "mean"),
    )
    .reset_index()
)

baseline["route_share_all_slow"] = (
    baseline["all_slow_points"] / baseline["all_slow_points"].sum()
)

# Keep only road-network congestion events
road_events = events[
    events["event_type"] == "road_network_congestion_candidate"
][["event_id"]].copy()

if road_events.empty:
    print("No road-network congestion events found. Route diagnostics will contain zero road-cluster exposure.")

road_cluster_points = cluster_points.merge(
    road_events,
    on="event_id",
    how="inner",
)

road_by_route = (
    road_cluster_points.groupby(route_col)
    .agg(
        road_cluster_points=("slow_score", "count"),
        road_events_entered=("event_id", "nunique"),
        mean_slow_score_in_road_clusters=("slow_score", "mean"),
        mean_rolling_speed_in_road_clusters=("rolling_speed_kmh", "mean"),
    )
    .reset_index()
)

road_by_route["route_share_road_clusters"] = (
    road_by_route["road_cluster_points"]
    / road_by_route["road_cluster_points"].sum()
)

total_road_points = road_by_route["road_cluster_points"].sum()

if total_road_points > 0:
    road_by_route["route_share_road_clusters"] = (
        road_by_route["road_cluster_points"] / total_road_points
    )
else:
    road_by_route["route_share_road_clusters"] = 0

diagnostics = baseline.merge(
    road_by_route,
    on=route_col,
    how="left",
)

fill_cols = [
    "road_cluster_points",
    "road_events_entered",
    "route_share_road_clusters",
]

for col in fill_cols:
    diagnostics[col] = diagnostics[col].fillna(0)

diagnostics["overrepresentation_ratio"] = (
    diagnostics["route_share_road_clusters"]
    / diagnostics["route_share_all_slow"]
)

diagnostics["share_of_route_slow_points_in_road_clusters"] = (
    diagnostics["road_cluster_points"]
    / diagnostics["all_slow_points"]
)

diagnostics = diagnostics.sort_values(
    [
        "overrepresentation_ratio",
        "road_events_entered",
        "road_cluster_points",
    ],
    ascending=[False, False, False],
)

# Add readable route_short_name labels for output tables and figures.
diagnostics = add_route_short_names(diagnostics, route_col)

output_file = "cluster_analysis/processed_data/clusters/route_congestion_diagnostics.csv"
diagnostics.to_csv(output_file, index=False)

print(f"Saved route diagnostics: {output_file}")

show_cols = [
    "route_short_name",
    route_col,
    "all_slow_points",
    "road_cluster_points",
    "road_events_entered",
    "share_of_route_slow_points_in_road_clusters",
    "route_share_all_slow",
    "route_share_road_clusters",
    "overrepresentation_ratio",
    "mean_slow_score_in_road_clusters",
]

print(diagnostics[show_cols].head(25).to_string(index=False))

# ------------------------------------------------------------
# Plot: observed vs. expected route exposure in road clusters
# ------------------------------------------------------------

def plot_route_overrepresentation(
    diagnostics,
    route_col,
    output_dir="cluster_analysis/processed_data/clusters",
    top_n=15,
    min_road_cluster_points=1,
):
    """
    Make a dumbbell plot comparing each route's expected and observed share.

    Expected share = route_share_all_slow
    Observed share = route_share_road_clusters

    If observed > expected, the route is overrepresented in road-network
    congestion clusters relative to its overall slow-point activity.
    """
    plot_df = diagnostics.copy()
    plot_df = plot_df[plot_df["road_cluster_points"] >= min_road_cluster_points].copy()

    if plot_df.empty:
        print("No routes with road-cluster points to plot.")
        return

    plot_df = plot_df.sort_values(
        ["overrepresentation_ratio", "road_events_entered", "road_cluster_points"],
        ascending=[False, False, False],
    ).head(top_n)

    # Reverse order so the highest-ranked route appears at the top.
    plot_df = plot_df.iloc[::-1].copy()

    if "route_short_name" in plot_df.columns:
        plot_df["route_label"] = plot_df["route_short_name"].astype(str)
    else:
        plot_df["route_label"] = plot_df[route_col].apply(shorten_route_id)
    plot_df["expected_pct"] = 100 * plot_df["route_share_all_slow"]
    plot_df["observed_pct"] = 100 * plot_df["route_share_road_clusters"]

    fig_height = max(5.0, 0.42 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    y = range(len(plot_df))

    # Connecting lines show the shift from expected to observed share.
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.plot(
            [row["expected_pct"], row["observed_pct"]],
            [yi, yi],
            linewidth=1.5,
            alpha=0.75,
        )

    ax.scatter(
        plot_df["expected_pct"],
        list(y),
        marker="o",
        s=45,
        label="Expected share: all slow points",
        zorder=3,
    )
    ax.scatter(
        plot_df["observed_pct"],
        list(y),
        marker="s",
        s=45,
        label="Observed share: road-cluster points",
        zorder=3,
    )

    # Add compact overrepresentation-ratio labels at the observed point.
    x_span = max(plot_df["expected_pct"].max(), plot_df["observed_pct"].max())
    label_pad = max(0.08, 0.015 * x_span)

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.text(
            row["observed_pct"] + label_pad,
            yi,
            f"{row['overrepresentation_ratio']:.2f}x",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["route_label"])
    ax.set_xlabel("Share of Points (%)")
    ax.set_ylabel("Route Name")
    ax.set_title("Route Over-Representation in Road-Network Congestion Clusters")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.0,
        -0.16,
        "Expected share is the route's share of all slow observations; observed share is its share of road-network cluster observations. Labels show observed/expected ratio.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_file = output_dir / "route_overrepresentation_expected_vs_observed_short_names.png"
    pdf_file = output_dir / "route_overrepresentation_expected_vs_observed_short_names.pdf"

    fig.tight_layout()
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved route overrepresentation plot: {png_file}")
    print(f"Saved route overrepresentation plot: {pdf_file}")


plot_route_overrepresentation(
    diagnostics=diagnostics,
    route_col=route_col,
    output_dir="cluster_analysis/processed_data/clusters",
    top_n=15,
    min_road_cluster_points=1,
)
