import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from pathlib import Path
from shapely import wkt
from config import (
    FOCUS_BAIRRO,
    IBGE_BAIRROS,
    METRIC_CRS,
    OSM_EDGES_FILE,
    SLOW_SCORE_POINTS_FILE,
    PREPARED_GTFS_STOPS,
    PREPARED_GTFS_TERMINALS,
    CLUSTER_FRAMES_FILE,
    OUTPUT_GIF,
    ANIMATION_START_TIME,
    ANIMATION_END_TIME,
    FRAME_FREQ,
    EVENT_SUMMARY_FILE,
    CMAP_NAME,
    POINT_SIZE,
    POINT_ALPHA,
    SHOW_REGULAR_STOPS,
    SHOW_TERMINALS,
    SHOW_CLUSTER_LABELS,
    REGULAR_STOP_SIZE,
    REGULAR_STOP_COLOR,
    REGULAR_STOP_ALPHA,
    TERMINAL_STOP_SIZE,
    TERMINAL_STOP_COLOR,
    TERMINAL_STOP_ALPHA,
)
from utils import load_focus_bairro, safe_read_table, filter_gdf_to_bairro, get_row_value

MAX_CLUSTER_LABELS = 4
LABEL_MIN_BUSES = 2

MAP_BACKGROUND = "#eeeeee"
ROAD_COLOR = "#cfcfcf"
BOUNDARY_COLOR = "black"
CLUSTER_OUTLINE_COLOR = "black"

SHOW_SIDE_PANEL = True

CLUSTER_TYPE_COLUMNS = [
    "cluster_type",
    "event_type",
    "event_class",
    "classification",
    "class_label"
]
CLUSTER_COLORS = {
    "road_network_congestion_candidate": "#1f77b4",
    "road_network_candidate": "#1f77b4",
    "road_slow_candidate": "#1f77b4",

    "stop_or_terminal_dwell": "#d62728",
    "stop_or_terminal_dwell_candidate": "#d62728",
    "dwell_or_stop_service": "#d62728",
    "terminal_or_layover": "#d62728",
    "stop_dwell_candidate": "#d62728",
    "terminal_dwell_candidate": "#d62728",

    "mixed_or_uncertain": "#7f7f7f",
    "unclassified": "black",
}
CLUSTER_LABELS = {
    "road_network_congestion_candidate": "Road-Network Congestion",
    "road_network_candidate": "Road-Network Congestion",
    "road_slow_candidate": "Road Slow Cluster",

    "stop_or_terminal_dwell": "Stop/Terminal Dwell",
    "stop_or_terminal_dwell_candidate": "Stop/Terminal Dwell",
    "dwell_or_stop_service": "Stop/Terminal Dwell",
    "terminal_or_layover": "Stop/Terminal Dwell",
    "stop_dwell_candidate": "Stop/Terminal Dwell",
    "terminal_dwell_candidate": "Stop/Terminal Dwell",
    
    "mixed_or_uncertain": "Mixed/Uncertain",
    "unclassified": "Unclassified",
}
DEFAULT_CLUSTER_COLOR = "black"
SHOW_CLUSTER_TYPE_LEGEND = False

LEGEND_JPG = Path(OUTPUT_GIF).with_name(Path(OUTPUT_GIF).stem + "_legend.jpg")

DRAW_LEGEND_ON_GIF = False
DRAW_COLORBAR_ON_GIF = False

def normalize_cluster_type(value):
    if pd.isna(value):
        return "unclassified"

    s = str(value).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")

    aliases = {
        "road_congestion": "road_network_congestion_candidate",
        "road_network_congestion": "road_network_congestion_candidate",
        "road_candidate": "road_network_congestion_candidate",
        "road_network_candidate": "road_network_congestion_candidate",
        "road_slow": "road_network_congestion_candidate",
        "road_slow_candidate": "road_network_congestion_candidate",

        "stop_or_terminal_dwell_candidate": "stop_or_terminal_dwell",
        "stop_terminal_dwell": "stop_or_terminal_dwell",
        "stop_dwell": "stop_or_terminal_dwell",
        "stop_dwell_candidate": "stop_or_terminal_dwell",
        "dwell_or_stop_service": "stop_or_terminal_dwell",
        "terminal": "stop_or_terminal_dwell",
        "terminal_layover": "stop_or_terminal_dwell",
        "terminal_or_layover": "stop_or_terminal_dwell",
        "terminal_dwell": "stop_or_terminal_dwell",
        "terminal_dwell_candidate": "stop_or_terminal_dwell",

        "mixed": "mixed_or_uncertain",
        "uncertain": "mixed_or_uncertain",
        "mixed_uncertain": "mixed_or_uncertain",
    }

    return aliases.get(s, s)

def get_cluster_type(row):
    # First try explicit classification columns.
    for col in CLUSTER_TYPE_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            return normalize_cluster_type(row[col])

    # Fallback: try boolean/count columns if they exist.
    fallback_rules = [
        ("road_network_congestion_candidate", "road_network_congestion_candidate"),
        ("road_slow_candidate", "road_network_congestion_candidate"),
        ("terminal_or_layover", "terminal_or_layover"),
        ("terminal_dwell_candidate", "terminal_or_layover"),
        ("stop_dwell_candidate", "stop_dwell_candidate"),
        ("route_operation_candidate", "route_operation_candidate"),
        ("mixed_or_uncertain", "mixed_or_uncertain"),
    ]

    for col, label in fallback_rules:
        if col in row.index and pd.notna(row[col]):
            value = row[col]

            if isinstance(value, bool) and value:
                return label

            if isinstance(value, (int, float)) and value > 0:
                return label

            if isinstance(value, str) and value.strip().lower() in ["true", "yes", "1"]:
                return label

    return "unclassified"

def get_cluster_color(row):
    cluster_type = get_cluster_type(row)
    return CLUSTER_COLORS.get(cluster_type, DEFAULT_CLUSTER_COLOR)

# Places labels besides clusters
def get_label_offset(row, centroid, xmin, xmax, ymin, ymax):
    event_id = int(get_row_value(row, "event_id", 0))
    mid_x = (xmin + xmax) / 2

    horizontal_side = 1 if centroid.x < mid_x else -1

    vertical_offsets = [-34, 0, 34]
    vertical_offset = vertical_offsets[event_id % len(vertical_offsets)]

    x_offset = horizontal_side * 72

    return x_offset, vertical_offset

def merge_event_types_into_clusters(clusters):
    if clusters.empty:
        return clusters

    if not EVENT_SUMMARY_FILE.exists():
        print(f"Warning: event summary file not found: {EVENT_SUMMARY_FILE}")
        print("Cluster rings may remain black because event_type was not merged.")
        return clusters

    event_summary = pd.read_csv(EVENT_SUMMARY_FILE)

    if "event_id" not in clusters.columns:
        print("Warning: clusters file has no event_id column.")
        return clusters

    if "event_id" not in event_summary.columns:
        print("Warning: event summary file has no event_id column.")
        return clusters

    type_cols = [
        col for col in CLUSTER_TYPE_COLUMNS
        if col in event_summary.columns
    ]

    if not type_cols:
        print("Warning: event summary file has no event_type/classification column.")
        print(f"Available columns: {list(event_summary.columns)}")
        return clusters

    # Prefer event_type if it exists.
    preferred_type_col = "event_type" if "event_type" in type_cols else type_cols[0]

    event_summary_small = event_summary[["event_id", preferred_type_col]].copy()
    event_summary_small = event_summary_small.rename(
        columns={preferred_type_col: "event_type_from_summary"}
    )

    clusters = clusters.copy()
    clusters["event_id"] = pd.to_numeric(clusters["event_id"], errors="coerce")
    event_summary_small["event_id"] = pd.to_numeric(
        event_summary_small["event_id"],
        errors="coerce",
    )

    clusters = clusters.merge(
        event_summary_small,
        on="event_id",
        how="left",
    )

    if "event_type" in clusters.columns:
        clusters["event_type"] = clusters["event_type"].fillna(
            clusters["event_type_from_summary"]
        )
    else:
        clusters["event_type"] = clusters["event_type_from_summary"]

    clusters = clusters.drop(columns=["event_type_from_summary"])

    print("Merged event types into cluster frames.")
    print("Cluster event_type counts:")
    print(clusters["event_type"].value_counts(dropna=False))

    return clusters

def draw_hull(ax, row):
    hull = wkt.loads(row["hull_wkt"])

    if hull.is_empty:
        return

    cluster_type = get_cluster_type(row)
    color = get_cluster_color(row)

    linestyle = "--" if cluster_type == "mixed_or_uncertain" else "-"

    if hull.geom_type == "Polygon":
        xh, yh = hull.exterior.xy
        ax.plot(
            xh,
            yh,
            color=color,
            linewidth=2.3,
            alpha=0.95,
            linestyle=linestyle,
            zorder=7,
        )

    elif hull.geom_type == "MultiPolygon":
        for polygon in hull.geoms:
            xh, yh = polygon.exterior.xy
            ax.plot(
                xh,
                yh,
                color=color,
                linewidth=2.3,
                alpha=0.95,
                linestyle=linestyle,
                zorder=7,
            )

def select_labeled_clusters(clusters_df):
    if clusters_df.empty:
        return clusters_df
    
    df = clusters_df.copy()

    if "num_buses" not in df.columns:
        df["num_buses"] = 0
    
    if "num_lines" not in df.columns:
        df["num_lines"] = 0
    
    if "mean_slow_score" not in df.columns:
        df["mean_slow_score"] = 0
    
    df = df[df["num_buses"] >= LABEL_MIN_BUSES].copy()

    if df.empty:
        return df
    
    df = df.sort_values(
        by=["num_buses", "num_lines", "mean_slow_score"],
        ascending=[False, False, False]
    )
    return df.head(MAX_CLUSTER_LABELS)

def draw_side_panel(
        info_ax, 
        t,
        total_points,
        active_buses,
        active_edges,
        raw_slow_points,
        stop_dwell_points,
        terminal_dwell_points,
        road_slow_points,
        cluster_count,
        largest_cluster_size,
        stops_count,
        terminals_count,
):
        
    info_ax.clear()
    info_ax.set_facecolor(MAP_BACKGROUND)
    info_ax.set_axis_off()
    info_ax.text(
        0.0,
        0.98,
        "Frame Summary",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        transform=info_ax.transAxes,
    )
    info_ax.text(
        0.0,
        0.92,
        f"{t:%Y-%m-%d %H:%M}",
        ha="left",
        va="top",
        fontsize=11,
        transform=info_ax.transAxes
    )
    summary_text = (
        f"Active buses: {active_buses}\n"
        f"GPS points: {total_points}\n"
        f"Assigned OSM edges: {active_edges}\n\n"
        f"Raw slow points: {raw_slow_points}\n"
        f"Road-candidate points: {road_slow_points}\n"
        f"Stop-dwell points: {stop_dwell_points}\n"
        f"Terminal-dwell points: {terminal_dwell_points}\n\n"
        f"Slow clusters: {cluster_count}\n"
        f"Largest cluster: {largest_cluster_size} buses"
    )
    info_ax.text(
        0.0,
        0.82,
        summary_text,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
        transform=info_ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=8)
    )
    context_text = (
        "Map context\n"
        f"GTFS stops: {stops_count}\n"
        f"GTFS terminals: {terminals_count}\n\n"
        "Gray Dots = Bus Stops\n"
        "Red Stars = Terminals\n"
        "Labels show largest clusters only"
    )
    info_ax.text(
        0.0,
        0.34,
        context_text,
        ha="left",
        va="top",
        fontsize=9,
        linespacing=1.35,
        transform=info_ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=8)
    )

def build_map_legend_handles(include_cluster_types=True):
    handles = []

    if SHOW_REGULAR_STOPS:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=REGULAR_STOP_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.4,
                alpha=REGULAR_STOP_ALPHA,
                markersize=6,
                label="Bus Stops",
            )
        )

    if SHOW_TERMINALS:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markerfacecolor=TERMINAL_STOP_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.7,
                alpha=TERMINAL_STOP_ALPHA,
                markersize=11,
                label="Terminals",
            )
        )

    if include_cluster_types:
        cluster_legend_order = [
            "road_network_congestion_candidate",
            "stop_or_terminal_dwell",
            "mixed_or_uncertain",
        ]

        for cluster_type in cluster_legend_order:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=CLUSTER_COLORS.get(cluster_type, DEFAULT_CLUSTER_COLOR),
                    linewidth=2.3,
                    linestyle="--" if cluster_type == "mixed_or_uncertain" else "-",
                    label=CLUSTER_LABELS.get(cluster_type, cluster_type),
                )
            )

    return handles

def add_map_legend(ax):
    handles = build_map_legend_handles(
        include_cluster_types=SHOW_CLUSTER_TYPE_LEGEND
    )

    if not handles:
        return

    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        title="Map Legend",
        title_fontsize=11,
    )

def save_legend_jpg(output_file):
    fig = plt.figure(figsize=(6.0, 4.2))
    fig.patch.set_facecolor("white")

    legend_ax = fig.add_axes([0.06, 0.34, 0.88, 0.60])
    legend_ax.set_axis_off()

    handles = build_map_legend_handles(include_cluster_types=True)

    legend_ax.legend(
        handles=handles,
        loc="center left",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=11,
        title="Map Legend",
        title_fontsize=12,
    )

    cbar_ax = fig.add_axes([0.12, 0.16, 0.76, 0.07])

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=CMAP_NAME)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Slow-Speed Score: 0 = Free-Flow, 1 = Stopped/Very Slow",
        fontsize=10,
    )
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

def draw_frame(
    frame_idx,
    ax,
    all_bins,
    frame_data,
    cluster_data,
    edges_plot,
    focus_bairro_metric,
    stops_bairro,
    terminals_bairro,
    xmin,
    ymin,
    xmax,
    ymax,
    pad_x,
    pad_y,
):
    ax.clear()
    ax.set_facecolor(MAP_BACKGROUND)

    t = all_bins[frame_idx]

    edges_plot.plot(
        ax=ax,
        color=ROAD_COLOR,
        linewidth=0.5,
        zorder=1,
    )

    if SHOW_REGULAR_STOPS and not stops_bairro.empty:
        stops_bairro.plot(
            ax=ax,
            markersize=REGULAR_STOP_SIZE,
            color=REGULAR_STOP_COLOR,
            alpha=REGULAR_STOP_ALPHA,
            edgecolor="black",
            linewidth=0.2,
            zorder=2,
        )

    if SHOW_TERMINALS and not terminals_bairro.empty:
        terminals_bairro.plot(
            ax=ax,
            markersize=TERMINAL_STOP_SIZE,
            marker="*",
            color=TERMINAL_STOP_COLOR,
            alpha=TERMINAL_STOP_ALPHA,
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )

    if t in frame_data:
        points_df = frame_data[t]

        ax.scatter(
            points_df["snap_x"],
            points_df["snap_y"],
            s=POINT_SIZE,
            c=points_df["slow_score"],
            cmap=CMAP_NAME,
            vmin=0,
            vmax=1,
            alpha=POINT_ALPHA,
            edgecolors="black",
            linewidths=0.15,
            zorder=5,
        )

    largest_cluster_size = 0

    if t in cluster_data:
        clusters_df = cluster_data[t]
        cluster_count = len(clusters_df)

        # Draw all DBSCAN cluster rings, color-coded by event type.
        for _, row in clusters_df.iterrows():
            draw_hull(ax, row)

            num_buses = int(get_row_value(row, "num_buses", 0))
            largest_cluster_size = max(largest_cluster_size, num_buses)

        # Label only the largest clusters.
        # This block must be outside the ring-drawing loop.
        if SHOW_CLUSTER_LABELS:
            labeled_clusters = select_labeled_clusters(clusters_df)

            for _, row in labeled_clusters.iterrows():
                hull = wkt.loads(row["hull_wkt"])
                centroid = hull.centroid

                event_id = int(get_row_value(row, "event_id", -1))
                num_buses = int(get_row_value(row, "num_buses", 0))
                num_lines = int(get_row_value(row, "num_lines", 0))
                mean_slow = float(get_row_value(row, "mean_slow_score", 0))

                label_text = (
                    f"E{event_id}\n"
                    f"{num_buses} buses\n"
                    f"{num_lines} lines\n"
                    f"{mean_slow:.2f}"
                )

                ring_color = get_cluster_color(row)
                x_offset, y_offset = get_label_offset(row, centroid, xmin, xmax, ymin, ymax)

                txt = ax.annotate(
                    label_text,
                    xy=(centroid.x, centroid.y),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.90,
                        edgecolor="none",
                        pad=3
                    ),
                    arrowprops=dict(
                        arrowstyle="-",
                        color=ring_color,
                        linewidth=1.4,
                        alpha=0.95,
                        shrinkA=4,
                        shrinkB=4
                    ),
                    zorder=9,
                    annotation_clip=False
                )

                txt.set_path_effects([
                    pe.withStroke(linewidth=1.5, foreground="white")
                ])

    focus_bairro_metric.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=1.4, zorder=4)

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Snapped Bus GPS Slow-Speed Clusters in {FOCUS_BAIRRO} — {t}", fontsize=14, pad=14)
    ax.set_axis_off()
    if DRAW_LEGEND_ON_GIF:
        add_map_legend(ax)

# Main
print("Loading animation inputs...")

points = safe_read_table(SLOW_SCORE_POINTS_FILE)
points["time_bin"] = pd.to_datetime(points["time_bin"])
points = points[(points["time_bin"] >= ANIMATION_START_TIME) & (points["time_bin"] <= ANIMATION_END_TIME)].copy()

edges_plot = gpd.read_file(OSM_EDGES_FILE).to_crs(METRIC_CRS)

focus_bairro = load_focus_bairro(IBGE_BAIRROS, FOCUS_BAIRRO)
focus_bairro_metric = focus_bairro.to_crs(METRIC_CRS)

stops = gpd.read_file(PREPARED_GTFS_STOPS).to_crs(METRIC_CRS)
terminals = gpd.read_file(PREPARED_GTFS_TERMINALS).to_crs(METRIC_CRS)

stops_bairro = filter_gdf_to_bairro(stops, focus_bairro_metric)
terminals_bairro = filter_gdf_to_bairro(terminals, focus_bairro_metric)

if CLUSTER_FRAMES_FILE.exists():
    clusters = pd.read_csv(CLUSTER_FRAMES_FILE)
    clusters["time_bin"] = pd.to_datetime(clusters["time_bin"])

    clusters = clusters[
        (clusters["time_bin"] >= ANIMATION_START_TIME)
        & (clusters["time_bin"] <= ANIMATION_END_TIME)
    ].copy()

else:
    clusters = pd.DataFrame()

clusters = merge_event_types_into_clusters(clusters)

all_bins = pd.date_range(
    points["time_bin"].min(),
    points["time_bin"].max(),
    freq=FRAME_FREQ,
)

frame_data = {
    t: df.copy()
    for t, df in points.groupby("time_bin")
}

if not clusters.empty:
    cluster_data = {
        t: df.copy()
        for t, df in clusters.groupby("time_bin")
    }
else:
    cluster_data = {}

xmin, ymin, xmax, ymax = edges_plot.total_bounds

pad_x = (xmax - xmin) * 0.03
pad_y = (ymax - ymin) * 0.03

print(f"Frames: {len(all_bins)}")
print("Building animation...")

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(MAP_BACKGROUND)
ax.set_facecolor(MAP_BACKGROUND)

save_legend_jpg(LEGEND_JPG)
print(f"Saved separate legend JPG: {LEGEND_JPG}")

if DRAW_COLORBAR_ON_GIF:
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=CMAP_NAME)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.05)
    cbar.set_label(
        "Slow-Speed Score\n0 = Free-Flow\n1 = Stopped/Very Slow",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)

ani = animation.FuncAnimation(
    fig,
    draw_frame,
    frames=len(all_bins),
    interval=250,
    repeat=False,
    fargs=(
        ax,
        all_bins,
        frame_data,
        cluster_data,
        edges_plot,
        focus_bairro_metric,
        stops_bairro,
        terminals_bairro,
        xmin,
        ymin,
        xmax,
        ymax,
        pad_x,
        pad_y,
    ),
)

print(f"Saving GIF to {OUTPUT_GIF}...")

ani.save(OUTPUT_GIF, writer="pillow", fps=4)

plt.close(fig)

print("Done.")
print(f"Saved: {OUTPUT_GIF}")