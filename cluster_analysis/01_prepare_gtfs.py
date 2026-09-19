import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from config import (
    GTFS_FEED_DIRS,
    METRIC_CRS,
    PREPARED_GTFS_STOPS,
    PREPARED_GTFS_TERMINALS,
    PREPARED_GTFS_SHAPES,
    PREPARED_GTFS_ROUTES,
    PREPARED_GTFS_TRIPS,
    PREPARED_GTFS_ROUTE_SHAPES,
)

def load_gtfs_file(filename, required=True):
    if not GTFS_FEED_DIRS:
        raise ValueError(
            "No GTFS feed folders found. Check GTFS_ROOT / GTFS_FEED_DIRS in config.py."
        )

    tables = []

    for feed_dir in GTFS_FEED_DIRS:
        path = feed_dir / filename

        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required GTFS file: {path}")

            print(f"Optional GTFS file not found: {path}")
            continue

        df = pd.read_csv(path, dtype=str)
        df["feed_id"] = feed_dir.name
        tables.append(df)

    if not tables:
        return None

    return pd.concat(tables, ignore_index=True)

def prefix_id_column(df, col):
    if col not in df.columns:
        return df

    original_col = f"{col}_original"

    if original_col not in df.columns:
        df[original_col] = df[col]

    mask = df[col].notna() & (df[col].astype(str).str.strip() != "")

    df.loc[mask, col] = (
        df.loc[mask, "feed_id"].astype(str)
        + "::"
        + df.loc[mask, original_col].astype(str)
    )

    return df

def prepare_stops():
    print("Preparing GTFS stops from multiple feeds...")

    stops = load_gtfs_file("stops.txt", required=True)

    needed = ["stop_id", "stop_lat", "stop_lon"]
    missing = [col for col in needed if col not in stops.columns]

    if missing:
        raise ValueError(f"stops.txt missing required columns: {missing}")

    stops = prefix_id_column(stops, "stop_id")

    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")

    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )

    stops_gdf = stops_gdf.to_crs(METRIC_CRS)
    stops_gdf.to_file(PREPARED_GTFS_STOPS, driver="GPKG")

    print(f"GTFS stops saved: {PREPARED_GTFS_STOPS}")
    print(f"GTFS stops: {len(stops_gdf):,}")
    print("Stops by feed:")
    print(stops_gdf["feed_id"].value_counts())

    return stops_gdf

def prepare_routes_and_trips():
    print("Preparing GTFS routes/trips from multiple feeds...")

    routes = load_gtfs_file("routes.txt", required=True)
    trips = load_gtfs_file("trips.txt", required=True)

    if "route_id" not in routes.columns:
        raise ValueError("routes.txt missing required column: route_id")

    if "route_id" not in trips.columns:
        raise ValueError("trips.txt missing required column: route_id")

    if "trip_id" not in trips.columns:
        raise ValueError("trips.txt missing required column: trip_id")

    routes = prefix_id_column(routes, "route_id")

    trips = prefix_id_column(trips, "trip_id")
    trips = prefix_id_column(trips, "route_id")

    if "shape_id" in trips.columns:
        trips = prefix_id_column(trips, "shape_id")

    routes.to_csv(PREPARED_GTFS_ROUTES, index=False)
    trips.to_csv(PREPARED_GTFS_TRIPS, index=False)

    print(f"GTFS routes saved: {PREPARED_GTFS_ROUTES}")
    print(f"GTFS trips saved: {PREPARED_GTFS_TRIPS}")

    print("Routes by feed:")
    print(routes["feed_id"].value_counts())

    print("Trips by feed:")
    print(trips["feed_id"].value_counts())

    return routes, trips

def prepare_terminals(stops_gdf):
    print("Preparing GTFS terminal stops from stop_times.txt...")

    stop_times = load_gtfs_file("stop_times.txt", required=True)

    needed = ["trip_id", "stop_id", "stop_sequence"]
    missing = [col for col in needed if col not in stop_times.columns]

    if missing:
        raise ValueError(f"stop_times.txt missing required columns: {missing}")

    stop_times = prefix_id_column(stop_times, "trip_id")
    stop_times = prefix_id_column(stop_times, "stop_id")

    stop_times["stop_sequence_num"] = pd.to_numeric(
        stop_times["stop_sequence"],
        errors="coerce",
    )

    stop_times = stop_times.dropna(subset=["stop_sequence_num"]).copy()
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence_num"])

    terminal_pairs = (
        stop_times.groupby("trip_id")
        .agg(
            first_stop_id=("stop_id", "first"),
            last_stop_id=("stop_id", "last"),
        )
        .reset_index()
    )

    terminal_stop_ids = set(terminal_pairs["first_stop_id"].astype(str))
    terminal_stop_ids.update(terminal_pairs["last_stop_id"].astype(str))

    terminals_gdf = stops_gdf[
        stops_gdf["stop_id"].astype(str).isin(terminal_stop_ids)
    ].copy()

    terminals_gdf.to_file(PREPARED_GTFS_TERMINALS, driver="GPKG")

    print(f"GTFS terminals saved: {PREPARED_GTFS_TERMINALS}")
    print(f"GTFS terminal stops: {len(terminals_gdf):,}")

    return terminals_gdf

def prepare_shapes(trips):
    print("Preparing GTFS shapes from multiple feeds...")

    shapes = load_gtfs_file("shapes.txt", required=False)

    if shapes is None:
        print("No shapes.txt found in any selected GTFS feed.")

        empty = gpd.GeoDataFrame(
            {
                "shape_id": [],
                "shape_id_original": [],
                "feed_id": [],
            },
            geometry=[],
            crs=METRIC_CRS,
        )

        empty.to_file(PREPARED_GTFS_SHAPES, driver="GPKG")

        pd.DataFrame(
            columns=["route_id", "direction_id", "shape_id", "trip_count"]
        ).to_csv(PREPARED_GTFS_ROUTE_SHAPES, index=False)

        return empty

    needed = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    missing = [col for col in needed if col not in shapes.columns]

    if missing:
        raise ValueError(f"shapes.txt missing required columns: {missing}")

    shapes = prefix_id_column(shapes, "shape_id")

    shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    shapes["shape_pt_sequence_num"] = pd.to_numeric(
        shapes["shape_pt_sequence"],
        errors="coerce",
    )

    shapes = shapes.dropna(
        subset=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence_num"]
    ).copy()

    shape_rows = []

    for shape_id, group in shapes.groupby("shape_id"):
        group = group.sort_values("shape_pt_sequence_num")

        coords = list(zip(group["shape_pt_lon"], group["shape_pt_lat"]))

        if len(coords) < 2:
            continue

        shape_rows.append(
            {
                "shape_id": str(shape_id),
                "shape_id_original": str(group["shape_id_original"].iloc[0]),
                "feed_id": str(group["feed_id"].iloc[0]),
                "geometry": LineString(coords),
            }
        )

    shapes_gdf = gpd.GeoDataFrame(shape_rows, geometry="geometry", crs="EPSG:4326")
    shapes_gdf = shapes_gdf.to_crs(METRIC_CRS)

    shapes_gdf.to_file(PREPARED_GTFS_SHAPES, driver="GPKG")

    print(f"GTFS shapes saved: {PREPARED_GTFS_SHAPES}")
    print(f"GTFS shapes: {len(shapes_gdf):,}")

    if "shape_id" not in trips.columns:
        print("trips.txt has no shape_id column. Skipping route-shape lookup.")

        pd.DataFrame(
            columns=["route_id", "direction_id", "shape_id", "trip_count"]
        ).to_csv(PREPARED_GTFS_ROUTE_SHAPES, index=False)

        return shapes_gdf

    if "direction_id" not in trips.columns:
        trips["direction_id"] = "unknown"

    usable_trips = trips.dropna(subset=["route_id", "shape_id"]).copy()
    usable_trips = usable_trips[
        usable_trips["shape_id"].astype(str).str.strip() != ""
    ].copy()

    lookup = (
        usable_trips.groupby(["route_id", "direction_id", "shape_id"])
        .size()
        .reset_index(name="trip_count")
        .sort_values(
            ["route_id", "direction_id", "trip_count"],
            ascending=[True, True, False],
        )
    )

    dominant_lookup = (
        lookup.groupby(["route_id", "direction_id"])
        .head(1)
        .reset_index(drop=True)
    )

    dominant_lookup.to_csv(PREPARED_GTFS_ROUTE_SHAPES, index=False)

    print(f"Route-shape lookup saved: {PREPARED_GTFS_ROUTE_SHAPES}")

    return shapes_gdf

# Main
print("Selected GTFS feeds:")

for feed_dir in GTFS_FEED_DIRS:
    print(f"  - {feed_dir}")

stops_gdf = prepare_stops()
routes, trips = prepare_routes_and_trips()

prepare_terminals(stops_gdf)
prepare_shapes(trips)

print("Done preparing combined GTFS.")
