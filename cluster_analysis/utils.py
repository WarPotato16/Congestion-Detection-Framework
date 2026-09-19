import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# In kilometers
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = ((np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def read_point_json_as_gdf(path):
    try:
        gdf = gpd.read_file(path)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        return gdf

    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        rows = []

        if isinstance(data, dict):
            for stop_id, values in data.items():
                if not isinstance(values, dict):
                    continue

                lat = values.get("lat")
                lng = values.get("lng")

                if lat is None or lng is None:
                    continue

                row = values.copy()
                row["stop_id"] = str(stop_id)
                row["geometry"] = Point(float(lng), float(lat))
                rows.append(row)
            
        elif isinstance(data, list):
            for i, values in enumerate(data):
                if not isinstance(values, dict):
                    continue
                    
                lat = values.get("lat")
                lng = values.get("lng")

                if lat is None or lng is None:
                    continue

                row = values.copy()
                row["stop_id"] = str(i)
                row["geometry"] = Point(float(lng), float(lat))
                rows.append(row)

        if not rows:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    
def load_focus_bairro(ibge_bairros_path, focus_bairro):
    bairros = gpd.read_file(ibge_bairros_path)
    niteroi_bairros = (bairros[bairros["NM_MUN"].astype(str).str.upper() == "NITERÓI"].copy().to_crs("EPSG:4326"))
    focus = niteroi_bairros[niteroi_bairros["NM_BAIRRO"].astype(str).str.upper() == focus_bairro.upper()].copy()
    if focus.empty:
        raise ValueError(f"Bairro '{focus_bairro}' not found.")

    return focus

def make_edge_uid(u, v, key):
    return f"{int(u)}|{int(v)}|{int(key)}"

def split_edge_uid(edge_uid):
    u, v, key = str(edge_uid).split("|")
    return int(u), int(v), int(key)

def normalized_entropy(values):
    values = pd.Series(values).dropna().astype(str)
    if values.empty:
        return 0.0

    counts = values.value_counts()
    probs = counts / counts.sum()
    if len(probs) <= 1:
        return 0.0

    entropy = -(probs * np.log(probs)).sum()
    max_entropy = np.log(len(probs))
    if max_entropy <= 0:
        return 0.0

    return float(entropy / max_entropy)

def largest_category_share(values):
    values = pd.Series(values).dropna().astype(str)
    if values.empty:
        return 0.0

    counts = values.value_counts()
    return float(counts.max() / counts.sum())

def safe_to_parquet(df, path):
    try:
        df.to_parquet(path, index=False)
        print(f"Saved: {path}")

    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        print(f"Could not save parquet because: {exc}")
        print(f"Saving CSV fallback instead: {csv_path}")
        df.to_csv(csv_path, index=False)

def safe_read_table(path):
    if path.exists():
        return pd.read_parquet(path)

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Could not find {path} or {csv_path}")

def count_true(df, col, fallback=None):
    if col in df.columns:
        return int(df[col].fillna(False).astype(bool).sum())

    if fallback is not None:
        return int(fallback.fillna(False).astype(bool).sum())

    return 0

def safe_nunique(df, col):
    if col not in df.columns:
        return 0

    return int(df[col].nunique())

def filter_gdf_to_bairro(gdf, focus_bairro):
    if gdf.empty:
        return gdf

    focus = focus_bairro.to_crs(gdf.crs)
    return gpd.clip(gdf, focus)

def get_row_value(row, col, default=0):
    if col in row.index and pd.notna(row[col]):
        return row[col]

    return default