"""Reading and writing the compact multi-layer AOI stack and output rasters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

NODATA_I16 = -32768

# name -> scale factor applied when storing as int16 (value_stored = value * scale)
LAYER_SCALES = {
    # Sentinel-2 summer median surface reflectance (0..1)
    "B02": 10000, "B03": 10000, "B04": 10000, "B05": 10000, "B06": 10000,
    "B07": 10000, "B8A": 10000, "B11": 10000, "B12": 10000,
    # Sentinel-2 winter median (only the bands needed for seasonal contrast)
    "W_B04": 10000, "W_B8A": 10000, "W_B11": 10000,
    "VEG_PERSIST": 10000,   # fraction of clear summer dates with NDVI > 0.25
    "S2_NOBS": 1,           # number of clear summer observations
    "LST_SUMMER": 100,      # Landsat land surface temperature, deg C
    "DEM": 100,             # Copernicus GLO-30 elevation, m
    "WORLDCOVER": 1,        # ESA WorldCover class code
    # output layers (results/fingerprint_and_scores.tif)
    "ECE_dSm": 100, "LST_C": 100, "WATERLOG": 10000, "BUILT_DENSITY": 10000,
    "WATER_NEED_mm": 10, "SAVING_mm": 10, "STABILITY": 10000,
}


def write_stack(path, layers: dict, transform: Affine, crs: str, meta: dict | None = None):
    """Write named float layers as a compressed int16 GeoTIFF with scales in tags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(layers)
    h, w = next(iter(layers.values())).shape
    profile = dict(driver="GTiff", height=h, width=w, count=len(names), dtype="int16",
                   crs=crs, transform=transform, nodata=NODATA_I16, compress="deflate",
                   predictor=2, zlevel=9, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        for i, n in enumerate(names, start=1):
            scale = LAYER_SCALES.get(n, 1)
            arr = np.asarray(layers[n], dtype="float64") * scale
            out = np.where(np.isfinite(arr), np.clip(np.round(arr), -32767, 32767), NODATA_I16)
            dst.write(out.astype("int16"), i)
            dst.set_band_description(i, n)
            dst.update_tags(i, scale=str(scale))
        dst.update_tags(layer_names=json.dumps(names))
    if meta is not None:
        meta_path = path.with_name(path.stem + "_metadata.json")
        meta_path.write_text(json.dumps(meta, indent=2, default=str))


def read_stack(path):
    """Return (layers: dict[name -> float32 array with NaN], profile)."""
    with rasterio.open(path) as src:
        names = json.loads(src.tags().get("layer_names", "[]")) or list(src.descriptions)
        layers = {}
        for i, n in enumerate(names, start=1):
            arr = src.read(i).astype("float32")
            scale = float(src.tags(i).get("scale", LAYER_SCALES.get(n, 1)))
            arr[arr == NODATA_I16] = np.nan
            layers[n] = arr / scale
        profile = src.profile.copy()
    return layers, profile


def write_float_raster(path, arrays: dict, profile: dict):
    """Write one or more float32 layers (e.g. suitability maps) as a GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(arrays)
    p = profile.copy()
    p.update(count=len(names), dtype="float32", nodata=np.nan, compress="deflate",
             predictor=3, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **p) as dst:
        for i, n in enumerate(names, start=1):
            dst.write(np.asarray(arrays[n], dtype="float32"), i)
            dst.set_band_description(i, n)


def write_class_raster(path, arr, profile: dict, labels: dict | None = None):
    """Write a uint8 categorical raster (255 = nodata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(count=1, dtype="uint8", nodata=255, compress="deflate", tiled=True,
             blockxsize=256, blockysize=256, predictor=1)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.asarray(arr, dtype="uint8"), 1)
        if labels:
            dst.update_tags(labels=json.dumps({int(k): v for k, v in labels.items()}))


def pixel_to_lonlat(rows, cols, profile):
    """Convert pixel indices to WGS84 lon/lat."""
    from pyproj import Transformer
    t = profile["transform"]
    xs, ys = rasterio.transform.xy(t, rows, cols)
    tr = Transformer.from_crs(profile["crs"], "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.asarray(xs), np.asarray(ys))
    return np.asarray(lon), np.asarray(lat)
