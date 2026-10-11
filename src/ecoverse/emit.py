"""NASA EMIT hyperspectral data over the Bahrain AOI.

EMIT (Earth Surface Mineral Dust Source Investigation, on the ISS) records
285 bands from 381 to 2493 nm at 60 m. Its L2A surface reflectance product is
free (NASA Earthdata login required for download, no usage restrictions).

Two parts:
  * download + extract (online, once): find EMIT scenes covering the AOI,
    download the reflectance + mask files, cut a small window around the AOI,
    compute narrow-band features and simulated Sentinel-2 bands, keep a PCA of
    the full spectra, and save a compact .npz (a few MB) to data/sample_input.
  * analysis (offline, from the committed .npz): pixel table for
    cross-validation, and resampling of every layer onto the 20 m AOI grid.

Credentials: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD as environment
variables. They are never written to disk or into notebook outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .indices import S2_SRF, bad_band_mask, hyperspectral_features, simulate_s2

S2_NAMES = list(S2_SRF)
FEATURES = ["GYPSUM_1750", "CLAY_2200", "CARBONATE_2340", "REP", "CANOPY_WATER_1200"]
EMIT_SHORT_NAME = "EMITL2ARFL"


# ---------------------------------------------------------------------------
# reading EMIT netCDF4 files (they are HDF5 underneath, so h5py is enough)
# ---------------------------------------------------------------------------
def _find(f, name):
    """Return the first dataset in an HDF5 file whose path ends with `name`."""
    hits = []
    f.visititems(lambda p, o: hits.append(p) if (p == name or p.endswith("/" + name)) and hasattr(o, "shape") else None)
    if not hits:
        raise KeyError(f"{name} not found in {f.filename}")
    return f[sorted(hits, key=len)[0]]


def _window(lat, lon, bbox, pad_deg=0.01):
    lo0, la0, lo1, la1 = bbox
    inside = (lon >= lo0 - pad_deg) & (lon <= lo1 + pad_deg) & (lat >= la0 - pad_deg) & (lat <= la1 + pad_deg)
    if not inside.any():
        return None
    r, c = np.nonzero(inside)
    return slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1)


def extract_window(rfl_path, mask_path, bbox, granule_id, out_dir, n_pcs=20,
                   seed=config.RANDOM_SEED, extra_meta=None):
    """Cut the AOI window from an EMIT L2A scene and save a compact subset."""
    import h5py
    from sklearn.decomposition import PCA

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(rfl_path, "r") as f:
        lat = _find(f, "lat")[:]
        lon = _find(f, "lon")[:]
        win = _window(lat, lon, bbox)
        if win is None:
            return None
        rs, cs = win
        wl = _find(f, "wavelengths")[:].astype("float64")
        try:
            good = _find(f, "good_wavelengths")[:].astype(bool)
        except KeyError:
            good = np.ones(wl.size, bool)
        refl = _find(f, "reflectance")[rs, cs, :].astype("float32")      # (h, w, bands)
    lat, lon = lat[rs, cs], lon[rs, cs]
    refl[(refl < -0.5) | (refl > 1.5)] = np.nan                           # fill value -9999
    keep = good & ~bad_band_mask(wl)
    cube = np.moveaxis(refl[:, :, keep], -1, 0)                           # (bands, h, w)
    wlk = wl[keep]

    valid = np.isfinite(cube).mean(axis=0) > 0.9
    if mask_path is not None and Path(mask_path).exists():
        with h5py.File(mask_path, "r") as f:
            m = _find(f, "mask")[rs, cs, :]
            try:
                names = [n.decode() if isinstance(n, bytes) else str(n) for n in _find(f, "mask_bands")[:]]
            except KeyError:
                names = []
        low = [n.lower() for n in names]
        if any("aggregate" in n for n in low):
            bad = m[:, :, [i for i, n in enumerate(low) if "aggregate" in n][0]] > 0
        else:  # cloud, cirrus, dilated cloud flags (first, second and fifth layers in EMIT V001)
            idx = [i for i, n in enumerate(low) if any(k in n for k in ("cloud", "cirrus"))] or [0, 1, 4]
            bad = (m[:, :, idx] > 0).any(axis=2)
        valid &= ~bad
    cube[:, ~valid] = np.nan

    s2 = simulate_s2(cube, wlk)
    hs = hyperspectral_features(cube, wlk)

    # PCA of the full (cleaned) spectrum, so mean spectra can be reconstructed later
    flat = cube.reshape(cube.shape[0], -1).T
    ok = np.isfinite(flat).all(axis=1)
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(np.flatnonzero(ok), size=min(20000, ok.sum()), replace=False)
    pca = PCA(n_components=n_pcs, random_state=seed).fit(flat[fit_idx])
    pcs = np.full((flat.shape[0], n_pcs), np.nan, dtype="float32")
    pcs[ok] = pca.transform(flat[ok])
    pcs = pcs.reshape(cube.shape[1], cube.shape[2], n_pcs)

    lo0, la0, lo1, la1 = bbox
    in_aoi = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    meta_valid_pct = float(100 * valid[in_aoi].mean()) if in_aoi.any() else 0.0
    pad = 0.005                                     # keep a ~500 m margin, drop the rest of the swath
    keep_px = (lon >= lo0 - pad) & (lon <= lo1 + pad) & (lat >= la0 - pad) & (lat <= la1 + pad)
    valid &= keep_px
    meta = {"granule": granule_id, "sensor": "NASA EMIT (ISS)", "product": "EMIT L2A surface reflectance (EMITL2ARFL)",
            "window_shape": list(cube.shape[1:]), "n_bands_total": int(wl.size), "n_bands_used": int(keep.sum()),
            "aoi_valid_pct": meta_valid_pct,
            "pixel_size_m": 60, "pca_explained_var": pca.explained_variance_ratio_.round(5).tolist(),
            "licence": "NASA EMIT data: free and open, no restrictions (cite LP DAAC / NASA JPL)"}
    if extra_meta:
        meta.update(extra_meta)
    np.savez_compressed(
        out_dir / f"{granule_id}.npz",
        lat=lat.astype("float32"), lon=lon.astype("float32"), valid=valid,
        **{k: np.where(valid, v, np.nan).astype("float32" if k == "REP" else "float16") for k, v in {**s2, **hs}.items()},
        pcs=np.where(valid[:, :, None], pcs, np.nan).astype("float16"), pca_components=pca.components_.astype("float32"),
        pca_mean=pca.mean_.astype("float32"), wavelengths=wlk.astype("float32"))
    (out_dir / f"{granule_id}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def fetch_emit(bbox=config.AOI_DEFAULT.bbox, out_dir=None, min_valid_pct=60.0, max_tries=4, log=print):
    """Search, download and extract the clearest EMIT scene over the AOI."""
    import os
    import tempfile
    import earthaccess

    out_dir = Path(out_dir or config.EMIT_DIR)
    if not (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")):
        raise RuntimeError("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD (free account: urs.earthdata.nasa.gov)")
    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(short_name=EMIT_SHORT_NAME, bounding_box=tuple(bbox),
                                      temporal=config.EMIT_TEMPORAL, count=500)
    if not results:
        raise RuntimeError("No EMIT scenes found over the AOI")

    def info(g):
        u = g["umm"]
        name = u.get("GranuleUR", "")
        cc = u.get("CloudCover")
        cc = float(cc) if cc is not None else 50.0
        when = u.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime", "")
        return name, cc, when

    ranked = sorted(results, key=lambda g: ("_002_" not in info(g)[0], info(g)[1]))
    log(f"{len(results)} EMIT scenes intersect the AOI; trying the clearest first")
    tmp = Path(tempfile.mkdtemp(prefix="emit_"))
    for g in ranked[:max_tries]:
        name, cc, when = info(g)
        links = [l for l in g.data_links() if l.endswith(".nc")]
        rfl = [l for l in links if Path(l).name.startswith("EMIT_L2A_RFL_")]
        msk = [l for l in links if Path(l).name.startswith("EMIT_L2A_MASK_")]
        if not rfl:
            continue
        log(f"  {name}  {when}  scene cloud {cc:.0f}% -> downloading")
        files = earthaccess.download(rfl[:1] + msk[:1], local_path=str(tmp))
        rp = next((Path(p) for p in files if Path(p).name.startswith("EMIT_L2A_RFL_")), None)
        mp = next((Path(p) for p in files if Path(p).name.startswith("EMIT_L2A_MASK_")), None)
        meta = extract_window(rp, mp, bbox, name, out_dir,
                              extra_meta={"datetime": when, "scene_cloud_pct": cc, "source_files": [Path(x).name for x in files]})
        for p in files:
            Path(p).unlink(missing_ok=True)
        if meta is None:
            log("    scene does not cover the AOI"); continue
        log(f"    AOI valid (cloud-free) {meta['aoi_valid_pct']:.0f}%")
        if meta["aoi_valid_pct"] >= min_valid_pct:
            for old in out_dir.glob("*_meta.json"):        # keep only the chosen scene
                if old.name != f"{name}_meta.json":
                    old.unlink(); (out_dir / old.name.replace("_meta.json", ".npz")).unlink(missing_ok=True)
            return meta
        (out_dir / f"{name}.npz").unlink(missing_ok=True); (out_dir / f"{name}_meta.json").unlink(missing_ok=True)
    raise RuntimeError("No sufficiently cloud-free EMIT scene found; lower min_valid_pct")


# ---------------------------------------------------------------------------
# analysis (offline)
# ---------------------------------------------------------------------------
def load_emit(emit_dir=None):
    emit_dir = Path(emit_dir or config.EMIT_DIR)
    metas = sorted(emit_dir.glob("*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No EMIT subset in {emit_dir}. Run scripts/fetch_data.py --only-emit")
    meta = json.loads(metas[0].read_text())
    e = {k: (v.astype("float32") if v.dtype == np.float16 and k != "pcs" else v)
         for k, v in np.load(emit_dir / f"{meta['granule']}.npz").items()}
    return e, meta


def pixel_table(e, meta) -> pd.DataFrame:
    """One row per valid EMIT pixel: simulated S2 bands, features, PCA scores."""
    valid = e["valid"]
    r, c = np.nonzero(valid)
    df = pd.DataFrame({"row": r, "col": c})
    for k in S2_NAMES + FEATURES:
        df[k] = e[k][r, c]
    pcs = e["pcs"][r, c].astype("float32")
    for i in range(pcs.shape[1]):
        df[f"PC{i+1:02d}"] = pcs[:, i]
    df["scene"] = meta["granule"]
    return df.dropna(subset=S2_NAMES).reset_index(drop=True)


def to_aoi_grid(e, profile, max_dist_m=45.0):
    """Nearest-neighbour resample of every EMIT layer onto the 20 m AOI grid."""
    import rasterio
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    H, W = profile["height"], profile["width"]
    rows, cols = np.mgrid[0:H, 0:W]
    xs, ys = rasterio.transform.xy(profile["transform"], rows.ravel(), cols.ravel())
    tr = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    ex, ey = tr.transform(e["lon"].ravel().astype("float64"), e["lat"].ravel().astype("float64"))
    ok = e["valid"].ravel() & np.isfinite(ex)
    tree = cKDTree(np.c_[ex[ok], ey[ok]])
    d, idx = tree.query(np.c_[np.asarray(xs), np.asarray(ys)], distance_upper_bound=max_dist_m)
    hit = np.isfinite(d)
    src = np.flatnonzero(ok)
    out = {}
    for k in S2_NAMES + FEATURES:
        v = np.full(H * W, np.nan, dtype="float32")
        v[hit] = e[k].ravel()[src[idx[hit]]]
        out[k] = v.reshape(H, W)
    out["EMIT_VALID"] = hit.reshape(H, W)
    return out


def mean_spectra(e, labels_by_pixel: pd.Series, df: pd.DataFrame):
    """Reconstruct the mean spectrum of each class from PCA scores."""
    pcs = [c for c in df.columns if c.startswith("PC")]
    out = {}
    for k in sorted(labels_by_pixel.unique()):
        s = np.nanmean(df.loc[labels_by_pixel.index[labels_by_pixel == k], pcs].to_numpy(), axis=0)
        out[k] = s @ e["pca_components"] + e["pca_mean"]
    return e["wavelengths"], out
