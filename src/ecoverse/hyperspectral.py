"""Hyperspectral calibration: "hyperspectral teaches multispectral".

Bahrain is not in Planet Tanager's open archive and Satellite 813 data is not
available during the PoC. So we:

1. Take open Tanager scenes over climatically similar Gulf sites
   (coastal sabkha in Abu Dhabi, the arid city of Riyadh).
2. Compute narrow-band features only a hyperspectral sensor can see
   (gypsum 1750 nm, clay 2200 nm, carbonate 2340 nm, red-edge position,
   canopy water 1200 nm).
3. Simulate what Sentinel-2 would have recorded for the same pixels.
4. Learn Sentinel-2 -> hyperspectral-feature models with *spatially blocked*
   cross-validation, which tells us honestly how much of each hyperspectral
   signal Sentinel-2 can recover (R2) and how much it cannot (1 - R2: the gap
   Satellite 813 fills).
5. Apply only the well-recovered models to Bahrain Sentinel-2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

from . import config
from .indices import (S2_SRF, bad_band_mask, hyperspectral_features, nd, simulate_s2)

H5_ROOT = "HDFEOS/GRIDS/HYP/Data Fields"
S2_NAMES = list(S2_SRF)
TARGETS = {
    "GYPSUM_1750": "soil",
    "CLAY_2200": "soil",
    "CARBONATE_2340": "soil",
    "REP": "veg",
    "CANOPY_WATER_1200": "veg",
}


# ---------------------------------------------------------------------------
# Extraction (runs once, online, inside scripts/fetch_data.py)
# ---------------------------------------------------------------------------
def tanager_wavelengths(item: dict, sr_key: str) -> np.ndarray:
    bands = item["assets"][sr_key].get("bands", [])
    wl = [b["eo:center_wavelength"] for b in bands if "eo:center_wavelength" in b]
    return np.asarray(wl, dtype="float64") * 1000.0  # um -> nm


def extract_tanager_scene(h5_path, item: dict, sr_key: str, out_dir, label: str,
                          stride_maps: int = 4, n_sample: int = 12000, n_pcs: int = 20,
                          clip_size: int = 32, seed: int = config.RANDOM_SEED, block_rows: int = 48):
    """Compute features from a downloaded Tanager SR HDF5 and save a compact
    subset that is small enough to commit to the repository."""
    import h5py

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wl_all = tanager_wavelengths(item, sr_key)
    good = ~bad_band_mask(wl_all)
    gidx = np.where(good)[0]
    wl = wl_all[gidx]
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        sr = f[f"{H5_ROOT}/surface_reflectance"]
        nb, H, W = sr.shape
        if nb != len(wl_all):
            raise ValueError(f"cube has {nb} bands but STAC lists {len(wl_all)}")
        cloud = f[f"{H5_ROOT}/beta_cloud_mask"][:]
        nodata = f[f"{H5_ROOT}/nodata_pixels"][:] if "nodata_pixels" in f[H5_ROOT] else np.zeros((H, W))
        valid = (cloud == 0) & (nodata == 0)

        maps = {k: np.full((H, W), np.nan, dtype="float32") for k in S2_NAMES + list(TARGETS)}
        for r0 in range(0, H, block_rows):
            r1 = min(H, r0 + block_rows)
            blk = sr[:, r0:r1, :][gidx].astype("float32")
            blk[(blk < 0) | (blk > 1.5)] = np.nan
            blk[:, ~valid[r0:r1]] = np.nan
            s2 = simulate_s2(blk, wl)
            hs = hyperspectral_features(blk, wl)
            for k, v in {**s2, **hs}.items():
                maps[k][r0:r1] = v
        bad = ~np.isfinite(maps["B04"]) | (maps["B04"] <= 0)
        for k in maps:
            maps[k][bad | ~valid] = np.nan

        ndvi = nd(maps["B8A"], maps["B04"])
        mndwi = nd(maps["B03"], maps["B11"])
        land = valid & np.isfinite(ndvi) & (mndwi < 0.0)
        # Pixel sample for model training (row/col kept for spatial blocking)
        rr, cc = np.nonzero(land)
        k = min(n_sample, rr.size)
        pick = rng.choice(rr.size, size=k, replace=False)
        rr, cc = rr[pick], cc[pick]
        order = np.lexsort((cc, rr))
        rr, cc = rr[order], cc[order]
        spectra = np.empty((k, gidx.size), dtype="float32")
        # read sampled spectra row-block by row-block (fast with h5py)
        for r0 in range(0, H, block_rows):
            sel = (rr >= r0) & (rr < r0 + block_rows)
            if not sel.any():
                continue
            blk = sr[:, r0:min(H, r0 + block_rows), :][gidx].astype("float32")
            spectra[sel] = blk[:, rr[sel] - r0, cc[sel]].T
        spectra = np.clip(np.nan_to_num(spectra, nan=0.0), 0, 1.5)

        # small clip around the land pixel with the highest local variety
        cr = int(np.clip(np.median(rr), clip_size // 2, H - clip_size // 2))
        ccn = int(np.clip(np.median(cc), clip_size // 2, W - clip_size // 2))
        clip = sr[:, cr - clip_size // 2: cr + clip_size // 2,
                  ccn - clip_size // 2: ccn + clip_size // 2][gidx].astype("float16")

    pca = PCA(n_components=n_pcs, random_state=seed).fit(spectra)
    table = pd.DataFrame({"row": rr, "col": cc})
    for kname in S2_NAMES + list(TARGETS):
        table[kname] = maps[kname][rr, cc]
    scores = pca.transform(spectra)
    for i in range(n_pcs):
        table[f"PC{i+1:02d}"] = scores[:, i].astype("float32")
    table["scene"] = item["id"]
    sid = item["id"]
    table.to_csv(out_dir / f"{sid}_pixels.csv.gz", index=False, float_format="%.5f")
    np.savez_compressed(
        out_dir / f"{sid}_maps.npz",
        **{k: v[::stride_maps, ::stride_maps].astype("float16") for k, v in maps.items()},
        land=land[::stride_maps, ::stride_maps], wavelengths=wl,
        pca_components=pca.components_.astype("float32"), pca_mean=pca.mean_.astype("float32"),
        clip=clip, clip_origin=np.array([cr - clip_size // 2, ccn - clip_size // 2]),
    )
    meta = {"scene_id": sid, "label": label, "collection": item.get("collection"),
            "datetime": item["properties"].get("datetime"), "bbox": item.get("bbox"),
            "sr_asset": sr_key, "n_bands_total": int(len(wl_all)), "n_bands_used": int(gidx.size),
            "shape": [int(H), int(W)], "valid_pct": float(100 * valid.mean()),
            "land_pct": float(100 * land.mean()), "n_sampled_pixels": int(k),
            "map_stride": stride_maps, "pca_explained_var": pca.explained_variance_ratio_.round(5).tolist(),
            "licence": "Planet Tanager open data, CC BY 4.0 (see Planet open-data terms)"}
    (out_dir / f"{sid}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Analysis (offline, from committed files)
# ---------------------------------------------------------------------------
def load_tanager_tables(tdir=config.TANAGER_DIR):
    tdir = Path(tdir)
    tables, metas, maps = [], {}, {}
    for meta_p in sorted(tdir.glob("*_meta.json")):
        meta = json.loads(meta_p.read_text())
        sid = meta["scene_id"]
        tables.append(pd.read_csv(tdir / f"{sid}_pixels.csv.gz"))
        metas[sid] = meta
        maps[sid] = dict(np.load(tdir / f"{sid}_maps.npz"))
    if not tables:
        raise FileNotFoundError(f"No Tanager subsets in {tdir}. Run scripts/fetch_data.py first.")
    return pd.concat(tables, ignore_index=True), metas, maps


def s2_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sentinel-2 predictors: the 9 bands plus standard indices."""
    X = df[S2_NAMES].astype("float32").copy()
    X["NDVI"] = nd(X["B8A"], X["B04"])
    X["NDRE"] = nd(X["B8A"], X["B05"])
    X["NDMI"] = nd(X["B8A"], X["B11"])
    X["SWIR_RATIO"] = nd(X["B11"], X["B12"])
    X["SI"] = np.sqrt(np.clip(X["B02"] * X["B04"], 0, None))
    return X


def _subset(df, target):
    ndvi = nd(df["B8A"], df["B04"])
    sel = np.isfinite(df[target]) & np.isfinite(df[S2_NAMES]).all(axis=1)
    sel &= (ndvi < 0.2) if TARGETS[target] == "soil" else (ndvi >= 0.3)
    return df[sel]


def spatial_groups(df, block_px: int = 32):
    return (df["scene"].astype(str) + "_" + (df["row"] // block_px).astype(str) + "_" +
            (df["col"] // block_px).astype(str))


def evaluate_transfer(df: pd.DataFrame, n_splits: int = 5, seed: int = config.RANDOM_SEED):
    """Spatially-blocked CV of Sentinel-2 -> hyperspectral-feature models."""
    rows = []
    for target in TARGETS:
        d = _subset(df, target)
        if len(d) < 200:
            rows.append({"target": target, "n": len(d)}); continue
        X = s2_feature_frame(d).to_numpy(); y = d[target].to_numpy()
        groups = spatial_groups(d)
        gkf = GroupKFold(n_splits=n_splits)
        pred_gb = np.empty_like(y); pred_lin = np.empty_like(y)
        for tr, te in gkf.split(X, y, groups):
            gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed)
            gb.fit(X[tr], y[tr]); pred_gb[te] = gb.predict(X[te])
            lin = LinearRegression().fit(X[tr], y[tr]); pred_lin[te] = lin.predict(X[te])
        rows.append({"target": target, "pixels": TARGETS[target], "n": len(d),
                     "r2_linear": r2_score(y, pred_lin), "r2_boosted": r2_score(y, pred_gb),
                     "mae_boosted": mean_absolute_error(y, pred_gb),
                     "hyperspectral_only_info_pct": 100 * (1 - max(r2_score(y, pred_gb), 0))})
    return pd.DataFrame(rows)


def fit_transfer_model(df: pd.DataFrame, target: str, seed: int = config.RANDOM_SEED):
    d = _subset(df, target)
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed)
    gb.fit(s2_feature_frame(d).to_numpy(), d[target].to_numpy())
    return gb


def predict_on_s2(model, bands: dict) -> np.ndarray:
    shape = bands["B04"].shape
    df = pd.DataFrame({k: bands[k].ravel() for k in S2_NAMES})
    ok = np.isfinite(df.to_numpy()).all(axis=1)
    out = np.full(df.shape[0], np.nan, dtype="float32")
    if ok.any():
        out[ok] = model.predict(s2_feature_frame(df[ok]).to_numpy())
    return out.reshape(shape)


def soil_class_recovery(df: pd.DataFrame, k: int = 6, n_splits: int = 5, seed: int = config.RANDOM_SEED):
    """How many hyperspectral soil-surface types can Sentinel-2 tell apart?

    Cluster bare-ground pixels on their hyperspectral absorption depths
    (gypsum, clay, carbonate, 1200 nm water), then try to predict those clusters from Sentinel-2 bands with
    spatially-blocked CV. Balanced accuracy near 1/k means Sentinel-2 is
    blind to the distinction; near 1 means it is not needed.
    """
    ndvi = nd(df["B8A"], df["B04"])
    feats = ["GYPSUM_1750", "CLAY_2200", "CARBONATE_2340", "CANOPY_WATER_1200"]
    d = df[(ndvi < 0.2)].dropna(subset=S2_NAMES + feats)
    # Narrow-band absorption depths are physically comparable across scenes
    # (PCA scores are not: each scene has its own PCA basis).
    Z = d[feats].to_numpy()
    Z = np.clip((Z - np.median(Z, 0)) / (Z.std(0) + 1e-9), -4, 4)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    X = s2_feature_frame(d).to_numpy()
    groups = spatial_groups(d)
    pred = np.empty_like(labels)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, labels, groups):
        clf = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed).fit(X[tr], labels[tr])
        pred[te] = clf.predict(X[te])
    acc = balanced_accuracy_score(labels, pred)
    cm = pd.crosstab(pd.Series(labels, name="hyperspectral class"), pd.Series(pred, name="predicted from S2"),
                     normalize="index")
    return {"k": k, "balanced_accuracy": acc, "chance": 1 / k, "confusion": cm,
            "labels": labels, "index": d.index.to_numpy()}
