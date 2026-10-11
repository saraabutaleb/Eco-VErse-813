"""Uncertainty, robustness and the satellite -> ground-probe loop.

1. Monte Carlo: perturb every uncertain input (salinity, heat, pH,
   waterlogging) and count how often each pixel keeps the same
   recommendation -> a *stability* map.
2. Probe planning: send a two-person team with a low-cost probe kit to the
   few sites where a measurement would change the most decisions.
3. Fusion: blend probe readings into the satellite prior (simple residual
   kriging with a Gaussian kernel), shrinking uncertainty around each probe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .io import pixel_to_lonlat
from .recommend import run_engine


def monte_carlo(fp: dict, plants: pd.DataFrame, base, n: int = 40, seed: int = config.RANDOM_SEED):
    """Return stability (fraction of draws keeping the base hero species) and
    a DataFrame of per-draw totals (for confidence intervals)."""
    rng = np.random.default_rng(seed)
    mask = base.mask
    N = mask.sum()
    same = np.zeros(N, dtype="float32")
    draws = []
    ece0 = fp["ECE"][mask]
    sd = fp["ECE_LOG_SD"][mask]
    heat0 = np.nan_to_num(fp["HEAT_PCT"][mask], nan=0.5)
    wl0 = fp["WATERLOG"][mask]
    ph_mu, ph_sd = config.REGIONAL_PH_PRIOR
    for _ in range(n):
        global_shift = rng.normal(0, 0.25)             # whole-AOI calibration error
        ece = ece0 * np.exp(global_shift + rng.normal(0, 1, N) * sd)
        ov = {
            "ECE": ece.astype("float32"),
            "HEAT_PCT": np.clip(heat0 + rng.normal(0, 0.1, N), 0, 1).astype("float32"),
            "WATERLOG": np.clip(wl0 + rng.normal(0, 0.15, N), 0, 1).astype("float32"),
            "PH": (rng.normal(ph_mu, ph_sd) + rng.normal(0, 0.15, N)).astype("float32"),
        }
        et0 = rng.uniform(*config.ET0_RANGE)
        r = run_engine(fp, plants, mask=mask, overrides=ov, et0=et0)
        same += (r.hero == base.hero)
        draws.append({"et0": et0, "mean_saving_mm": float(np.nanmean(r.saving_mm)),
                      "pct_hero_changed": float(100 * np.mean(r.hero != base.hero))})
    return same / n, pd.DataFrame(draws)


def select_probe_sites(fp: dict, base, stability: np.ndarray, profile: dict,
                       n_sites: int = 12, min_spacing_m: float = 500.0, res_m: float = 20.0):
    """Greedy information-driven site selection.

    value = (1 - stability) x decision weight, where decision weight is
    higher on today's irrigated urban greenery (where a change of species
    saves the most water) and on open, plantable ground.
    """
    mask = base.mask
    inst = 1.0 - stability
    irrigated = fp["irrigated"][mask].astype("float32")
    built = fp["BUILT_DENSITY"][mask]
    weight = 1.0 + 1.5 * irrigated + 0.5 * (built < 0.6)
    value2d = np.zeros(mask.shape, dtype="float32")
    value2d[mask] = inst * weight
    value2d[fp["intertidal"]] *= 0.3          # tidal flats are hard to access on foot
    rows, cols = np.nonzero(mask)
    picked = []
    v = value2d.copy()
    rad = int(np.ceil(min_spacing_m / res_m))
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    disk = (yy ** 2 + xx ** 2) <= rad ** 2
    for _ in range(n_sites):
        k = int(np.argmax(v))
        if v.flat[k] <= 0:
            break
        r, c = np.unravel_index(k, v.shape)
        picked.append((r, c, float(value2d[r, c])))
        r0, r1 = max(0, r - rad), min(v.shape[0], r + rad + 1)
        c0, c1 = max(0, c - rad), min(v.shape[1], c + rad + 1)
        sub = disk[(r0 - r + rad):(r1 - r + rad), (c0 - c + rad):(c1 - c + rad)]
        v[r0:r1, c0:c1][sub] = 0
    if not picked:
        return pd.DataFrame()
    pr = np.array([p[0] for p in picked]); pc = np.array([p[1] for p in picked])
    lon, lat = pixel_to_lonlat(pr, pc, profile)
    df = pd.DataFrame({"row": pr, "col": pc, "lon": lon.round(6), "lat": lat.round(6),
                       "value": [p[2] for p in picked]})
    df["ece_prior_dSm"] = fp["ECE"][pr, pc].round(1)
    df["lst_c"] = fp["LST"][pr, pc].round(1)
    df["irrigated_now"] = fp["irrigated"][pr, pc]
    df = _route_order(df)
    df.insert(0, "visit_order", np.arange(1, len(df) + 1))
    df["site_id"] = [f"P{i:02d}" for i in df["visit_order"]]
    return df


def _route_order(df):
    """Nearest-neighbour route so two students can walk/drive it in one go."""
    xy = df[["lon", "lat"]].to_numpy()
    start = int(np.argmin(xy[:, 1]))  # start at southernmost site
    order, left = [start], set(range(len(df))) - {start}
    while left:
        last = xy[order[-1]]
        nxt = min(left, key=lambda j: np.hypot(*(xy[j] - last)))
        order.append(nxt); left.remove(nxt)
    return df.iloc[order].reset_index(drop=True)


def fuse_ground_readings(fp: dict, readings: pd.DataFrame, range_m: float = 600.0,
                         prior_strength: float = 0.5, res_m: float = 20.0) -> dict:
    """Return a copy of fp with ECe and pH updated by probe readings.

    readings needs columns: row, col, and any of ece_dSm, ph.
    """
    out = dict(fp)
    h, w = fp["ECE"].shape
    yy, xx = np.mgrid[0:h, 0:w]
    for var, sd_key, is_log in (("ece_dSm", "ECE_LOG_SD", True), ("ph", "PH_SD", False)):
        if var not in readings or readings[var].isna().all():
            continue
        key = "ECE" if var == "ece_dSm" else "PH"
        prior = np.log(fp[key]) if is_log else fp[key].astype("float64")
        wsum = np.zeros((h, w)); rsum = np.zeros((h, w))
        for _, r in readings.dropna(subset=[var]).iterrows():
            obs = np.log(r[var]) if is_log else r[var]
            resid = obs - prior[int(r.row), int(r.col)]
            d2 = ((yy - r.row) ** 2 + (xx - r.col) ** 2) * res_m ** 2
            wk = np.exp(-d2 / (2 * range_m ** 2))
            wsum += wk; rsum += wk * resid
        post = prior + rsum / (wsum + prior_strength)
        out[key] = (np.exp(post) if is_log else post).astype("float32")
        out[sd_key] = (fp[sd_key] * np.sqrt(prior_strength / (wsum + prior_strength))).astype("float32")
    return out


def ec15_to_ece(ec_1to5, texture: str = "loam"):
    """Convert a cheap 1:5 soil:water EC reading to saturated-paste ECe.

    Texture factors after Slavich & Petterson (1993). Values are approximate;
    calibrate against a lab ECe for a few samples when possible.
    """
    factor = {"sand": 17.0, "loamy_sand": 13.8, "sandy_loam": 9.5, "loam": 8.6,
              "clay_loam": 8.6, "clay": 6.0}[texture]
    return ec_1to5 * factor


def prepare_readings(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Turn a filled-in ground_readings.csv (lat/lon, EC 1:5) into the
    row/col + ECe form that fuse_ground_readings expects."""
    import rasterio
    from pyproj import Transformer
    d = df.copy()
    if d.get("row") is None or d["row"].isna().any():
        tr = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
        x, y = tr.transform(d["lon"].to_numpy(), d["lat"].to_numpy())
        r, c = rasterio.transform.rowcol(profile["transform"], x, y)
        d["row"], d["col"] = np.asarray(r), np.asarray(c)
    if "ece_dSm" not in d:
        d["ece_dSm"] = np.nan
    need = d["ece_dSm"].isna() & d.get("ec_1to5_dSm", pd.Series(np.nan, index=d.index)).notna()
    d.loc[need, "ece_dSm"] = [ec15_to_ece(e, t if isinstance(t, str) else "loam")
                              for e, t in zip(d.loc[need, "ec_1to5_dSm"], d.loc[need, "texture"])]
    inside = (d["row"] >= 0) & (d["row"] < profile["height"]) & (d["col"] >= 0) & (d["col"] < profile["width"])
    return d[inside].reset_index(drop=True)
