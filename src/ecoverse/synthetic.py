"""SYNTHETIC test fixtures - for automated tests only.

These arrays are generated from simple shapes and noise so the pipeline can
be tested without internet. They are NOT observations and must never be
presented as results. The notebooks refuse to use them unless explicitly
asked (ECOVERSE_ALLOW_SYNTHETIC=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from .indices import S2_SRF, hyperspectral_features, simulate_s2


def synthetic_stack(h=180, w=200, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    sea = xx > 0.85 - 0.1 * yy
    bay = ((xx - 0.55) ** 2 + (yy - 0.6) ** 2) < 0.012
    water = sea | bay
    dist = np.minimum(np.abs(xx - (0.85 - 0.1 * yy)), np.sqrt((xx - 0.55) ** 2 + (yy - 0.6) ** 2))
    urban = (((xx - 0.3) ** 2 + (yy - 0.3) ** 2) < 0.05) & ~water
    farms = (yy > 0.75) & (xx < 0.4)
    mangrove = (dist < 0.05) & (np.abs(yy - 0.6) < 0.12) & ~water & (xx > 0.45) & (xx < 0.65)
    veg = farms | mangrove | (urban & (rng.random((h, w)) < 0.15))
    dem = np.clip(12 * dist + rng.normal(0, 0.3, (h, w)), 0, None)
    n = lambda s: rng.normal(0, s, (h, w))  # noqa: E731
    base = 0.28 + n(0.02)
    L = {}
    L["B02"] = np.where(water, 0.05, np.where(veg, 0.04, base * 0.55)) + n(0.005)
    L["B03"] = np.where(water, 0.07, np.where(veg, 0.07, base * 0.75)) + n(0.005)
    L["B04"] = np.where(water, 0.04, np.where(veg, 0.05, base * 0.95)) + n(0.005)
    L["B05"] = np.where(water, 0.03, np.where(veg, 0.12, base)) + n(0.005)
    L["B06"] = np.where(water, 0.02, np.where(veg, 0.28, base * 1.02)) + n(0.005)
    L["B07"] = np.where(water, 0.02, np.where(veg, 0.33, base * 1.04)) + n(0.005)
    L["B8A"] = np.where(water, 0.02, np.where(veg, 0.36, base * 1.05)) + n(0.005)
    L["B11"] = np.where(water, 0.01, np.where(veg, 0.22, base * 1.25)) + n(0.005)
    L["B12"] = np.where(water, 0.01, np.where(veg, 0.12, base * 1.1)) + n(0.005)
    L["W_B04"] = L["B04"] * 0.95
    L["W_B8A"] = L["B8A"] * 1.05
    L["W_B11"] = L["B11"]
    L["VEG_PERSIST"] = np.where(veg, 0.9, 0.05) + n(0.03)
    L["S2_NOBS"] = np.full((h, w), 25.0)
    L["LST_SUMMER"] = np.where(water, 33, np.where(veg, 40, np.where(urban, 50, 47))) + n(1.0)
    L["DEM"] = dem
    wc = np.full((h, w), 60)
    wc[urban] = 50; wc[farms] = 40; wc[water] = 80; wc[mangrove] = 95
    L["WORLDCOVER"] = wc.astype("float32")
    L = {k: np.clip(v, -0.5, None).astype("float32") if k.startswith(("B", "W_")) else v.astype("float32")
         for k, v in L.items()}
    transform = from_origin(450000, 2900000, 20, 20)
    return L, transform, "EPSG:32639"


def synthetic_tanager_table(n=4000, seed=2):
    """Synthetic hyperspectral pixel table with the same columns as the real one."""
    rng = np.random.default_rng(seed)
    wl = np.arange(400, 2450, 5.0)
    wl = wl[~(((wl >= 1350) & (wl <= 1450)) | ((wl >= 1800) & (wl <= 1950)))]
    cover = rng.random(n)
    gyp = rng.random(n) * (cover < 0.4)
    soil = 0.25 + 0.1 * (wl / 2500)[:, None] + 0.02 * rng.normal(size=(1, n))
    soil = soil - gyp[None, :] * 0.08 * np.exp(-0.5 * ((wl - 1750) / 12) ** 2)[:, None]
    vegsp = 0.05 + 0.4 / (1 + np.exp(-(wl - 715) / 12))[:, None] * np.ones((1, n))
    cube = (1 - cover) * soil + cover * vegsp
    s2 = simulate_s2(cube, wl)
    hs = hyperspectral_features(cube, wl)
    df = pd.DataFrame({k: v for k, v in {**s2, **hs}.items()})
    df["row"] = rng.integers(0, 600, n); df["col"] = rng.integers(0, 600, n)
    for i in range(20):
        df[f"PC{i+1:02d}"] = rng.normal(size=n) + (gyp * 3 if i == 0 else 0)
    df["scene"] = "SYNTHETIC"
    return df[list(S2_SRF) + list(hs) + ["row", "col"] + [f"PC{i+1:02d}" for i in range(20)] + ["scene"]]
