"""The Ecoverse *environmental fingerprint*.

For every 20 m pixel we build a small, interpretable vector describing the
conditions a plant has to survive there:

    heat, salinity (as an ECe estimate), waterlogging / shallow saline
    groundwater, surface moisture, survival evidence (vegetation that persists
    through summer), irrigation dependence, urban/infrastructure density,
    loose sand, intertidal status.

All layers are satellite-derived *proxies*. Ground-probe readings (see
`uncertainty.fuse_ground_readings`) pull the estimates toward measured values.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

from . import config
from .indices import s2_indices

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]

# ESA WorldCover 2021 class codes
WC = {"tree": 10, "shrub": 20, "grass": 30, "crop": 40, "built": 50, "bare": 60,
      "snow": 70, "water": 80, "wetland": 90, "mangrove": 95, "moss": 100}


def pct_rank(x, mask):
    """Percentile rank (0..1) of x computed over pixels in mask; NaN elsewhere."""
    out = np.full(x.shape, np.nan, dtype="float32")
    v = x[mask & np.isfinite(x)]
    if v.size == 0:
        return out
    order = np.sort(v)
    sel = mask & np.isfinite(x)
    out[sel] = np.searchsorted(order, x[sel], side="right") / order.size
    return out


def local_fraction(mask, radius_px):
    """Fraction of True pixels in a (2r+1)^2 window."""
    k = 2 * radius_px + 1
    return ndimage.uniform_filter(mask.astype("float32"), size=k, mode="nearest")


def build_fingerprint(layers: dict, res_m: float = 20.0, gypsum_proxy=None) -> dict:
    """Derive every fingerprint layer from the input stack.

    Parameters
    ----------
    layers : dict from io.read_stack
    res_m : pixel size in metres
    gypsum_proxy : optional 2-D array, the hyperspectral-calibrated gypsum
        (sulfate) proxy predicted from Sentinel-2 (see hyperspectral.py).
    """
    b = {k: layers[k] for k in S2_BANDS}
    idx = s2_indices(b)
    wc = np.nan_to_num(layers["WORLDCOVER"], nan=0).astype(int)
    dem = layers["DEM"]
    lst = layers["LST_SUMMER"]
    valid = np.isfinite(b["B04"]) & np.isfinite(b["B8A"])

    # --- water, land, intertidal ------------------------------------------
    water = valid & ((wc == WC["water"]) | ((idx["MNDWI"] > 0.15) & (idx["NDVI"] < 0.1)))
    land = valid & ~water
    dist_sea_m = ndimage.distance_transform_edt(~water) * res_m
    mangrove_now = land & ((wc == WC["mangrove"]) | ((wc == WC["wetland"]) & (idx["NDVI"] > 0.3)))
    intertidal = land & (
        mangrove_now
        | (wc == WC["wetland"])
        | ((dist_sea_m <= 200) & (np.nan_to_num(dem, nan=99) <= 1.5) & (wc != WC["built"]))
    )

    # --- heat ---------------------------------------------------------------
    heat_pct = pct_rank(lst, land)

    # --- vegetation, survival evidence, irrigation dependence -------------
    ndvi_s = idx["NDVI"]
    ndvi_w = (layers["W_B8A"] - layers["W_B04"]) / (layers["W_B8A"] + layers["W_B04"] + 1e-6)
    persist = np.clip(np.nan_to_num(layers["VEG_PERSIST"], nan=0), 0, 1)
    veg_frac = np.clip((ndvi_s - 0.08) / (0.6 - 0.08), 0, 1)  # crude cover fraction
    # Summer has ~no rain in Bahrain: vegetation that stays green through
    # June-August and is not tidal is almost certainly irrigated or
    # groundwater-fed. That is the greenery Ecoverse can make cheaper.
    irrigated = land & ~intertidal & (persist >= 0.6) & (ndvi_s >= 0.3)
    irrigation_dependence = np.where(land & ~intertidal, persist * veg_frac, 0).astype("float32")

    # --- urban / infrastructure density -----------------------------------
    built = land & (wc == WC["built"])
    built_density = local_fraction(built, radius_px=max(1, int(round(50 / res_m))))

    # --- salinity pressure -> ECe ----------------------------------------
    bare_like = land & (ndvi_s < 0.2) & ~built
    spectral_sal = pct_rank(idx["SI"], bare_like)
    if gypsum_proxy is not None:
        mineral = pct_rank(gypsum_proxy, bare_like)
    else:
        mineral = pct_rank(idx["SWIR_RATIO"], bare_like)
    lowland = np.clip((4.0 - np.nan_to_num(dem, nan=4.0)) / 4.0, 0, 1)
    coastal = np.exp(-dist_sea_m / 1500.0)

    w_spec = 0.35 * (1 - veg_frac)          # spectra only speak for bare ground
    w_min = 0.20 * (1 - veg_frac)
    w_low, w_coast = 0.25, 0.20
    num = (w_spec * np.nan_to_num(spectral_sal, nan=0.5)
           + w_min * np.nan_to_num(mineral, nan=0.5)
           + w_low * lowland + w_coast * coastal)
    den = w_spec + w_min + w_low + w_coast
    sal_pressure = np.clip(num / den, 0, 1).astype("float32")
    # Irrigated soils are leached; tidal soils are seawater-saturated.
    sal_pressure = np.where(irrigated, sal_pressure * 0.8, sal_pressure)
    sal_pressure = np.where(intertidal, np.maximum(sal_pressure, 0.9), sal_pressure)
    ece = config.ECE_PRIOR_MIN * (config.ECE_PRIOR_MAX / config.ECE_PRIOR_MIN) ** sal_pressure
    # Uncertainty (sd of log ECe): larger where spectra are blind (vegetated,
    # built) and where the evidence terms disagree.
    disagreement = np.nanstd(np.stack([np.nan_to_num(spectral_sal, nan=0.5), lowland, coastal]), axis=0)
    ece_log_sd = (0.35 + 0.35 * veg_frac + 0.25 * built_density + 0.4 * disagreement).astype("float32")

    # --- waterlogging / shallow saline groundwater -----------------------
    waterlog = np.clip(lowland * np.exp(-dist_sea_m / 2500.0) * 1.6, 0, 1).astype("float32")
    waterlog = np.where(intertidal, 1.0, waterlog)

    # --- loose sand (needed by dune specialists) -------------------------
    sand = (land & (wc == WC["bare"]) & (np.nan_to_num(dem, nan=0) > 3)
            & (np.nan_to_num(spectral_sal, nan=1) < 0.6) & (built_density < 0.2)).astype("float32")

    fp = {
        "valid": valid, "water": water, "land": land, "intertidal": intertidal,
        "mangrove_now": mangrove_now, "built": built, "irrigated": irrigated,
        "LST": lst, "HEAT_PCT": heat_pct, "NDVI_SUMMER": ndvi_s, "NDVI_WINTER": ndvi_w,
        "NDRE": idx["NDRE"], "NDMI": idx["NDMI"], "MNDWI": idx["MNDWI"], "NDBI": idx["NDBI"],
        "SI": idx["SI"], "ALBEDO": idx["ALBEDO"],
        "VEG_PERSIST": persist, "VEG_FRAC": veg_frac,
        "IRRIGATION_DEPENDENCE": irrigation_dependence,
        "BUILT_DENSITY": built_density, "DIST_SEA_M": dist_sea_m.astype("float32"),
        "DEM": dem, "SAL_PRESSURE": sal_pressure, "ECE": ece.astype("float32"),
        "ECE_LOG_SD": ece_log_sd, "WATERLOG": waterlog, "SAND": sand,
        "PH": np.full(lst.shape, config.REGIONAL_PH_PRIOR[0], dtype="float32"),
        "PH_SD": np.full(lst.shape, config.REGIONAL_PH_PRIOR[1], dtype="float32"),
    }
    return fp


FINGERPRINT_AXES = [
    ("LST", "Summer surface temp (°C)"),
    ("ECE", "Salinity estimate ECe (dS/m)"),
    ("WATERLOG", "Shallow saline groundwater risk"),
    ("NDMI", "Surface moisture (NDMI)"),
    ("VEG_PERSIST", "Summer vegetation persistence"),
    ("BUILT_DENSITY", "Urban density"),
]


def habitat_archetypes(fp: dict, k: int = 6, seed: int = config.RANDOM_SEED, sample: int = 60000):
    """Cluster land pixels into k micro-habitat archetypes using the fingerprint.

    Returns (labels array with 255 for non-land, centroids DataFrame).
    """
    import pandas as pd

    land = fp["land"] & np.isfinite(fp["LST"]) & np.isfinite(fp["ECE"])
    cols = [a for a, _ in FINGERPRINT_AXES]
    X = np.stack([np.log(fp[c][land]) if c == "ECE" else fp[c][land] for c in cols], axis=1)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z[fit_idx])
    lab = km.predict(Z)
    labels = np.full(land.shape, 255, dtype="uint8")
    labels[land] = lab
    cent = km.cluster_centers_ * sd + mu
    df = pd.DataFrame(cent, columns=cols)
    df["ECE"] = np.exp(df["ECE"])
    df["share_pct"] = np.bincount(lab, minlength=k) / len(lab) * 100
    df["name"] = [_name_archetype(r, df) for _, r in df.iterrows()]
    return labels, df


def _name_archetype(r, df):
    """Human-readable name from a centroid, relative to the other centroids."""
    parts = []
    hot = r["LST"] >= df["LST"].quantile(0.6)
    parts.append("Hot" if hot else "Cooler")
    if r["ECE"] >= 20:
        parts.append("saline")
    elif r["ECE"] <= 8:
        parts.append("low-salt")
    if r["WATERLOG"] >= 0.5:
        parts.append("wet coastal flat")
    elif r["BUILT_DENSITY"] >= 0.5:
        parts.append("dense urban")
    elif r["VEG_PERSIST"] >= 0.5:
        parts.append("irrigated green")
    else:
        parts.append("open ground")
    return " ".join(parts)
