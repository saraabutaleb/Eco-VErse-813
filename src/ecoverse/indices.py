"""Spectral indices for multispectral (Sentinel-2) and hyperspectral (Tanager /
Satellite 813) data.

Every index here is an *indicator*, not a direct measurement. The docstrings
say what physical signal each one is a proxy for.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def nd(a, b):
    """Normalised difference (a - b) / (a + b), safe against divide-by-zero."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    return (a - b) / (a + b + EPS)


# ---------------------------------------------------------------------------
# Sentinel-2 (surface reflectance 0..1). Band names follow the L2A product.
# ---------------------------------------------------------------------------
def s2_indices(b: dict) -> dict:
    """Compute the multispectral indices used in the environmental fingerprint.

    Parameters
    ----------
    b : dict of 2-D arrays keyed by Sentinel-2 band name
        (B02, B03, B04, B05, B06, B07, B8A, B11, B12), reflectance 0..1.
    """
    out = {}
    out["NDVI"] = nd(b["B8A"], b["B04"])                       # green biomass
    out["NDRE"] = nd(b["B8A"], b["B05"])                       # chlorophyll / N stress
    out["NDMI"] = nd(b["B8A"], b["B11"])                       # canopy / surface moisture
    out["MNDWI"] = nd(b["B03"], b["B11"])                      # open water
    out["NDBI"] = nd(b["B11"], b["B8A"])                       # built-up / bare
    # Salinity indices from the soil-salinity literature (Khan et al. 2001;
    # Douaoui et al. 2006). Bright, white salt crusts raise visible reflectance.
    out["SI"] = np.sqrt(np.clip(b["B02"] * b["B04"], 0, None))
    out["SI2"] = np.sqrt(np.clip(b["B03"] ** 2 + b["B04"] ** 2, 0, None))
    # SWIR ratio: gypsum/halite-rich sabkha surfaces are bright in SWIR1 but
    # absorb near 2.2 um (B12) less than clay; a weak mineralogy hint only.
    out["SWIR_RATIO"] = nd(b["B11"], b["B12"])
    out["ALBEDO"] = np.mean(np.stack([b[k] for k in ("B02", "B03", "B04", "B8A", "B11", "B12")]), axis=0)
    return out


# ---------------------------------------------------------------------------
# Sentinel-2 spectral response, approximated as Gaussians (centre, FWHM in nm).
# Used to simulate what Sentinel-2 would have seen from a hyperspectral pixel.
# ---------------------------------------------------------------------------
S2_SRF = {
    "B02": (492.4, 66.0), "B03": (559.8, 36.0), "B04": (664.6, 31.0),
    "B05": (704.1, 15.0), "B06": (740.5, 15.0), "B07": (782.8, 20.0),
    "B8A": (864.7, 21.0), "B11": (1613.7, 91.0), "B12": (2202.4, 175.0),
}


def simulate_s2(cube: np.ndarray, wl_nm: np.ndarray) -> dict:
    """Convolve a hyperspectral cube (bands, ...) to Sentinel-2-like bands."""
    out = {}
    finite = np.isfinite(cube)
    filled = np.where(finite, cube, 0.0)
    for name, (c, fwhm) in S2_SRF.items():
        sigma = fwhm / 2.3548
        w = np.exp(-0.5 * ((wl_nm - c) / sigma) ** 2)
        w[w < 1e-3] = 0.0
        if w.sum() == 0:
            raise ValueError(f"No hyperspectral bands overlap Sentinel-2 {name}")
        num = np.tensordot(w, filled, axes=(0, 0))
        den = np.tensordot(w, finite.astype("float32"), axes=(0, 0))
        # pixel valid only if >= 60 % of the response-weighted bands are finite
        out[name] = np.where(den >= 0.6 * w.sum(), num / np.maximum(den, 1e-12), np.nan).astype("float32")
    return out


# ---------------------------------------------------------------------------
# Hyperspectral features: things a handful of broad bands cannot resolve.
# ---------------------------------------------------------------------------
def _band(cube, wl, target, half_width=6.0):
    """Mean reflectance of bands within +/- half_width nm of target."""
    sel = np.abs(wl - target) <= half_width
    if not sel.any():
        sel = np.zeros_like(wl, dtype=bool)
        sel[np.argmin(np.abs(wl - target))] = True
    return np.nanmean(cube[sel], axis=0)


def band_depth(cube, wl, left, centre_lo, centre_hi, right):
    """Continuum-removed absorption depth.

    A straight 'continuum' line is drawn between the shoulders at `left` and
    `right` nm; the depth is 1 - R_min / R_continuum inside the absorption
    window [centre_lo, centre_hi]. 0 = no absorption feature.
    """
    rl, rr = _band(cube, wl, left), _band(cube, wl, right)
    sel = (wl >= centre_lo) & (wl <= centre_hi)
    if not sel.any():
        raise ValueError("absorption window has no bands")
    depths = []
    for i in np.where(sel)[0]:
        t = (wl[i] - left) / (right - left)
        cont = rl + t * (rr - rl)
        depths.append(1.0 - cube[i] / (cont + EPS))
    return np.nanmax(np.stack(depths), axis=0).astype("float32")


def red_edge_position(cube, wl):
    """Red-edge inflection point (nm) by 4-point linear interpolation
    (Guyot & Baret 1988). Shifts to shorter wavelengths under chlorophyll
    stress (heat, salt, drought) before NDVI changes."""
    r670, r700 = _band(cube, wl, 670), _band(cube, wl, 700)
    r740, r780 = _band(cube, wl, 740), _band(cube, wl, 780)
    rre = (r670 + r780) / 2.0
    rep = 700.0 + 40.0 * (rre - r700) / (r740 - r700 + EPS)
    return np.clip(rep, 680, 760).astype("float32")


HYPERSPECTRAL_FEATURES = {
    # name: (description, function)
    "REP": "Red-edge position (nm): early chlorophyll stress",
    "CANOPY_WATER_1200": "1200 nm liquid-water absorption depth: plant water status",
    "GYPSUM_1750": "1750 nm absorption depth: gypsum/sulfate (sabkha) soils",
    "CLAY_2200": "2200 nm Al-OH absorption depth: clay content",
    "CARBONATE_2340": "2340 nm absorption depth: calcareous (high-pH) soils",
}


def hyperspectral_features(cube: np.ndarray, wl_nm: np.ndarray) -> dict:
    """Narrow-band features that need a hyperspectral sensor."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # masked (all-NaN) pixels
        return {
            "REP": red_edge_position(cube, wl_nm),
            "CANOPY_WATER_1200": band_depth(cube, wl_nm, 1090, 1170, 1220, 1265),
            "GYPSUM_1750": band_depth(cube, wl_nm, 1690, 1735, 1765, 1790),
            "CLAY_2200": band_depth(cube, wl_nm, 2120, 2185, 2225, 2265),
            "CARBONATE_2340": band_depth(cube, wl_nm, 2270, 2320, 2355, 2395),
        }


def bad_band_mask(wl_nm: np.ndarray) -> np.ndarray:
    """True for bands to drop: atmospheric water-vapour troughs and the noisy
    SWIR detector edge (same rule as the hackathon starter notebooks)."""
    wv = ((wl_nm >= 1350) & (wl_nm <= 1450)) | ((wl_nm >= 1800) & (wl_nm <= 1950))
    return wv | (wl_nm > 2450) | (wl_nm < 400)
