"""Recommendation engine: from fingerprint to the lowest-maintenance plant
guild (canopy + shrub + groundcover) that can survive each pixel, plus the
irrigation water it would save versus business-as-usual planting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .plants import LAYERS, suitability_matrix, utility


def env_vectors(fp: dict, mask: np.ndarray, overrides: dict | None = None) -> dict:
    """Flatten the fingerprint over `mask` into the vectors the plant model uses."""
    env = {
        "ECE": fp["ECE"][mask],
        "HEAT_PCT": np.nan_to_num(fp["HEAT_PCT"][mask], nan=0.5),
        "WATERLOG": fp["WATERLOG"][mask],
        "INTERTIDAL": fp["intertidal"][mask],
        "SAND": fp["SAND"][mask],
        "PH": fp["PH"][mask],
        "BUILT_DENSITY": fp["BUILT_DENSITY"][mask],
    }
    if overrides:
        env.update(overrides)
    return {k: np.asarray(v) for k, v in env.items()}


@dataclass
class EngineResult:
    mask: np.ndarray            # 2-D bool: pixels evaluated (land)
    surv: np.ndarray            # S x N survival scores
    limiting: np.ndarray        # S x N index of the limiting factor
    factor_names: list
    guild: dict                 # layer -> N species index (-1 = none)
    hero: np.ndarray            # N: single best species index (-1 = none)
    hero_surv: np.ndarray       # N
    landscape_factor: np.ndarray  # N: water plant factor of the guild
    water_mm: np.ndarray        # N: irrigation need, mm/yr
    saving_mm: np.ndarray       # N: saving vs baseline, mm/yr
    n_survivors: np.ndarray     # N: number of native species that survive
    plants: pd.DataFrame

    def to_2d(self, vec, fill=np.nan, dtype="float32"):
        out = np.full(self.mask.shape, fill, dtype=dtype)
        out[self.mask] = vec
        return out


def run_engine(fp: dict, plants: pd.DataFrame, mask: np.ndarray | None = None,
               overrides: dict | None = None, et0: float = config.ET0_MM_YR) -> EngineResult:
    if mask is None:
        mask = fp["land"] & np.isfinite(fp["ECE"])
    env = env_vectors(fp, mask, overrides)
    surv, limiting, fnames = suitability_matrix(plants, env)
    util = utility(plants, surv, env["HEAT_PCT"])

    natives = plants["comparison"].to_numpy() == 0
    util_native = np.where(natives[:, None], util, -np.inf)
    layer_of = plants["layer"].to_numpy()

    guild = {}
    for layer in LAYERS:
        u = np.where((layer_of == layer)[:, None], util_native, -np.inf)
        best = np.argmax(u, axis=0)
        ok = np.isfinite(u[best, np.arange(u.shape[1])])
        guild[layer] = np.where(ok, best, -1).astype("int16")

    hero = np.argmax(util_native, axis=0)
    hero_ok = np.isfinite(util_native[hero, np.arange(util_native.shape[1])])
    hero = np.where(hero_ok, hero, -1).astype("int16")
    hero_surv = np.where(hero_ok, surv[np.maximum(hero, 0), np.arange(surv.shape[1])], 0.0)

    pf = plants["plant_factor"].to_numpy()
    num = np.zeros(mask.sum(), dtype="float32")
    den = np.zeros(mask.sum(), dtype="float32")
    for layer in LAYERS:
        g = guild[layer]
        present = g >= 0
        c = config.GUILD_COVER[layer]
        num += np.where(present, c * pf[np.maximum(g, 0)], 0)
        den += np.where(present, c, 0)
    lf = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan).astype("float32")
    water_mm = lf * et0
    saving_mm = (config.BASELINE_PLANT_FACTOR - lf) * et0
    n_surv = (surv[natives] >= config.SURVIVAL_THRESHOLD).sum(axis=0).astype("int16")

    return EngineResult(mask, surv, limiting, fnames, guild, hero, hero_surv.astype("float32"),
                        lf, water_mm.astype("float32"), saving_mm.astype("float32"), n_surv, plants)


def species_summary(res: EngineResult, weights: np.ndarray | None = None) -> pd.DataFrame:
    """Share of evaluated area where each species survives / is recommended."""
    p = res.plants
    N = res.surv.shape[1]
    w = np.ones(N) if weights is None else weights
    tot = w.sum()
    rows = []
    for i, r in p.iterrows():
        survives = res.surv[i] >= config.SURVIVAL_THRESHOLD
        chosen = np.zeros(N, dtype=bool)
        for layer in LAYERS:
            chosen |= res.guild[layer] == i
        lim = np.bincount(res.limiting[i][~survives], minlength=len(res.factor_names))
        rows.append({
            "species_id": r.species_id, "scientific_name": r.scientific_name,
            "common_name": r.common_name, "layer": r.layer,
            "comparison": bool(r.comparison), "plant_factor": r.plant_factor,
            "survives_pct_area": 100 * (w * survives).sum() / tot,
            "recommended_pct_area": 100 * (w * chosen).sum() / tot,
            "main_limit_where_fails": res.factor_names[int(np.argmax(lim))] if lim.sum() else "-",
        })
    return pd.DataFrame(rows).sort_values(["comparison", "recommended_pct_area"],
                                          ascending=[True, False]).reset_index(drop=True)


def guild_codes(res: EngineResult):
    """Encode each pixel's guild as a single integer id + lookup table."""
    keys = np.stack([res.guild[l] for l in LAYERS], axis=1)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    names = res.plants["common_name"].to_numpy()

    def nm(i):
        return names[i] if i >= 0 else "—"

    table = pd.DataFrame({
        "guild_id": np.arange(len(uniq)),
        "canopy": [nm(k[0]) for k in uniq],
        "shrub": [nm(k[1]) for k in uniq],
        "ground": [nm(k[2]) for k in uniq],
        "pixels": counts,
    })
    table["share_pct"] = 100 * table["pixels"] / table["pixels"].sum()
    return inv.ravel().astype("int32"), table.sort_values("pixels", ascending=False).reset_index(drop=True)


def water_budget(res: EngineResult, fp: dict, res_m: float = 20.0) -> pd.DataFrame:
    """Irrigation water (m3/yr) for two scenarios over the irrigated greenery
    that exists today (replacement scenario) and over all plantable land."""
    px_ha = (res_m * res_m) / 10000.0
    irrigated = fp["irrigated"][res.mask]
    urban = fp["BUILT_DENSITY"][res.mask] >= 0.25
    has = np.isfinite(res.water_mm)
    rows = []
    for label, sel in [("Urban irrigated greenery (replace)", irrigated & urban & has),
                       ("All irrigated greenery incl. farms", irrigated & has),
                       ("All evaluated land (green from scratch)", has)]:
        area_ha = sel.sum() * px_ha
        base = config.BASELINE_PLANT_FACTOR * config.ET0_MM_YR * 10 * area_ha
        eco = np.nansum(res.water_mm[sel]) * 10 * px_ha
        rows.append({"scenario": label, "area_ha": area_ha,
                     "baseline_m3_yr": base, "ecoverse_m3_yr": eco,
                     "saving_m3_yr": base - eco,
                     "saving_pct": 100 * (base - eco) / base if base else np.nan})
    return pd.DataFrame(rows)
