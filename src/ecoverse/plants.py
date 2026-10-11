"""Plant knowledge base and per-pixel survival (suitability) model.

Suitability is a product of transparent 0..1 response curves, one per
environmental factor. A product means one fatal factor (e.g. salinity far
beyond a species' survival limit) cannot be compensated by good scores
elsewhere, which matches how plants actually fail.

"Survive naturally" here means: survive after establishment with no more than
occasional deep watering. Species with low drought tolerance are therefore
penalised, because in Bahrain they only persist under routine irrigation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

DROUGHT_FACTOR = {1: 0.3, 2: 0.5, 3: 0.9, 4: 1.0}
LAYERS = ["canopy", "shrub", "ground"]


def load_plants(path=config.PLANTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["comparison"] = df["comparison"].astype(int)
    return df.reset_index(drop=True)


def ramp_down(x, ok, zero):
    """1 when x <= ok, falling linearly to 0 at x >= zero."""
    return np.clip((zero - x) / max(zero - ok, 1e-6), 0.0, 1.0)


def factor_curves(row: pd.Series, env: dict) -> dict:
    """Per-factor suitability arrays for one species.

    env keys (1-D arrays over land pixels): ECE, HEAT_PCT, WATERLOG,
    INTERTIDAL (bool), SAND, PH, BUILT_DENSITY.
    """
    f = {}
    f["salinity"] = ramp_down(env["ECE"], row.ece_comfort_dSm, row.ece_survival_dSm)

    # Heat: every row in the KB survives Bahrain air temperatures; what differs
    # is how badly the hottest (sealed, unshaded) surfaces hurt less-adapted
    # species.
    excess = np.clip((env["HEAT_PCT"] - 0.5) * 2, 0, 1)
    f["heat"] = 1.0 - {4: 0.0, 3: 0.4, 2: 0.7, 1: 0.9}[int(row.heat_tolerance)] * excess

    if row.habitat == "tidal":
        f["drought"] = np.ones_like(env["ECE"])
    else:
        f["drought"] = np.full_like(env["ECE"], DROUGHT_FACTOR[int(row.drought_tolerance)])
        if row.habitat == "wet_saline":
            # wet-sabkha specialists drink the shallow groundwater itself
            f["drought"] = np.maximum(f["drought"], np.clip(env["WATERLOG"], 0, 1))

    f["waterlog"] = 1.0 - 0.9 * env["WATERLOG"] * (1.0 - row.waterlog_tolerance)

    tidal = env["INTERTIDAL"].astype("float32")
    if row.habitat == "tidal":
        f["habitat"] = tidal
    elif row.habitat == "wet_saline":
        f["habitat"] = np.clip(np.maximum(env["WATERLOG"] / 0.4, tidal), 0, 1)
    elif row.habitat == "sand":
        f["habitat"] = 0.2 + 0.8 * env["SAND"]
    else:
        # Dryland species cannot be planted in the intertidal zone unless
        # they tolerate waterlogging (salt-marsh edge species).
        f["habitat"] = 1.0 - tidal * (1.0 - row.waterlog_tolerance)

    f["ph"] = ramp_down(env["PH"], row.ph_max, row.ph_max + 1.0) * \
        np.clip(1.0 - (row.ph_min - env["PH"]), 0, 1)

    root_pen = {0: 0.0, 1: 0.2, 2: 0.7}[int(row.root_risk)]
    f["infrastructure"] = 1.0 - root_pen * env["BUILT_DENSITY"]
    return {k: np.asarray(v, dtype="float32") for k, v in f.items()}


def suitability_matrix(plants: pd.DataFrame, env: dict):
    """Return (S x N survival matrix, dict of limiting factor index S x N)."""
    S = len(plants)
    N = env["ECE"].shape[0]
    surv = np.empty((S, N), dtype="float32")
    limiting = np.empty((S, N), dtype="uint8")
    names = None
    for i, row in plants.iterrows():
        curves = factor_curves(row, env)
        names = list(curves)
        stack = np.stack([curves[n] for n in names])
        surv[i] = np.prod(stack, axis=0)
        limiting[i] = np.argmin(stack, axis=0)
    return surv, limiting, names


def utility(plants: pd.DataFrame, surv: np.ndarray, heat_pct: np.ndarray) -> np.ndarray:
    """Recommendation utility: survive first, then minimise water and upkeep,
    and reward shade where it is hottest (urban cooling)."""
    pf = plants["plant_factor"].to_numpy()[:, None]
    maint = plants["maintenance"].to_numpy()[:, None]
    shade = (plants["shade"].to_numpy()[:, None] / 3.0)
    heat_need = np.nan_to_num(heat_pct, nan=0.5)[None, :]
    u = surv - 0.6 * pf - 0.08 * (maint - 1) + 0.15 * shade * heat_need
    u = np.where(surv >= config.SURVIVAL_THRESHOLD, u, -np.inf)
    return u.astype("float32")
