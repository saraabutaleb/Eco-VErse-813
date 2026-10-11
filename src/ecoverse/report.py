"""Site report cards: the product a municipality planner actually reads."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .plants import LAYERS
from .viz import CAT, INK, INK2, MUTED, SURFACE


def site_report(fp, res, row, col):
    """Plain-language summary of one pixel."""
    idx = np.flatnonzero(res.mask.ravel() == True)  # noqa: E712
    flat = row * res.mask.shape[1] + col
    k = np.searchsorted(idx, flat)
    if k >= idx.size or idx[k] != flat:
        return None
    p = res.plants
    guild = {l: (p.loc[res.guild[l][k], "common_name"] if res.guild[l][k] >= 0 else "none survives")
             for l in LAYERS}
    surv = res.surv[:, k]
    natives = p["comparison"] == 0
    top = p[natives].assign(s=surv[natives.to_numpy()]).sort_values("s", ascending=False).head(5)
    conoc = p.index[p["species_id"] == "conocarpus"]
    return {
        "row": int(row), "col": int(col),
        "lst_c": float(fp["LST"][row, col]), "ece": float(fp["ECE"][row, col]),
        "ece_range": (float(fp["ECE"][row, col] * np.exp(-fp["ECE_LOG_SD"][row, col])),
                      float(fp["ECE"][row, col] * np.exp(fp["ECE_LOG_SD"][row, col]))),
        "waterlog": float(fp["WATERLOG"][row, col]), "built": float(fp["BUILT_DENSITY"][row, col]),
        "persist": float(fp["VEG_PERSIST"][row, col]), "irrigated": bool(fp["irrigated"][row, col]),
        "intertidal": bool(fp["intertidal"][row, col]),
        "guild": guild, "water_mm": float(res.water_mm[k]), "saving_mm": float(res.saving_mm[k]),
        "top": list(zip(top["common_name"], top["s"].round(2))),
        "conocarpus_surv": float(surv[conoc[0]]) if len(conoc) else np.nan,
    }


def fig_site_cards(reports, titles):
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 6.4))
    axes = np.atleast_1d(axes)
    for ax, r, t in zip(axes, reports, titles):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f3f2ee", zorder=0))
        y = 0.95
        ax.text(0.05, y, t, fontsize=12, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
        y -= 0.09
        facts = [
            ("Summer surface", f"{r['lst_c']:.1f} °C"),
            ("Salinity (prior)", f"{r['ece']:.0f} dS/m  ({r['ece_range'][0]:.0f}–{r['ece_range'][1]:.0f})"),
            ("Saline groundwater", f"{r['waterlog']:.2f}"),
            ("Urban density", f"{r['built']:.2f}"),
            ("Greenery now", "irrigated" if r["irrigated"] else ("tidal" if r["intertidal"] else "none / sparse")),
        ]
        for k, v in facts:
            ax.text(0.05, y, k, fontsize=9, color=INK2, transform=ax.transAxes, va="top")
            ax.text(0.95, y, v, fontsize=9, color=INK, transform=ax.transAxes, va="top", ha="right")
            y -= 0.055
        y -= 0.02
        ax.text(0.05, y, "Recommended guild", fontsize=10, fontweight="bold", color=CAT[0],
                transform=ax.transAxes, va="top"); y -= 0.065
        for layer in LAYERS:
            ax.text(0.05, y, layer.capitalize(), fontsize=9, color=INK2, transform=ax.transAxes, va="top")
            ax.text(0.95, y, r["guild"][layer], fontsize=9, color=INK, transform=ax.transAxes, va="top",
                    ha="right", wrap=True)
            y -= 0.055
        y -= 0.02
        if np.isfinite(r["water_mm"]):
            ax.text(0.05, y, f"Water need  {r['water_mm']:.0f} mm/yr", fontsize=9.5, color=INK,
                    transform=ax.transAxes, va="top"); y -= 0.05
            ax.text(0.05, y, f"Saves {r['saving_mm']:.0f} mm/yr  "
                    f"(= {r['saving_mm']*10:.0f} m³ per ha)", fontsize=9.5, color=CAT[0],
                    fontweight="bold", transform=ax.transAxes, va="top"); y -= 0.06
        else:
            ax.text(0.05, y, textwrap.fill("No species survives without irrigation: leave as natural "
                    "sabkha or use engineered shade instead of planting.", 48), fontsize=9.5,
                    color=CAT[1], fontweight="bold", transform=ax.transAxes, va="top"); y -= 0.11
        ax.text(0.05, y, f"Conocarpus survival score: {r['conocarpus_surv']:.2f}", fontsize=9,
                color=MUTED, transform=ax.transAxes, va="top")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def pick_example_sites(fp, res, seed=config.RANDOM_SEED):
    """Pick contrasting, representative sites: hottest dense urban, most
    saline open ground, irrigated urban greenery, and intertidal edge."""
    m = res.mask
    def best(score, cond):
        s = np.where(cond & m, score, -np.inf)
        if not np.isfinite(s).any():
            return None
        return np.unravel_index(int(np.nanargmax(s)), s.shape)
    lst = np.nan_to_num(fp["LST"], nan=-99)
    picks = [
        ("Hot dense urban block", best(lst, fp["BUILT_DENSITY"] > 0.7)),
        ("Saline open ground", best(fp["ECE"], ~fp["intertidal"] & (fp["BUILT_DENSITY"] < 0.3) & ~fp["irrigated"])),
        ("Irrigated urban greenery", best(fp["IRRIGATION_DEPENDENCE"] + 0.3 * fp["BUILT_DENSITY"], fp["irrigated"])),
        ("Coastal / intertidal edge", best(np.nan_to_num(fp["NDVI_SUMMER"], nan=-1), fp["intertidal"])),
    ]
    return [(t, p) for t, p in picks if p is not None]
