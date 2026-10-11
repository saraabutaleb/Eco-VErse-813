"""Figures. Every function saves a PNG to results/ and returns the Figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch

from . import config

# Validated categorical order (fixed order, never cycled) + "Other" grey
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b9b8b2"
NONE = "#e9e8e4"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fde3d4", "#f4a07a", "#d95926", "#7a2a0b"])
SEQ_GREEN = LinearSegmentedColormap.from_list("seq_green", ["#e3f3e3", "#7cc47c", "#008300", "#004d00"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#86b6ef", "#e9e8e4", "#f4a07a", "#a83a10"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "figure.dpi": 72, "savefig.dpi": 110,
})


def save(fig, name, results_dir=config.RESULTS_DIR):
    p = Path(results_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    return p


def _map(ax, arr, title, cmap, vmin=None, vmax=None, label=None, mask=None):
    a = np.array(arr, dtype="float32")
    if mask is not None:
        a = np.where(mask, a, np.nan)
    cm = matplotlib.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cm = cm.copy(); cm.set_bad(NONE)
    im = ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    if label:
        cb.set_label(label, color=INK2)
    return im


def scalebar(ax, res_m, km=2):
    n = km * 1000 / res_m
    x0, y0 = 10, ax.get_ylim()[0] - 15
    ax.plot([x0, x0 + n], [y0, y0], color=INK, lw=3, solid_capstyle="butt")
    ax.text(x0 + n / 2, y0 - 8, f"{km} km", ha="center", va="bottom", color=INK, fontsize=8)


def true_colour(layers, gain=3.2):
    rgb = np.dstack([layers["B04"], layers["B03"], layers["B02"]])
    rgb = np.clip(rgb * gain, 0, 1) ** 0.85
    return np.nan_to_num(rgb, nan=0.9)


def fig_inputs(layers, fp, res_m=20):
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    ax[0, 0].imshow(true_colour(layers)); ax[0, 0].set_title("Sentinel-2 true colour (summer median)", loc="left")
    ax[0, 0].axis("off"); scalebar(ax[0, 0], res_m)
    _map(ax[0, 1], fp["LST"], "Landsat land surface temperature, summer (°C)", SEQ_ORANGE,
         np.nanpercentile(fp["LST"], 2), np.nanpercentile(fp["LST"], 98), "°C", mask=fp["land"])
    _map(ax[1, 0], fp["NDVI_SUMMER"], "NDVI, summer median", SEQ_GREEN, 0, 0.7, "NDVI", mask=fp["valid"])
    _map(ax[1, 1], fp["DEM"], "Elevation, Copernicus GLO-30 (m)", SEQ_BLUE, 0,
         np.nanpercentile(fp["DEM"], 98), "m", mask=fp["land"])
    fig.suptitle("Inputs: open satellite data over the area of interest", x=0.02, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_fingerprint(fp):
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    L = fp["land"]
    _map(ax[0, 0], fp["ECE"], "Salinity estimate ECe (dS/m, satellite prior)", SEQ_ORANGE, 3, 50, "dS/m", L)
    _map(ax[0, 1], fp["LST"], "Heat: summer surface temperature (°C)", SEQ_ORANGE,
         np.nanpercentile(fp["LST"][L], 2), np.nanpercentile(fp["LST"][L], 98), "°C", L)
    _map(ax[0, 2], fp["WATERLOG"], "Shallow saline groundwater risk", SEQ_BLUE, 0, 1, "0–1", L)
    _map(ax[1, 0], fp["VEG_PERSIST"], "Survival evidence: summer-green persistence", SEQ_GREEN, 0, 1,
         "share of clear dates", L)
    _map(ax[1, 1], fp["IRRIGATION_DEPENDENCE"], "Irrigation dependence of existing greenery",
         SEQ_BLUE, 0, 1, "0–1", L)
    _map(ax[1, 2], fp["BUILT_DENSITY"], "Urban density (roots vs. pipes risk)", SEQ_BLUE, 0, 1, "built share", L)
    fig.suptitle("The environmental fingerprint: six layers per 20 m pixel", x=0.02, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def categorical_map(ax, codes2d, labels, title, max_classes=8):
    """codes2d: int array, -1/255 = none. labels: list indexed by code."""
    codes = np.asarray(codes2d)
    valid = (codes >= 0) & (codes < len(labels))
    vals, counts = np.unique(codes[valid], return_counts=True)
    order = vals[np.argsort(-counts)]
    keep = order[:max_classes - 1] if len(order) > max_classes else order
    disp = np.full(codes.shape, -1, dtype=int)
    colours, legend = [], []
    for i, c in enumerate(keep):
        disp[codes == c] = i
        colours.append(CAT[i]); legend.append(labels[c])
    if len(order) > len(keep):
        disp[valid & ~np.isin(codes, keep)] = len(keep)
        colours.append(OTHER); legend.append("Other")
    cmap = ListedColormap([NONE] + colours)
    norm = BoundaryNorm(np.arange(-1.5, len(colours) + 0.5), cmap.N)
    ax.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Patch(color=c, label=l) for c, l in zip(colours, legend)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return legend


def fig_recommendation(res, fp, stability):
    names = list(res.plants["common_name"])
    hero2d = res.to_2d(res.hero.astype("float32"), fill=-1).astype(int)
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    a0 = fig.add_subplot(gs[0, 0])
    categorical_map(a0, hero2d, names, "Lowest-maintenance species that survives here")
    a1 = fig.add_subplot(gs[0, 1])
    _map(a1, res.to_2d(res.saving_mm), "Irrigation saved vs. Conocarpus + lawn (mm/yr)",
         SEQ_BLUE, 0, config.BASELINE_PLANT_FACTOR * config.ET0_MM_YR, "mm/yr")
    a2 = fig.add_subplot(gs[1, 0])
    _map(a2, res.to_2d(res.n_survivors.astype("float32")), "How many native species survive",
         SEQ_GREEN, 0, max(1, int(res.n_survivors.max())), "species")
    a3 = fig.add_subplot(gs[1, 1])
    _map(a3, res.to_2d(stability), "Recommendation stability (Monte Carlo)", SEQ_GREEN, 0, 1,
         "share of draws unchanged")
    fig.suptitle("Ecoverse recommendation", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_species_bars(summary):
    s = summary.copy()
    s["label"] = s["common_name"].str.replace(" (COMPARISON)", "", regex=False)
    s = s.sort_values("survives_pct_area")
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(s) + 1.5))
    y = np.arange(len(s))
    col = np.where(s["comparison"], CAT[1], CAT[0])
    ax.barh(y, s["survives_pct_area"], color=col, height=0.62)
    ax.set_yticks(y, s["label"]); ax.set_xlabel("% of evaluated land where it survives without routine irrigation")
    ax.set_xlim(0, 100)
    for yi, v in zip(y, s["survives_pct_area"]):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
    ax.legend(handles=[Patch(color=CAT[0], label="Native / adapted"),
                       Patch(color=CAT[1], label="Common urban planting (comparison)")],
              loc="lower right")
    ax.set_title("Survival across the area of interest", loc="left")
    fig.tight_layout()
    return fig


def fig_water(budget):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(budget))
    ax.barh(y + 0.18, budget["baseline_m3_yr"] / 1e6, height=0.34, color=CAT[1], label="Conocarpus + lawn")
    ax.barh(y - 0.18, budget["ecoverse_m3_yr"] / 1e6, height=0.34, color=CAT[0], label="Ecoverse guild")
    ax.set_yticks(y, [f"{s}\n{a:,.0f} ha" for s, a in zip(budget["scenario"], budget["area_ha"])])
    ax.set_xlabel("Irrigation water, million m³ per year")
    for yi, (b, e, p) in enumerate(zip(budget["baseline_m3_yr"], budget["ecoverse_m3_yr"], budget["saving_pct"])):
        ax.text(b / 1e6, yi + 0.18, f"  {b/1e6:.2f}", va="center", fontsize=8, color=INK2)
        ax.text(e / 1e6, yi - 0.18, f"  {e/1e6:.2f}  (−{p:.0f}%)", va="center", fontsize=8, color=INK2)
    ax.legend(loc="lower right"); ax.invert_yaxis()
    ax.set_title(f"Water budget (ET0 = {config.ET0_MM_YR:.0f} mm/yr, WUCOLS plant factors)", loc="left")
    fig.tight_layout()
    return fig


def fig_probes(fp, res, stability, probes):
    fig, ax = plt.subplots(figsize=(10, 9))
    _map(ax, res.to_2d(1 - stability), "Where a probe reading changes the most decisions",
         SEQ_ORANGE, 0, 1, "instability")
    if len(probes):
        ax.plot(probes["col"], probes["row"], "-", color=INK, lw=1.2, alpha=0.7)
        ax.scatter(probes["col"], probes["row"], s=90, c=CAT[0], edgecolors=SURFACE, linewidths=2, zorder=3)
        for _, r in probes.iterrows():
            ax.annotate(r["site_id"], (r["col"], r["row"]), xytext=(6, 6), textcoords="offset points",
                        fontsize=9, color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_archetypes(labels, cent):
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
    a0 = fig.add_subplot(gs[0, 0])
    names = [f"{i}: {n}" for i, n in enumerate(cent["name"])]
    categorical_map(a0, labels.astype(int), names, "Micro-habitat archetypes (k-means on fingerprint)")
    a1 = fig.add_subplot(gs[0, 1]); a1.axis("off")
    show = cent[["name", "LST", "ECE", "WATERLOG", "NDMI", "VEG_PERSIST", "BUILT_DENSITY", "share_pct"]].copy()
    show.columns = ["Archetype", "LST °C", "ECe dS/m", "Wet", "NDMI", "Green persist", "Urban", "Area %"]
    cell = [[str(i)] + [f"{v:.2f}" if isinstance(v, float) else v for v in r]
            for i, r in zip(range(len(show)), show.itertuples(index=False))]
    t = a1.table(cellText=cell, colLabels=["#"] + list(show.columns), loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.6)
    for (r, c), cl in t.get_celld().items():
        cl.set_edgecolor("#e0dfda")
        if r == 0:
            cl.set_text_props(fontweight="bold")
    fig.tight_layout()
    return fig
