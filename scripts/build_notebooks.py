#!/usr/bin/env python
"""Generates the notebooks from this single source of truth (keeps them diff-able).
Run once after editing; then execute them with scripts/run_all.py."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


SETUP = r'''
import os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN medians at scene edges

from ecoverse import config, io, fingerprint, plants, recommend, uncertainty, hyperspectral, viz, report
np.random.seed(config.RANDOM_SEED)
pd.set_option("display.width", 160, "display.max_columns", 20, "display.precision", 3)
print("Repository root:", ROOT.name)
print("Results folder :", config.RESULTS_DIR.relative_to(ROOT) if config.RESULTS_DIR.is_relative_to(ROOT) else config.RESULTS_DIR)
'''


def nb_main():
    c = []
    c.append(md(r'''
# Ecoverse 813 — main analysis
**Theme 05 · Sustainable Urban Planning & Smart Cities** · Arab Youth Space Hackathon: 813 Challenge

**Problem statement.** *We want to **identify** the lowest-maintenance native plant combination that can survive each 20 m patch of
**northern-central Bahrain** (Manama – Tubli Bay – Isa Town – Sitra) under **summer 2025** conditions, so that **municipal
landscaping planners** can decide **what to plant (or replant) where**, instead of keeping unsuitable trees alive with irrigation.*

**What this notebook does (in order)**
1. Loads the committed, clipped open-data stack (Sentinel-2, Landsat thermal, Copernicus DEM, ESA WorldCover).
2. Pulls in the *hyperspectral calibration* learned on open Planet Tanager Gulf scenes (notebook 01).
3. Builds the **environmental fingerprint**: heat, salinity, saline groundwater, moisture, survival evidence, irrigation dependence, urban density.
4. Scores 26 species (21 native/adapted + 5 common urban plantings) for survival, then picks the **lowest-water, lowest-upkeep guild** (canopy + shrub + groundcover).
5. Estimates **irrigation water saved** vs. today's typical Conocarpus + lawn planting.
6. Runs a **Monte Carlo robustness test** and plans **where two students with a low-cost probe should measure next**.
7. Validates what can be validated, and states limitations.

Runtime ≈ 2–4 min on a laptop, no GPU. Inputs: `data/sample_input/`. Outputs: `results/`.
'''))
    c.append(code(SETUP))
    c.append(md("## 1 · Load the input stack and its provenance"))
    c.append(code(r'''
if not config.STACK_TIF.exists():
    raise FileNotFoundError(f"{config.STACK_TIF} missing - run `python scripts/fetch_data.py` (see README)")
layers, profile = io.read_stack(config.STACK_TIF)
meta = json.loads(config.STACK_META.read_text())
if meta.get("synthetic") and os.environ.get("ECOVERSE_ALLOW_SYNTHETIC") != "1":
    raise RuntimeError("Synthetic test fixture detected - refusing to produce results from it.")
RES = abs(profile["transform"].a)
print(f"AOI   : {meta['aoi']['name']}")
print(f"Grid  : {profile['width']} x {profile['height']} px at {RES:.0f} m, {profile['crs']}")
print(f"Layers: {', '.join(layers)}")
prov = pd.DataFrame([{"source": k, "collection": v.get("collection"), "period": v.get("period", "-"),
                      "scenes": v.get("n_scenes", len(v.get("item_ids", [])))} for k, v in meta["sources"].items()])
prov
'''))
    c.append(md("### Quality control\nCloud/shadow pixels were removed scene by scene with the Sentinel-2 SCL mask and the Landsat QA_PIXEL bits before compositing (see `src/ecoverse/fetch.py`). Here we check what is left."))
    c.append(code(r'''
nobs = layers["S2_NOBS"]
qc = {
    "Sentinel-2 valid pixels (%)": 100 * np.isfinite(layers["B04"]).mean(),
    "Clear summer observations per pixel (median)": float(np.nanmedian(nobs)),
    "Clear summer observations per pixel (5th pct)": float(np.nanpercentile(nobs, 5)),
    "Landsat LST coverage (%)": 100 * np.isfinite(layers["LST_SUMMER"]).mean(),
    "Summer LST range, 2-98 pct (°C)": f"{np.nanpercentile(layers['LST_SUMMER'],2):.1f} – {np.nanpercentile(layers['LST_SUMMER'],98):.1f}",
    "DEM range (m)": f"{np.nanmin(layers['DEM']):.1f} – {np.nanmax(layers['DEM']):.1f}",
}
pd.Series(qc, name="value").to_frame()
'''))
    c.append(md(r'''
## 2 · Hyperspectral calibration ("hyperspectral teaches multispectral")
Bahrain is not in Tanager's open archive and Satellite 813 data is only available in incubation. Notebook 01 learns, on open Tanager scenes
of Gulf coastal sabkha and an arid Gulf city, which narrow-band signals Sentinel-2 can and cannot reproduce. We only use a transferred
signal in Bahrain if spatially-blocked cross-validation shows Sentinel-2 recovers it well (R² ≥ 0.4). Otherwise the fingerprint falls back
to a Sentinel-2-only SWIR proxy, and the gap is reported as exactly what Satellite 813 must fill.
'''))
    c.append(code(r'''
gypsum_proxy, transfer = None, None
try:
    tdf, tmeta, _ = hyperspectral.load_tanager_tables()
    transfer = hyperspectral.evaluate_transfer(tdf)
    display(transfer.round(3))
    r2 = transfer.set_index("target").loc["GYPSUM_1750", "r2_boosted"]
    if r2 >= 0.4:
        model = hyperspectral.fit_transfer_model(tdf, "GYPSUM_1750")
        gypsum_proxy = hyperspectral.predict_on_s2(model, {k: layers[k] for k in hyperspectral.S2_NAMES})
        print(f"Gypsum (1750 nm) signal recovered from Sentinel-2 with CV R² = {r2:.2f} -> used in the salinity layer")
    else:
        print(f"Gypsum CV R² = {r2:.2f} < 0.4 -> NOT transferred; Sentinel-2 SWIR ratio used instead (gap for Satellite 813)")
except FileNotFoundError as e:
    print("No Tanager calibration data:", e)
'''))
    c.append(md("## 3 · The environmental fingerprint"))
    c.append(code(r'''
fp = fingerprint.build_fingerprint(layers, res_m=RES, gypsum_proxy=gypsum_proxy)
fig = viz.fig_inputs(layers, fp, RES); viz.save(fig, "01_inputs.png"); plt.show()
fig = viz.fig_fingerprint(fp); viz.save(fig, "02_fingerprint.png"); plt.show()
L = fp["land"]
print(f"Land pixels: {L.sum():,} ({L.sum()*RES*RES/1e6:.1f} km²) · intertidal: {fp['intertidal'].sum():,} · "
      f"irrigated greenery today: {fp['irrigated'].sum():,} px ({fp['irrigated'].sum()*RES*RES/1e4:.0f} ha)")
'''))
    c.append(md("### Micro-habitat archetypes\nClustering the fingerprint shows the AOI is a patchwork of a few recurring habitats — the unit a planner can reason about."))
    c.append(code(r'''
arche, cent = fingerprint.habitat_archetypes(fp, k=6)
fig = viz.fig_archetypes(arche, cent); viz.save(fig, "03_archetypes.png"); plt.show()
cent.round(2)
'''))
    c.append(md(r'''
## 4 · Survival model and recommendation
Each species gets a 0–1 response curve per factor (salinity vs. its literature ECe comfort/survival limits, heat, drought, waterlogging,
habitat, pH, root-vs-infrastructure risk). Survival = product of curves, so one fatal factor cannot be averaged away.
A species *survives* at ≥ 0.6. Among survivors, the engine minimises water (WUCOLS plant factor) and upkeep, and rewards shade where it is hottest.
'''))
    c.append(code(r'''
kb = plants.load_plants()
res = recommend.run_engine(fp, kb)
summary = recommend.species_summary(res)
summary.to_csv(config.RESULTS_DIR / "species_summary.csv", index=False)
fig = viz.fig_species_bars(summary); viz.save(fig, "04_species_survival.png"); plt.show()
summary[["common_name", "layer", "plant_factor", "survives_pct_area", "recommended_pct_area", "main_limit_where_fails"]].round(1)
'''))
    c.append(code(r'''
guild_id, guilds = recommend.guild_codes(res)
guilds.head(10).round(1)
'''))
    c.append(md("## 5 · Robustness: Monte Carlo over every uncertain input\nSalinity (per-pixel log-normal + an AOI-wide calibration shift), heat, waterlogging, pH and ET0 are perturbed; we count how often each pixel keeps the same recommendation."))
    c.append(code(r'''
run = config.RunConfig()
stability, draws = uncertainty.monte_carlo(fp, kb, res, n=run.n_monte_carlo, seed=run.seed)
print(f"Median stability {np.median(stability):.2f}; {100*np.mean(stability>=0.8):.0f}% of land pixels keep the same species in ≥80% of draws")
fig = viz.fig_recommendation(res, fp, stability); viz.save(fig, "05_recommendation.png"); plt.show()
'''))
    c.append(md("## 6 · Water: what this saves"))
    c.append(code(r'''
budget = recommend.water_budget(res, fp, RES)
budget.to_csv(config.RESULTS_DIR / "water_budget.csv", index=False)
fig = viz.fig_water(budget); viz.save(fig, "06_water_budget.png"); plt.show()
lo, hi = np.percentile(draws["mean_saving_mm"], [5, 95])
print(f"Mean saving per irrigated m²: {np.nanmean(res.saving_mm):.0f} mm/yr (90% MC interval {lo:.0f}–{hi:.0f})")
budget.round(1)
'''))
    c.append(md("## 7 · Site report cards — the product a planner reads"))
    c.append(code(r'''
sites = report.pick_example_sites(fp, res)
reps = [report.site_report(fp, res, int(p[0]), int(p[1])) for _, p in sites]
fig = report.fig_site_cards(reps, [t for t, _ in sites]); viz.save(fig, "07_site_cards.png"); plt.show()
'''))
    c.append(md(r'''
## 8 · Closing the loop with ground probes
Satellites see the surface; roots feel the soil. A ~USD 100 kit (EC 1:5 probe, pH probe, capacitive moisture, RS-485 NPK, IR thermometer,
phone GPS) measures what orbit cannot. The engine picks the sites where a reading would **change the most decisions**, and orders them
into a one-afternoon route.
'''))
    c.append(code(r'''
probes = uncertainty.select_probe_sites(fp, res, stability, profile, n_sites=run.n_probe_sites,
                                        min_spacing_m=run.probe_min_spacing_m, res_m=RES)
probes.to_csv(config.RESULTS_DIR / "probe_plan.csv", index=False)
fig = viz.fig_probes(fp, res, stability, probes); viz.save(fig, "08_probe_plan.png"); plt.show()
probes
'''))
    c.append(md(r'''
### What-if: how one probe reading updates the map
**The readings below are hypothetical** (no field campaign has happened yet). They show the mechanism: a measured ECe and pH at the
first probe site are blended into the satellite prior (Gaussian-kernel residual correction, 600 m range), uncertainty shrinks around
the probe, and recommendations nearby are re-computed. Real readings go in `data/ground/ground_readings.csv` (template provided).
'''))
    c.append(code(r'''
gpath = config.GROUND_DIR / "ground_readings.csv"
real = pd.read_csv(gpath) if gpath.exists() else pd.DataFrame()
if len(real):
    readings = uncertainty.prepare_readings(real, profile)
    print(f"Using {len(readings)} REAL ground readings from {gpath}")
else:
    p0 = probes.iloc[0]
    prior = float(fp["ECE"][int(p0.row), int(p0.col)])
    readings = pd.DataFrame([{"row": p0.row, "col": p0.col, "ece_dSm": prior * 3.0, "ph": 8.6}])
    print(f"HYPOTHETICAL reading at {p0.site_id}: ECe {prior:.1f} -> {prior*3:.1f} dS/m, pH 8.6")
fp2 = uncertainty.fuse_ground_readings(fp, readings, res_m=RES)
res2 = recommend.run_engine(fp2, kb, mask=res.mask)
changed = res2.hero != res.hero
print(f"Recommendations changed on {changed.sum():,} pixels ({changed.sum()*RES*RES/1e4:.1f} ha) around the probe")
r0, c0 = int(readings.iloc[0].row), int(readings.iloc[0].col)
win = np.s_[max(0, r0-60):r0+60, max(0, c0-60):c0+60]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
names = list(kb["common_name"])
viz.categorical_map(ax[0], res.to_2d(res.hero.astype("float32"), -1).astype(int)[win], names, "Before probe")
viz.categorical_map(ax[1], res2.to_2d(res2.hero.astype("float32"), -1).astype(int)[win], names, "After probe (what-if)")
fig.tight_layout(); viz.save(fig, "09_probe_whatif.png"); plt.show()
'''))
    c.append(md(r'''
## 9 · Validation — what we can and cannot check today
There is no ground-truth map of "where each species survives" for Bahrain, so a single accuracy number would be misleading. Instead:
- **Independent reference check:** ESA WorldCover maps mangroves independently of our engine. Grey mangrove (*Avicennia marina*) should be a
  survivor there and nowhere inland.
- **Physical consistency:** greener pixels should be cooler in summer (negative LST–NDVI correlation).
- **Robustness:** Monte Carlo stability (section 5).
- **Regulatory sanity check:** Bahrain banned new Conocarpus planting in 2022 for infrastructure damage and water use; the model, which knows
  nothing about the ban, should rank it poorly in dense urban fabric.
'''))
    c.append(code(r'''
from scipy.stats import spearmanr
wc = np.nan_to_num(layers["WORLDCOVER"]).astype(int)
qi = kb.index[kb.species_id == "qurm"][0]
qsurv = res.to_2d(res.surv[qi], 0.0)
mang = (wc == 95) & res.mask
val = {}
if mang.sum():
    val["WorldCover mangrove pixels where grey mangrove survives (%)"] = 100 * (qsurv[mang] >= config.SURVIVAL_THRESHOLD).mean()
inland = res.mask & (fp["DIST_SEA_M"] > 1000)
val["Inland (>1 km from sea) pixels where grey mangrove is recommended (%)"] = 100 * (res.to_2d(res.hero.astype('float32'), -1)[inland] == qi).mean()
ok = L & np.isfinite(fp["LST"]) & np.isfinite(fp["NDVI_SUMMER"])
rho = spearmanr(fp["NDVI_SUMMER"][ok], fp["LST"][ok]).statistic
val["Spearman rho, summer NDVI vs LST (expect < 0)"] = rho
ci = kb.index[kb.species_id == "conocarpus"][0]
dense = (fp["BUILT_DENSITY"][res.mask] > 0.6)
val["Conocarpus survives without routine irrigation in dense urban pixels (%)"] = 100 * (res.surv[ci][dense] >= config.SURVIVAL_THRESHOLD).mean()
val["Pixels with ≥1 surviving native species (%)"] = 100 * (res.n_survivors > 0).mean()
validation = pd.Series(val, name="value").round(2).to_frame(); validation
'''))
    c.append(md("## 10 · Save outputs"))
    c.append(code(r'''
out = config.RESULTS_DIR
io.write_class_raster(out / "recommended_species.tif", res.to_2d(res.hero.astype("float32"), 255).astype("uint8"),
                      profile, labels=dict(enumerate(kb["common_name"])))
io.write_stack(out / "fingerprint_and_scores.tif", {
    "ECE_dSm": fp["ECE"], "LST_C": fp["LST"], "WATERLOG": fp["WATERLOG"], "VEG_PERSIST": fp["VEG_PERSIST"],
    "BUILT_DENSITY": fp["BUILT_DENSITY"], "WATER_NEED_mm": res.to_2d(res.water_mm), "SAVING_mm": res.to_2d(res.saving_mm),
    "STABILITY": res.to_2d(stability)}, profile["transform"], profile["crs"])   # int16 + scale tags
guilds.to_csv(out / "guilds.csv", index=False)
validation.to_csv(out / "validation.csv")
if transfer is not None:
    transfer.to_csv(out / "hyperspectral_transfer_cv.csv", index=False)
top = summary[~summary.comparison].head(5)
headline = {
    "aoi": meta["aoi"]["name"], "land_km2": float(L.sum() * RES * RES / 1e6),
    "irrigated_greenery_ha": float(fp["irrigated"].sum() * RES * RES / 1e4),
    "n_species": int(len(kb)), "n_native": int((kb.comparison == 0).sum()),
    "pct_land_with_native_survivor": float(100 * (res.n_survivors > 0).mean()),
    "top_species": top[["common_name", "recommended_pct_area"]].round(1).values.tolist(),
    "mean_saving_mm": float(np.nanmean(res.saving_mm)),
    "saving_mm_ci90": [float(lo), float(hi)],
    "water_budget": budget.round(1).to_dict(orient="records"),
    "median_stability": float(np.median(stability)),
    "pct_stable_80": float(100 * np.mean(stability >= 0.8)),
    "validation": {k: float(v) for k, v in val.items()},
    "transfer_cv": None if transfer is None else transfer.round(3).to_dict(orient="records"),
    "gypsum_transferred": gypsum_proxy is not None,
    "n_probe_sites": int(len(probes)),
    "data_period_summer": meta["sources"]["sentinel2_summer"]["period"],
    "n_s2_scenes_summer": meta["sources"]["sentinel2_summer"]["n_scenes"],
}
(out / "headline.json").write_text(json.dumps(headline, indent=2))
print(json.dumps({k: headline[k] for k in ["land_km2", "irrigated_greenery_ha", "pct_land_with_native_survivor",
                                            "mean_saving_mm", "median_stability"]}, indent=1))
sorted(p.name for p in out.iterdir())
'''))
    c.append(md(r'''
## 11 · Limitations (read before using the maps)
- **Salinity is a satellite prior, not a measurement.** It blends spectral salinity indices, elevation and distance to sea, mapped to ECe
  with literature anchors (FAO AQUASTAT Bahrain). Under vegetation and buildings the spectra say little; uncertainty is mapped and fed to Monte Carlo.
- **Species tolerances** come from published trials and regional floras; several are expert estimates (marked `E` in the trait table) and need
  review by a Gulf horticulturist. Hydroponic NaCl tolerance overstates field tolerance, so comfort limits are deliberately conservative.
- **pH and nutrients are not observable from orbit**; a regional calcareous-soil prior is used until probes measure them.
- **20 m pixels mix** roads, roofs and verges. Recommendations are for the planting conditions typical of the pixel, not a specific tree pit.
- **Hyperspectral calibration uses Gulf analogue scenes**, not Bahrain; transfer across sites and between Tanager-simulated and real Sentinel-2
  reflectance adds error. Satellite 813 coverage of Bahrain would remove this step.
- **Irrigated-greenery detection** assumes summer-green, non-tidal vegetation is irrigated (Bahrain summer rainfall ≈ 0); date-palm farms are
  included in one water scenario and excluded from the urban replacement scenario.
- **Water savings** use WUCOLS-style plant factors × ET0 (1,850 mm/yr) and assume established plants; establishment-year irrigation is extra.
'''))
    return c


def nb_hyper():
    c = []
    c.append(md(r'''
# Ecoverse 813 — 01 · Hyperspectral calibration on Gulf analogue scenes
**Question:** which soil and plant signals that matter for plant survival in Bahrain *need* a hyperspectral sensor (Tanager today,
Satellite 813 tomorrow), and which can Sentinel-2 already approximate?

**Data:** open Planet Tanager surface-reflectance scenes (426 bands, 380–2500 nm, ~30 m) over a **coastal sabkha on Abu Dhabi's Al Dhafra coast**
and **Riyadh** — the closest open-archive analogues to Bahrain's saline coast and arid city. `scripts/fetch_data.py` downloaded each
HDF5, removed water-vapour and noisy edge bands, and saved a compact subset (pixel sample with Sentinel-2-simulated bands, narrow-band
features and 20 principal components of the full spectrum, plus thinned feature maps) to `data/sample_input/tanager_gulf/`.

**Method:** continuum-removed absorption depths (gypsum 1750 nm, clay 2200 nm, carbonate 2340 nm, canopy water 1200 nm) and red-edge
position; Sentinel-2 simulated with Gaussian spectral response functions; spatially-blocked 5-fold cross-validation (32 × 32 px blocks)
so near-identical neighbouring pixels never sit on both sides of a split.
'''))
    c.append(code(SETUP))
    c.append(code(r'''
tdf, tmeta, tmaps = hyperspectral.load_tanager_tables()
pd.DataFrame(tmeta).T[["label", "datetime", "n_bands_total", "n_bands_used", "valid_pct", "land_pct", "n_sampled_pixels"]]
'''))
    c.append(md("## 1 · Feature maps that only hyperspectral resolves"))
    c.append(code(r'''
for sid, m in tmaps.items():
    land = m["land"].astype(bool)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    rgb = np.dstack([m["B04"], m["B03"], m["B02"]]).astype("float32")
    ax[0].imshow(np.clip(np.nan_to_num(rgb, nan=1) * 3.2, 0, 1)); ax[0].set_title("Simulated S2 true colour", loc="left"); ax[0].axis("off")
    specs = [("GYPSUM_1750", "Gypsum depth (1750 nm)", viz.SEQ_ORANGE), ("CARBONATE_2340", "Carbonate depth (2340 nm)", viz.SEQ_BLUE),
             ("CLAY_2200", "Clay Al-OH depth (2200 nm)", viz.SEQ_BLUE), ("REP", "Red-edge position (nm)", viz.SEQ_GREEN)]
    for a, (k, t, cm) in zip(ax[1:], specs):
        arr = m[k].astype("float32"); arr[~land] = np.nan
        v = arr[np.isfinite(arr)]
        viz._map(a, arr, t, cm, *(np.percentile(v, [2, 98]) if v.size else (0, 1)))
    fig.suptitle(tmeta[sid]["label"], x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); viz.save(fig, f"10_hs_features_{sid}.png"); plt.show()
'''))
    c.append(md("## 2 · Hyperspectral soil-surface types and their spectra\nBare-ground pixels from all scenes are clustered on four narrow-band absorption depths (gypsum, clay, carbonate, 1200 nm water) that only a hyperspectral sensor resolves; mean spectra per class are reconstructed from the first scene's PCA basis."))
    c.append(code(r'''
rec = hyperspectral.soil_class_recovery(tdf, k=6)
sid0 = next(iter(tmaps)); m0 = tmaps[sid0]
wl = m0["wavelengths"]; comps = m0["pca_components"]; mean = m0["pca_mean"]
pcs = [c for c in tdf.columns if c.startswith("PC")]
sub = tdf.loc[rec["index"]]
fig, ax = plt.subplots(figsize=(12, 5))
for k in range(rec["k"]):
    sel = rec["labels"] == k
    # PCA bases differ per scene; reconstruct only with this scene's basis
    s = sub[sel & (sub["scene"] == sid0).to_numpy()]
    if len(s) < 20: continue
    spec = s[pcs].to_numpy().mean(0) @ comps + mean
    gaps = np.r_[False, np.diff(wl) > 20]
    y = spec.copy(); y[gaps] = np.nan
    ax.plot(wl, y, color=viz.CAT[k], lw=1.6, label=f"class {k} (n={len(s)})")
for lo_, hi_, name in [(1735, 1765, "gypsum"), (2185, 2225, "clay"), (2320, 2355, "carbonate"), (1170, 1220, "water")]:
    ax.axvspan(lo_, hi_, color="#e9e8e4", zorder=0); ax.text((lo_+hi_)/2, ax.get_ylim()[1]*0.98, name, ha="center", va="top", fontsize=8, color=viz.INK2)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Surface reflectance"); ax.legend(ncol=3, fontsize=8)
ax.set_title(f"Mean spectra of hyperspectral bare-ground classes · {tmeta[sid0]['label']}", loc="left")
fig.tight_layout(); viz.save(fig, "11_hs_soil_spectra.png"); plt.show()
'''))
    c.append(md("## 3 · Can Sentinel-2 see the same thing? (spatially-blocked cross-validation)"))
    c.append(code(r'''
cv = hyperspectral.evaluate_transfer(tdf)
cv.to_csv(config.RESULTS_DIR / "hyperspectral_transfer_cv.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 4))
y = np.arange(len(cv))
ax.barh(y + 0.18, cv["r2_linear"].clip(lower=0), height=0.34, color=viz.CAT[1], label="Linear on S2 bands")
ax.barh(y - 0.18, cv["r2_boosted"].clip(lower=0), height=0.34, color=viz.CAT[0], label="Gradient boosting on S2 bands")
ax.set_yticks(y, cv["target"]); ax.set_xlim(0, 1); ax.axvline(0.4, color=viz.MUTED, ls="--", lw=1)
ax.text(0.41, len(cv) - 0.5, "transfer threshold", fontsize=8, color=viz.INK2)
ax.set_xlabel("Cross-validated R² (spatial blocks)"); ax.legend(loc="lower right"); ax.invert_yaxis()
ax.set_title("How much of each hyperspectral signal Sentinel-2 can recover", loc="left")
fig.tight_layout(); viz.save(fig, "12_hs_transfer_r2.png"); plt.show()
cv.round(3)
'''))
    c.append(code(r'''
print(f"Sentinel-2 recovers the {rec['k']} hyperspectral soil classes with balanced accuracy "
      f"{rec['balanced_accuracy']:.2f} (chance = {rec['chance']:.2f})")
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(rec["confusion"].to_numpy(), cmap=viz.SEQ_BLUE, vmin=0, vmax=1)
ax.set_xlabel("Predicted from Sentinel-2"); ax.set_ylabel("Hyperspectral class")
ax.set_xticks(range(rec["k"])); ax.set_yticks(range(rec["k"]))
for i in range(rec["k"]):
    for j in range(rec["k"]):
        v = rec["confusion"].to_numpy()[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else viz.INK)
plt.colorbar(im, ax=ax, fraction=0.04); ax.set_title("Soil-class recovery (row-normalised)", loc="left")
fig.tight_layout(); viz.save(fig, "13_hs_soil_confusion.png"); plt.show()
json.dump({"soil_class_balanced_accuracy": rec["balanced_accuracy"], "chance": rec["chance"], "k": rec["k"]},
          open(config.RESULTS_DIR / "hyperspectral_soil_recovery.json", "w"), indent=2)
'''))
    c.append(md(r'''
## 4 · What this means for Ecoverse
- Targets with **high R²** can be approximated from free Sentinel-2 everywhere in Bahrain today; the main notebook transfers them.
- Targets with **low R²** (the "hyperspectral-only information" column) and soil classes Sentinel-2 confuses are precisely the
  measurements that justify **Satellite 813**: they decide between, e.g., a gypsum-rich sabkha crust (where only halophytes like
  *Suaeda*, *Arthrocnemum* or *Salvadora* survive) and a bright but non-saline sand (where *Haloxylon* or *Pennisetum* are cheaper).
- Caveats: two analogue scenes; Tanager-simulated Sentinel-2 is not identical to real Sentinel-2 L2A reflectance; no field spectra.
'''))
    return c


def nb_colab():
    c = []
    c.append(md(r'''
# Ecoverse 813 — run everything (Google Colab or any machine with internet)
This notebook downloads the open data, runs both analysis notebooks, builds the slides PDF and packages the results.
Runtime ≈ 15–30 min (most of it is downloading two Tanager hyperspectral scenes of ~1–5 GB each). No keys or accounts needed.
'''))
    c.append(code(r'''
import os, subprocess, sys
from pathlib import Path
REPO = "https://github.com/saraabutaleb/Eco-verse-813.git"   # <- change after you push
if not Path("src/ecoverse").exists() and not Path("../src/ecoverse").exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
    os.chdir(Path(REPO).stem)
elif Path("../src/ecoverse").exists():
    os.chdir("..")
print("Working in", Path.cwd())
'''))
    c.append(code(r'''
%pip install -q -r requirements.txt
'''))
    c.append(code(r'''
!python scripts/fetch_data.py
'''))
    c.append(code(r'''
!python scripts/run_all.py
'''))
    c.append(code(r'''
# Package the generated inputs/outputs so you can copy them into your local clone and commit them
!zip -qr ecoverse_outputs.zip data/sample_input results docs/slides.pdf notebooks README.md
try:
    from google.colab import files; files.download("ecoverse_outputs.zip")
except ImportError:
    print("Saved ecoverse_outputs.zip")
'''))
    return c


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb["metadata"]["language_info"] = {"name": "python"}
    p = NB / name
    nbf.write(nb, p)
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    NB.mkdir(exist_ok=True)
    write("00_run_everything_colab.ipynb", nb_colab())
    write("01_hyperspectral_gulf_calibration.ipynb", nb_hyper())
    write("02_main_analysis.ipynb", nb_main())
