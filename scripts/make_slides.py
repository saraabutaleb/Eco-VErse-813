#!/usr/bin/env python
"""Build docs/slides.pdf from the real outputs in results/ (run after the notebooks).

Every number on the slides is read from results/headline.json and every chart
is a figure the notebooks produced - nothing is typed in by hand.
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecoverse import config  # noqa: E402

RES = config.RESULTS_DIR
OUT = config.DOCS_DIR / "slides.pdf"
INK, INK2, MUTED, ACCENT, ACCENT2, BG, PANEL = "#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#1baf7a", "#fcfcfb", "#f3f2ee"
W, H = 13.333, 7.5


def new_slide(title, kicker=None, n=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    if kicker:
        fig.text(0.05, 0.925, kicker.upper(), fontsize=10, color=ACCENT, fontweight="bold")
    fig.text(0.05, 0.86, title, fontsize=24, color=INK, fontweight="bold")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.835, 0.835], color="#e0dfda", lw=1))
    if n:
        fig.text(0.95, 0.03, f"Ecoverse 813 · {n}", fontsize=8, color=MUTED, ha="right")
    return fig


def bullets(fig, items, x=0.05, y=0.78, w=0.42, size=13, gap=0.012):
    chars = int(w * 150 * 13 / size)
    for it in items:
        sub = it.startswith("  ")
        txt = textwrap.fill(it.strip(), chars - (4 if sub else 0))
        fig.text(x + (0.025 if sub else 0), y, ("– " if sub else "• ") + txt, fontsize=size - (1.5 if sub else 0),
                 color=INK2 if sub else INK, va="top", linespacing=1.3)
        y -= (txt.count("\n") + 1) * size * 1.3 / (H * 72) + gap
    return y


def image(fig, name, rect, caption=None):
    p = RES / name
    ax = fig.add_axes(rect); ax.axis("off")
    if p.exists():
        ax.imshow(plt.imread(p))
    else:
        ax.text(0.5, 0.5, f"[{name} not produced]", ha="center", va="center", color=MUTED)
    if caption:
        fig.text(rect[0], rect[1] - 0.03, caption, fontsize=9, color=INK2, va="top")


def stat(fig, x, y, value, label, color=ACCENT):
    fig.text(x, y, value, fontsize=30, fontweight="bold", color=color, va="top")
    fig.text(x, y - 0.085, textwrap.fill(label, 30), fontsize=10, color=INK2, va="top")


def box(fig, x, y, w, h, title, body, color):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor=PANEL, edgecolor=color, lw=1.5))
    fig.text(x + 0.01, y + h - 0.02, title, fontsize=11.5, fontweight="bold", color=color, va="top")
    fig.text(x + 0.01, y + h - 0.065, textwrap.fill(body, int(w * 135)), fontsize=9, color=INK, va="top",
             linespacing=1.3)


def arrow(fig, x0, y0, x1, y1):
    fig.add_artist(matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                                      arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))


def main():
    hl = json.loads((RES / "headline.json").read_text())
    team = json.loads((config.DOCS_DIR / "team.json").read_text())
    wb = {r["scenario"]: r for r in hl["water_budget"]}
    urban = next((r for k, r in wb.items() if k.startswith("Urban")), None)
    total_irr = next((r for k, r in wb.items() if k.startswith("All irrigated")), None)
    tcv = {r["target"]: r for r in (hl.get("transfer_cv") or []) if "r2_boosted" in r}
    soil_rec = RES / "hyperspectral_soil_recovery.json"
    soil = json.loads(soil_rec.read_text()) if soil_rec.exists() else None
    n = 0
    with PdfPages(OUT) as pdf:
        # 1 Title
        fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor(BG)
        fig.text(0.06, 0.70, "Ecoverse 813", fontsize=54, fontweight="bold", color=INK)
        fig.text(0.06, 0.61, "Plant what the ground already wants.", fontsize=24, color=ACCENT)
        fig.text(0.06, 0.56, "Satellite environmental fingerprints that pick\nthe lowest-maintenance native plants able to\nsurvive each patch of urban Bahrain.", fontsize=15, color=INK2, linespacing=1.4, va="top")
        fig.text(0.06, 0.32, f"{team['theme']}\nCountry: {team['country']}\nTeam {team['team_name']}: " +
                 ", ".join(m["name"] for m in team["members"]), fontsize=13, color=INK, linespacing=1.6, va="top")
        fig.text(0.06, 0.08, "Arab Youth Space Hackathon · 813 Challenge · Proof of Concept", fontsize=11, color=MUTED)
        image(fig, "05_recommendation.png", [0.60, 0.14, 0.37, 0.62])
        pdf.savefig(fig); plt.close(fig); n += 1

        # 2 Problem
        fig = new_slide("Bahrain greens its cities by fighting its own soil", "The problem", n + 1)
        bullets(fig, [
            "Reference evapotranspiration ≈ 1,850 mm/yr against ≈ 75 mm of rain: most urban plantings survive only with irrigation (Fanack Water; FAO AQUASTAT).",
            "Soils are saline and calcareous: irrigated soils 4–12 dS/m, abandoned farmland up to 60 dS/m, shallow saline groundwater on the coast (FAO AQUASTAT).",
            "The default urban tree, Conocarpus, was banned for new planting in 2022 because its roots break pipes, drains and walls (GDN Online).",
            "Species are still chosen from generic lists, not from the conditions of the actual site — so planners overpay in water, maintenance and failed plantings.",
            "Who is affected: municipalities and their landscaping budgets, the water authority, and residents of the hottest, least-green neighbourhoods.",
        ], w=0.5)
        stat(fig, 0.62, 0.75, f"{hl['irrigated_greenery_ha']:,.0f} ha", "of irrigated greenery detected in our area of interest")
        stat(fig, 0.62, 0.50, f"{hl['land_km2']:,.0f} km²", "of land analysed at 20 m resolution")
        stat(fig, 0.62, 0.25, f"{hl['n_native']} species", f"native or Gulf-adapted, scored against {hl['n_species'] - hl['n_native']} common urban plantings", ACCENT2)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 3 Use case
        fig = new_slide("Who uses Ecoverse and what decision it changes", "Business use case", n + 1)
        box(fig, 0.05, 0.47, 0.28, 0.32, "User", "Municipal landscaping planners (Ministry of Municipalities Affairs & Agriculture, "
            "the four municipalities), real-estate developers and landscape contractors.", ACCENT)
        box(fig, 0.36, 0.47, 0.28, 0.32, "Decision", "Which species — canopy, shrub, groundcover — to plant or replant on a given "
            "site, and which existing irrigated plantings to convert first.", ACCENT)
        box(fig, 0.67, 0.47, 0.28, 0.32, "Today", "Standard species lists, nursery availability and habit; failures are "
            "masked by extra irrigation and maintenance.", "#eb6834")
        box(fig, 0.05, 0.10, 0.44, 0.30, "What Ecoverse delivers", "A district map of the lowest-maintenance guild that survives each 20 m "
            "patch; a one-page site card per project; the water it saves; a field-probe route that targets the uncertain sites.", ACCENT2)
        box(fig, 0.51, 0.10, 0.44, 0.30, "Why it scales", "Runs on free Sentinel-2/Landsat now, Satellite 813 hyperspectral later; the ground "
            "kit costs ≈ USD 100; one analyst can cover a municipality. Revenue: per-district subscription or per-project site report.", ACCENT2)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 4 What exists
        fig = new_slide("What already exists — and what is new here", "Originality", n + 1)
        box(fig, 0.05, 0.50, 0.28, 0.29, "Native-plant guides", "Gulf botanic and municipal lists (e.g. Qatar, UAE) say WHICH species are adapted, "
            "but not WHERE each one fits; they are static PDFs.", MUTED)
        box(fig, 0.36, 0.50, 0.28, 0.29, "Tree-canopy & heat tools", "Urban forestry tools say WHERE to add trees (heat, equity) "
            "but assume a temperate species palette and irrigation.", MUTED)
        box(fig, 0.67, 0.50, 0.28, 0.29, "Agricultural salinity mapping", "Biosaline programmes map salinity to match crops on farms; "
            "they target yield, not low-input urban landscapes.", MUTED)
        bullets(fig, [
            "Ecoverse joins the three: a per-pixel fingerprint → a species GUILD that survives with minimal water → the water it saves.",
            "Hyperspectral over Bahrain: NASA EMIT measures saline gypsum soil directly, and shows what Sentinel-2 misses.",
            "Uncertainty-driven field loop: the map tells two students where one probe reading changes the most decisions, then updates itself.",
        ], y=0.40, w=0.9)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 5 Data
        fig = new_slide("Open data only — and where hyperspectral earns its place", "Data", n + 1)
        bullets(fig, [
            f"Sentinel-2 L2A (ESA Copernicus): {hl['n_s2_scenes_summer']} clear summer scenes, {hl['data_period_summer']}, 20 m; plus a winter composite.",
            "Landsat 8/9 Collection 2 surface temperature (USGS): summer heat.",
            "Copernicus DEM GLO-30: elevation → shallow saline groundwater risk.",
            "ESA WorldCover 2021: land-cover context and an independent mangrove reference.",
            f"NASA EMIT hyperspectral (285 bands, 60 m) over the same Bahrain AOI: {hl.get('hs_source') or 'not available'}.",
            "26-species trait table from FAO, ICBA, peer-reviewed salinity trials and Gulf floras (sources per row).",
        ], w=0.5, size=12)
        if tcv:
            lines = [f"{k}: CV R² = {v['r2_boosted']:.2f}" for k, v in tcv.items()]
            fig.text(0.6, 0.78, "Hyperspectral signal recoverable from Sentinel-2\n(spatial-block cross-validation)",
                     fontsize=12, fontweight="bold", color=INK, va="top")
            fig.text(0.6, 0.68, "\n".join(lines), fontsize=12, color=INK2, va="top", linespacing=1.6, family="monospace")
            msg = ("EMIT gypsum (1750 nm) used directly in the Bahrain salinity layer." if hl.get("gypsum_transferred")
                   else "Gypsum signal NOT recoverable from Sentinel-2 → a measured gap for Satellite 813.")
            fig.text(0.6, 0.38, textwrap.fill(msg, 50), fontsize=12, color=ACCENT, va="top", fontweight="bold")
        if soil:
            fig.text(0.6, 0.27, textwrap.fill(f"Sentinel-2 separates the {soil['k']} hyperspectral soil classes with balanced accuracy "
                     f"{soil['soil_class_balanced_accuracy']:.2f} (chance {soil['chance']:.2f}).", 50), fontsize=12, color=INK2, va="top")
        pdf.savefig(fig); plt.close(fig); n += 1

        # 6 Approach
        fig = new_slide("Approach", "Workflow", n + 1)
        steps = [("1 · Open data", "S2 summer/winter, Landsat LST, DEM, WorldCover, clipped to the AOI"),
                 ("2 · Hyperspectral (EMIT)", "Gypsum, clay, carbonate, plant stress over Bahrain; what S2 can recover"),
                 ("3 · Fingerprint", "heat · salinity (ECe) · saline groundwater · moisture · survival evidence · urban density"),
                 ("4 · Survival model", "26 species × factor response curves → survival score"),
                 ("5 · Guild optimiser", "lowest water + upkeep, shade where hottest"),
                 ("6 · Robustness & probes", "Monte Carlo stability → probe route → map update")]
        xs = [0.05, 0.36, 0.67]
        for i, (t, b) in enumerate(steps):
            x = xs[i % 3]; y = 0.50 if i < 3 else 0.15
            box(fig, x, y, 0.28, 0.25, t, b, ACCENT if i != 1 else ACCENT2)
            if i % 3 != 2:
                arrow(fig, x + 0.285, y + 0.125, x + 0.305, y + 0.125)
        arrow(fig, 0.81, 0.495, 0.19, 0.405)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 7 Fingerprint
        fig = new_slide("The environmental fingerprint of every 20 m patch", "Output 1", n + 1)
        image(fig, "02_fingerprint.png", [0.05, 0.06, 0.9, 0.74])
        pdf.savefig(fig); plt.close(fig); n += 1

        # 8 Recommendation
        fig = new_slide("What to plant where — and how sure we are", "Output 2", n + 1)
        image(fig, "05_recommendation.png", [0.03, 0.05, 0.62, 0.76])
        top = "\n".join(f"{nm}: {pct:.0f}% of land" for nm, pct in hl["top_species"][:5])
        fig.text(0.67, 0.78, "Most-recommended natives", fontsize=13, fontweight="bold", color=INK, va="top")
        fig.text(0.67, 0.72, top, fontsize=11.5, color=INK2, va="top", linespacing=1.6)
        stat(fig, 0.67, 0.42, f"{hl['pct_land_with_native_survivor']:.0f}%", "of land has at least one native that survives without routine irrigation")
        stat(fig, 0.67, 0.22, f"{hl['pct_stable_80']:.0f}%", "of land keeps the same pick in ≥80% of Monte Carlo draws", ACCENT2)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 9 Site cards
        fig = new_slide("The site card a planner actually reads", "Output 3", n + 1)
        image(fig, "07_site_cards.png", [0.03, 0.06, 0.94, 0.74])
        pdf.savefig(fig); plt.close(fig); n += 1

        # 10 Water
        fig = new_slide("Water saved vs. Conocarpus + lawn", "Output 4", n + 1)
        image(fig, "06_water_budget.png", [0.03, 0.40, 0.6, 0.40])
        image(fig, "04_species_survival.png", [0.64, 0.05, 0.33, 0.76])
        if urban:
            stat(fig, 0.05, 0.33, f"{urban['saving_m3_yr']/1e6:.2f} M m³/yr",
                 f"saved by converting the {urban['area_ha']:,.0f} ha of urban irrigated greenery in this AOI ({urban['saving_pct']:.0f}% less water)")
        lo, hi = hl["saving_mm_ci90"]
        fig.text(0.35, 0.30, textwrap.fill(f"Mean saving {hl['mean_saving_mm']:.0f} mm/yr per planted m² "
                 f"(90% Monte Carlo interval {lo:.0f}–{hi:.0f}).", 40), fontsize=11, color=INK2, va="top")
        pdf.savefig(fig); plt.close(fig); n += 1

        # 11 Hyperspectral
        if (RES / "12_hs_transfer_r2.png").exists():
            fig = new_slide("Hyperspectral: what Sentinel-2 cannot see", "Output 5", n + 1)
            fig.text(0.05, 0.80, f"Same Bahrain AOI, {hl.get('hs_source') or 'NASA EMIT'}. The gypsum layer in the salinity map is measured by EMIT;\n"
                     "bars show how much of each hyperspectral signal free Sentinel-2 can recover (the rest needs a hyperspectral sensor).", fontsize=10, color=INK2, va="top")
            image(fig, "12_hs_transfer_r2.png", [0.03, 0.40, 0.48, 0.36])
            image(fig, "11_hs_soil_spectra.png", [0.51, 0.40, 0.46, 0.36])
            image(fig, "10_emit_features_bahrain.png", [0.03, 0.05, 0.94, 0.33])
            pdf.savefig(fig); plt.close(fig); n += 1

        # 12 Probe loop
        fig = new_slide("Closing the loop: satellites tell students where to measure", "Output 6", n + 1)
        image(fig, "08_probe_plan.png", [0.03, 0.05, 0.45, 0.76])
        image(fig, "09_probe_whatif.png", [0.49, 0.36, 0.48, 0.42], "What-if with a HYPOTHETICAL reading: the map updates around the probe")
        bullets(fig, [f"{hl['n_probe_sites']} sites chosen where a reading flips the most decisions, ordered into one route.",
                      "Kit ≈ USD 100: EC 1:5 probe, pH probe, capacitive moisture, NPK sensor, IR thermometer, phone GPS.",
                      "Readings fuse into the map (residual kriging) and shrink uncertainty locally."], x=0.5, y=0.29, w=0.46, size=11)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 13 Validation & limitations
        fig = new_slide("Validation and limitations", "Honesty slide", n + 1)
        v = hl["validation"]
        notes = {"Conocarpus": " (follows from its low drought rating: a consistency check, not independent evidence)",
                 "Spearman": " (weak at 20 m; LST is 100 m Landsat resampled)"}
        vl = [f"{k}: {val:.2f}" + next((t for p_, t in notes.items() if k.startswith(p_)), "") for k, val in v.items()]
        fig.text(0.05, 0.78, "Checks we can run today", fontsize=13, fontweight="bold", color=INK, va="top")
        bullets(fig, vl + [f"Median Monte Carlo stability: {hl['median_stability']:.2f}"], y=0.72, w=0.42, size=11)
        fig.text(0.52, 0.78, "Limitations", fontsize=13, fontweight="bold", color=INK, va="top")
        bullets(fig, [
            "Salinity is a satellite prior anchored to FAO ranges, not a measurement — probes calibrate it.",
            "Several species tolerances are expert estimates; a Gulf horticulturist must review the table.",
            "pH and nutrients are invisible from orbit; a regional prior is used.",
            "20 m pixels mix roads, roofs and verges.",
            "Hyperspectral data is one EMIT date (June 2023) at 60 m; Sentinel-2 is 2025 at 20 m.",
            "Water savings assume established plants and WUCOLS plant factors.",
        ], x=0.52, y=0.72, w=0.43, size=11)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 14 Impact
        v_all = hl["validation"]
        fig = new_slide("Impact", "Why it matters", n + 1)
        if total_irr:
            stat(fig, 0.05, 0.75, f"{total_irr['saving_pct']:.0f}%", "modelled irrigation saving where natives replace high-water plantings")
        stat(fig, 0.37, 0.75, f"{hl['n_native']}", "native or Gulf-adapted species brought into urban planting decisions", ACCENT2)
        ck = next((k for k in v_all if k.startswith("Conocarpus")), None)
        if ck:
            pass
        if urban:
            stat(fig, 0.69, 0.75, f"{urban['saving_m3_yr']/1e6:.1f} M m³", "modelled water saved per year by converting this AOI's urban irrigated greenery", "#eb6834")
        bullets(fig, [
            "Water security: less desalinated / groundwater spent on ornamentals.",
            "Lower maintenance budgets: fewer replacements, less pruning and fertiliser.",
            "Cooler streets: shade-giving natives are prioritised where surfaces are hottest.",
            "Biodiversity and heritage: ghaf, sidr, arak and halophytes return to the city.",
            "Transferable to every Gulf city with the same open data and a ≈ USD 100 kit.",
        ], y=0.45, w=0.9)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 15 Next steps
        fig = new_slide("Next steps for incubation", "Roadmap", n + 1)
        box(fig, 0.05, 0.47, 0.28, 0.32, "Satellite 813", "Repeat, higher-resolution hyperspectral coverage of Bahrain "
            "to track soil salinity and plant stress over time, beyond a single EMIT date.", ACCENT)
        box(fig, 0.36, 0.47, 0.28, 0.32, "Field campaign", "50 probe sites + lab ECe on 10 samples to calibrate the salinity prior; "
            "survival audit of existing plantings.", ACCENT)
        box(fig, 0.67, 0.47, 0.28, 0.32, "Expert review", "Gulf horticulturists and nurseries validate the trait table and add "
            "availability and cost per species.", ACCENT)
        box(fig, 0.05, 0.10, 0.44, 0.30, "Pilot", "One municipality, one district: plant two demonstration verges "
            "(Ecoverse guild vs. standard) and measure water and survival for 12 months.", ACCENT2)
        box(fig, 0.51, 0.10, 0.44, 0.30, "Product", "Web map + site-card generator on gIQ; upload probe readings from a phone; "
            "export planting schedules for contractors.", ACCENT2)
        pdf.savefig(fig); plt.close(fig); n += 1
    print(f"Wrote {OUT} ({n} slides)")


if __name__ == "__main__":
    main()
