# Ecoverse 813 — plant what the ground already wants

**Team:** Ecoverse 813 (see [Team](#10-team-licence-and-attribution)) · **Country:** Kingdom of Bahrain ·
**Theme 05 — Sustainable Urban Planning & Smart Cities** · Arab Youth Space Hackathon: 813 Challenge (Proof of Concept)

**One line:** Ecoverse builds a satellite *environmental fingerprint* (heat, salinity, saline groundwater, moisture,
survival evidence, urban density) for every 20 m patch of northern-central Bahrain and recommends the
**lowest-maintenance native plant guild that can survive there**, together with the irrigation water it saves and
a field-probe route that targets the places where the satellite estimate is least certain.

![Recommendation](results/05_recommendation.png)

---

## 1. Title and one-line summary
See above. The PoC runs on open data only, all over Bahrain (Sentinel-2, Landsat, Copernicus DEM, ESA WorldCover and NASA EMIT
hyperspectral), and is built so that **Satellite 813** hyperspectral data drops in during incubation.

## 2. The business use case
| | |
|---|---|
| **User** | Municipal landscaping planners (Ministry of Municipalities Affairs & Agriculture and the four municipalities), real-estate developers, landscape contractors. |
| **Decision** | Which canopy tree, shrub and groundcover to plant — or replant — on a given site, and which existing irrigated plantings to convert first. |
| **What they use today** | Generic species lists, nursery availability and habit. Failures are hidden by extra irrigation, replacement and pruning. |
| **What Ecoverse gives them** | A district map of the lowest-water guild that survives each patch, a one-page **site card** per project, the water saved vs. business-as-usual, and a **probe route** for the sites that need a ground check. |
| **Why it can scale** | Free satellite data now; Satellite 813 later; a ≈ USD 100 probe kit; one analyst per municipality. Revenue: per-district subscription or per-project site report. |

## 3. The problem
- Bahrain's reference evapotranspiration is about **1,850 mm/yr** against **≈ 75 mm** of rain ([Fanack Water](https://water.fanack.com/bahrain/water-resources-bahrain/); [FAO AQUASTAT](https://www.fao.org/4/w4356e/w4356e09.HTM)). Ornamental plantings survive on desalinated water or groundwater.
- Soils are **saline and calcareous**: irrigated soils 4–12 dS/m, abandoned farmland up to 60 dS/m, shallow saline groundwater along the coast (FAO AQUASTAT).
- The default fast-growing street tree, *Conocarpus*, was **banned for new planting in Bahrain in 2022** because its roots damage pipes, drains and walls ([GDN Online](https://www.gdnonline.com/Details/1372822)); Qatar did the same in 2017.
- Native species that already survive these conditions (ghaf, sidr, arak, rimth, halophytes) exist, but there is **no site-specific way** to know which one fits which patch.

**Why satellites:** the deciding factors — heat, surface salinity, saline groundwater, whether vegetation survives the summer
without rain — vary block by block and change over seasons. Only satellites observe all of them, everywhere, repeatedly, at
near-zero marginal cost. Hyperspectral sensors add soil mineralogy (gypsum, carbonate, clay) and early plant stress that broad
bands cannot resolve — which is exactly what Satellite 813 provides.

## 4. Data used
| Dataset | Provider / access | Dates | Processing level | Licence |
|---|---|---|---|---|
| Sentinel-2 MSI L2A, bands B02–B07, B8A, B11, B12, SCL | ESA Copernicus via Microsoft Planetary Computer STAC (`sentinel-2-l2a`) | Summer 2025-06-01 → 08-31; winter 2025-12-01 → 2026-02-28; scene cloud < 20 % | L2A surface reflectance, baseline ≥ 04.00 (offset −1000 applied), SCL-masked, per-pixel median | Copernicus free & open |
| Landsat 8/9 Collection 2 L2, `lwir11` + `QA_PIXEL` | USGS via Planetary Computer (`landsat-c2-l2`) | Summer 2025; cloud < 30 % | Surface temperature (scale 0.00341802, offset 149 K), QA-masked median | Public domain |
| Copernicus DEM GLO-30 | ESA / Airbus via Planetary Computer (`cop-dem-glo-30`) | 2021 release | 30 m DSM, resampled to 20 m | Copernicus DEM licence |
| ESA WorldCover | ESA via Planetary Computer (`esa-worldcover`) | 2021 (v200) | 10 m land cover, mode-resampled | CC BY 4.0 |
| NASA EMIT L2A surface reflectance (`EMITL2ARFL`, V002) | NASA JPL / LP DAAC via NASA Earthdata (`earthaccess`); granule `EMIT_L2A_RFL_002_20230617T111139` | 2023-06-17 (0 % cloud over the AOI) | L2A surface reflectance, 285 bands 381–2493 nm, 60 m; water-vapour and > 2450 nm bands removed (237 used); window cut to the AOI | NASA open data, no restrictions |
| Plant trait table (26 species) | FAO, ICBA Biosaline, peer-reviewed salinity trials, Gulf floras — one source URL per row in `data/plants/bahrain_plant_traits.csv` | — | Curated; `salinity_basis` F = from source, F/E = converted, E = expert estimate | Facts cited with attribution |

Exact AOI, dates and limits are in [`src/ecoverse/config.py`](src/ecoverse/config.py); every STAC item ID actually used is
recorded in `data/sample_input/bahrain_aoi_stack_metadata.json` and `data/sample_input/emit_bahrain/*_meta.json`.

**AOI:** lon 50.50–50.64 E, lat 26.12–26.24 N (≈ 14 × 13 km): Manama, Tubli Bay mangroves, Isa Town, Sitra — a deliberate
gradient from intertidal flats to dense urban fabric to irrigated date-palm belts.

## 5. Technical approach (execution order)
```
open data ─► hyperspectral bridge ─► fingerprint ─► survival model ─► guild optimiser ─► water budget
                                                                           │
                                                   Monte Carlo stability ◄─┘─► probe route ─► ground fusion ─► updated map
```
1. **Compositing** (`fetch.py`): SCL/QA-masked medians for summer and winter; *summer-green persistence* = share of clear
   summer dates with NDVI > 0.25 (Bahrain gets ~no summer rain, so persistent green that is not tidal is irrigated or groundwater-fed).
2. **Hyperspectral over Bahrain** (`emit.py`, `hyperspectral.py`, notebook 01): from the NASA EMIT scene of the same AOI compute
   continuum-removed absorption depths — gypsum 1750 nm, clay 2200 nm, carbonate 2340 nm, liquid water 1200 nm — and red-edge
   position; resample them to the 20 m grid (nearest neighbour, ≤ 45 m). The **gypsum depth goes directly into the salinity layer**.
   Simulate Sentinel-2 from the EMIT spectra and train gradient-boosted S2 → feature models with **5-fold spatially-blocked CV**
   (32 × 32 px blocks ≈ 1.9 km); the unexplained share of each feature is what only a hyperspectral sensor (Satellite 813) sees.
   The few grid cells EMIT does not cover are filled from that model only if its CV R² ≥ 0.4. Also: cluster bare-ground pixels on
   the absorption depths and measure how well Sentinel-2 can recover those soil classes.
3. **Fingerprint** (`fingerprint.py`): NDVI, NDRE, NDMI, MNDWI, NDBI, salinity indices SI = √(B·R), SI2, SWIR ratio;
   distance to sea; intertidal mask; **salinity pressure** (spectral salinity + mineral proxy on bare ground, low elevation,
   coastal proximity) mapped log-linearly to an ECe prior of 3–50 dS/m with per-pixel uncertainty; waterlogging risk;
   urban density (50 m window); k-means **micro-habitat archetypes**.
4. **Survival model** (`plants.py`): per species, 0–1 response curves for salinity (ECe comfort → survival limit), heat,
   drought, waterlogging, habitat (tidal / wet-sabkha / sand), pH and root-vs-infrastructure risk; survival = product.
5. **Guild optimiser** (`recommend.py`): among natives with survival ≥ 0.6, per layer maximise
   `survival − 0.6·plant_factor − 0.08·(maintenance−1) + 0.15·shade·heat_need`.
6. **Water** : WUCOLS-style plant factor × ET0 1,850 mm/yr, guild cover 40/30/30 %, vs. baseline Conocarpus + lawn (factor 0.8).
7. **Robustness** (`uncertainty.py`): 40-draw Monte Carlo over salinity (per-pixel + AOI-wide shift), heat, waterlogging,
   pH and ET0 (1,800–2,100 mm) → stability map and confidence interval on savings.
8. **Probe loop**: greedy selection of 12 sites ≥ 500 m apart maximising instability × decision value, nearest-neighbour
   route; readings fused by Gaussian-kernel residual kriging (600 m range), shrinking uncertainty locally.

Random seed 813 everywhere.

## 6. Installation
Requires **Python 3.12** (3.13 also tested).
```bash
git clone https://github.com/saraabutaleb/Eco-verse-813.git
cd Eco-verse-813
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
No keys are needed to reproduce the results from the committed sample data. Re-downloading the NASA EMIT scene needs a free
NASA Earthdata account: set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` as environment variables (never commit them).

## 7. How to run
**Reproduce the headline result from the committed sample data (offline, ≈ 3 min, no GPU):**
```bash
jupyter lab notebooks/02_main_analysis.ipynb      # Run All
```
It reads `data/sample_input/bahrain_aoi_stack.tif` and `data/sample_input/emit_bahrain/`, and writes figures, GeoTIFFs and CSVs to `results/`.
The hyperspectral analysis of Bahrain is `notebooks/01_hyperspectral_bahrain_emit.ipynb` (≈ 1 min).

**Re-run everything from scratch** (downloads data, ≈ 15–20 min; the EMIT scene is ≈ 2 GB before it is cut to the AOI):
```bash
export EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=...   # free NASA Earthdata account, for EMIT only
python scripts/fetch_data.py          # writes data/sample_input/ (add --skip-emit for a 5-min run without EMIT)
python scripts/run_all.py             # executes notebooks 01 and 02, builds docs/slides.pdf, refreshes this README's results
pytest -q                             # offline unit tests on synthetic fixtures
```
Or open [`notebooks/00_run_everything_colab.ipynb`](notebooks/00_run_everything_colab.ipynb) in Google Colab.

**What to edit:** AOI, dates, cloud limits, EMIT search window → `src/ecoverse/config.py`; species and tolerances →
`data/plants/bahrain_plant_traits.csv`; field readings → copy `data/ground/ground_readings_TEMPLATE.csv` to
`data/ground/ground_readings.csv`.

## 8. Example input and output
- **Input:** `data/sample_input/bahrain_aoi_stack.tif` — 17-layer, 20 m, int16 GeoTIFF clipped to the AOI (layer names and scale factors in the tags);
  `data/sample_input/emit_bahrain/` — NASA EMIT window over the AOI (absorption-depth features, simulated Sentinel-2 bands,
  20 PCA scores of the full spectrum per pixel, metadata).
- **Outputs (results/):** `recommended_species.tif`, `fingerprint_and_scores.tif` (ECe, LST, waterlogging, persistence,
  urban density, water need, saving, stability), `species_summary.csv`, `guilds.csv`, `water_budget.csv`, `probe_plan.csv`,
  `validation.csv`, `hyperspectral_transfer_cv.csv`, `headline.json`, and the figures below.

| | |
|---|---|
| ![Fingerprint](results/02_fingerprint.png) | ![Site cards](results/07_site_cards.png) |
| ![Water](results/06_water_budget.png) | ![Probe plan](results/08_probe_plan.png) |
| ![Hyperspectral transfer](results/12_hs_transfer_r2.png) | ![Soil spectra](results/11_hs_soil_spectra.png) |

## 9. Results and limitations
<!-- RESULTS:START -->

_Auto-generated by `scripts/update_readme.py` from `results/headline.json` (do not edit by hand)._

| Metric | Value |
|---|---|
| Land analysed | 149.5 km² at 20 m |
| Irrigated greenery detected | 1,108 ha |
| Land with ≥1 native survivor (no routine irrigation) | 100.0 % |
| Mean irrigation saving vs. Conocarpus + lawn | 1285 mm/yr (90 % MC interval 1249–1432) |
| Median recommendation stability (Monte Carlo) | 0.93 |
| Land keeping the same pick in ≥80 % of draws | 87.9 % |
| Probe sites planned | 12 |

**Water budget**

| Scenario | Area (ha) | Baseline (m³/yr) | Ecoverse (m³/yr) | Saving |
|---|---|---|---|---|
| Urban irrigated greenery (replace) | 851 | 12,595,984 | 1,596,868 | 87 % |
| All irrigated greenery incl. farms | 1,108 | 16,395,440 | 2,128,610 | 87 % |
| All evaluated land (green from scratch) | 14,952 | 221,283,088 | 29,227,396 | 87 % |

**Most-recommended native species (share of land)**

1. Arak / Toothbrush tree — 96.9 %
2. Ghaf — 77.4 %
3. Rimth — 44.3 %
4. Harm / Qatar bean caper — 34.3 %
5. Sea lavender — 14.4 %

**Hyperspectral → Sentinel-2 transfer (spatial-block CV)**

| Hyperspectral feature | Pixels | n | Linear R² | Boosted R² | Info only hyperspectral sees |
|---|---|---|---|---|---|
| GYPSUM_1750 | soil | 35,175 | 0.24 | 0.43 | 57 % |
| CLAY_2200 | soil | 35,175 | 0.17 | 0.27 | 73 % |
| CARBONATE_2340 | soil | 35,175 | 0.15 | 0.39 | 61 % |
| REP | veg | 1,128 | 0.85 | 0.65 | 35 % |
| CANOPY_WATER_1200 | veg | 1,128 | 0.88 | 0.85 | 15 % |

EMIT gypsum depth (1750 nm) used in the Bahrain salinity layer: **True** (NASA EMIT EMIT_L2A_RFL_002_20230617T111139 (2023-06-17))

Sentinel-2 recovers the 6 hyperspectral soil classes with balanced accuracy **0.58** (chance 0.17).

**Validation checks**

- WorldCover mangrove pixels where grey mangrove survives (%): **100.00**
- Inland (>1 km from sea) pixels where grey mangrove is recommended (%): **0.00**
- Spearman rho, summer NDVI vs LST (expect < 0): **-0.07**
- Conocarpus survives without routine irrigation in dense urban pixels (%): **0.00**
- Pixels with ≥1 surviving native species (%): **99.99**
- EMIT coverage of the AOI grid (%): **95.68**
- EMIT vs Sentinel-2 red reflectance, Pearson r (co-registration check): **0.80**
- EMIT vs Sentinel-2 NIR reflectance, Pearson r (co-registration check): **0.89**

<!-- RESULTS:END -->

**How we validated.** No map of "where each species survives" exists for Bahrain, so a single accuracy number would be
misleading. We report: (i) an **independent reference check** against ESA WorldCover mangroves (grey mangrove should survive
there and never be recommended inland); (ii) **physical consistency** (summer NDVI vs. land surface temperature should be
negatively correlated); (iii) **Monte Carlo stability**; (iv) for the hyperspectral bridge, **spatially-blocked cross-validation**
so neighbouring pixels never leak across folds. For external context: a Qatar GORD project reported native landscapes using
roughly 70 % less irrigation than standard park ornamentals (non-peer-reviewed) — our modelled savings should be read against that.

**Reading the validation checks honestly.** The Conocarpus result (0 %) follows directly from its low drought-tolerance
rating in the trait table, so it is a consistency check, not independent evidence. The summer NDVI–LST correlation is
negative as expected but weak (ρ ≈ −0.07), because Landsat surface temperature is 100 m data resampled to 20 m. The
water savings are modelled from WUCOLS-style plant factors and ET0, not metered. All data, including the hyperspectral
data (NASA EMIT, 17 June 2023), covers the same Bahrain AOI. EMIT and Sentinel-2 agree well where they overlap (red r ≈ 0.80,
NIR r ≈ 0.89), which confirms the two are correctly co-registered.

**Where it breaks.**
- Salinity is a *satellite prior* anchored to FAO ranges, not a measurement; it is weakest under vegetation and buildings
  (uncertainty is mapped and propagated). Ground probes are designed in to fix this.
- Several species tolerances are expert estimates (`salinity_basis = E`); lab NaCl trials overstate field tolerance, so
  comfort limits are conservative. A Gulf horticulturist must review the table before operational use.
- pH and nutrients cannot be seen from orbit — a regional calcareous prior (pH 8.1 ± 0.35) is used until probed.
- 20 m pixels mix roads, roofs and verges; recommendations describe the typical planting condition of the patch.
- Hyperspectral data is a single EMIT date (17 June 2023) at 60 m, while Sentinel-2/Landsat are summer 2025 at 20 m; soil
  mineralogy changes slowly, but construction since 2023 is not reflected in the gypsum layer. The S2 → feature models use
  Sentinel-2 bands simulated from EMIT, not co-incident real Sentinel-2. The vegetation targets rest on 1,128 vegetated pixels
  and the soil targets on 35,175.
- Water savings assume established plants and WUCOLS-style plant factors; establishment-year irrigation is extra.
- Irrigated-greenery detection includes date-palm farms; the urban replacement scenario excludes them using urban density.

More: [`docs/concept_review.md`](docs/concept_review.md) — idea refinement, weaknesses, similar solutions, probe-kit bill of materials.

## 10. Team, licence and attribution
| Member | Role |
|---|---|
| Enam Abutaleb ([@EnamAbutaleb](https://github.com/EnamAbutaleb)) | Team leader |
| Sara Abutaleb | Participant |

- **Code:** MIT licence (see `LICENSE`).
- **Data:** Copernicus Sentinel-2 data (ESA) and Copernicus DEM; Landsat courtesy of USGS; ESA WorldCover 2021 (CC BY 4.0);
  NASA EMIT L2A reflectance (NASA JPL, distributed by LP DAAC; no restrictions); accessed through Microsoft Planetary Computer
  and NASA Earthdata.
- **Built on:** the hackathon starter notebooks ([813-hyperspectral-hackathon](https://github.com/Tnecniv-Teikram/813-hyperspectral-hackathon)) for
  hyperspectral band screening and index conventions; earthaccess, odc-stac, rasterio, scikit-learn.
- **Plant knowledge:** FAO salt-tolerance tables, ICBA *Salt Tolerant Plants of the UAE*, UAE Flora, Qatar e-Nature and the
  peer-reviewed trials cited per species in the trait table.
