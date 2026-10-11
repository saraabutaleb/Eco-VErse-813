# Ecoverse 813 — concept review

## The refined idea in one paragraph
Instead of asking "how do we keep this tree alive here?", Ecoverse asks "what would survive here on its own?". Satellites
measure, for every 20 m patch of a Bahraini city, the conditions a plant must endure — summer heat, salinity, saline
groundwater, moisture and whether anything already stays green without rain. A survival model scores 21 native/adapted
Gulf species (and 5 common urban plantings for comparison) against those conditions and picks the **lowest-water,
lowest-upkeep guild** (canopy + shrub + groundcover) that survives. Because satellite salinity is uncertain, the map also
tells a two-person team **where one ground reading would change the most decisions**, and updates itself when the reading
comes in.

## What changed from the original pitch (and why)
| Original | Refined | Reason |
|---|---|---|
| "Use Satellite 813 data" | Open data now; Tanager Gulf scenes as a **hyperspectral bridge**; Satellite 813 slots in during incubation | Satellite 813 is not guaranteed in the PoC phase — a PoC built on it would not run. |
| Ground sensors everywhere | Satellite first, **probes only where uncertainty changes the decision** | Makes it affordable and scalable for a two-person team. |
| Single best plant | **Guild** (canopy + shrub + groundcover) | A real planting needs layers; it also lets salt-tolerant groundcovers work where trees cannot. |
| "Environmental fingerprint" as a concept | Six concrete, mapped layers + **habitat archetypes** | Judges and planners need to see what it is. |
| Implicit benefit | **Water budget** vs. Conocarpus + lawn, with Monte Carlo interval | Turns the idea into a number a ministry can budget against. |

## Weaknesses to be ready for (and our answers)
1. **"Salinity from space is unreliable."** True under vegetation and buildings. We map the uncertainty, propagate it
   (Monte Carlo) and use it to place probes. The hyperspectral bridge measures how much better a hyperspectral sensor does.
2. **"Your plant tolerances are guesses."** Each row cites a source; estimated values are flagged `E`. A horticulturist
   review is the first incubation task.
3. **"20 m is too coarse for a tree pit."** The product is a planning screen (which species palette fits this street /
   district), with the site card and a probe reading for the actual pit.
4. **"Natives grow slowly and look sparse."** Correct for some (ghaf). The guild mixes faster cover (rimth, ikrish, sea
   lavender) with slow canopy, and the utility rewards shade where it is hottest.
5. **"Nurseries don't stock these."** A real constraint — add nursery availability/cost to the trait table in incubation.
6. **"Validation is weak."** Yes: no survival ground truth exists. We use an independent mangrove reference, physical
   consistency, robustness and a proposed 12-month pilot (two verges, Ecoverse vs. standard).

## Similar solutions and how Ecoverse differs
- **Native-plant guides** (Qatar GORD / Qatar e-Nature, UAE Flora, ICBA *Salt Tolerant Plants of the UAE*, municipal lists):
  say which species are adapted to the region; do not say *where* each fits, and are static.
- **Urban tree-canopy / heat tools** (e.g. i-Tree, tree-equity style planning maps): say *where* to add trees; assume a
  temperate palette and irrigation.
- **Salinity mapping for agriculture** (e.g. ICBA's work on saline farmland and crop diversification): matches crops to
  saline farms for yield, not low-input urban landscapes.
- **Precision-agriculture platforms**: monitor crops already planted rather than choose species.

Ecoverse's new combination: per-pixel *what + where + how much water*, a measured hyperspectral case for Satellite 813, and
an uncertainty-driven ground loop. Policy timing helps: Bahrain banned new Conocarpus in 2022, so municipalities need a
replacement palette now.

## Ground probe kit (≈ USD 100, indicative prices)
| Item | Measures | ≈ USD |
|---|---|---|
| Handheld EC/TDS meter + 1:5 soil-water method | Soil salinity (EC 1:5 → ECe via texture factor) | 15 |
| Soil pH meter (or pH pen + distilled water) | pH | 15 |
| Capacitive soil-moisture sensor + ESP32 | Volumetric moisture | 12 |
| RS-485 soil NPK sensor | Indicative N, P, K | 35 |
| IR thermometer | Surface temperature (cross-check with Landsat) | 15 |
| Phone GPS + spreadsheet template | Location, photo, notes | 0 |

The fusion step is in `src/ecoverse/uncertainty.py`; the field template is `data/ground/ground_readings_TEMPLATE.csv`.

## Business model sketch
- **Customers:** municipalities, ministry landscaping programmes, master developers, landscape contractors.
- **Offer:** district screening map (subscription), per-project site report, probe campaign as a service.
- **Cost base:** open data + compute + a probe kit; Satellite 813 data via the incubation partnership.
- **Value metric:** m³ of irrigation water and maintenance hours saved per hectare.
