"""Central configuration for Ecoverse 813.

Everything a reviewer might want to change (area of interest, dates, grid,
constants) lives here so the notebooks never hard-code values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the repository root -> reproducible on any machine)
# ----------------------------------------------------------------------------
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
# The two env overrides exist only so the offline test harness can point the
# notebooks at a synthetic fixture without touching the real folders.
SAMPLE_DIR = Path(os.environ.get("ECOVERSE_SAMPLE_DIR", DATA_DIR / "sample_input"))
PLANTS_CSV = DATA_DIR / "plants" / "bahrain_plant_traits.csv"
GROUND_DIR = DATA_DIR / "ground"
RESULTS_DIR = Path(os.environ.get("ECOVERSE_RESULTS_DIR", REPO_ROOT / "results"))
DOCS_DIR = REPO_ROOT / "docs"

STACK_TIF = SAMPLE_DIR / "bahrain_aoi_stack.tif"          # multi-layer 20 m input
STACK_META = SAMPLE_DIR / "bahrain_aoi_stack_metadata.json"
TANAGER_DIR = SAMPLE_DIR / "tanager_gulf"                  # (not used: earlier Gulf-analogue approach)
EMIT_DIR = SAMPLE_DIR / "emit_bahrain"                     # NASA EMIT hyperspectral subset over the AOI

RANDOM_SEED = 813


# ----------------------------------------------------------------------------
# Area of interest: northern-central Bahrain
# Covers a full environmental gradient on purpose: Tubli Bay mangroves and
# intertidal flats, reclaimed coastal land (Sitra), dense urban fabric
# (Manama / Isa Town) and inland date-palm belts.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AOI:
    name: str = "Northern-central Bahrain (Manama - Tubli Bay - Isa Town - Sitra)"
    bbox: tuple = (50.50, 26.12, 50.64, 26.24)  # lon_min, lat_min, lon_max, lat_max
    crs: str = "EPSG:32639"                       # UTM zone 39N (metres)
    resolution: int = 20                          # metres


AOI_DEFAULT = AOI()


@dataclass(frozen=True)
class Periods:
    # Hottest, driest months: the stress test every plant must survive.
    summer: str = "2025-06-01/2025-08-31"
    # Coolest months: used for the seasonal-contrast (irrigation dependence) signal.
    winter: str = "2025-12-01/2026-02-28"
    max_cloud: int = 20


PERIODS_DEFAULT = Periods()


# ----------------------------------------------------------------------------
# Hyperspectral calibration ("Gulf analogue") scenes from Planet Tanager's
# OPEN archive. Bahrain itself is not in the open archive, so we learn
# hyperspectral soil/vegetation signals on climatically similar Gulf scenes
# (coastal sabkha of Abu Dhabi's Al Dhafra coast + arid city of Riyadh) and
# transfer them to Sentinel-2. Satellite 813 data replaces this in incubation.
# ----------------------------------------------------------------------------
TANAGER_STAC_ROOT = "https://www.planet.com/data/stac/tanager-core-imagery"
TANAGER_SCENES = [
    {"collection": "coastal-water-bodies", "id": "20250511_074311_00_4001",
     "label": "Al Dhafra coast, Abu Dhabi (coastal sabkha)"},
    {"collection": "urban", "id": "20250515_080954_16_4001",
     "label": "Riyadh (arid city, irrigated parks)"},
    {"collection": "natural-lands", "id": "20250608_074757_58_4001",
     "label": "Rub al Khali margin (desert sand)"},
]
GULF_SEARCH_BBOX = (45.0, 22.0, 57.0, 28.5)  # used to auto-discover more scenes

# NASA EMIT search window (EMIT has imaged Bahrain since August 2022)
EMIT_TEMPORAL = ("2022-08-01", "2026-09-30")

# ----------------------------------------------------------------------------
# Agronomic constants
# ----------------------------------------------------------------------------
ET0_MM_YR = 1850.0          # Bahrain reference evapotranspiration (Fanack Water)
ET0_RANGE = (1800.0, 2100.0)
RAINFALL_MM_YR = 75.0       # ~70-80 mm/yr, effectively negligible for irrigation
REGIONAL_PH_PRIOR = (8.1, 0.35)   # mean, sd: calcareous Gulf soils (prior until probed)

# Salinity prior: maps a 0..1 "salinity pressure" score to ECe (dS/m) on a log
# scale. 0 -> 3 dS/m (irrigated garden soil), 0.5 -> ~12 dS/m, 1 -> 50 dS/m
# (abandoned farmland / sabkha). Anchors from FAO AQUASTAT Bahrain profile.
ECE_PRIOR_MIN = 3.0
ECE_PRIOR_MAX = 50.0

# Baseline "business as usual" planting for water comparisons:
# Conocarpus canopy over a bermudagrass lawn (both WUCOLS 'high').
BASELINE_PLANT_FACTOR = 0.8

# Guild layer cover fractions used to compute a site's landscape water factor.
GUILD_COVER = {"canopy": 0.4, "shrub": 0.3, "ground": 0.3}

# A species "survives" a pixel when its suitability is at or above this.
SURVIVAL_THRESHOLD = 0.6


@dataclass
class RunConfig:
    aoi: AOI = field(default_factory=AOI)
    periods: Periods = field(default_factory=Periods)
    seed: int = RANDOM_SEED
    n_monte_carlo: int = 40
    n_probe_sites: int = 12
    probe_min_spacing_m: float = 500.0
