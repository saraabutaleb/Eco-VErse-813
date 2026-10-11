#!/usr/bin/env python
"""Download and prepare all example input data for Ecoverse 813.

Needs internet. Sentinel-2, Landsat, DEM and WorldCover need no account. The NASA EMIT
hyperspectral step needs a free NASA Earthdata login, read from the environment variables
EARTHDATA_USERNAME and EARTHDATA_PASSWORD (never stored). Writes only small, clipped,
openly-licensed files into data/sample_input/ (safe to commit).

    python scripts/fetch_data.py               # everything (~10-20 min)
    python scripts/fetch_data.py --skip-emit   # Sentinel-2/Landsat/DEM/WorldCover only (~3-6 min)
    python scripts/fetch_data.py --only-emit   # NASA EMIT hyperspectral subset only

Exact parameters (AOI, dates, cloud limit, scene IDs) are in src/ecoverse/config.py
and are recorded, with every STAC item ID / EMIT granule used, in
data/sample_input/bahrain_aoi_stack_metadata.json and data/sample_input/emit_bahrain/*_meta.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoverse import config, emit, fetch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-emit", action="store_true")
    ap.add_argument("--only-emit", action="store_true")
    args = ap.parse_args()

    run = config.RunConfig()
    if not args.only_emit:
        fetch.fetch_aoi_stack(run)
    if not args.skip_emit:
        meta = emit.fetch_emit(run.aoi.bbox)
        print(f"EMIT: {meta['granule']} {meta.get('datetime')} AOI cloud-free {meta['aoi_valid_pct']:.0f}%")
    print("Done. Now run the notebooks (see README).")


if __name__ == "__main__":
    main()
