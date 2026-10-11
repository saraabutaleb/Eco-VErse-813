#!/usr/bin/env python
"""Download and prepare all example input data for Ecoverse 813.

Needs internet; no accounts, keys or tokens. Writes only small, clipped,
openly-licensed files into data/sample_input/ (safe to commit).

    python scripts/fetch_data.py                 # everything (~10-25 min, Tanager is the slow part)
    python scripts/fetch_data.py --skip-tanager  # Sentinel-2/Landsat/DEM/WorldCover only (~3-6 min)
    python scripts/fetch_data.py --only-tanager
    python scripts/fetch_data.py --discover-tanager   # list open Tanager scenes inside the Gulf

Exact parameters (AOI, dates, cloud limit, scene IDs) are in src/ecoverse/config.py
and are recorded, with every STAC item ID used, in
data/sample_input/bahrain_aoi_stack_metadata.json and tanager_gulf/scenes.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoverse import config, fetch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tanager", action="store_true")
    ap.add_argument("--only-tanager", action="store_true")
    ap.add_argument("--discover-tanager", action="store_true")
    ap.add_argument("--keep-h5", action="store_true", help="keep the multi-GB Tanager HDF5 in data/cache")
    args = ap.parse_args()

    if args.discover_tanager:
        fetch.discover_gulf_scenes()
        return

    run = config.RunConfig()
    if not args.only_tanager:
        fetch.fetch_aoi_stack(run)
    if not args.skip_tanager:
        metas = fetch.fetch_tanager(keep_h5=args.keep_h5)
        if not metas:
            print("No Tanager scene could be prepared. Try --discover-tanager and edit "
                  "TANAGER_SCENES in src/ecoverse/config.py.")
    print("Done. Now run the notebooks (see README).")


if __name__ == "__main__":
    main()
