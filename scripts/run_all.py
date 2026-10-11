#!/usr/bin/env python
"""Execute the analysis notebooks in place (outputs saved inside them), then
build docs/slides.pdf and refresh the README results block.

    python scripts/run_all.py
"""
import subprocess
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecoverse import config  # noqa: E402


def execute(nb_name, timeout=1800):
    path = ROOT / "notebooks" / nb_name
    nb = nbformat.read(path, as_version=4)
    print(f"Executing {nb_name} ...", flush=True)
    ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)
    print(f"  ok -> {path.relative_to(ROOT)}")


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if any(config.TANAGER_DIR.glob("*_meta.json")):
        execute("01_hyperspectral_gulf_calibration.ipynb")
    else:
        print("No Tanager subsets found - skipping notebook 01 (run scripts/fetch_data.py)")
    execute("02_main_analysis.ipynb")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_slides.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "update_readme.py")], check=True)


if __name__ == "__main__":
    main()
