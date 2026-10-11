# Finish and submit (deadline 11 Oct 2026, 23:59 Bahrain time)

Everything is built and tested. These steps produce the real results on a machine with internet and get them into GitHub.
Budget about 1–1.5 hours, most of it waiting for downloads.

## A. Laptop route (recommended)
1. Install Python 3.12 and Git.
2. Unzip this folder, open a terminal inside `ecoverse-813/`:
   ```bash
   python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pytest -q                                             # should print "9 passed"
   python scripts/fetch_data.py                          # 15–30 min; downloads open data only
   python scripts/run_all.py                             # runs both notebooks, builds docs/slides.pdf, fills README results
   ```
   If time is short: `python scripts/fetch_data.py --skip-tanager` (≈ 5 min). Everything still runs; the hyperspectral
   slide and notebook 01 are skipped — you lose the hyperspectral bonus, so try the full run first.
3. Edit `docs/team.json` (names, roles, contact) and the Team table at the bottom of `README.md`, then re-run
   `python scripts/make_slides.py`.
4. Open `results/` and `docs/slides.pdf` and look at every figure. Check the numbers make sense before you present them.

## B. Colab route (if the laptop install fails)
Push the repo to GitHub first (step C), put your repo URL in `notebooks/00_run_everything_colab.ipynb`, open it in Colab,
Run all, download `ecoverse_outputs.zip`, unzip it over your local copy, then commit.

## C. GitHub
```bash
git add -A
git commit -m "Ecoverse 813 PoC: real data, results and slides"
git branch -M main
git remote add origin https://github.com/saraabutaleb/Eco-verse-813.git
git push -u origin main
```
- Make the repo **public**, open the link in a private/incognito window to confirm it loads.
- Replace `YOUR-TEAM` in `README.md`, `docs/team.json` and notebook 00.
- Check `data/sample_input/` is a few MB to ~25 MB (no `.h5` files — `.gitignore` blocks them).

## D. Submission form
Team + all members registered · country · **Theme 05 Sustainable Urban Planning & Smart Cities** · GitHub link ·
attach `docs/slides.pdf`. Submit with time to spare.

## If something fails
- `fetch_data.py` Sentinel-2 error "Only N scenes": raise `max_cloud` or widen `summer` in `src/ecoverse/config.py`.
- Tanager scene fails to load: `python scripts/fetch_data.py --discover-tanager`, copy a listed Gulf scene into
  `TANAGER_SCENES` in `config.py`, then `python scripts/fetch_data.py --only-tanager`.
- A notebook cell fails: the full error is in the terminal; fix and run `python scripts/run_all.py` again.
