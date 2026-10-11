"""Offline tests on SYNTHETIC fixtures (no internet, no real data).

Run:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoverse import fingerprint, hyperspectral, io, plants, recommend, uncertainty  # noqa: E402
from ecoverse.indices import band_depth, red_edge_position, simulate_s2  # noqa: E402
from ecoverse.synthetic import synthetic_stack, synthetic_tanager_table  # noqa: E402


def _engine(tmp_path):
    L, tr, crs = synthetic_stack()
    p = tmp_path / "stack.tif"
    io.write_stack(p, L, tr, crs, meta={"synthetic": True})
    layers, profile = io.read_stack(p)
    for k in L:  # int16 round trip within scale tolerance
        np.testing.assert_allclose(np.nan_to_num(layers[k]), np.nan_to_num(L[k]),
                                   atol=2.0 / io.LAYER_SCALES.get(k, 1) + 1e-6)
    fp = fingerprint.build_fingerprint(layers)
    kb = plants.load_plants()
    res = recommend.run_engine(fp, kb)
    return layers, profile, fp, kb, res


def test_plant_kb_is_consistent():
    kb = plants.load_plants()
    assert kb["species_id"].is_unique
    assert (kb["ece_survival_dSm"] > kb["ece_comfort_dSm"]).all()
    assert set(kb["layer"]) <= {"canopy", "shrub", "ground"}
    assert kb["plant_factor"].between(0, 1).all()


def test_salinity_curve():
    row = plants.load_plants().set_index("species_id").loc["bermuda"]
    env = {"ECE": np.array([3.0, 6.9, 14.7, 22.5, 40.0]), "HEAT_PCT": np.zeros(5),
           "WATERLOG": np.zeros(5), "INTERTIDAL": np.zeros(5, bool), "SAND": np.zeros(5),
           "PH": np.full(5, 8.0), "BUILT_DENSITY": np.zeros(5)}
    f = plants.factor_curves(row, env)["salinity"]
    assert f[0] == 1 and f[1] == 1 and 0 < f[2] < 1 and f[3] == 0 and f[4] == 0


def test_mangrove_only_in_intertidal(tmp_path):
    _, _, fp, kb, res = _engine(tmp_path)
    i = kb.index[kb.species_id == "qurm"][0]
    inter = fp["intertidal"][res.mask]
    assert (res.surv[i][~inter] == 0).all()
    assert (res.surv[i][inter] > 0).any()


def test_engine_outputs_and_water(tmp_path):
    _, profile, fp, kb, res = _engine(tmp_path)
    assert res.hero.shape == (res.mask.sum(),)
    assert (res.hero >= 0).mean() > 0.5
    ok = np.isfinite(res.saving_mm)
    assert (res.saving_mm[ok] > 0).all()            # natives always beat the baseline
    bud = recommend.water_budget(res, fp)
    assert (bud["saving_pct"].dropna() > 0).all()
    summ = recommend.species_summary(res)
    assert summ.loc[summ.species_id == "conocarpus", "recommended_pct_area"].item() == 0


def test_uncertainty_and_probes(tmp_path):
    _, profile, fp, kb, res = _engine(tmp_path)
    stab, draws = uncertainty.monte_carlo(fp, kb, res, n=4)
    assert stab.shape == res.hero.shape and np.all((stab >= 0) & (stab <= 1))
    probes = uncertainty.select_probe_sites(fp, res, stab, profile, n_sites=5)
    assert 1 <= len(probes) <= 5
    readings = probes.iloc[:1].assign(ece_dSm=30.0, ph=7.9)[["row", "col", "ece_dSm", "ph"]]
    fp2 = uncertainty.fuse_ground_readings(fp, readings)
    r0, c0 = int(probes.row[0]), int(probes.col[0])
    assert abs(np.log(fp2["ECE"][r0, c0]) - np.log(30.0)) < abs(np.log(fp["ECE"][r0, c0]) - np.log(30.0))
    assert fp2["ECE_LOG_SD"][r0, c0] < fp["ECE_LOG_SD"][r0, c0]


def test_archetypes(tmp_path):
    _, _, fp, _, _ = _engine(tmp_path)
    lab, cent = fingerprint.habitat_archetypes(fp, k=4)
    assert len(cent) == 4 and set(np.unique(lab)) <= {0, 1, 2, 3, 255}


def test_hyperspectral_features_physics():
    wl = np.arange(400, 2450, 5.0)
    flat = np.full((wl.size, 3), 0.3)
    dip = flat.copy()
    dip[:, 1] -= 0.1 * np.exp(-0.5 * ((wl - 1750) / 10) ** 2)
    d = band_depth(dip, wl, 1690, 1735, 1765, 1790)
    assert d[0] < 0.01 and d[1] > 0.1
    veg = 0.05 + 0.4 / (1 + np.exp(-(wl - 715) / 10))
    stressed = 0.05 + 0.4 / (1 + np.exp(-(wl - 700) / 10))
    rep = red_edge_position(np.stack([veg, stressed], 1), wl)
    assert rep[0] > rep[1]                    # stress shifts the red edge blue-ward
    s2 = simulate_s2(np.stack([veg, flat[:, 0]], 1), wl)
    assert s2["B8A"][0] > s2["B04"][0]


def test_transfer_cv_runs():
    df = synthetic_tanager_table()
    out = hyperspectral.evaluate_transfer(df, n_splits=3)
    assert "r2_boosted" in out
    rec = hyperspectral.soil_class_recovery(df, k=3, n_splits=3)
    assert 0 <= rec["balanced_accuracy"] <= 1


def test_prepare_readings(tmp_path):
    _, profile, fp, kb, res = _engine(tmp_path)
    from ecoverse.io import pixel_to_lonlat
    lon, lat = pixel_to_lonlat([10], [20], profile)
    df = pd.DataFrame({"lat": lat, "lon": lon, "ec_1to5_dSm": [1.0], "texture": ["loam"], "ph": [8.0]})
    out = uncertainty.prepare_readings(df, profile)
    assert int(out.row[0]) == 10 and int(out.col[0]) == 20
    assert abs(out.ece_dSm[0] - 8.6) < 1e-6
