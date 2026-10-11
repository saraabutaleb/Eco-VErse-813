"""Online data retrieval (run once by scripts/fetch_data.py).

All sources are open and need no credentials:
  * Sentinel-2 L2A          - Microsoft Planetary Computer STAC (ESA Copernicus)
  * Landsat 8/9 C2 L2 (ST)  - Microsoft Planetary Computer STAC (USGS)
  * Copernicus DEM GLO-30   - Microsoft Planetary Computer STAC
  * ESA WorldCover 2021     - Microsoft Planetary Computer STAC
  * Planet Tanager (open)   - Planet open STAC catalogue
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import requests

from . import config

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
SCL_VALID = (4, 5, 6, 7)   # vegetation, bare, water, unclassified


def _catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)


def _search(collection, bbox, datetime=None, query=None, limit=200):
    items = _catalog().search(collections=[collection], bbox=bbox, datetime=datetime,
                              query=query, max_items=limit).item_collection()
    return list(items)


def _load(items, bands, aoi, resampling):
    import odc.stac
    return odc.stac.load(items, bands=bands, crs=aoi.crs, resolution=aoi.resolution,
                         bbox=aoi.bbox, groupby="solar_day", resampling=resampling,
                         fail_on_error=False)


def sentinel2_composite(aoi, period, max_cloud, bands=S2_BANDS, persistence=True, log=print):
    items = _search("sentinel-2-l2a", aoi.bbox, period, {"eo:cloud_cover": {"lt": max_cloud}})
    items = [it for it in items
             if float(str(it.properties.get("s2:processing_baseline", "0"))) >= 4.0]
    if len(items) < 3:
        raise RuntimeError(f"Only {len(items)} Sentinel-2 scenes for {period}; widen dates or cloud limit")
    log(f"  Sentinel-2 {period}: {len(items)} scenes")
    ds = _load(items, list(bands) + ["SCL"], aoi, {"*": "bilinear", "SCL": "nearest"})
    scl = ds["SCL"].values
    valid = np.isin(scl, SCL_VALID)
    out = {}
    for b in bands:
        dn = ds[b].values.astype("float32")
        refl = (dn - 1000.0) / 10000.0          # processing baseline >= 04.00 offset
        refl[(dn == 0) | ~valid] = np.nan
        out[b] = np.nanmedian(refl, axis=0)
    if persistence:
        red = (ds["B04"].values.astype("float32") - 1000) / 1e4
        nir = (ds["B8A"].values.astype("float32") - 1000) / 1e4
        ndvi = (nir - red) / (nir + red + 1e-6)
        land_obs = valid & (scl != 6)
        green = (ndvi > 0.25) & land_obs
        nobs = land_obs.sum(axis=0)
        out["VEG_PERSIST"] = np.where(nobs > 0, green.sum(axis=0) / np.maximum(nobs, 1), np.nan)
        out["S2_NOBS"] = nobs.astype("float32")
    geobox = ds.odc.geobox
    prov = {"collection": "sentinel-2-l2a", "period": period, "n_scenes": len(items),
            "max_cloud_pct": max_cloud, "item_ids": [it.id for it in items]}
    return out, geobox, prov


def landsat_lst(aoi, period, max_cloud=30, log=print):
    try:
        items = _search("landsat-c2-l2", aoi.bbox, period,
                        {"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": ["landsat-8", "landsat-9"]}})
    except Exception:  # noqa: BLE001  (older STAC servers reject the "in" operator)
        items = [it for it in _search("landsat-c2-l2", aoi.bbox, period, {"eo:cloud_cover": {"lt": max_cloud}})
                 if it.properties.get("platform") in ("landsat-8", "landsat-9")]
    if not items:
        raise RuntimeError(f"No Landsat scenes for {period}")
    log(f"  Landsat 8/9 {period}: {len(items)} scenes")
    ds = _load(items, ["lwir11", "qa_pixel"], aoi, {"*": "bilinear", "qa_pixel": "nearest"})
    dn = ds["lwir11"].values.astype("float32")
    qa = ds["qa_pixel"].values.astype("uint16")
    bad = (dn == 0) | ((qa & 0b11111) != 0)   # fill, dilated cloud, cirrus, cloud, shadow
    lst = dn * 0.00341802 + 149.0 - 273.15
    lst[bad] = np.nan
    prov = {"collection": "landsat-c2-l2 (lwir11 surface temperature)", "period": period,
            "n_scenes": len(items), "item_ids": [it.id for it in items]}
    return np.nanmedian(lst, axis=0), prov


def single_layer(collection, band, aoi, resampling="bilinear"):
    items = _search(collection, aoi.bbox)
    if not items:
        raise RuntimeError(f"Nothing found in {collection}")
    items = sorted(items, key=lambda it: str(it.datetime or it.properties.get("start_datetime", "")))
    ds = _load(items, [band], aoi, {"*": resampling})
    da = ds[band]
    arr = da.values.astype("float32")
    nod = da.attrs.get("nodata")
    if nod is not None:
        arr[arr == nod] = np.nan
    if arr.ndim == 3:
        # DEM tiles mosaic across slices; for WorldCover keep the newest year.
        arr = arr[-1] if collection == "esa-worldcover" else np.nanmax(arr, axis=0)
    prov = {"collection": collection, "asset": band, "item_ids": [it.id for it in items]}
    return arr, prov


def fetch_aoi_stack(run: config.RunConfig, log=print):
    """Download and composite everything for the AOI into one layer dict."""
    from .io import write_stack
    aoi, per = run.aoi, run.periods
    log("Fetching Sentinel-2 summer composite ...")
    summer, geobox, p_s = sentinel2_composite(aoi, per.summer, per.max_cloud, log=log)
    log("Fetching Sentinel-2 winter composite ...")
    winter, _, p_w = sentinel2_composite(aoi, per.winter, per.max_cloud,
                                         bands=["B04", "B8A", "B11"], persistence=False, log=log)
    log("Fetching Landsat summer land surface temperature ...")
    lst, p_l = landsat_lst(aoi, per.summer, log=log)
    log("Fetching Copernicus DEM ...")
    dem, p_d = single_layer("cop-dem-glo-30", "data", aoi)
    log("Fetching ESA WorldCover ...")
    wc, p_wc = single_layer("esa-worldcover", "map", aoi, resampling="mode")

    layers = {**{b: summer[b] for b in S2_BANDS},
              "W_B04": winter["B04"], "W_B8A": winter["B8A"], "W_B11": winter["B11"],
              "VEG_PERSIST": summer["VEG_PERSIST"], "S2_NOBS": summer["S2_NOBS"],
              "LST_SUMMER": lst, "DEM": dem, "WORLDCOVER": wc}
    meta = {"aoi": {"name": aoi.name, "bbox_wgs84": aoi.bbox, "crs": aoi.crs,
                    "resolution_m": aoi.resolution},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "layers": list(layers),
            "sources": {"sentinel2_summer": p_s, "sentinel2_winter": p_w, "landsat_lst": p_l,
                        "dem": p_d, "worldcover": p_wc},
            "licences": {
                "Sentinel-2": "Copernicus Sentinel data, free and open (Copernicus licence)",
                "Landsat": "USGS, public domain",
                "Copernicus DEM GLO-30": "Copernicus DEM licence (free, attribution required)",
                "ESA WorldCover": "CC BY 4.0, (c) ESA WorldCover project 2021",
            }}
    write_stack(config.STACK_TIF, layers, geobox.affine, aoi.crs, meta)
    log(f"Saved {config.STACK_TIF}")
    return meta


# ---------------------------------------------------------------------------
# Planet Tanager open archive
# ---------------------------------------------------------------------------
def tanager_item(collection: str, item_id: str) -> dict:
    url = f"{config.TANAGER_STAC_ROOT}/{collection}/{item_id}/{item_id}.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    item = r.json()
    item.setdefault("collection", collection)
    return item


def _intersects(b, q):
    return not (b[2] < q[0] or b[0] > q[2] or b[3] < q[1] or b[1] > q[3])


def discover_gulf_scenes(collections=("coastal-water-bodies", "urban", "natural-lands",
                                      "agriculture", "energy-mining"), log=print):
    """Scan Tanager collections for scenes inside the Gulf search box."""
    found = []
    for col in collections:
        try:
            c = requests.get(f"{config.TANAGER_STAC_ROOT}/{col}/collection.json", timeout=60).json()
        except Exception as e:  # noqa: BLE001
            log(f"  skip {col}: {e}"); continue
        for link in c.get("links", []):
            if link.get("rel") != "item":
                continue
            href = link["href"]
            if not href.startswith("http"):
                href = f"{config.TANAGER_STAC_ROOT}/{col}/{href.lstrip('./')}"
            try:
                it = requests.get(href, timeout=60).json()
            except Exception:  # noqa: BLE001
                continue
            if it.get("bbox") and _intersects(it["bbox"], config.GULF_SEARCH_BBOX):
                found.append({"collection": col, "id": it["id"], "label": f"{col} {it['bbox']}"})
                log(f"  Gulf scene: {col}/{it['id']} bbox={np.round(it['bbox'], 3).tolist()}")
    return found


def download(url, dest: Path, log=print, chunk=8 * 1024 * 1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  already downloaded: {dest.name}"); return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)); done = 0; t0 = time.time()
        with open(tmp, "wb") as f:
            for blk in r.iter_content(chunk_size=chunk):
                f.write(blk); done += len(blk)
                if total and done % (256 * 1024 * 1024) < chunk:
                    log(f"    {done/1e9:.2f}/{total/1e9:.2f} GB ({done/1e6/(time.time()-t0+1e-9):.0f} MB/s)")
    tmp.rename(dest)
    return dest


def fetch_tanager(scenes=None, keep_h5=False, log=print):
    from .hyperspectral import extract_tanager_scene
    scenes = scenes or config.TANAGER_SCENES
    cache = config.DATA_DIR / "cache"
    metas = []
    for s in scenes:
        try:
            item = tanager_item(s["collection"], s["id"])
        except Exception as e:  # noqa: BLE001
            log(f"  could not load {s['id']}: {e}"); continue
        if not _intersects(item["bbox"], config.GULF_SEARCH_BBOX):
            log(f"  WARNING {s['id']} is outside the Gulf search box: {item['bbox']}")
        sr_key = "ortho_sr_hdf5" if "ortho_sr_hdf5" in item["assets"] else "basic_sr_hdf5"
        url = item["assets"][sr_key]["href"]
        log(f"Tanager {s['id']} ({s['label']}) -> {sr_key}")
        h5 = download(url, cache / Path(url).name, log=log)
        meta = extract_tanager_scene(h5, item, sr_key, config.TANAGER_DIR, s["label"])
        log(f"  extracted: valid {meta['valid_pct']:.1f}%, land {meta['land_pct']:.1f}%")
        metas.append(meta)
        if not keep_h5:
            h5.unlink(missing_ok=True)
    (config.TANAGER_DIR / "scenes.json").write_text(json.dumps(metas, indent=2))
    return metas
