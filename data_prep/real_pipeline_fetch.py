# -*- coding: utf-8 -*-
"""
Real-data pipeline, fetch stage (long-running, resumable).

For each district in district_targets.csv, fetch:
  * SoilGrids soil properties (nitrogen, pH, SOC, sand, clay, CEC) at the coordinate
  * Open-Meteo climate normals (annual mean temp, mean humidity, total rainfall)
and append to data/real/real_dataset.csv. Re-running resumes where it stopped.

SoilGrids is throttled (~5 req/min) so this is slow by design.
"""
import os
import time
import datetime
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = os.path.join(ROOT, "data", "real", "district_targets.csv")
OUT = os.path.join(ROOT, "data", "real", "real_dataset.csv")

SG_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SG_PROPS = ["nitrogen", "phh2o", "soc", "sand", "clay", "cec"]
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
SG_DELAY = 1          # small pause; SoilGrids' own ~13s latency does most of the spacing
MAX_DISTRICTS = 160   # cap for a practical, state-spread subset
# district lat/lon are city centres (urban-masked -> null); try likely-rural offsets
# FIRST, centroid last, and cap attempts so one bad district can't stall the run.
OFFSETS = [(0.25, 0), (0, 0.25), (-0.25, 0), (0, -0.25), (0, 0)]


def _sg_once(lat, lon):
    params = [("lon", lon), ("lat", lat), ("depth", "0-5cm"), ("value", "mean")]
    params += [("property", p) for p in SG_PROPS]
    try:
        r = requests.get(SG_URL, params=params, timeout=60)
        if r.status_code != 200:
            return None
        layers = r.json()["properties"]["layers"]
        out = {f"soil_{l['name']}": l["depths"][0]["values"]["mean"] for l in layers}
        if all(v is None for v in out.values()):
            return None
        return out
    except Exception:
        return None


def soilgrids(lat, lon):
    """Try the centroid, then nearby rural offsets (urban centres return null)."""
    for i, (dla, dlo) in enumerate(OFFSETS):
        out = _sg_once(lat + dla, lon + dlo)
        if out is not None:
            return out
        if i < len(OFFSETS) - 1:
            time.sleep(SG_DELAY)
    return None


def climate(lat, lon):
    end = datetime.date.today().year - 1
    params = {"latitude": lat, "longitude": lon,
              "start_date": f"{end-4}-01-01", "end_date": f"{end}-12-31",
              "daily": "temperature_2m_mean,precipitation_sum",
              "hourly": "relative_humidity_2m", "timezone": "auto"}
    try:
        d = requests.get(ARCHIVE, params=params, timeout=60).json()
        dd = d.get("daily", {})
        temps = [t for t in dd.get("temperature_2m_mean", []) if t is not None]
        precs = [p for p in dd.get("precipitation_sum", []) if p is not None]
        rhs = [h for h in d.get("hourly", {}).get("relative_humidity_2m", []) if h is not None]
        yrs = max(1, (end - (end - 4) + 1))
        return {"clim_temp": round(sum(temps)/len(temps), 1) if temps else None,
                "clim_rainfall": round(sum(precs)/yrs, 0) if precs else None,   # annual mm
                "clim_humidity": round(sum(rhs)/len(rhs), 1) if rhs else None}
    except Exception:
        return None


def main():
    tg = pd.read_csv(TARGETS)
    # spread across states: up to ~8 districts/state, richer crop sets first
    tg = tg.sort_values(["state", "n_crops"], ascending=[True, False])
    tg = tg.groupby("state").head(8).reset_index(drop=True).head(MAX_DISTRICTS)

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev["lat"], prev["lon"]))
        print(f"resuming: {len(done)} districts already fetched")

    header_written = os.path.exists(OUT)
    n = 0
    for _, row in tg.iterrows():
        if (row["lat"], row["lon"]) in done:
            continue
        soil = soilgrids(row["lat"], row["lon"])
        time.sleep(SG_DELAY)
        if soil is None:
            continue
        clim = climate(row["lat"], row["lon"])
        if clim is None or clim["clim_temp"] is None:
            continue
        rec = {"district": row["district"], "state": row["state"], "lat": row["lat"],
               "lon": row["lon"], "crops": row["crops"], **soil, **clim}
        pd.DataFrame([rec]).to_csv(OUT, mode="a", header=not header_written, index=False)
        header_written = True
        n += 1
        if n % 10 == 0:
            print(f"  fetched {n} new districts...", flush=True)
    print(f"DONE. fetched {n} new districts. total file: {OUT}")


if __name__ == "__main__":
    main()
