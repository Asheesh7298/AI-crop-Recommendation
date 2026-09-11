"""
FastAPI backend for the Crop Recommendation AI web app.
Final Year Project: System using IoT and Crop Recommendation AI
Team: Asheesh Kumar (2300970130034), Aditya Chaudhary (2300970130009)

Serves the custom frontend and a JSON API. POST /api/predict is also the
endpoint the IoT ESP32 module will send sensor readings to.

Run:  python main.py         (or:  uvicorn main:app --reload)
Then open http://localhost:5000
"""

import os
import time
import math
from datetime import date
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # load OPENWEATHER_API_KEY from .env before anything else

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crops_meta import FEATURES, FEATURE_RANGES, emoji, month_name
import fertility, fertilizer, season as season_mod, weather as weather_mod
import yield_profit, irrigation, rotation, explain, cultivation, economics

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
MODEL_PATH = os.path.join(BASE, "crop_model.pkl")
REAL_MODEL_PATH = os.path.join(BASE, "real_model.pkl")
_bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
_real = joblib.load(REAL_MODEL_PATH) if os.path.exists(REAL_MODEL_PATH) else None
_latest = None  # most recent IoT device reading (kept in memory)

app = FastAPI(title="Crop Recommendation AI")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class SoilInput(BaseModel):
    N: float = 90
    P: float = 42
    K: float = 43
    temperature: float = 21
    humidity: float = 82
    ph: float = 6.5
    rainfall: float = 200
    soil_moisture: Optional[float] = None
    month: Optional[int] = None  # month the farmer intends to start farming (1-12)
    city: Optional[str] = None   # location, for the real-cultivation cross-check


def range_warnings(values):
    """Flag any input outside the training-data range so a bad sensor/weather
    value cannot silently drive a wrong recommendation."""
    warns = []
    for feat, val in values.items():
        lo, hi = FEATURE_RANGES.get(feat, (None, None))
        if lo is None or val is None:
            continue
        if val < lo or val > hi:
            warns.append(f"{feat} = {val} is outside the usual range ({lo}–{hi}); "
                         f"the recommendation may be less reliable.")
    return warns


def predict_ranked(bundle, row_df, top_n=6):
    model = bundle["model"]
    x = bundle["scaler"].transform(row_df) if bundle["uses_scaled"] else row_df
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        if bundle["model_name"] == "XGBoost" and bundle.get("label_encoder") is not None:
            classes = bundle["label_encoder"].classes_
        else:
            classes = model.classes_
        order = np.argsort(proba)[::-1][:top_n]
        return [(classes[i], float(proba[i])) for i in order]
    pred = model.predict(x)[0]
    if bundle["model_name"] == "XGBoost" and bundle.get("label_encoder") is not None:
        pred = bundle["label_encoder"].inverse_transform([pred])[0]
    return [(pred, 1.0)]


# ---------------- static pages ----------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/charts/{name}")
def charts(name: str):
    path = os.path.join(BASE, name)
    if name.endswith(".png") and os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "not_found"}, status_code=404)


# ---------------- API ----------------
@app.get("/api/meta")
def meta():
    if _bundle is None:
        return JSONResponse({"error": "model_not_trained"}, status_code=503)
    return {
        "model": _bundle["model_name"],
        "accuracy": _bundle["accuracy"],
        "all_results": _bundle.get("all_results", {}),
        "cv_results": _bundle.get("cv_results", {}),
        "cv_folds": _bundle.get("cv_folds"),
        "n_samples": _bundle.get("n_samples"),
        "data_source": _bundle.get("data_source", "unknown"),
        "season": season_mod.current_season(),
        "weather_key": weather_mod.has_key(),
    }


@app.get("/api/weather")
def api_weather(city: str = "Delhi", month: Optional[int] = None):
    res = weather_mod.get_weather(city, month)
    if "error" not in res:
        m = res.get("month") or date.today().month
        res["season"] = season_mod.season_of_month(int(m))
    return res


# ---------- Real-data model: SoilGrids soil + climate -> crops actually grown ----------
def _haversine(la1, lo1, la2, lo2):
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_district(lat, lon):
    best, bestd = None, 1e18
    for row in _real["cache"]:
        d = _haversine(lat, lon, row["lat"], row["lon"])
        if d < bestd:
            bestd, best = d, row
    return best, round(bestd)


@app.get("/api/real_meta")
def real_meta():
    if _real is None:
        return JSONResponse({"error": "real_model_not_trained"}, status_code=503)
    return {"crops": _real["crops"], "n_districts": _real["n_districts"]}


@app.get("/api/recommend_real")
def recommend_real(city: str = "Lucknow", month: Optional[int] = None):
    """Location-driven recommendation from the real-data model (nearest surveyed
    district's SoilGrids soil + climate -> multi-label crop suitability)."""
    if _real is None:
        return JSONResponse({"error": "real_model_not_trained"}, status_code=503)
    geo = weather_mod._geocode(city)
    if not geo:
        return {"error": "geocode_failed", "message": f"Could not locate '{city}'. Check the spelling."}
    lat, lon, name, admin = geo
    nd, km = _nearest_district(lat, lon)
    # climate live for the ACTUAL location (fast, accurate); soil from nearest survey
    live = weather_mod.annual_climate(lat, lon)
    soil_feats = _real["soil_features"]; clim_feats = _real["clim_features"]
    soil_vals = {f: nd[f] for f in soil_feats}
    clim_vals = {f: (live[f] if live and f in live else nd[f]) for f in clim_feats}
    x = [[{**soil_vals, **clim_vals}[f] for f in _real["features"]]]
    crops = _real["crops"]
    proba = _real["model"].predict_proba(x)
    scores = [float(p[0][1]) if p.shape[1] == 2 else float(p[0][0]) for p in proba]
    order = sorted(range(len(crops)), key=lambda i: scores[i], reverse=True)[:6]
    top = [{"crop": crops[i], "emoji": emoji(crops[i]), "score": round(scores[i], 3),
            "yield_q_ha": economics.yield_q_ha(crops[i])[0]} for i in order]
    plant_month = int(month) if month and 1 <= int(month) <= 12 else date.today().month
    return {
        "location": name, "state": admin,
        "nearest_survey": {"district": nd["district"], "state": nd["state"], "km": km},
        "soil_climate": {
            "soil_pH": round(nd["soil_phh2o"] / 10, 1),
            "soil_nitrogen": nd["soil_nitrogen"],
            "sand_pct": round(nd["soil_sand"] / 10),
            "clay_pct": round(nd["soil_clay"] / 10),
            "temperature": clim_vals["clim_temp"],
            "humidity": clim_vals["clim_humidity"],
            "rainfall_annual": clim_vals["clim_rainfall"],
        },
        "climate_source": "live (Open-Meteo)" if live else "nearest survey",
        "top": top,
        "grounding": cultivation.check(top[0]["crop"], admin),
        "season": season_mod.season_of_month(plant_month),
        "month_name": month_name(plant_month),
        "model": {"name": "Real-data multi-label (SoilGrids + APY)", "n_districts": _real["n_districts"]},
    }


def analyze(N, P, K, temperature, humidity, ph, rainfall, soil_moisture, month=None, city=None):
    """Run the full pipeline for one reading. Shared by /api/predict and /api/ingest.
    `month` is the month (1-12) the farmer intends to start farming; the season
    filter and sowing-time advice are computed for that month, defaulting to now.
    `city` (optional) drives the real-world cultivation cross-check."""
    plant_month = month if month and 1 <= int(month) <= 12 else date.today().month
    plant_month = int(plant_month)
    season_now = season_mod.season_of_month(plant_month)

    frep = fertility.assess(N, P, K, ph)                                   # Stage 1
    ni_val, ni_cls = fertility.nutrient_index(N, P, K)
    row = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
    ranked = predict_ranked(_bundle, row, top_n=6)                         # Stage 2
    # The model's top match for the given conditions IS the recommendation.
    # Season is advisory only (never overrides a confident prediction).
    best_crop, best_p = ranked[0]
    top = [{"crop": c, "emoji": emoji(c), "confidence": round(p, 4),
            "in_season": season_mod.is_in_season(c, season_now)} for c, p in ranked[:4]]
    # If the recommended crop is out of season, offer in-season alternatives.
    season_alt = [{"crop": c, "emoji": emoji(c), "confidence": round(p, 4)}
                  for c, p in ranked[1:]
                  if season_mod.is_in_season(c, season_now)][:3]
    best_in_season = season_mod.is_in_season(best_crop, season_now)

    candidates = [c for c, _ in ranked[:4]]                                # Stage 3
    fert_costs = {c: fertilizer.estimated_cost(c, N, P, K) for c in candidates}
    profit = yield_profit.rank_by_profit(list(ranked[:4]), fert_costs)

    # Real-world grounding: is the recommended crop actually grown in this state?
    grounding = cultivation.check(best_crop, weather_mod.get_state(city)) if city else None

    pairs = explain.explain_instance(_bundle, row.values[0]) or []         # Stage 4
    explanation = [{"feature": f, "value": round(float(v), 4)} for f, v in pairs]

    warnings = range_warnings({"N": N, "P": P, "K": K, "temperature": temperature,
                               "humidity": humidity, "ph": ph, "rainfall": rainfall})

    return {
        "season": season_now,
        "month": plant_month,
        "month_name": month_name(plant_month),
        "fertility": {"items": frep, "overall": fertility.overall(frep),
                      "index": {"value": ni_val, "class": ni_cls}},
        "best": {"crop": best_crop, "emoji": emoji(best_crop),
                 "confidence": round(best_p, 4),
                 "in_season": best_in_season,
                 "season_note": season_mod.season_note(best_crop, season_now)},
        "timing": season_mod.timing_verdict(best_crop, plant_month),
        "season_alt": season_alt,
        "grounding": grounding,
        "top": top,
        "profit": profit,
        "fertilizer": {"tips": fertilizer.advise(best_crop, N, P, K),
                       "cost": fertilizer.estimated_cost(best_crop, N, P, K)},
        "irrigation": irrigation.advise(best_crop, rainfall, soil_moisture),
        "rotation": rotation.suggest(best_crop),
        "explain": explanation,
        "warnings": warnings,
        "model": {"name": _bundle["model_name"], "accuracy": _bundle["accuracy"]},
    }


@app.post("/api/predict")
def api_predict(inp: SoilInput):
    if _bundle is None:
        return JSONResponse({"error": "model_not_trained"}, status_code=503)
    return analyze(inp.N, inp.P, inp.K, inp.temperature, inp.humidity,
                   inp.ph, inp.rainfall, inp.soil_moisture, inp.month, inp.city)


# ---------------- IoT device endpoints ----------------
class SensorReading(SoilInput):
    """What an ESP32 field device POSTs to /api/ingest."""
    device_id: str = "esp32-01"


@app.post("/api/ingest")
def api_ingest(reading: SensorReading, x_device_token: Optional[str] = Header(default=None)):
    """
    Ingestion endpoint for IoT sensor rigs. An ESP32 reads NPK/pH/DHT22/
    soil-moisture and POSTs the JSON here. We store it as the latest reading
    (so the dashboard's live mode can show it) and return the recommendation
    back to the device. Optional auth: set DEVICE_TOKEN to require a matching
    X-Device-Token header.
    """
    token = os.environ.get("DEVICE_TOKEN")
    if token and x_device_token != token:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if _bundle is None:
        return JSONResponse({"error": "model_not_trained"}, status_code=503)

    result = analyze(reading.N, reading.P, reading.K, reading.temperature,
                     reading.humidity, reading.ph, reading.rainfall,
                     reading.soil_moisture, reading.month, reading.city)

    global _latest
    _latest = {
        "device_id": reading.device_id,
        "ts": time.time(),
        "reading": {"N": reading.N, "P": reading.P, "K": reading.K,
                    "ph": reading.ph, "soil_moisture": reading.soil_moisture,
                    "temperature": reading.temperature, "humidity": reading.humidity,
                    "rainfall": reading.rainfall},
        "best": result["best"],
    }
    return {"stored": True, "device_id": reading.device_id, "result": result}


@app.get("/api/latest")
def api_latest():
    """Latest device reading, for the dashboard's live-sensor mode."""
    if _latest is None:
        return {"reading": None}
    return {**_latest, "age_seconds": round(time.time() - _latest["ts"], 1)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))  # hosts (Render) inject $PORT
    print(f"Crop Recommendation AI  ->  http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
