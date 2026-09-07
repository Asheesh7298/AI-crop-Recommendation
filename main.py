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
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # load OPENWEATHER_API_KEY from .env before anything else

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crops_meta import FEATURES, emoji
import fertility, fertilizer, season as season_mod, weather as weather_mod
import yield_profit, irrigation, rotation, explain

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
MODEL_PATH = os.path.join(BASE, "crop_model.pkl")
_bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

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
        "data_source": _bundle.get("data_source", "unknown"),
        "season": season_mod.current_season(),
        "weather_key": weather_mod.has_key(),
    }


@app.get("/api/weather")
def api_weather(city: str = "Delhi"):
    return weather_mod.get_weather(city)


@app.post("/api/predict")
def api_predict(inp: SoilInput):
    if _bundle is None:
        return JSONResponse({"error": "model_not_trained"}, status_code=503)

    N, P, K = inp.N, inp.P, inp.K
    ph, rainfall = inp.ph, inp.rainfall
    season_now = season_mod.current_season()

    # Stage 1 - fertility
    frep = fertility.assess(N, P, K, ph)

    # Stage 2 - recommendation
    row = pd.DataFrame([[N, P, K, inp.temperature, inp.humidity, ph, rainfall]], columns=FEATURES)
    ranked = predict_ranked(_bundle, row, top_n=6)
    in_season = [(c, p) for c, p in ranked if season_mod.is_in_season(c, season_now)]
    display = in_season if in_season else ranked
    best_crop, best_p = display[0]
    top = [{"crop": c, "emoji": emoji(c), "confidence": round(p, 4),
            "in_season": season_mod.is_in_season(c, season_now)} for c, p in display[:4]]

    # Stage 3 - optimization
    candidates = [c for c, _ in display[:4]]
    fert_costs = {c: fertilizer.estimated_cost(c, N, P, K, _bundle["crop_means"]) for c in candidates}
    profit = yield_profit.rank_by_profit(candidates, fert_costs)

    # Stage 4 - explanation
    pairs = explain.explain_instance(_bundle, row.values[0]) or []
    explanation = [{"feature": f, "value": round(float(v), 4)} for f, v in pairs]

    return {
        "season": season_now,
        "fertility": {"items": frep, "overall": fertility.overall(frep)},
        "best": {"crop": best_crop, "emoji": emoji(best_crop),
                 "confidence": round(best_p, 4),
                 "season_note": season_mod.season_note(best_crop, season_now)},
        "top": top,
        "profit": profit,
        "fertilizer": {"tips": fertilizer.advise(best_crop, N, P, K, _bundle["crop_means"]),
                       "cost": fertilizer.estimated_cost(best_crop, N, P, K, _bundle["crop_means"])},
        "irrigation": irrigation.advise(best_crop, rainfall, inp.soil_moisture),
        "rotation": rotation.suggest(best_crop),
        "explain": explanation,
        "model": {"name": _bundle["model_name"], "accuracy": _bundle["accuracy"]},
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))  # hosts (Render) inject $PORT
    print(f"Crop Recommendation AI  ->  http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
