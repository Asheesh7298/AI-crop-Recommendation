# -*- coding: utf-8 -*-
"""
Train the deployable REAL-DATA crop model (multi-label, ~30 field crops) on the
coordinate-linked dataset (SoilGrids soil + Open-Meteo climate -> crops actually
grown). Saves real_model.pkl for the app's "By location" mode.

The district rows themselves double as a soil+climate cache: at inference the app
finds the nearest surveyed district to the farmer's location (instant, no live
SoilGrids call) and feeds its soil+climate to the model.
"""
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
APY = os.path.join(ROOT, "data", "real", "india_agri.csv")
REAL = os.path.join(ROOT, "data", "real", "real_dataset.csv")
OUT = os.path.join(ROOT, "real_model.pkl")

KEEP = {
 "Rice": "rice", "Wheat": "wheat", "Maize": "maize", "Cotton(lint)": "cotton", "Sugarcane": "sugarcane",
 "Gram": "chickpea", "Arhar/Tur": "pigeonpeas", "Moong(Green Gram)": "mungbean", "Urad": "blackgram",
 "Masoor": "lentil", "Groundnut": "groundnut", "Rapeseed &Mustard": "mustard", "Soyabean": "soybean",
 "Bajra": "pearlmillet", "Jowar": "sorghum", "Ragi": "fingermillet", "Potato": "potato", "Onion": "onion",
 "Sesamum": "sesamum", "Sunflower": "sunflower", "Jute": "jute", "Barley": "barley",
 "Dry chillies": "chillies", "Peas & beans (Pulses)": "peasbeans", "Small millets": "smallmillets",
 "Linseed": "linseed", "Sweet potato": "sweetpotato", "Safflower": "safflower", "Castor seed": "castor",
 "Horse-gram": "horsegram",
}
MIN_SHARE = 5.0
SOIL = ["soil_nitrogen", "soil_phh2o", "soil_soc", "soil_sand", "soil_clay", "soil_cec"]
CLIM = ["clim_temp", "clim_rainfall", "clim_humidity"]
FEATURES = SOIL + CLIM
norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())


def derive_targets():
    apy = pd.read_csv(APY); apy.columns = [c.strip() for c in apy.columns]
    for c in ("District", "Crop"):
        apy[c] = apy[c].astype(str).str.strip()
    apy = apy.dropna(subset=["Production"])
    apy = apy[(apy["Area"] > 0) & (apy["Crop_Year"] >= 2010) & (apy["Crop"].isin(KEEP))]
    apy["our"] = apy["Crop"].map(KEEP)
    tgt = {}
    for dist, g in apy.groupby("District"):
        area = g.groupby("our")["Area"].sum(); tot = area.sum()
        if tot <= 0:
            continue
        crops = sorted([c for c, a in area.items() if a / tot * 100 >= MIN_SHARE])
        if crops:
            tgt[norm(dist)] = crops
    return tgt


def main():
    df = pd.read_csv(REAL)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    tgt = derive_targets()
    df["cs"] = df["district"].map(lambda d: tgt.get(norm(d)))
    df = df[df["cs"].notna()].reset_index(drop=True)

    crops = sorted({c for s in df["cs"] for c in s})
    Y = np.array([[1 if c in s else 0 for c in crops] for s in df["cs"]])
    X = df[FEATURES].values
    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    model.fit(X, Y)

    # district cache for nearest-location lookup at inference
    cache = df[["district", "state", "lat", "lon"] + FEATURES].to_dict(orient="records")

    joblib.dump({"model": model, "crops": crops, "features": FEATURES,
                 "soil_features": SOIL, "clim_features": CLIM,
                 "n_districts": int(len(df)), "cache": cache}, OUT)
    print(f"[save] {OUT}")
    print(f"  districts: {len(df)} | crops: {len(crops)} | features: {len(FEATURES)}")
    print(f"  crops: {crops}")


if __name__ == "__main__":
    main()
