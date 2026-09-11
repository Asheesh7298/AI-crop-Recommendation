# -*- coding: utf-8 -*-
"""Regenerate the real cultivation cross-check + yields for the 30-crop field set,
so the app's real-data mode can ground its recommendations. Writes:
  data/real/state_crops.json  (per-state crop area share)
  data/real/crop_yield.json    (real median yield q/ha)
"""
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APY = os.path.join(ROOT, "data", "real", "india_agri.csv")
OUT = os.path.join(ROOT, "data", "real")

CROP_MAP = {
 "Rice": "rice", "Wheat": "wheat", "Maize": "maize", "Cotton(lint)": "cotton", "Sugarcane": "sugarcane",
 "Gram": "chickpea", "Arhar/Tur": "pigeonpeas", "Moong(Green Gram)": "mungbean", "Urad": "blackgram",
 "Masoor": "lentil", "Groundnut": "groundnut", "Rapeseed &Mustard": "mustard", "Soyabean": "soybean",
 "Bajra": "pearlmillet", "Jowar": "sorghum", "Ragi": "fingermillet", "Potato": "potato", "Onion": "onion",
 "Sesamum": "sesamum", "Sunflower": "sunflower", "Jute": "jute", "Barley": "barley",
 "Dry chillies": "chillies", "Peas & beans (Pulses)": "peasbeans", "Small millets": "smallmillets",
 "Linseed": "linseed", "Sweet potato": "sweetpotato", "Safflower": "safflower", "Castor seed": "castor",
 "Horse-gram": "horsegram",
}
REV = {v: k for k, v in CROP_MAP.items()}

df = pd.read_csv(APY); df.columns = [c.strip() for c in df.columns]
for c in ("State", "District", "Crop"):
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["Production"])
df = df[(df["Area"] > 0) & (df["Crop_Year"] >= 2010) & (df["Crop"].isin(CROP_MAP.keys()))].copy()
df["our"] = df["Crop"].map(CROP_MAP)
df["yq"] = (df["Production"] / df["Area"]) * 10.0
df = df[(df["yq"] > 0) & (df["yq"] < 2000)]

# real yields
yields = {c: round(float(g["yq"].median()), 1) for c, g in df.groupby("our")}

# per-state cultivation shares
sa = df.groupby(["State", "our"])["Area"].sum().reset_index()
state_crops = {}
for st, g in sa.groupby("State"):
    tot = g["Area"].sum()
    if tot <= 0:
        continue
    shares = {r["our"]: round(r["Area"] / tot * 100, 1) for _, r in g.iterrows()}
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    state_crops[st] = {"shares": shares, "ranks": {c: i + 1 for i, (c, _) in enumerate(ranked)},
                       "total_crops": len(ranked)}

json.dump(yields, open(os.path.join(OUT, "crop_yield.json"), "w"), indent=1)
json.dump(state_crops, open(os.path.join(OUT, "state_crops.json"), "w"), indent=1)
print(f"crops with yields: {len(yields)} | states: {len(state_crops)}")
print("yields:", yields)
