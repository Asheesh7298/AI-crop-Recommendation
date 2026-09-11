# -*- coding: utf-8 -*-
"""
Real-data pipeline, offline stage.

Builds the multi-label targets and picks the district subset to query, joining:
  * APY district crop statistics  -> which crops are actually grown per district
  * district lat/lon               -> coordinates for SoilGrids + climate queries

Output: data/real/district_targets.csv  (district, state, lat, lon, n_crops, crops)
The next stage (fetch) adds SoilGrids soil + Open-Meteo climate columns.
"""
import os
import re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APY = os.path.join(ROOT, "data", "real", "india_agri.csv")
LATLON = os.path.join(ROOT, "data", "real", "districts_latlon.csv")
OUT = os.path.join(ROOT, "data", "real", "district_targets.csv")

# major field crops we keep as labels (present in APY, real cultivation)
KEEP = {
 "Rice": "rice", "Wheat": "wheat", "Maize": "maize", "Cotton(lint)": "cotton",
 "Sugarcane": "sugarcane", "Gram": "chickpea", "Arhar/Tur": "pigeonpeas",
 "Moong(Green Gram)": "mungbean", "Urad": "blackgram", "Masoor": "lentil",
 "Groundnut": "groundnut", "Rapeseed &Mustard": "mustard", "Soyabean": "soybean",
 "Bajra": "pearlmillet", "Jowar": "sorghum", "Ragi": "fingermillet",
 "Potato": "potato", "Onion": "onion", "Sesamum": "sesamum", "Sunflower": "sunflower",
 "Jute": "jute", "Barley": "barley",
}
MIN_SHARE = 5.0   # a crop counts as "grown here" if >= 5% of the district's kept-crop area


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def main():
    apy = pd.read_csv(APY); apy.columns = [c.strip() for c in apy.columns]
    for c in ("District", "Crop", "State"):
        apy[c] = apy[c].astype(str).str.strip()
    apy = apy.dropna(subset=["Production"])
    apy = apy[(apy["Area"] > 0) & (apy["Crop_Year"] >= 2010) & (apy["Crop"].isin(KEEP))]
    apy["our"] = apy["Crop"].map(KEEP)

    ll = pd.read_csv(LATLON); ll.columns = [c.strip() for c in ll.columns]
    ll["key"] = ll["District"].map(norm)
    latlon = {r["key"]: (r["Latitude"], r["Longitude"], r["State"]) for _, r in ll.iterrows()}

    rows = []
    for dist, g in apy.groupby("District"):
        key = norm(dist)
        if key not in latlon:
            continue
        lat, lon, st = latlon[key]
        area = g.groupby("our")["Area"].sum()
        total = area.sum()
        if total <= 0:
            continue
        crops = sorted([c for c, a in area.items() if a / total * 100 >= MIN_SHARE])
        if len(crops) < 2:                      # need a few crops to be a useful multi-label row
            continue
        rows.append({"district": dist, "state": st, "lat": round(float(lat), 3),
                     "lon": round(float(lon), 3), "n_crops": len(crops), "crops": "|".join(crops)})

    out = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"])
    out.to_csv(OUT, index=False)
    print(f"matched districts with coords + crop sets: {len(out)}")
    print("states covered:", out['state'].nunique())
    print("avg crops/district:", round(out['n_crops'].mean(), 1))
    from collections import Counter
    cc = Counter(c for s in out['crops'] for c in s.split("|"))
    print("crop frequency across districts:", dict(cc.most_common()))


if __name__ == "__main__":
    main()
