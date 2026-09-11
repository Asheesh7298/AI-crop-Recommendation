# -*- coding: utf-8 -*-
"""
30-crop test: widen the real-data model's crop set from 22 -> 30 (APY has 55),
re-derive multi-label targets for the already-fetched districts (no re-fetch),
and validate with leave-one-state-out spatial CV. Honest top-k metrics.
"""
import os, re
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APY = os.path.join(ROOT, "data", "real", "india_agri.csv")
REAL = os.path.join(ROOT, "data", "real", "real_dataset.csv")

KEEP = {  # 30 crops (22 major + 8 more with good district coverage)
 "Rice":"rice","Wheat":"wheat","Maize":"maize","Cotton(lint)":"cotton","Sugarcane":"sugarcane",
 "Gram":"chickpea","Arhar/Tur":"pigeonpeas","Moong(Green Gram)":"mungbean","Urad":"blackgram",
 "Masoor":"lentil","Groundnut":"groundnut","Rapeseed &Mustard":"mustard","Soyabean":"soybean",
 "Bajra":"pearlmillet","Jowar":"sorghum","Ragi":"fingermillet","Potato":"potato","Onion":"onion",
 "Sesamum":"sesamum","Sunflower":"sunflower","Jute":"jute","Barley":"barley",
 # +8
 "Dry chillies":"chillies","Peas & beans (Pulses)":"peasbeans","Small millets":"smallmillets",
 "Linseed":"linseed","Sweet potato":"sweetpotato","Safflower":"safflower","Castor seed":"castor",
 "Horse-gram":"horsegram",
}
MIN_SHARE = 5.0
SOIL = ["soil_nitrogen","soil_phh2o","soil_soc","soil_sand","soil_clay","soil_cec"]
CLIM = ["clim_temp","clim_rainfall","clim_humidity"]
FEATURES = SOIL + CLIM
norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())


def derive_targets():
    apy = pd.read_csv(APY); apy.columns=[c.strip() for c in apy.columns]
    for c in ("District","Crop"): apy[c]=apy[c].astype(str).str.strip()
    apy=apy.dropna(subset=["Production"])
    apy=apy[(apy["Area"]>0)&(apy["Crop_Year"]>=2010)&(apy["Crop"].isin(KEEP))]
    apy["our"]=apy["Crop"].map(KEEP)
    tgt={}
    for dist,g in apy.groupby("District"):
        area=g.groupby("our")["Area"].sum(); tot=area.sum()
        if tot<=0: continue
        crops=sorted([c for c,a in area.items() if a/tot*100>=MIN_SHARE])
        if crops: tgt[norm(dist)]=crops
    return tgt


def main():
    df=pd.read_csv(REAL)
    for c in FEATURES: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=FEATURES).reset_index(drop=True)
    tgt=derive_targets()
    df["cs"]=df["district"].map(lambda d: tgt.get(norm(d)))
    df=df[df["cs"].notna()].reset_index(drop=True)
    crops=sorted({c for s in df["cs"] for c in s})
    Y=np.array([[1 if c in s else 0 for c in crops] for s in df["cs"]])
    X=df[FEATURES].values; states=df["state"].values
    print(f"districts: {len(df)} | crops: {len(crops)} | avg crops/district: {Y.sum(1).mean():.1f}")
    print("crops:", crops)

    counts=Counter(states); test_states=[s for s,n in counts.items() if n>=4]
    agg={}; n_eval=0
    for st in test_states:
        te=states==st; tr=~te
        m=MultiOutputClassifier(RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1))
        m.fit(X[tr],Y[tr]); proba=m.predict_proba(X[te])
        P=np.column_stack([p[:,1] if p.shape[1]==2 else p[:,0] for p in proba])
        order=np.argsort(P,axis=1)[:,::-1]
        for i,row in enumerate(np.where(te)[0]):
            true=set(np.where(Y[row]==1)[0])
            if not true: continue
            for k in (1,3,5):
                inter=len(set(order[i,:k])&true)
                agg[f"hit@{k}"]=agg.get(f"hit@{k}",0)+(1 if inter>0 else 0)
                if k==3: agg["prec@3"]=agg.get("prec@3",0)+inter/3
            n_eval+=1
    print(f"\n=== 30-CROP leave-one-state-out spatial CV ({len(test_states)} states, {n_eval} districts) ===")
    for k in ["hit@1","hit@3","hit@5","prec@3"]:
        print(f"  {k:8s}: {agg[k]/n_eval*100:.1f}%")


if __name__=="__main__":
    main()
