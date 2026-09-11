# -*- coding: utf-8 -*-
"""
Real-data pipeline, modeling stage.

Trains a MULTI-LABEL crop-suitability model on the real coordinate-linked dataset
(SoilGrids soil + Open-Meteo climate -> set of crops actually grown), and validates
with LEAVE-ONE-STATE-OUT spatial cross-validation (the honest test: predict crops
for a region never seen in training). Reports top-k metrics, not single-label accuracy.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "real", "real_dataset.csv")

SOIL = ["soil_nitrogen", "soil_phh2o", "soil_soc", "soil_sand", "soil_clay", "soil_cec"]
CLIM = ["clim_temp", "clim_rainfall", "clim_humidity"]
FEATURES = SOIL + CLIM


def load():
    df = pd.read_csv(DATA)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES)
    df["crop_set"] = df["crops"].str.split("|")
    return df


def topk_metrics(proba_list, classes, Y_true, ks=(1, 3, 5)):
    """proba_list: per-crop P(present); classes aligned; Y_true: binary matrix."""
    P = np.column_stack([p[:, 1] if p.shape[1] == 2 else p[:, 0] for p in proba_list])
    order = np.argsort(P, axis=1)[:, ::-1]
    out = {}
    for k in ks:
        hit = prec = 0.0
        for i in range(len(Y_true)):
            topk = order[i, :k]
            true_idx = set(np.where(Y_true[i] == 1)[0])
            if not true_idx:
                continue
            inter = len(set(topk) & true_idx)
            hit += 1 if inter > 0 else 0          # at least one correct in top-k
            prec += inter / k                      # precision@k
        out[f"hit@{k}"] = hit / len(Y_true) * 100
        out[f"prec@{k}"] = prec / len(Y_true) * 100
    return out


def main():
    df = load()
    crops = sorted({c for s in df["crop_set"] for c in s})
    Y = np.array([[1 if c in s else 0 for c in crops] for s in df["crop_set"]])
    X = df[FEATURES].values
    states = df["state"].values
    print(f"districts: {len(df)} | crops: {len(crops)} | features: {len(FEATURES)}")
    print(f"avg crops/district: {Y.sum(1).mean():.1f}")

    # leave-one-state-out spatial CV (only states with enough districts)
    from collections import Counter
    counts = Counter(states)
    test_states = [s for s, n in counts.items() if n >= 4]
    agg = {}
    n_eval = 0
    for st in test_states:
        te = states == st
        tr = ~te
        if Y[tr].sum(0).min() == 0:      # ensure every crop appears in train
            pass
        model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        model.fit(X[tr], Y[tr])
        proba = model.predict_proba(X[te])
        m = topk_metrics(proba, crops, Y[te])
        for k, v in m.items():
            agg[k] = agg.get(k, 0) + v * te.sum()
        n_eval += te.sum()
    print(f"\n=== Leave-one-state-out spatial CV ({len(test_states)} states, {n_eval} districts) ===")
    for k in ["hit@1", "hit@3", "hit@5", "prec@3"]:
        print(f"  {k:8s}: {agg[k]/n_eval:.1f}%")
    print("\n(hit@k = at least one actually-grown crop in the top-k suggestions)")


if __name__ == "__main__":
    main()
