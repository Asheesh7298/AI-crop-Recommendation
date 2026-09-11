"""
Crop Recommendation - Model Training & Comparison
Final Year Project: System using IoT and Crop Recommendation AI
Team: Asheesh Kumar (2300970130034), Aditya Chaudhary (2300970130009)
Supervisor: Dr. Meenakshi Yadav

Trains and COMPARES several models on the crop-recommendation dataset:
  - ELM (Extreme Learning Machine)  <- the method used in the reference papers
  - Random Forest
  - XGBoost
  - Decision Tree, KNN, Naive Bayes, SVM  (extra baselines)

Saves the best model as crop_model.pkl plus per-crop nutrient means
(used later for fertilizer-gap advice) and report charts.

Uses the real Kaggle 'Crop_recommendation.csv' if present, otherwise a
synthetic fallback so the pipeline always runs.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from elm import ELMClassifier
from crops_meta import FEATURES

try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except ImportError:
    HAVE_SMOTE = False

CSV_PATH = "Crop_recommendation.csv"
AUG_PER_CLASS = 300      # SMOTE target samples per crop for the expanded dataset

# per-crop agronomic centres for the synthetic fallback only
CROP_PROFILES = {
    "rice": (80, 48, 40, 24, 82, 6.4, 236), "maize": (78, 48, 20, 22, 65, 6.2, 84),
    "chickpea": (40, 68, 80, 18, 16, 7.3, 80), "kidneybeans": (21, 67, 20, 20, 22, 5.7, 105),
    "pigeonpeas": (21, 68, 20, 27, 48, 5.8, 149), "mothbeans": (21, 48, 20, 28, 53, 6.8, 51),
    "mungbean": (21, 47, 20, 28, 85, 6.7, 48), "blackgram": (40, 67, 19, 30, 65, 7.1, 68),
    "lentil": (19, 68, 19, 25, 65, 6.9, 46), "pomegranate": (19, 18, 40, 22, 90, 6.4, 107),
    "banana": (100, 82, 50, 27, 80, 6.0, 105), "mango": (20, 27, 30, 31, 50, 5.7, 95),
    "grapes": (23, 132, 200, 24, 82, 6.0, 69), "watermelon": (99, 17, 50, 26, 85, 6.5, 51),
    "muskmelon": (100, 18, 50, 28, 92, 6.4, 25), "apple": (21, 134, 200, 23, 92, 5.9, 113),
    "orange": (20, 17, 10, 23, 92, 7.0, 110), "papaya": (50, 59, 50, 34, 92, 6.7, 143),
    "coconut": (22, 17, 31, 27, 95, 5.9, 175), "cotton": (118, 46, 20, 24, 80, 6.9, 80),
    "jute": (78, 47, 40, 25, 80, 6.7, 175), "coffee": (101, 29, 30, 26, 59, 6.8, 158),
}
NOISE = np.array([12, 12, 12, 3, 8, 0.4, 30])


def make_synthetic(n_per_crop=100, seed=42):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for crop, centre in CROP_PROFILES.items():
        s = rng.normal(np.array(centre, float), NOISE, size=(n_per_crop, len(FEATURES)))
        s = np.clip(s, 0, None)
        s[:, 5] = np.clip(s[:, 5], 3.5, 9.5)
        rows.append(s)
        labels += [crop] * n_per_crop
    df = pd.DataFrame(np.vstack(rows), columns=FEATURES)
    df["label"] = labels
    return df


def load_data():
    if os.path.exists(CSV_PATH):
        print(f"[data] Found {CSV_PATH} -> training on the REAL Kaggle dataset.")
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        return df, "real"
    print(f"[data] {CSV_PATH} not found -> using SYNTHETIC data.")
    return make_synthetic(), "synthetic"


def cross_validate(make_models, X, y, scaled_models, k=10):
    """Stratified k-fold CV, done manually so the custom ELM works and so the
    scaler is fit inside each fold (no leakage). Returns {name: (mean%, std%)}."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    Xv, yv = X.values, y.values
    accs = {name: [] for name in make_models()}
    for tr, te in skf.split(Xv, yv):
        Xtr, Xte, ytr, yte = Xv[tr], Xv[te], yv[tr], yv[te]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        le = LabelEncoder().fit(ytr)
        for name, model in make_models().items():
            use_scaled = name in scaled_models
            a, b = (Xtr_s, Xte_s) if use_scaled else (Xtr, Xte)
            if name == "XGBoost":
                model.fit(a, le.transform(ytr))
                pred = le.inverse_transform(model.predict(b))
            else:
                model.fit(a, ytr)
                pred = model.predict(b)
            accs[name].append(accuracy_score(yte, pred))
    return {n: (float(np.mean(v) * 100), float(np.std(v) * 100)) for n, v in accs.items()}


def smote_expand(X, y, per_class, seed=42):
    """Enlarge the dataset with SMOTE synthetic samples (the established, citable
    way to grow this crop dataset — cf. Zenodo 10.5281/zenodo.19709807). Each
    crop is oversampled up to `per_class`; classes already larger are kept."""
    counts = pd.Series(y).value_counts()
    strategy = {c: max(per_class, int(counts[c])) for c in counts.index}
    k = max(1, min(5, int(counts.min()) - 1))
    Xa, ya = SMOTE(sampling_strategy=strategy, random_state=seed,
                   k_neighbors=k).fit_resample(X, y)
    return Xa, ya


def main():
    df, source = load_data()
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    # XGBoost needs integer labels
    le = LabelEncoder().fit(y_train)

    def make_models():
        return {
            "ELM": ELMClassifier(n_hidden=200, activation="sigmoid"),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.2,
                subsample=0.9, eval_metric="mlogloss", random_state=42,
                tree_method="hist",
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True, random_state=42),
        }

    scaled_models = {"ELM", "KNN", "SVM"}
    results = {}
    print("\n=== Model comparison (test accuracy) ===")
    for name, model in make_models().items():
        use_scaled = name in scaled_models
        Xtr = Xtr_s if use_scaled else X_train
        Xte = Xte_s if use_scaled else X_test
        if name == "XGBoost":
            model.fit(Xtr, le.transform(y_train))
            preds = le.inverse_transform(model.predict(Xte))
        else:
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"  {name:15s}: {acc*100:.2f}%")

    # ---- stratified 10-fold cross-validation (robust estimate on small data) ----
    print("\n=== 10-fold cross-validation (mean ± std accuracy) ===")
    cv_results = cross_validate(make_models, X, y, scaled_models, k=10)
    for name in results:
        m, s = cv_results[name]
        print(f"  {name:15s}: {m:.2f}% ± {s:.2f}%")

    # ---- SMOTE augmentation: enlarge TRAIN only, evaluate on real held-out test ----
    aug_results = {}
    aug_train_size = None
    if HAVE_SMOTE:
        Xtr_aug, ytr_aug = smote_expand(X_train, y_train, per_class=AUG_PER_CLASS)
        aug_train_size = int(len(Xtr_aug))
        print(f"\n=== SMOTE-augmented training ({len(X_train)} -> {aug_train_size} rows), "
              f"tested on REAL held-out data ===")
        sc_aug = StandardScaler().fit(Xtr_aug)
        le_aug = LabelEncoder().fit(ytr_aug)
        for name, model in make_models().items():
            use_scaled = name in scaled_models
            a = sc_aug.transform(Xtr_aug) if use_scaled else Xtr_aug
            b = sc_aug.transform(X_test) if use_scaled else X_test
            if name == "XGBoost":
                model.fit(a, le_aug.transform(ytr_aug))
                preds = le_aug.inverse_transform(model.predict(b))
            else:
                model.fit(a, ytr_aug)
                preds = model.predict(b)
            aug_results[name] = accuracy_score(y_test, preds)
            print(f"  {name:15s}: {aug_results[name]*100:.2f}%")
        # save the full expanded dataset for the record / report
        Xful, yful = smote_expand(X, y, per_class=AUG_PER_CLASS)
        aug_df = pd.DataFrame(Xful, columns=FEATURES); aug_df["label"] = list(yful)
        os.makedirs("data", exist_ok=True)
        aug_df.to_csv(os.path.join("data", "crop_recommendation_smote.csv"), index=False)
        print(f"[save] data/crop_recommendation_smote.csv ({len(aug_df)} rows)")
    else:
        print("\n[skip] SMOTE augmentation (install imbalanced-learn to enable).")

    best_name = max(results, key=results.get)
    print(f"\n[best] {best_name} "
          f"(holdout {results[best_name]*100:.2f}%, "
          f"CV {cv_results[best_name][0]:.2f}% ± {cv_results[best_name][1]:.2f}%)")

    # retrain best on ALL data for deployment
    best_uses_scaled = best_name in scaled_models
    final_scaler = StandardScaler().fit(X)
    Xall = final_scaler.transform(X) if best_uses_scaled else X
    final_model = make_models()[best_name]
    if best_name == "XGBoost":
        le_all = LabelEncoder().fit(y)
        final_model.fit(Xall, le_all.transform(y))
    else:
        le_all = None
        final_model.fit(Xall, y)

    # per-crop nutrient means -> used by fertilizer.py for gap analysis
    crop_means = df.groupby("label")[FEATURES].mean().to_dict(orient="index")

    joblib.dump(
        {
            "model": final_model,
            "scaler": final_scaler,
            "uses_scaled": best_uses_scaled,
            "label_encoder": le_all,       # only for XGBoost
            "features": FEATURES,
            "classes": sorted(y.unique().tolist()),
            "model_name": best_name,
            "accuracy": results[best_name],
            "data_source": source,
            "all_results": results,
            "cv_results": cv_results,          # {name: (mean%, std%)} from 10-fold CV
            "cv_folds": 10,
            "n_samples": int(len(df)),
            "aug_results": aug_results,        # accuracy on real test, SMOTE-augmented training
            "aug_train_size": aug_train_size,
            "crop_means": crop_means,
            "X_sample": X.sample(min(200, len(X)), random_state=1).values,  # SHAP background
        },
        "crop_model.pkl",
    )
    print("[save] crop_model.pkl written.")

    # ---- charts ----
    plt.figure(figsize=(8.5, 4.5))
    names = list(results.keys())
    means = [cv_results[n][0] for n in names]
    stds = [cv_results[n][1] for n in names]
    colors = ["#2e7d32" if n == best_name else "#90a4ae" for n in names]
    bars = plt.bar(names, means, yerr=stds, capsize=4, color=colors,
                   error_kw={"ecolor": "#37474f", "elinewidth": 1})
    plt.ylabel("Accuracy (%)"); plt.ylim(0, 105)
    plt.title("Model Comparison — 10-fold CV accuracy (mean ± std)")
    plt.xticks(rotation=20, ha="right")
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width()/2, m + 2.5, f"{m:.1f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig("model_comparison.png", dpi=130); plt.close()

    # confusion matrix of best model
    if best_name == "XGBoost":
        preds = le_all.inverse_transform(
            final_model.predict(final_scaler.transform(X_test) if best_uses_scaled else X_test)
        )
    else:
        preds = final_model.predict(
            final_scaler.transform(X_test) if best_uses_scaled else X_test
        )
    labels_sorted = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, cmap="Greens"); plt.colorbar(fraction=0.046)
    plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=90, fontsize=7)
    plt.yticks(range(len(labels_sorted)), labels_sorted, fontsize=7)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=130); plt.close()

    # feature importance (if available)
    if hasattr(final_model, "feature_importances_"):
        imp = final_model.feature_importances_
        order = np.argsort(imp)[::-1]
        plt.figure(figsize=(7, 4))
        plt.bar([FEATURES[i] for i in order], imp[order], color="#1565c0")
        plt.ylabel("Importance"); plt.title(f"Feature Importance — {best_name}")
        plt.tight_layout(); plt.savefig("feature_importance.png", dpi=130); plt.close()

    print("[save] charts: model_comparison.png, confusion_matrix.png, feature_importance.png")
    print("\nDone. Run the app with:  python main.py   (then open http://localhost:5000)")


if __name__ == "__main__":
    main()
