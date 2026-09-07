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

from sklearn.model_selection import train_test_split
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

CSV_PATH = "Crop_recommendation.csv"

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

    best_name = max(results, key=results.get)
    print(f"\n[best] {best_name} ({results[best_name]*100:.2f}%)")

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
            "crop_means": crop_means,
            "X_sample": X.sample(min(200, len(X)), random_state=1).values,  # SHAP background
        },
        "crop_model.pkl",
    )
    print("[save] crop_model.pkl written.")

    # ---- charts ----
    plt.figure(figsize=(8.5, 4.5))
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    colors = ["#2e7d32" if n == best_name else "#90a4ae" for n in names]
    bars = plt.bar(names, accs, color=colors)
    plt.ylabel("Accuracy (%)"); plt.ylim(0, 105)
    plt.title("Model Comparison — ELM vs RF vs XGBoost (and baselines)")
    plt.xticks(rotation=20, ha="right")
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width()/2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
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
    print("\nDone. Run the app with:  python -m streamlit run app.py")


if __name__ == "__main__":
    main()
