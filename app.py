"""
Crop Recommendation AI - Smart Precision-Agriculture Dashboard
Final Year Project: System using IoT and Crop Recommendation AI
Team: Asheesh Kumar (2300970130034), Aditya Chaudhary (2300970130009)
Supervisor: Dr. Meenakshi Yadav

Pipeline: location/weather + soil -> fertility assessment -> season-aware
crop recommendation -> yield/profit + fertilizer + irrigation + rotation
-> SHAP explanation.  Models compared: ELM vs Random Forest vs XGBoost.

Run:  python -m streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from crops_meta import FEATURES, emoji, CROPS
import fertility, fertilizer, season as season_mod, weather as weather_mod
import yield_profit, irrigation, rotation, explain

st.set_page_config(page_title="Crop Recommendation AI", page_icon="🌱", layout="wide")
MODEL_PATH = "crop_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_ranked(bundle, row_df, top_n=5):
    """Return list of (crop, probability) sorted desc."""
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
    # no proba -> single prediction
    pred = model.predict(x)[0]
    if bundle["model_name"] == "XGBoost" and bundle.get("label_encoder") is not None:
        pred = bundle["label_encoder"].inverse_transform([pred])[0]
    return [(pred, 1.0)]


bundle = load_model()

# ---------------- header ----------------
st.title("🌱 Crop Recommendation AI — Smart Precision Agriculture")
st.caption(
    "Final Year Project · System using IoT and Crop Recommendation AI · "
    "Asheesh Kumar (2300970130034) & Aditya Chaudhary (2300970130009) · "
    "Supervisor: Dr. Meenakshi Yadav"
)

if bundle is None:
    st.error("Model not found. Run  `python train_model.py`  first.")
    st.stop()

season_now = season_mod.current_season()
src = bundle.get("data_source", "unknown")
res = bundle.get("all_results", {})
res_str = " · ".join(f"{k} {v*100:.1f}%" for k, v in res.items()) if res else ""

c1, c2, c3 = st.columns(3)
c1.metric("Best model", bundle["model_name"], f"{bundle['accuracy']*100:.2f}% acc")
c2.metric("Current season", season_now)
c3.metric("Dataset", "Real (Kaggle)" if src == "real" else "Synthetic")
if res_str:
    st.caption(f"Model comparison — {res_str}")
if src == "synthetic":
    st.warning("Running on synthetic data. Add real `Crop_recommendation.csv` and re-run `train_model.py`.")

# ---------------- sidebar inputs ----------------
st.sidebar.header("📍 Location & Weather")
city = st.sidebar.text_input("City (for live weather)", value="Delhi")
if st.sidebar.button("🌦️ Fetch live weather", use_container_width=True):
    w = weather_mod.get_weather(city)
    if "error" in w:
        msg = {"no_api_key": "No OpenWeather API key set (OPENWEATHER_API_KEY). Using manual values.",
               "requests_not_installed": "`requests` not installed. Using manual values."}.get(
            w["error"], f"Weather unavailable ({w['error']}). Using manual values.")
        st.sidebar.info(msg)
    else:
        st.session_state["temperature"] = w["temperature"]
        st.session_state["humidity"] = w["humidity"]
        st.session_state["rainfall"] = w["rainfall"]
        st.sidebar.success(f"Loaded {w['source']}: {w['temperature']}°C, {w['humidity']}% RH")

st.sidebar.header("🔬 Soil (from sensors / manual)")
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 145, 42)
K = st.sidebar.slider("Potassium (K)", 0, 205, 43)
ph = st.sidebar.slider("Soil pH", 3.5, 9.5, 6.5, 0.1)
soil_moisture = st.sidebar.slider("Soil moisture (%) — optional IoT", 0, 100, 45)

st.sidebar.header("🌤️ Weather (auto or manual)")
temperature = st.sidebar.slider("Temperature (°C)", 5.0, 45.0,
                                float(st.session_state.get("temperature", 21.0)), 0.1)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0,
                             float(st.session_state.get("humidity", 82.0)), 0.1)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0,
                             float(st.session_state.get("rainfall", 200.0)), 1.0)

go = st.sidebar.button("🌾 Recommend Crop", type="primary", use_container_width=True)

# ---------------- Stage 1: soil fertility ----------------
st.subheader("🧪 Stage 1 — Soil Fertility Assessment")
report = fertility.assess(N, P, K, ph)
fcols = st.columns(len(report))
badge = {"Low": "🔴", "Medium": "🟡", "High": "🟢",
         "Acidic": "🟠", "Neutral": "🟢", "Alkaline": "🔵"}
for col, (name, level) in zip(fcols, report.items()):
    col.metric(name, f"{badge.get(level,'⚪')} {level}")
st.caption(f"Overall: **{fertility.overall(report)}**")

st.divider()

if not go:
    st.info("Set soil & weather in the sidebar (or fetch live weather), then click **Recommend Crop**.")
    st.stop()

# ---------------- run prediction ----------------
row = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
ranked = predict_ranked(bundle, row, top_n=6)

# split season-appropriate vs off-season
in_season = [(c, p) for c, p in ranked if season_mod.is_in_season(c, season_now)]
display = in_season if in_season else ranked
best_crop, best_p = display[0]

# ---------------- Stage 2: crop recommendation ----------------
st.subheader("🎯 Stage 2 — Crop Recommendation (season-aware)")
left, right = st.columns([1, 1])
with left:
    st.markdown(f"## {emoji(best_crop)} **{best_crop.capitalize()}**")
    st.caption(season_mod.season_note(best_crop, season_now))
    st.progress(best_p)
    st.write(f"Model confidence: **{best_p*100:.1f}%**")
with right:
    st.markdown("**Top matches this season:**")
    for c, p in display[:4]:
        flag = "" if season_mod.is_in_season(c, season_now) else " (off-season)"
        st.write(f"{emoji(c)} {c.capitalize()} — {p*100:.1f}%{flag}")
        st.progress(p)

st.divider()

# ---------------- Stage 3: optimization ----------------
st.subheader("💡 Stage 3 — Optimization")

candidates = [c for c, _ in display[:4]]
fert_costs = {c: fertilizer.estimated_cost(c, N, P, K, bundle["crop_means"]) for c in candidates}
profit_rows = yield_profit.rank_by_profit(candidates, fert_costs)

o1, o2 = st.columns(2)
with o1:
    st.markdown("**💰 Profitability ranking (indicative)**")
    dfp = pd.DataFrame([{
        "Crop": f"{r['emoji']} {r['crop']}",
        "Yield (q/ha)": r["yield_q_ha"],
        "Price (₹/q)": r["price_q"],
        "Fert ₹/ha": r["fert_cost"],
        "Net ₹/ha": r["net_profit"],
    } for r in profit_rows])
    st.dataframe(dfp, hide_index=True, use_container_width=True)
    top_profit = profit_rows[0]
    if top_profit["crop"] != best_crop:
        st.caption(f"💵 Most *profitable* among suitable crops: "
                   f"{top_profit['emoji']} **{top_profit['crop']}** (₹{top_profit['net_profit']:,}/ha).")

with o2:
    st.markdown("**🧴 Fertilizer advice for " + best_crop + "**")
    for tip in fertilizer.advise(best_crop, N, P, K, bundle["crop_means"]):
        st.write("- " + tip)
    cost = fertilizer.estimated_cost(best_crop, N, P, K, bundle["crop_means"])
    st.caption(f"Estimated extra fertilizer cost: **₹{cost:,}/ha** (indicative).")

o3, o4 = st.columns(2)
with o3:
    st.markdown("**💧 Irrigation advice**")
    for tip in irrigation.advise(best_crop, rainfall, soil_moisture):
        st.write("- " + tip)
with o4:
    st.markdown("**🔄 Crop rotation suggestion**")
    rot = rotation.suggest(best_crop)
    st.write(rot["reason"])
    st.write("Next-season options: " +
             ", ".join(f"{p['emoji']} {p['crop']}" for p in rot["picks"]))

st.divider()

# ---------------- Stage 4: explainability ----------------
st.subheader("🧠 Stage 4 — Why this crop? (Explainable AI)")
pairs = explain.explain_instance(bundle, row.values[0])
if pairs:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(feats, vals, color=colors)
    ax.set_title(f"Feature contributions to '{best_crop}'")
    ax.axvline(0, color="#333", linewidth=0.8)
    st.pyplot(fig)
    top_feat = pairs[0][0]
    st.caption(f"Biggest driver: **{top_feat}**. Green = pushed toward this crop, red = against.")
else:
    st.caption("Explanation unavailable for this model.")

st.divider()

# ---------------- model insights ----------------
st.subheader("📊 Model Insights (for report / viva)")
ic1, ic2, ic3 = st.columns(3)
for col, img, cap in [
    (ic1, "model_comparison.png", "ELM vs RF vs XGBoost"),
    (ic2, "feature_importance.png", "Feature importance"),
    (ic3, "confusion_matrix.png", "Confusion matrix"),
]:
    if os.path.exists(img):
        col.image(img, caption=cap, use_container_width=True)
    else:
        col.caption(f"({cap} — run train_model.py)")

with st.expander("ℹ️ About & IoT future scope"):
    st.markdown(
        """
        **Pipeline:** location/weather (OpenWeather) + soil sensors → soil-fertility
        assessment → season-aware crop recommendation → yield/profit optimization,
        fertilizer, irrigation & rotation advice → SHAP explanation. Models compared:
        **ELM** (reference-paper method) vs **Random Forest** vs **XGBoost**.

        **IoT module (future scope):** an ESP32 with NPK, pH, DHT22 and soil-moisture
        sensors POSTs live readings to a `/predict` endpoint, so recommendations happen
        automatically in the field. This dashboard accepts manual or sensor values
        interchangeably.
        """
    )
