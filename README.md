<div align="center">

# 🌱 System using IoT and Crop Recommendation AI

**A soil-to-harvest decision engine — soil-fertility assessment, crop recommendation with
sowing-time guidance, real-world cultivation cross-check, real profitability &amp; fertilizer
optimization, and an explainable AI account of every call.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-EB0000)
![Deploy](https://img.shields.io/badge/deploy-Render-46E3B7?logo=render&logoColor=white)

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Machine Learning &amp; Results](#-machine-learning--results)
5. [Data](#-data)
6. [Tech Stack](#-tech-stack)
7. [Project Structure](#-project-structure)
8. [Getting Started](#-getting-started)
9. [Configuration](#-configuration)
10. [How It Works — the Pipeline](#-how-it-works--the-pipeline)
11. [API Reference](#-api-reference)
12. [IoT Module](#-iot-module)
13. [Deployment](#-deployment-render)
14. [Honest Limitations](#-honest-limitations)
15. [Research Background](#-research-background)
16. [Roadmap](#-roadmap--future-scope)
17. [Team](#-team)
18. [License](#-license)

---

## 🌾 Overview

Choosing the right crop depends on **soil chemistry** (nitrogen, phosphorus, potassium, pH) and
**local climate**. This project turns those readings into a concrete, defensible planting plan,
and — importantly — **grounds every recommendation in real data**:

> **Soil &amp; climate readings → fertility assessment (Parker's Nutrient Index) → crop recommendation
> → real-world cultivation cross-check → real profitability, fertilizer, irrigation &amp; rotation
> advice → SHAP explanation.**

The crop model is trained on the benchmark Kaggle dataset, but the **economics, fertilizer doses,
and a real-world "is this actually grown here?" cross-check are all built from real Government of
India data** (see [Honest Limitations](#-honest-limitations)). It is **software-sufficient today
and IoT-ready by design**: the same JSON endpoint the web app uses is the ingestion path for an
ESP32 field-sensor node.

---

## ✨ Key Features

- **🧪 Soil fertility grading** — N, P, K and pH rated Low / Medium / High, combined into an overall
  **Parker's Nutrient Index** (standard soil-science method).
- **🎯 Crop recommendation** — the model recommends the best of **22 crops** from the soil + climate
  readings you provide, with top matches and confidence.
- **📅 Planting-month &amp; sowing-time advice** — pick the month you plan to sow; the system says whether
  it's the right time and, if not, **when to start** (Kharif / Rabi / Zaid aware).
- **✅ Real-world cross-check** — every recommendation is checked against **real Govt-of-India
  cultivation data**: *"Rice — major crop in Uttar Pradesh, grown on ~68% of cropland."*
- **💰 Real profitability** — ranks crops by `real yield × real price − fertilizer cost`, using
  **real district yields (APY statistics)** and **real MSP 2024-25 prices**.
- **🧴 Real fertilizer advice** — **ICAR recommended N-P₂O₅-K₂O doses**, soil-test adjusted, converted
  to actual urea / DAP / MOP quantities with an indicative cost.
- **💧 Irrigation &amp; 🔄 rotation** — water schedule from crop need + rainfall; nitrogen-fixing
  rotation suggestion.
- **🧠 Explainable AI (SHAP)** — shows *why* each crop was recommended, feature by feature.
- **🌦️ Seasonal climate (advisory)** — **Open-Meteo** climate normals for your location &amp; month
  (free, **no API key**) shown as context; it does **not** override your field readings.
- **📊 Model study** — benchmarks **ELM vs Random Forest vs XGBoost** (+ baselines) with **10-fold
  cross-validation**.
- **🔌 IoT loop** — ESP32 firmware + device simulator + live-sensor dashboard mode.
- **🖥️ Custom web UI** — responsive single-page dashboard, light/dark theme (no Streamlit).

---

## 🏗 System Architecture

```
 ┌──────────────── INPUTS (field readings) ────────────────┐
 │  Soil: N, P, K, pH, moisture  (manual sliders OR ESP32)  │
 │  Field climate: temperature · humidity · rainfall        │
 │  Location (city) + planting month                        │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
   ╔═══════════════════ RECOMMENDATION PIPELINE ══════════════════╗
   ║  1. Soil fertility + Parker's Nutrient Index (fertility.py)   ║
   ║  2. Crop recommendation      (ELM / RF / XGBoost)            ║
   ║     sowing-time advice        (season.py)                    ║
   ║  3. Grounding + optimization                                 ║
   ║       • real-world cross-check   (cultivation.py + APY data)  ║
   ║       • real profitability       (yield_profit.py + economics)║
   ║       • real ICAR fertilizer     (fertilizer.py)             ║
   ║       • irrigation / rotation    (irrigation.py / rotation.py)║
   ║  4. Explainability — SHAP        (explain.py)                ║
   ╚════════════════════════════┬═════════════════════════════════╝
                             ▼
   Open-Meteo climate normals ──►  (advisory context only, never a model input)
                             ▼
   FastAPI backend (main.py)  ──►  Web dashboard (static/index.html)
            ▲                                   ▲
  POST /api/ingest                        GET /api/latest
            └────── ESP32 sensor node ──────────┘  (live sensor mode)
```

---

## 🤖 Machine Learning &amp; Results

Seven models are trained and compared. **ELM** is the method used in the project's reference
papers; **Random Forest** and **XGBoost** are the stronger modern baselines. Reported with
**stratified 10-fold cross-validation** (mean ± std) on the benchmark dataset:

| Model | 10-fold CV accuracy |
|-------|:-------------------:|
| **Random Forest** ⭐ (selected) | **99.59% ± 0.38%** |
| Naive Bayes | 99.50% ± 0.43% |
| XGBoost | 99.36% ± 0.46% |
| Decision Tree | 98.95% ± 0.54% |
| ELM (reference-paper method) | 98.45% ± 0.87% |
| SVM | 98.32% ± 1.00% |
| KNN | 97.59% ± 1.32% |

The training set is also enlarged with **SMOTE** (1,760 → 6,600 rows) as a robustness check;
accuracy holds at ~99% on the real held-out split. The best model is saved to `crop_model.pkl`,
and figures (`model_comparison.png`, `confusion_matrix.png`, `feature_importance.png`) are served
at `/charts/*`.

> ⚠️ These numbers are **cross-validation accuracy on the benchmark dataset**, not real-field
> validation — see [Honest Limitations](#-honest-limitations).

> **ELM** (`elm.py`) is implemented from scratch in NumPy: random hidden weights + a
> Moore–Penrose pseudo-inverse — no back-propagation, near-instant training.

---

## 📊 Data

**Crop model** — *Crop Recommendation Dataset* (Kaggle): 2,200 rows, 7 features, **22 crops**.
Source: <https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>

**Real grounding data** (in `data/real/`, derived from Government of India statistics):

| Layer | Source | Used for |
|-------|--------|----------|
| District Area-Production-Yield (345K rows, 707 districts, 1997–2020) | Govt of India APY / data.gov.in | real yields + cultivation cross-check |
| Minimum Support Prices (MSP) 2024-25 | CACP / Govt of India | real crop prices |
| ICAR / SAU recommended fertilizer doses | ICAR packages of practice | real fertilizer advice |
| Monthly climate normals | Open-Meteo Archive API (10-yr) | seasonal climate context |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend / API | **FastAPI** + Uvicorn |
| ML | scikit-learn, XGBoost, custom ELM (NumPy), imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Frontend | Vanilla HTML / CSS / JS (single page, no build step) |
| Data | pandas, NumPy; Kaggle dataset + Govt of India APY / MSP / ICAR |
| Climate | **Open-Meteo** (free, no API key) |
| IoT | ESP32 (Arduino / C++), REST over WiFi |
| Deployment | Render (free web service) |

---

## 📁 Project Structure

```
.
├── main.py               # FastAPI app: serves the UI + JSON API (run this)
├── train_model.py        # Trains & compares models (10-fold CV + SMOTE); saves model + charts
├── elm.py                # Extreme Learning Machine (papers' method), NumPy
├── crops_meta.py         # Crop knowledge base: season, sowing months, water, family, feature ranges
│
├── fertility.py          # Stage 1 — soil rating + Parker's Nutrient Index
├── season.py             # Kharif/Rabi/Zaid + sowing-time verdict ("right time / when to start")
├── cultivation.py        # Real-world cross-check (is this crop grown in the state?)
├── economics.py          # Real yields (APY) + real MSP / market prices
├── yield_profit.py       # Stage 3 — profitability ranking (real economics)
├── fertilizer.py         # Stage 3 — real ICAR doses → urea/DAP/MOP + cost
├── irrigation.py         # Stage 3 — water requirement & scheduling
├── rotation.py           # Stage 3 — next-season rotation suggestion
├── explain.py            # Stage 4 — SHAP feature contributions
├── weather.py            # Open-Meteo climate normals (advisory) + offline fallback
│
├── static/index.html     # The web dashboard (UI + client logic)
├── crop_model.pkl        # Trained model bundle (created by train_model.py)
├── Crop_recommendation.csv
├── data/
│   ├── climate_normals.csv          # offline climate fallback
│   └── real/                        # real Govt-of-India lookups (yields, MSP, cultivation)
├── model_comparison.png · confusion_matrix.png · feature_importance.png
│
├── iot/
│   ├── esp32_sensor.ino  # Flashable ESP32 firmware
│   ├── iot_simulator.py  # Device simulator (demo without hardware)
│   └── README.md         # Wiring, bill of materials, data flow
│
├── render.yaml           # Render deployment blueprint
├── .python-version       # Pins Python 3.12.6 for the host
├── requirements.txt
└── PRODUCT.md            # Product brief (audiences, purpose, principles)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 (3.11–3.13 also work locally) and `pip`

### 1. Install
```bash
git clone https://github.com/Asheesh7298/AI-crop-Recommendation.git
cd AI-crop-Recommendation
pip install -r requirements.txt
```

### 2. Train the model (creates `crop_model.pkl` + charts)
```bash
python train_model.py
```

### 3. Run the app
```bash
python main.py
```
Open **http://localhost:5000** — interactive API docs are at **http://localhost:5000/docs**.

Enter your soil &amp; climate readings, pick your **location** and **planting month**, then click
**Recommend crop**. The **Climate** button shows your location's seasonal weather as context.

---

## 🔧 Configuration

**No API key is required** — climate comes from the free, keyless **Open-Meteo** service, with a
bundled offline fallback (`data/climate_normals.csv`) so the app always works.

Optional IoT auth: set `DEVICE_TOKEN` (in a `.env` file or the environment) to require a matching
`X-Device-Token` header on `/api/ingest`. `.env` is git-ignored.

---

## ⚙️ How It Works — the Pipeline

**Stage 1 · Soil fertility** (`fertility.py`) — rates N, P, K and pH, and combines them into an
overall **Parker's Nutrient Index** (Low / Medium / High).

**Stage 2 · Crop recommendation** (`main.py` + model + `season.py`) — the trained model ranks the
22 crops by probability from the readings you provide; the top match is the recommendation. Season
is **advisory**: a sowing-time verdict says whether the chosen month is right, and if not, when to
start and which in-season crops to consider instead.

**Stage 3 · Grounding &amp; optimization**
- **Real-world cross-check** (`cultivation.py`) — is this crop actually grown in your state? (real APY data)
- **Profitability** (`yield_profit.py` + `economics.py`) — `real yield × real price − fertilizer cost`.
- **Fertilizer** (`fertilizer.py`) — ICAR recommended dose, soil-adjusted → urea / DAP / MOP + ₹/ha.
- **Irrigation** (`irrigation.py`) and **Rotation** (`rotation.py`).

**Stage 4 · Explainability** (`explain.py`) — SHAP feature contributions.

---

## 🔌 API Reference

Base URL: `http://localhost:5000` (or your deployed URL).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/` | The web dashboard |
| `GET`  | `/docs` | Interactive API docs (auto-generated) |
| `GET`  | `/api/meta` | Model name, CV accuracy, comparison, season |
| `GET`  | `/api/weather?city=Delhi&month=7` | Seasonal climate normals for a city &amp; month |
| `POST` | `/api/predict` | Full recommendation for one set of readings |
| `POST` | `/api/ingest` | **IoT** — a device pushes a reading; stored + recommendation returned |
| `GET`  | `/api/latest` | **IoT** — the most recent device reading (for live mode) |
| `GET`  | `/charts/<name>.png` | Training charts |

**`POST /api/predict`** request:
```json
{ "N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82,
  "ph": 6.5, "rainfall": 200, "soil_moisture": 45, "month": 7, "city": "Lucknow" }
```
**Response (abridged):**
```json
{
  "season": "Kharif", "month_name": "July",
  "fertility": { "items": { "Nitrogen (N)": "High", "...": "..." },
                 "overall": "Rich — good nutrient status (Nutrient Index 2.33)",
                 "index": { "value": 2.33, "class": "Medium" } },
  "best": { "crop": "rice", "confidence": 0.96, "in_season": true },
  "timing": { "ok": true, "message": "Right time to sow rice — July is within June–July." },
  "grounding": { "grown": true, "message": "Major crop in Uttar Pradesh — ~68% of cropland (rank #1)." },
  "profit": [ { "crop": "rice", "yield_q_ha": 23.1, "price_q": 2300, "fert_cost": 6479,
               "net_profit": 46651, "price_source": "MSP 2024-25" } ],
  "fertilizer": { "tips": ["Recommended dose ... N 75 · P₂O₅ 50 · K₂O 50 kg/ha ...",
                           "Apply about 121 kg Urea + 109 kg DAP + 83 kg MOP per hectare."], "cost": 6479 },
  "explain": [ { "feature": "rainfall", "value": 0.21 }, "..." ]
}
```

---

## 🌐 IoT Module

The IoT layer streams live sensor data to the app and drives the recommendation in real time.
Full details, wiring and bill of materials: **[`iot/README.md`](iot/README.md)**.

**Demo it without hardware:**
```bash
python main.py                              # terminal 1
python iot/iot_simulator.py                 # terminal 2 — behaves like an ESP32
python iot/iot_simulator.py --url https://your-app.onrender.com   # against the deployed app
```
Open the dashboard and click **Live sensor feed** — readings and the recommendation update as the
"device" streams. Endpoints: `POST /api/ingest` (device → server), `GET /api/latest` (server → dashboard).

---

## ☁️ Deployment (Render)

The repo includes a Render blueprint (`render.yaml`).

1. Push to GitHub.
2. [render.com](https://render.com) → sign in with GitHub → **New + → Blueprint** → pick the repo → **Apply**.
3. Open the `…onrender.com` URL when the build finishes.

No secrets are required (Open-Meteo needs no key). Python is pinned to 3.12.6; the free tier sleeps
after ~15 min idle (~30 s cold start); every `git push` auto-redeploys.

---

## ⚖️ Honest Limitations

Stated plainly, because they matter for interpreting the results:

- **The crop model is trained on the benchmark Kaggle dataset**, whose climate columns are
  idealized per-crop values rather than real ambient weather. So its accuracy figures are
  **cross-validation on that dataset — not validation on real fields.** Real weather is therefore
  used only as advisory context, never as a model input.
- **The real-world cross-check, yields, prices and fertilizer doses are real** (Govt of India APY,
  MSP 2024-25, ICAR) — this is what grounds the recommendations in reality.
- Crops not covered by the APY dataset (some fruits, coffee, kidney beans) fall back to
  **indicative** yields/prices, clearly marked in the UI.
- Irrigation and rotation are rule-based agronomic heuristics, presented as advice.

---

## 📚 Research Background

Grounded in recent literature on ML for soil/agriculture:

- **Sheeba et al. (2022)**, **Suchithra &amp; Pai (2020)** — ELM for soil fertility classification.
- **Escorcia-Gutierrez et al. (2022)** — ensemble deep learning for soil nutrients (benchmark).
- **Abdelmigid et al. (2022)** — Random Forest for nutrient prediction (supports the model choice).
- **Shastri et al. (2025)**, **Prity et al. (2024)** — crop recommendation with explainability.

**Contribution:** an ELM vs RF vs XGBoost study integrated into a decision-support prototype that
adds Parker's Nutrient Index, sowing-time guidance, a **real-cultivation cross-check**, and **real
economics/fertilizer** — closing the loop from readings to a grounded, explainable plan.

---

## 🗺 Roadmap / Future Scope

- Real ESP32 field deployment with calibrated sensors.
- A real soil→crop model from Soil Health Card district data (currently gated).
- Live mandi-price API (Agmarknet) alongside MSP.
- District-level climate + satellite crop-health (NDVI).
- Multi-language (Hindi / regional) field-phone UI.

---

## 👥 Team

**Project:** System using IoT and Crop Recommendation AI (Final-Year Project)

- **Asheesh Kumar** — 2300970130034
- **Aditya Chaudhary** — 2300970130009

**Supervisor:** Dr. Meenakshi Yadav

---

## 📄 License

Released for **academic and educational use** as a final-year project. The Crop Recommendation
dataset belongs to its original authors on Kaggle; the APY / MSP / ICAR data belong to the
Government of India and respective bodies. Indicative market prices (for non-MSP crops) are
illustrative, not live market data.
