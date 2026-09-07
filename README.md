<div align="center">

# 🌱 System using IoT and Crop Recommendation AI

**A soil-to-harvest decision engine — soil-fertility assessment, season-aware crop
recommendation, profit &amp; input optimization, and an explainable AI account of every call.**

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
5. [Dataset](#-dataset)
6. [Tech Stack](#-tech-stack)
7. [Project Structure](#-project-structure)
8. [Getting Started](#-getting-started)
9. [Configuration (.env)](#-configuration-env)
10. [How It Works — the Pipeline](#-how-it-works--the-pipeline)
11. [API Reference](#-api-reference)
12. [IoT Module](#-iot-module)
13. [Deployment](#-deployment-render)
14. [Research Background &amp; Novelty](#-research-background--novelty)
15. [Roadmap](#-roadmap--future-scope)
16. [Team](#-team)
17. [License](#-license)

---

## 🌾 Overview

Choosing the right crop is a high-stakes decision that depends on **soil chemistry**
(nitrogen, phosphorus, potassium, pH) and **local climate** (temperature, humidity,
rainfall). Most existing tools stop at classifying soil fertility; they never close the
loop to a concrete, defensible planting decision.

This project delivers that full loop as an **end-to-end system**:

> **Soil &amp; climate readings → fertility assessment → season-aware crop recommendation →
> yield / profit / fertilizer / irrigation / rotation optimization → SHAP explanation.**

It is **software-sufficient today and IoT-ready by design**: the same JSON endpoint the web
app uses is the ingestion path for an ESP32 field-sensor node, so real-time sensor data can
drive the recommendation with no code changes.

---

## ✨ Key Features

- **🧪 Soil fertility grading** — N, P, K and pH classified as Low / Medium / High (Neutral /
  Acidic / Alkaline for pH) with an overall status.
- **🎯 Season-aware crop recommendation** — predicts the best of **22 crops**, filtered by the
  current cropping season (Kharif / Rabi / Zaid), with top matches and model confidence.
- **📊 Model comparison** — benchmarks **ELM vs Random Forest vs XGBoost** (plus baselines); the
  best model is auto-selected and served.
- **💰 Profitability optimization** — ranks candidate crops by expected `yield × price − fertilizer cost`.
- **🧴 Fertilizer advice** — compares sensed vs required NPK and recommends what to add, with an
  estimated cost.
- **💧 Irrigation advice** — water requirement &amp; scheduling from moisture + rainfall.
- **🔄 Crop rotation** — suggests the next-season crop to preserve soil nutrients.
- **🧠 Explainable AI (SHAP)** — shows *why* each crop was recommended, feature by feature.
- **🌦️ Live weather** — auto-fills temperature / humidity / rainfall from a city via OpenWeather.
- **🔌 IoT loop** — ESP32 firmware + device simulator + live-sensor dashboard mode.
- **🖥️ Custom web UI** — a responsive single-page dashboard with a light/dark theme (no Streamlit).

---

## 🏗 System Architecture

```
 ┌──────────────── INPUTS ────────────────┐
 │  Soil sensors: N, P, K, pH, moisture    │        Weather API (OpenWeather)
 │  (manual sliders OR ESP32 device)       │        temperature · humidity · rainfall
 └───────────────────┬─────────────────────┘                 │
                     └───────────────┬─────────────────────────┘
                                     ▼
        ╔════════════════ RECOMMENDATION PIPELINE ════════════════╗
        ║  1. Soil fertility assessment      (fertility.py)        ║
        ║  2. Crop recommendation            (ELM / RF / XGBoost)  ║
        ║     season-aware filtering         (season.py)           ║
        ║  3. Optimization                                          ║
        ║       • profitability   (yield_profit.py)                ║
        ║       • fertilizer      (fertilizer.py)                  ║
        ║       • irrigation      (irrigation.py)                  ║
        ║       • rotation        (rotation.py)                    ║
        ║  4. Explainability — SHAP          (explain.py)          ║
        ╚═══════════════════════════┬══════════════════════════════╝
                                     ▼
        FastAPI backend (main.py)  ──►  Web dashboard (static/index.html)
                 ▲                                   ▲
     POST /api/ingest                         GET /api/latest
                 │                                   │
        ESP32 sensor node  ─────────────────────────┘  (live sensor mode)
        (iot/esp32_sensor.ino / iot/iot_simulator.py)
```

---

## 🤖 Machine Learning &amp; Results

Seven models are trained and compared on an 80/20 split of the real dataset. **ELM** is the
method used in the project's reference papers; **Random Forest** and **XGBoost** are the
stronger modern baselines it is measured against.

| Model | Test accuracy |
|-------|:-------------:|
| **Random Forest** ⭐ (selected) | **99.55%** |
| Naive Bayes | 99.55% |
| XGBoost | 98.64% |
| ELM (reference-paper method) | 98.41% |
| SVM | 98.41% |
| Decision Tree | 97.95% |
| KNN | 97.95% |

The best model is retrained on all data and saved to `crop_model.pkl`. Training also emits
report figures: `model_comparison.png`, `confusion_matrix.png`, `feature_importance.png`
(served in-app at `/charts/*`). **Rainfall, humidity and nitrogen** are consistently the most
influential features.

> **ELM** (`elm.py`) is implemented from scratch in NumPy: random hidden weights + a
> Moore–Penrose pseudo-inverse for the output layer — no back-propagation, near-instant training.

---

## 📊 Dataset

**Crop Recommendation Dataset** — 2,200 rows, 7 features, **22 crops**.

| Feature | Meaning | Sensor |
|---------|---------|--------|
| N, P, K | Soil nitrogen, phosphorus, potassium | NPK soil sensor |
| temperature | Air temperature (°C) | DHT22 / weather API |
| humidity | Relative humidity (%) | DHT22 / weather API |
| ph | Soil pH | pH sensor |
| rainfall | Rainfall (mm) | Rain gauge / weather API |
| **label** | **Recommended crop (target)** | — |

Source: <https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>
The CSV (`Crop_recommendation.csv`) ships with the repo. If it is missing, `train_model.py`
generates a realistic synthetic fallback so the pipeline always runs.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend / API | **FastAPI** + Uvicorn |
| ML | scikit-learn, XGBoost, custom ELM (NumPy) |
| Explainability | SHAP |
| Frontend | Vanilla HTML / CSS / JS (single page, no build step) |
| Data | pandas, NumPy; Kaggle Crop Recommendation dataset |
| Weather | OpenWeather Current Weather API |
| IoT | ESP32 (Arduino / C++), REST over WiFi |
| Deployment | Render (free web service) |

---

## 📁 Project Structure

```
.
├── main.py               # FastAPI app: serves the UI + JSON API (run this)
├── train_model.py        # Trains & compares ELM vs RF vs XGBoost; saves model + charts
├── elm.py                # Extreme Learning Machine (papers' method), NumPy
├── crops_meta.py         # Crop knowledge base: season, water, price, yield, family
│
├── fertility.py          # Stage 1 — soil fertility grading
├── season.py             # Kharif / Rabi / Zaid awareness + crop calendar
├── yield_profit.py       # Stage 3 — profitability ranking
├── fertilizer.py         # Stage 3 — fertilizer gap analysis + cost
├── irrigation.py         # Stage 3 — water requirement & scheduling
├── rotation.py           # Stage 3 — next-season rotation suggestion
├── explain.py            # Stage 4 — SHAP feature contributions
├── weather.py            # OpenWeather client (.env key) with graceful fallback
│
├── static/index.html     # The web dashboard (UI + client logic)
├── crop_model.pkl        # Trained model bundle (created by train_model.py)
├── Crop_recommendation.csv
├── model_comparison.png  · confusion_matrix.png · feature_importance.png
│
├── iot/
│   ├── esp32_sensor.ino  # Flashable ESP32 firmware
│   ├── iot_simulator.py  # Device simulator (demo without hardware)
│   └── README.md         # Wiring, bill of materials, data flow
│
├── app.py                # Optional Streamlit alternative UI
├── render.yaml           # Render deployment blueprint
├── .python-version       # Pins Python 3.12.6 for the host
├── requirements.txt
├── .env.example          # Copy to .env and add your OpenWeather key
└── PRODUCT.md            # Product brief (audiences, purpose, principles)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 (3.11–3.13 also work locally)
- `pip`

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

Enter soil readings (optionally **Use live weather**), then click **Recommend crop**.

---

## 🔧 Configuration (.env)

Live weather needs a free OpenWeather key. Copy the example and paste your key:

```bash
cp .env.example .env
```
```
OPENWEATHER_API_KEY=your_key_here
```

- Get a free key at <https://openweathermap.org/api> (the free *Current Weather Data* plan).
- A brand-new key can take **1–2 hours to activate**; until then the app shows a clear message
  and the manual weather sliders keep working.
- `.env` is **git-ignored** — the key is never committed.

Optional IoT auth: set `DEVICE_TOKEN` to require a matching `X-Device-Token` header on `/api/ingest`.

---

## ⚙️ How It Works — the Pipeline

**Stage 1 · Soil fertility** (`fertility.py`) — grades N, P, K and pH against threshold tables
adapted from the reference papers, and summarizes an overall status.

**Stage 2 · Crop recommendation** (`main.py` + `elm.py` + `season.py`) — the trained model
ranks all 22 crops by probability; results are filtered to the current cropping season, and the
top match + alternatives are returned with confidence.

**Stage 3 · Optimization**
- **Profitability** (`yield_profit.py`) — `yield × market price − fertilizer cost`, ranked.
- **Fertilizer** (`fertilizer.py`) — sensed vs required NPK → what to add + estimated ₹/ha.
- **Irrigation** (`irrigation.py`) — seasonal water need minus rainfall → a watering schedule,
  refined by soil-moisture.
- **Rotation** (`rotation.py`) — follow a nutrient-hungry cereal/cash crop with a nitrogen-fixing
  pulse, and vice-versa.

**Stage 4 · Explainability** (`explain.py`) — SHAP feature contributions show which readings
pushed the decision toward (or against) the recommended crop.

---

## 🔌 API Reference

Base URL: `http://localhost:5000` (or your deployed URL).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/` | The web dashboard |
| `GET`  | `/docs` | Interactive API docs (auto-generated) |
| `GET`  | `/api/meta` | Model name, accuracy, comparison, season, weather-key status |
| `GET`  | `/api/weather?city=Delhi` | Live temperature / humidity / rainfall for a city |
| `POST` | `/api/predict` | Full recommendation for one set of readings |
| `POST` | `/api/ingest` | **IoT** — a device pushes a reading; stored + recommendation returned |
| `GET`  | `/api/latest` | **IoT** — the most recent device reading (for live mode) |
| `GET`  | `/charts/<name>.png` | Training charts |

**`POST /api/predict`** request:
```json
{ "N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82,
  "ph": 6.5, "rainfall": 200, "soil_moisture": 45 }
```
**Response (abridged):**
```json
{
  "season": "Kharif",
  "fertility": { "items": { "Nitrogen (N)": "Medium", "...": "..." }, "overall": "Moderate — balanced fertility" },
  "best": { "crop": "rice", "emoji": "🌾", "confidence": 0.96, "season_note": "In season now (Kharif)." },
  "top": [ { "crop": "rice", "confidence": 0.96, "in_season": true }, "..." ],
  "profit": [ { "crop": "rice", "yield_q_ha": 40, "price_q": 2200, "fert_cost": 0, "net_profit": 88000 } ],
  "fertilizer": { "tips": ["..."], "cost": 0 },
  "irrigation": ["..."],
  "rotation": { "reason": "...", "picks": [ { "crop": "chickpea", "emoji": "🫘" } ] },
  "explain": [ { "feature": "rainfall", "value": 0.21 }, "..." ],
  "model": { "name": "Random Forest", "accuracy": 0.9955 }
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
# against the deployed app:
python iot/iot_simulator.py --url https://your-app.onrender.com
```
Then open the dashboard and click **Live sensor feed** — readings and the recommendation update
automatically as the "device" streams.

- **`iot/esp32_sensor.ino`** — flashable firmware: reads NPK (RS485), pH, DHT22, soil-moisture,
  POSTs to `/api/ingest`.
- **`iot/iot_simulator.py`** — a device simulator for hardware-free demos.
- **Endpoints** — `POST /api/ingest` (device → server), `GET /api/latest` (server → dashboard).

---

## ☁️ Deployment (Render)

The repo includes a Render blueprint (`render.yaml`).

1. Push to GitHub (done).
2. [render.com](https://render.com) → sign in with GitHub → **New + → Blueprint** → pick the repo → **Apply**.
3. In the service's **Environment** tab, set **`OPENWEATHER_API_KEY`** (secret, `sync: false`).
4. When the build finishes, open the `…onrender.com` URL.

Notes: Python is pinned to 3.12.6; the free tier sleeps after ~15 min idle (~30 s cold start);
every `git push` auto-redeploys.

---

## 📚 Research Background &amp; Novelty

The project is grounded in recent literature on ML for soil/agriculture, notably:

- **Sheeba et al. (2022)** and **Suchithra &amp; Pai (2020)** — ELM for soil nutrient / fertility
  classification (the ELM method and fertility tables here follow these).
- **Escorcia-Gutierrez et al. (2022)** — ensemble deep learning for soil nutrients (benchmark).
- **Abdelmigid et al. (2022)** — Random Forest / MARS for micronutrient prediction (supports the
  model choice).
- **Rekha et al. (2018)** — FYM / phosphorus / PSB field study (fertilizer &amp; rotation logic).

**Research gap addressed:** prior work classifies soil fertility but stops there. This system
closes the loop — from soil (and live sensor) data to a season-aware, explainable, economically
optimized crop recommendation, with the model choice justified by an explicit ELM vs RF vs
XGBoost comparison.

---

## 🗺 Roadmap / Future Scope

- Real ESP32 field deployment with calibrated sensors.
- Yield prediction as a trained regressor (currently table-driven).
- Live market-price API for profitability.
- NDVI / satellite crop-health (OpenWeather Agro API).
- Multi-language (Hindi / regional) and field-phone UI.
- Persistent device history &amp; per-device dashboards.

---

## 👥 Team

**Project:** System using IoT and Crop Recommendation AI (Final-Year Project)

- **Asheesh Kumar** — 2300970130034
- **Aditya Chaudhary** — 2300970130009

**Supervisor:** Dr. Meenakshi Yadav

---

## 📄 License

Released for **academic and educational use** as a final-year project. The Crop Recommendation
dataset belongs to its original authors on Kaggle. Market prices and yields in `crops_meta.py`
are indicative figures for demonstration, not live market data.
