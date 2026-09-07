# 🌱 System using IoT and Crop Recommendation AI

**Final Year Project — Smart Precision Agriculture**

| | |
|---|---|
| **Team** | Asheesh Kumar (2300970130034), Aditya Chaudhary (2300970130009) |
| **Supervisor** | Dr. Meenakshi Yadav |

An end-to-end system that turns soil + weather data into an actionable farming plan:
**soil-fertility assessment → season-aware crop recommendation → yield/profit
optimization → fertilizer, irrigation & rotation advice → an AI explanation of *why*.**
It benchmarks the reference papers' **ELM** against **Random Forest** and **XGBoost**.

---

## 🚀 Quick start (recommended — custom web app)

```bash
pip install -r requirements.txt
python train_model.py     # trains & compares ELM vs RF vs XGBoost, saves model + charts
python main.py            # FastAPI server -> open http://localhost:5000
```

Enter soil values (optionally **Use live weather**), then click **Recommend crop**.

The backend is **FastAPI** (`main.py`); the frontend is `static/index.html`.
Interactive API docs are auto-generated at **http://localhost:5000/docs**.
The **`POST /api/predict`** endpoint is also exactly what the IoT ESP32 module will send
sensor readings to.

> Run with autoreload during development: `uvicorn main:app --reload --port 5000`.
> A Streamlit version (`app.py`) is also included as an alternative.

## 🌦️ Weather key (`.env`)

Put your free OpenWeather key in a **`.env`** file in this folder:

```
OPENWEATHER_API_KEY=your_key_here
```

(Copy `.env.example` → `.env` and paste the key.) A brand-new key can take **1–2 hours to
activate** — until then you'll get a clear "key rejected (401)" message and the manual
weather sliders keep working. `.env` is git-ignored so the key never gets committed.

---

## 🧩 The pipeline

| Stage | What it does | File |
|-------|--------------|------|
| Inputs | Soil (N,P,K,pH,moisture) from sensors/manual; weather auto-filled by city | `weather.py` |
| 1. Soil fertility | Classify N/P/K/pH as Low/Med/High (papers' threshold tables) | `fertility.py` |
| 2. Crop recommendation | ELM/RF/XGBoost → best crop, top-N, **filtered by season** | `train_model.py`, `season.py` |
| 3. Optimization | Yield × price − cost ranking; fertilizer; irrigation; rotation | `yield_profit.py`, `fertilizer.py`, `irrigation.py`, `rotation.py` |
| 4. Explainability | SHAP "why this crop" chart | `explain.py` |

Central crop knowledge (season, water need, price, yield, family) lives in `crops_meta.py`.
The ELM implementation (papers' method) is in `elm.py`.

---

## 🤖 Model comparison (real Kaggle data)

| Model | Test accuracy |
|-------|---------------|
| Random Forest | **99.55%** |
| XGBoost | 98.64% |
| ELM (papers' method) | 98.41% |
| SVM | 98.41% |
| Naive Bayes | 99.55% |
| Decision Tree / KNN | 97.95% |

Best model is auto-selected and saved to `crop_model.pkl`. Charts:
`model_comparison.png`, `confusion_matrix.png`, `feature_importance.png`.

---

## 📊 Dataset

**Crop Recommendation Dataset** (2200 rows; N,P,K,temperature,humidity,pH,rainfall → 22 crops):
<https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>
Place `Crop_recommendation.csv` in this folder (already present). If missing, a synthetic
fallback keeps the pipeline running.

---

## 🌦️ Live weather (optional)

Get a free key at <https://openweathermap.org/api>, then:
```bash
setx OPENWEATHER_API_KEY "your_key"    # Windows — open a new terminal afterwards
```
Without a key, the app simply uses the manual weather sliders.

---

## 🔌 IoT module (future scope)

An **ESP32** with **NPK, pH, DHT22 and soil-moisture** sensors POSTs live readings to a
`/predict` endpoint for automatic, real-time recommendations. The dashboard already accepts
manual or sensor values interchangeably.

---

## 📚 Reference papers → project mapping

- Sheeba et al. 2022 (IoT + ELM soil micronutrients) and Suchithra & Pai 2020 (ELM tuning) → **ELM method + fertility tables**
- Escorcia-Gutierrez 2022 (ensemble DL) and Abdelmigid 2022 (RF/MARS) → **model benchmarks**
- Rekha et al. 2018 (FYM/P/PSB on green gram) → **fertilizer & rotation logic**

**Research gap addressed:** prior work classifies soil fertility but stops there; this project
closes the loop to an end-to-end crop + optimization recommendation.
