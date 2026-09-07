# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The design must serve three real audiences at once (confirmed):

- **Farmers** making a planting decision — decision-first, plain language, minimal jargon.
- **Agronomists / extension officers** advising farmers — tolerant of more data density and technical detail.
- **The final-year academic review panel** evaluating the project — needs visible rigor (model comparison, SHAP, metrics, methodology).

The product must stay farmer-usable while reading as rigorous to reviewers. Reach is **English, desktop-first** for now; field-phone and multi-language support are acknowledged future scope, not current requirements.

## Product Purpose

Turn a soil sample plus climate readings into a **season-aware crop-planting decision, with the reasoning and economics attached.** It exists to answer "what should I grow now, and why" — and to justify the answer. Success is a grower or advisor getting a trustworthy recommendation they understand and can act on, while a reviewer can see the method behind it.

## Positioning

Not a single-number crop classifier: a **staged decision pipeline** — soil-fertility assessment → season-aware ranked recommendation → profit / fertilizer / irrigation / rotation optimization → per-recommendation SHAP explanation — with the model choice justified by an explicit **ELM vs Random Forest vs XGBoost** comparison. The transparency ("why this crop") and season-awareness are what a bare classifier cannot truthfully claim.

## Operating Context

- Runs as a local **FastAPI web app** (`python main.py` → http://localhost:5000; auto docs at `/docs`), typically shown on a laptop/projector for review and used in a browser.
- Inputs: NPK, soil pH, soil moisture, temperature, humidity, rainfall — entered manually with sliders or pulled from **live weather** (OpenWeather; API key optional).
- The same **`/api/predict` JSON endpoint** is the intended future ingestion path for an ESP32 + NPK/pH/DHT22/soil-moisture IoT sensor rig.

## Capabilities and Constraints

Confirmed capabilities (present in the codebase):

- Soil-fertility grading from NPK + pH, combined into an overall grade.
- Season-aware, ranked crop recommendation with model confidence and off-season flagging.
- Optimization outputs: profitability table, fertilizer guidance + estimated cost, irrigation advice, crop-rotation suggestion.
- SHAP explanation of each recommendation ("why this crop").
- Model-comparison training artifacts (ELM vs Random Forest vs XGBoost; Random Forest selected, ~99.6% on the dataset) surfaced as insight charts.

Durable commitment (explicitly confirmed): **software-only today, IoT-ready by design** — the system runs with no hardware, and `/api/predict` is the documented future ESP32/IoT path.

Technical constraints:

- Implementation is a FastAPI JSON API plus a **single vanilla HTML/CSS/JS page** with no build step; Python 3.13, scikit-learn, xgboost, shap.
- Weather is optional — the app must work with manual values when no API key is set.

## Brand Commitments

- Product name: **Crop Recommendation AI** (project title: "System using IoT and Crop Recommendation AI").
- Authored by **Asheesh Kumar** (2300970130034) and **Aditya Chaudhary** (2300970130009); supervised by **Dr. Meenakshi Yadav**. These credits appear in the UI and must be preserved.
- No other brand, voice, or identity constraints have been established.

## Evidence on Hand

- Dataset: Kaggle `Crop_recommendation.csv` (2200 rows, 22 crops) in the repo.
- Trained model bundle `crop_model.pkl`; training charts `model_comparison.png`, `feature_importance.png`, `confusion_matrix.png` (served at `/charts/*`).
- No testimonials, external/real users, deployment claims, or benchmarks beyond this dataset exist — future work must not fabricate them.

## Product Principles

1. **Decision first, evidence attached** — lead with the recommendation, but keep the "why" (SHAP), the economics, and the season reasoning one glance away.
2. **Trustworthy over flashy** — every number is a real model output and explainable; never invent data.
3. **Serve grower and reviewer at once** — plain enough to act on, rigorous enough to defend.
4. **IoT-ready, software-sufficient** — the sensor future is designed for (`/api/predict`) but never required to run today.

## Accessibility & Inclusion

- English, desktop-first is the confirmed current scope. Keyboard access, visible focus, and WCAG-AA text contrast are already met in the frontend and must be kept.
- Multi-language (e.g. Hindi / regional) and low-literacy / field-phone support are recorded as future scope, not current requirements.
