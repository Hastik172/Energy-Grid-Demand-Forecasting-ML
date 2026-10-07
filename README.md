# GridSense — Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification
## 21CSC305P Machine Learning Mini Project (Unit 1 Foundations)

A complete, production-grade Machine Learning system designed specifically around **Unit 1 ML & Probability Theory** topics. Analyzes 8,760 hours of electrical load and renewable energy data, provides point forecasts and calibrated 90% empirical prediction intervals, and includes dedicated interactive labs for each syllabus feature.

---

## What's New: Full System Architecture (Matching Benchmark Standard)

The project has been elevated to match the standard of top-tier student projects (like **GlucoSense**), featuring:
1. **Interactive Multi-Tab Web Application (`backend/app.py` + `frontend/index.html`):**
   * **Landing Page:** Animated diurnal load waves, headline metrics (8,760 rows, 8 features, $R^2 = 0.809$, $89.95\%$ coverage).
   * **Data Tab:** Interactive raw dataset viewer and per-column descriptive statistics.
   * **Bayes Lab:** 100-dot visual icon arrays (Prior vs. Posterior), interactive condition pills, step-by-step Bayes formulas, and likelihood ratios.
   * **Distributions Lab:** Continuous random variable visualizer with dynamic **Quantile Slider $Q(p)$**, histogram densities, Gaussian & KDE fits, and moments.
   * **Curve-Fitting Lab (Bishop §1.1):** Interactive **Polynomial Order Slider ($M = 1$ to $9$)**, regularisation ($\ln \lambda$), sample size selector, and live train vs. test RMSE overfitting diagnostics.
   * **Forecaster / Screener:** Weather parameter sliders, preset test profiles, real-time demand prediction, 90% prediction interval bar, and feature contribution multipliers.
   * **Test-Set Report:** Test set evaluation ($R^2$, RMSE, MAE, coverage), parity plots, residual error distribution, and scikit-learn parity verification.
   * **All Graphs Gallery:** 12 project plots with tabbed explanations (`What it shows`, `How it's computed`, `How to read it`, `Where it's used in real grid operations`).
2. **FastAPI Backend & Swagger UI (`/docs`):**
   * Interactive OpenAPI documentation served live at `http://127.0.0.1:8001/docs`.
3. **From-Scratch Mathematical Engines (`backend/models_from_scratch.py`):**
   * Normal equation solver $(\mathbf{\Phi}^T \mathbf{\Phi} + \lambda \mathbf{I})\mathbf{w} = \mathbf{\Phi}^T \mathbf{t}$, Bayes inference, continuous sample moments with Bessel's correction, and Pearson correlation.
4. **Console Training Runner (`python -m backend.train`):**
   * Detailed logging of data loading, probability verification, model fitting, and graph generation.
5. **Automated Test Suite (`python -m pytest -v`):**
   * 17 automated tests verifying custom models against Scikit-learn and NumPy identities.
6. **Project Showcase & Output Screenshots (`OUTPUT_SCREENSHOTS_DOCUMENT.md` & `OUTPUT_SCREENSHOTS_SHOWCASE.html`):**
   * Full documentation mirroring the 14-page project presentation standard.

---

## Quickstart: How to Run

### 1. Launch the Interactive Web Dashboard & Swagger Docs
```powershell
uvicorn backend.app:app --reload --port 8001
```
* **Interactive Dashboard:** Open your browser at [http://127.0.0.1:8001](http://127.0.0.1:8001)
* **Interactive REST API Docs (Swagger UI):** Open [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### 2. Run the Full Training Pipeline (Console Log)
```powershell
python -m backend.train
```

### 3. Run Automated Tests
```powershell
python -m pytest -v
```
*(All 17 tests pass with 100% success rate validating from-scratch models against scikit-learn).*

### 4. Test Live Prediction via cURL / API
```powershell
curl -X POST http://127.0.0.1:8001/api/predict `
     -H "Content-Type: application/json" `
     -d '{"temperature": 34.0, "solar_generation": 4800, "wind_generation": 450, "hour": 14, "month": 7, "day_of_week": 2, "is_weekend": 0}'
```

### 5. View Output Screenshots Showcase
Open [`OUTPUT_SCREENSHOTS_SHOWCASE.html`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/OUTPUT_SCREENSHOTS_SHOWCASE.html) in your browser or read [`OUTPUT_SCREENSHOTS_DOCUMENT.md`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/OUTPUT_SCREENSHOTS_DOCUMENT.md).

---

## Key Results Summary

* **Test $R^2$ Score:** `0.8092` (Supervised feature-engineered polynomial model)
* **Test RMSE:** `1,570.08 MW` | **Test MAE:** `1,332.72 MW`
* **90% Empirical Prediction Interval Coverage:** `89.95%` (Well-calibrated uncertainty envelope)
* **Kolmogorov Addition Rule Discrepancy:** `0.00e+00` (Exact axiomatic verification)
* **Bayes Risk Multiplier:** `1.79x` (Heatwaves increase high-demand posterior to 44.8% vs. 25.0% prior)
* **Scikit-Learn vs. From-Scratch Parity:** $\max |w_{\text{ours}} - w_{\text{sklearn}}| = 2.73 \times 10^{-10}$
