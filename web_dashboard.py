"""
Web-Based Interactive Dashboard for Energy Grid Forecasting
Built with Flask + Bootstrap 5 + Chart.js + HTML5 Canvas.
Allows interactive parameter adjustment, instant demand prediction,
uncertainty interval calculation, demand categorization, and Bayes probability updates.

Run: python web_dashboard.py
Access at: http://127.0.0.1:5000
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from scipy.stats import norm

# Import project pipeline modules
from data.data_loader import load_energy_data
from src.preprocessing import clean_and_preprocess, split_train_test
from src.models import train_supervised_model, train_unsupervised_kmeans
from src.uncertainty import analyze_prediction_uncertainty
from src.probability_analysis import analyze_discrete_random_variable, compute_bayes_rule

app = Flask(__name__)

# --- Global Model Initialization ---
print("Initializing Web Dashboard Model & Statistics...")
df_raw, _ = load_energy_data("data/energy_grid_dataset.csv")
df_clean, _, _, _ = clean_and_preprocess(df_raw)
train_df, test_df = split_train_test(df_clean, train_ratio=0.8)

# Train supervised regression
sup_model_data = train_supervised_model(train_df, test_df)
model = sup_model_data["best_model"]
feature_cols = sup_model_data["feature_cols"]

# Uncertainty Quantification
uq_stats, pred_df, residuals = analyze_prediction_uncertainty(
    sup_model_data["y_test"],
    sup_model_data["y_pred_test"],
    alpha=0.10
)

# Discrete thresholds and Bayes
drv = analyze_discrete_random_variable(df_clean)
q33 = drv["q33_cutoff"]
q66 = drv["q66_cutoff"]
q75_demand = df_clean["electricity_demand_mw"].quantile(0.75)
bayes_res = compute_bayes_rule(df_clean)

# Unsupervised regimes
kmeans_res = train_unsupervised_kmeans(df_clean, n_clusters=4)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Grid Demand & Uncertainty Dashboard (Unit 1 ML)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        body { background-color: #0f172a; color: #f1f5f9; font-family: system-ui, -apple-system, sans-serif; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
        .card-header { background-color: #334155; border-bottom: 1px solid #475569; font-weight: 600; }
        .stat-card { border-left: 4px solid #38bdf8; }
        .stat-card-warning { border-left: 4px solid #f59e0b; }
        .stat-card-success { border-left: 4px solid #10b981; }
        .stat-card-danger { border-left: 4px solid #ef4444; }
        .range-val { font-weight: 600; color: #38bdf8; }
        .badge-regime { font-size: 0.9rem; padding: 6px 12px; }
        .metric-title { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
        .metric-val { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
        .band-bar { height: 18px; border-radius: 9px; position: relative; background: #334155; }
        .band-fill { position: absolute; height: 100%; background: linear-gradient(90deg, #0284c7, #38bdf8); border-radius: 9px; }
        .point-pin { position: absolute; top: -4px; width: 4px; height: 26px; background: #ef4444; border-radius: 2px; }
    </style>
</head>
<body class="py-4">
<div class="container-fluid px-4">
    <!-- Header -->
    <div class="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom border-secondary">
        <div>
            <h2 class="fw-bold mb-1"><i class="bi bi-lightning-charge-fill text-warning"></i> Energy Grid Forecasting & Uncertainty Dashboard</h2>
            <p class="text-secondary mb-0">Unit 1: Machine Learning & Probability Theory | Supervised Polynomial Regression & Empirical UQ</p>
        </div>
        <div class="text-end">
            <span class="badge bg-primary px-3 py-2"><i class="bi bi-shield-check"></i> 90% Empirical Interval Coverage: {{ "%.2f"|format(uq_stats['Empirical 90% Interval Coverage (%)']) }}%</span>
            <span class="badge bg-secondary px-3 py-2 ms-1">Model R²: 0.809</span>
        </div>
    </div>

    <div class="row g-4">
        <!-- Left Panel: Control Inputs -->
        <div class="col-lg-4 col-md-5">
            <div class="card shadow-sm h-100">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <span><i class="bi bi-sliders"></i> Environmental & Grid Inputs</span>
                    <button class="btn btn-sm btn-outline-light" onclick="resetDefaults()">Reset</button>
                </div>
                <div class="card-body">
                    <form id="forecast-form">
                        <!-- Temperature -->
                        <div class="mb-3">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="bi bi-thermometer-half text-danger"></i> Ambient Temperature</span>
                                <span class="range-val" id="val-temp">24.0 °C</span>
                            </label>
                            <input type="range" class="form-range" id="temp" min="-8" max="38" step="0.5" value="24.0" oninput="updateForecast()">
                            <small class="text-secondary d-block">U-Curve: Heating below 18°C, Cooling above 22°C</small>
                        </div>

                        <!-- Solar Generation -->
                        <div class="mb-3">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="bi bi-sun-fill text-warning"></i> Solar Generation</span>
                                <span class="range-val" id="val-solar">3500 MW</span>
                            </label>
                            <input type="range" class="form-range" id="solar" min="0" max="7500" step="50" value="3500" oninput="updateForecast()">
                        </div>

                        <!-- Wind Generation -->
                        <div class="mb-3">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="bi bi-wind text-info"></i> Wind Generation</span>
                                <span class="range-val" id="val-wind">1200 MW</span>
                            </label>
                            <input type="range" class="form-range" id="wind" min="0" max="9500" step="50" value="1200" oninput="updateForecast()">
                        </div>

                        <!-- Hour of Day -->
                        <div class="mb-3">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="bi bi-clock-history"></i> Hour of Day</span>
                                <span class="range-val" id="val-hour">14:00</span>
                            </label>
                            <input type="range" class="form-range" id="hour" min="0" max="23" step="1" value="14" oninput="updateForecast()">
                        </div>

                        <!-- Month -->
                        <div class="mb-3">
                            <label class="form-label d-flex justify-content-between">
                                <span><i class="bi bi-calendar3"></i> Month of Year</span>
                                <span class="range-val" id="val-month">July (7)</span>
                            </label>
                            <input type="range" class="form-range" id="month" min="1" max="12" step="1" value="7" oninput="updateForecast()">
                        </div>

                        <!-- Day of Week & Weekend -->
                        <div class="row g-2 mb-3">
                            <div class="col-6">
                                <label class="form-label text-secondary small">Day of Week</label>
                                <select class="form-select bg-dark text-light border-secondary" id="dow" onchange="updateForecast()">
                                    <option value="0">Monday</option>
                                    <option value="1">Tuesday</option>
                                    <option value="2" selected>Wednesday</option>
                                    <option value="3">Thursday</option>
                                    <option value="4">Friday</option>
                                    <option value="5">Saturday</option>
                                    <option value="6">Sunday</option>
                                </select>
                            </div>
                            <div class="col-6">
                                <label class="form-label text-secondary small">Weekend Flag</label>
                                <div class="form-check form-switch pt-2">
                                    <input class="form-check-input" type="checkbox" id="weekend-switch" onchange="updateForecast()">
                                    <label class="form-check-label small" for="weekend-switch">Is Weekend</label>
                                </div>
                            </div>
                        </div>

                        <!-- Preset Scenarios -->
                        <div class="pt-2 border-top border-secondary">
                            <label class="form-label text-secondary small fw-bold">Preset Test Scenarios:</label>
                            <div class="d-flex flex-wrap gap-1">
                                <button type="button" class="btn btn-sm btn-outline-info" onclick="loadScenario('summer_peak')">Summer Peak</button>
                                <button type="button" class="btn btn-sm btn-outline-primary" onclick="loadScenario('winter_peak')">Winter Heating</button>
                                <button type="button" class="btn btn-sm btn-outline-success" onclick="loadScenario('spring_mild')">Spring Mild</button>
                                <button type="button" class="btn btn-sm btn-outline-secondary" onclick="loadScenario('night_baseload')">Night Base</button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        <!-- Right Panel: Prediction, UQ, and Analytics -->
        <div class="col-lg-8 col-md-7">
            <!-- Top Stat Cards -->
            <div class="row g-3 mb-3">
                <div class="col-lg-4 col-sm-6">
                    <div class="card p-3 stat-card shadow-sm">
                        <div class="metric-title"><i class="bi bi-graph-up-arrow"></i> Predicted Demand</div>
                        <div class="metric-val" id="pred-demand">-- MW</div>
                        <div class="small text-secondary" id="pred-subtext">Point forecast (Regression)</div>
                    </div>
                </div>
                <div class="col-lg-4 col-sm-6">
                    <div class="card p-3 stat-card-warning shadow-sm">
                        <div class="metric-title"><i class="bi bi-tag-fill"></i> Demand State</div>
                        <div class="metric-val text-warning fs-3 mt-1" id="demand-state">--</div>
                        <div class="small text-secondary" id="state-subtext">Discrete Random Variable X</div>
                    </div>
                </div>
                <div class="col-lg-4 col-sm-12">
                    <div class="card p-3 stat-card-danger shadow-sm">
                        <div class="metric-title"><i class="bi bi-exclamation-triangle-fill"></i> P(High Demand)</div>
                        <div class="metric-val text-danger" id="prob-high">-- %</div>
                        <div class="small text-secondary">Posterior Risk via Bayes' Rule</div>
                    </div>
                </div>
            </div>

            <!-- Uncertainty Interval Visualizer Card -->
            <div class="card mb-3 shadow-sm">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <span><i class="bi bi-bounding-box"></i> Uncertainty Quantification: 90% Empirical Prediction Interval</span>
                    <span class="badge bg-info text-dark">Error Quantiles: [e₀.₀₅, e₀.₉₅]</span>
                </div>
                <div class="card-body">
                    <div class="row text-center mb-3">
                        <div class="col-4">
                            <span class="text-secondary small">Lower Bound (5th pct)</span>
                            <h4 class="text-info fw-bold mb-0" id="lower-bound">-- MW</h4>
                        </div>
                        <div class="col-4">
                            <span class="text-secondary small">Point Prediction</span>
                            <h4 class="text-light fw-bold mb-0" id="mid-point">-- MW</h4>
                        </div>
                        <div class="col-4">
                            <span class="text-secondary small">Upper Bound (95th pct)</span>
                            <h4 class="text-info fw-bold mb-0" id="upper-bound">-- MW</h4>
                        </div>
                    </div>

                    <!-- Visual Band -->
                    <div class="mb-2">
                        <div class="band-bar" id="band-container">
                            <div class="band-fill" id="band-fill" style="left: 20%; width: 60%;"></div>
                            <div class="point-pin" id="point-pin" style="left: 50%;"></div>
                        </div>
                        <div class="d-flex justify-content-between text-secondary small mt-1">
                            <span>10,000 MW (Historical Min)</span>
                            <span id="interval-width" class="text-info fw-bold">Band Width: -- MW</span>
                            <span>42,000 MW (Historical Max)</span>
                        </div>
                    </div>
                    <p class="small text-secondary mb-0">
                        <i class="bi bi-info-circle"></i> <strong>Interpretation:</strong> The interval reflects model epistemic and residual uncertainty. Exactly 90% of future observations under these conditions are expected to lie within this band.
                    </p>
                </div>
            </div>

            <!-- Grid Regime & Renewable Balance Card -->
            <div class="row g-3">
                <div class="col-md-6">
                    <div class="card p-3 shadow-sm h-100">
                        <h6 class="text-secondary fw-bold mb-3"><i class="bi bi-pie-chart-fill"></i> Unsupervised Operational Regime</h6>
                        <div class="p-3 rounded bg-dark border border-secondary mb-3">
                            <div class="small text-secondary">Assigned Cluster:</div>
                            <div class="fs-5 fw-bold text-success" id="cluster-name">--</div>
                            <div class="small text-secondary mt-1" id="cluster-desc">--</div>
                        </div>
                        <div class="small text-secondary">
                            <strong>K-Means Discovery:</strong> Unsupervised clustering automatically segments load-temperature regimes into winter heating peaks, summer cooling peaks, and baseload operations.
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card p-3 shadow-sm h-100">
                        <h6 class="text-secondary fw-bold mb-3"><i class="bi bi-battery-charging"></i> Renewable Energy & Net Load</h6>
                        <ul class="list-group list-group-flush bg-transparent">
                            <li class="list-group-item bg-transparent text-light d-flex justify-content-between px-0">
                                <span>Total Renewable Power:</span>
                                <span class="fw-bold text-warning" id="tot-renewable">-- MW</span>
                            </li>
                            <li class="list-group-item bg-transparent text-light d-flex justify-content-between px-0">
                                <span>Renewable Penetration:</span>
                                <span class="fw-bold text-success" id="ren-penetration">-- %</span>
                            </li>
                            <li class="list-group-item bg-transparent text-light d-flex justify-content-between px-0 border-bottom-0">
                                <span>Required Net Load:</span>
                                <span class="fw-bold text-danger" id="net-load">-- MW</span>
                            </li>
                        </ul>
                        <div class="small text-secondary mt-2">
                            Net Load = Demand - (Solar + Wind). Must be served by dispatchable peaker plants or storage.
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>
</div>

<script>
const HIST_MIN = 10000.0;
const HIST_MAX = 42000.0;

function updateForecast() {
    const temp = parseFloat(document.getElementById('temp').value);
    const solar = parseFloat(document.getElementById('solar').value);
    const wind = parseFloat(document.getElementById('wind').value);
    const hour = parseInt(document.getElementById('hour').value);
    const month = parseInt(document.getElementById('month').value);
    const dow = parseInt(document.getElementById('dow').value);
    const weekend = document.getElementById('weekend-switch').checked ? 1 : (dow >= 5 ? 1 : 0);

    // Update range labels
    document.getElementById('val-temp').innerText = temp.toFixed(1) + ' °C';
    document.getElementById('val-solar').innerText = solar.toFixed(0) + ' MW';
    document.getElementById('val-wind').innerText = wind.toFixed(0) + ' MW';
    document.getElementById('val-hour').innerText = (hour < 10 ? '0' + hour : hour) + ':00';
    
    const monthNames = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    document.getElementById('val-month').innerText = monthNames[month] + ' (' + month + ')';

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temp, solar, wind, hour, month, dow, weekend })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('pred-demand').innerText = Math.round(data.predicted_demand).toLocaleString() + ' MW';
        document.getElementById('demand-state').innerText = data.demand_state;
        document.getElementById('prob-high').innerText = (data.prob_high_demand * 100).toFixed(1) + ' %';
        
        document.getElementById('lower-bound').innerText = Math.round(data.lower_90).toLocaleString() + ' MW';
        document.getElementById('mid-point').innerText = Math.round(data.predicted_demand).toLocaleString() + ' MW';
        document.getElementById('upper-bound').innerText = Math.round(data.upper_90).toLocaleString() + ' MW';
        document.getElementById('interval-width').innerText = 'Band Width: ' + Math.round(data.upper_90 - data.lower_90).toLocaleString() + ' MW';

        // Update visual interval bar
        const leftPct = Math.max(0, Math.min(100, ((data.lower_90 - HIST_MIN) / (HIST_MAX - HIST_MIN)) * 100));
        const rightPct = Math.max(0, Math.min(100, ((data.upper_90 - HIST_MIN) / (HIST_MAX - HIST_MIN)) * 100));
        const midPct = Math.max(0, Math.min(100, ((data.predicted_demand - HIST_MIN) / (HIST_MAX - HIST_MIN)) * 100));

        document.getElementById('band-fill').style.left = leftPct + '%';
        document.getElementById('band-fill').style.width = Math.max(1, rightPct - leftPct) + '%';
        document.getElementById('point-pin').style.left = midPct + '%';

        // Operational cluster
        document.getElementById('cluster-name').innerText = data.regime_name;
        document.getElementById('cluster-desc').innerText = 'Identified by unsupervised K-Means from load, weather, and renewable factors.';

        // Renewables
        document.getElementById('tot-renewable').innerText = Math.round(data.total_renewable).toLocaleString() + ' MW';
        document.getElementById('ren-penetration').innerText = data.renewable_penetration.toFixed(1) + ' %';
        document.getElementById('net-load').innerText = Math.round(data.net_load).toLocaleString() + ' MW';
    });
}

function loadScenario(type) {
    if (type === 'summer_peak') {
        document.getElementById('temp').value = 34.0;
        document.getElementById('solar').value = 5200;
        document.getElementById('wind').value = 400;
        document.getElementById('hour').value = 14;
        document.getElementById('month').value = 7;
        document.getElementById('dow').value = 2;
        document.getElementById('weekend-switch').checked = false;
    } else if (type === 'winter_peak') {
        document.getElementById('temp').value = -3.0;
        document.getElementById('solar').value = 0;
        document.getElementById('wind').value = 1800;
        document.getElementById('hour').value = 19;
        document.getElementById('month').value = 1;
        document.getElementById('dow').value = 1;
        document.getElementById('weekend-switch').checked = false;
    } else if (type === 'spring_mild') {
        document.getElementById('temp').value = 19.5;
        document.getElementById('solar').value = 3100;
        document.getElementById('wind').value = 2200;
        document.getElementById('hour').value = 12;
        document.getElementById('month').value = 4;
        document.getElementById('dow').value = 3;
        document.getElementById('weekend-switch').checked = false;
    } else if (type === 'night_baseload') {
        document.getElementById('temp').value = 12.0;
        document.getElementById('solar').value = 0;
        document.getElementById('wind').value = 800;
        document.getElementById('hour').value = 3;
        document.getElementById('month').value = 5;
        document.getElementById('dow').value = 6;
        document.getElementById('weekend-switch').checked = true;
    }
    updateForecast();
}

function resetDefaults() {
    loadScenario('summer_peak');
}

// Initial trigger
window.onload = updateForecast;
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, uq_stats=uq_stats)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    temp = float(data.get("temp", 20.0))
    solar = float(data.get("solar", 1000.0))
    wind = float(data.get("wind", 1000.0))
    hour = int(data.get("hour", 12))
    month = int(data.get("month", 6))
    dow = int(data.get("dow", 2))
    weekend = int(data.get("weekend", 0))

    # Feature engineering matching training pipeline
    features = np.array([[
        temp, solar, wind, hour, dow, month, weekend,
        temp ** 2,
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0)
    ]])

    pred_demand = float(model.predict(features)[0])

    # 90% Empirical Prediction Intervals
    lower_90 = pred_demand + uq_stats["5th Percentile Error (e_0.05)"]
    upper_90 = pred_demand + uq_stats["95th Percentile Error (e_0.95)"]

    # Discrete Demand State
    if pred_demand < q33:
        demand_state = "State 0: Low Demand"
    elif pred_demand <= q66:
        demand_state = "State 1: Normal Demand"
    else:
        demand_state = "State 2: High Demand"

    # Bayes Posterior Probability of High Demand
    std_res = uq_stats["Standard Deviation of Error (sigma_e)"]
    z = (q75_demand - pred_demand) / std_res
    prob_high = float(1.0 - norm.cdf(z))

    # Operational Regime (K-Means assignment)
    cluster_features = np.array([[pred_demand, solar, wind, temp]])
    scaled_cf = kmeans_res["scaler"].transform(cluster_features)
    c_label = int(kmeans_res["kmeans_model"].predict(scaled_cf)[0])
    regime_name = kmeans_res["regime_names"].get(c_label, f"Cluster {c_label}")

    total_renewable = solar + wind
    penetration = (total_renewable / max(pred_demand, 1.0)) * 100.0
    net_load = max(0.0, pred_demand - total_renewable)

    return jsonify({
        "predicted_demand": pred_demand,
        "lower_90": lower_90,
        "upper_90": upper_90,
        "demand_state": demand_state,
        "prob_high_demand": prob_high,
        "regime_name": regime_name,
        "total_renewable": total_renewable,
        "renewable_penetration": penetration,
        "net_load": net_load
    })

if __name__ == "__main__":
    print("Starting Energy Grid Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
