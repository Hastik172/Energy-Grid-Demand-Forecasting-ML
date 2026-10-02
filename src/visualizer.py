"""
Visualizer Module for Energy Grid Forecasting Project
Generates all 12 required Unit-1 visualizations:
1. Demand time-series plot
2. Renewable-generation plot
3. Demand histogram
4. Probability density plot
5. Demand vs temperature scatter plot
6. Polynomial regression curve
7. Discrete probability/PMF plot
8. Bayes probability visualization
9. Clustering visualization
10. Actual vs predicted demand
11. Prediction-error distribution
12. Prediction interval/uncertainty plot
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def plot_01_demand_timeseries(df, output_dir="figures"):
    """Plot 1: Electricity Demand Time-Series over 1 year (with 2-week zoom)."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    
    # Full year
    ax1.plot(df["timestamp"], df["electricity_demand_mw"], color="#1f77b4", lw=0.8, alpha=0.85)
    ax1.set_title("1. Annual Hourly Electricity Demand (MW) - 2023 Full Time Series", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Demand (MW)", fontsize=11)
    ax1.axhline(df["electricity_demand_mw"].mean(), color="red", linestyle="--", lw=1.2, label=f"Mean Demand: {df['electricity_demand_mw'].mean():.1f} MW")
    ax1.legend(loc="upper right", frameon=True)
    
    # 2-Week zoom (Summer high-demand window: July 1 to July 15)
    summer_subset = df[(df["timestamp"] >= "2023-07-01") & (df["timestamp"] <= "2023-07-15")]
    ax2.plot(summer_subset["timestamp"], summer_subset["electricity_demand_mw"], color="#ff7f0e", lw=1.5, marker="o", markersize=2)
    ax2.set_title("Detailed 2-Week Diurnal Cycle Zoom (July 1 - July 15, 2023)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_ylabel("Demand (MW)", fontsize=11)
    ax2.set_xlabel("Timestamp", fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "01_demand_timeseries.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_02_renewable_generation(df, output_dir="figures"):
    """Plot 2: Solar and Wind Renewable Generation Time-Series."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 2-week sample window to clearly observe solar diurnal curve vs wind volatility
    sample_df = df[(df["timestamp"] >= "2023-05-01") & (df["timestamp"] <= "2023-05-15")]
    
    ax1.fill_between(sample_df["timestamp"], sample_df["solar_generation_mw"], color="#f39c12", alpha=0.6, label="Solar Generation (MW)")
    ax1.plot(sample_df["timestamp"], sample_df["solar_generation_mw"], color="#d68910", lw=1.2)
    ax1.set_title("2. Renewable Generation Profile: Solar vs Wind Dynamics (May 1 - May 15, 2023)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Solar Power (MW)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    
    ax2.fill_between(sample_df["timestamp"], sample_df["wind_generation_mw"], color="#2980b9", alpha=0.5, label="Wind Generation (MW)")
    ax2.plot(sample_df["timestamp"], sample_df["wind_generation_mw"], color="#1f618d", lw=1.2)
    ax2.set_ylabel("Wind Power (MW)", fontsize=11)
    ax2.set_xlabel("Timestamp", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "02_renewable_generation.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_03_demand_histogram(df, output_dir="figures"):
    """Plot 3: Histogram of Electricity Demand with Quantile Annotations."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = df["electricity_demand_mw"]
    q25 = data.quantile(0.25)
    q50 = data.quantile(0.50)
    q75 = data.quantile(0.75)
    
    sns.histplot(data, bins=50, kde=True, color="#2c3e50", ax=ax, stat="density", edgecolor="white", alpha=0.7)
    
    ax.axvline(q25, color="#27ae60", linestyle="--", lw=1.8, label=f"25th Percentile (Q1): {q25:.1f} MW")
    ax.axvline(q50, color="#e67e22", linestyle="-", lw=2.0, label=f"Median (50th): {q50:.1f} MW")
    ax.axvline(q75, color="#c0392b", linestyle="--", lw=1.8, label=f"75th Percentile (Q3): {q75:.1f} MW")
    
    ax.set_title("3. Empirical Distribution of Electricity Demand (MW)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Electricity Demand (MW)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "03_demand_histogram.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_04_probability_density(df, output_dir="figures"):
    """Plot 4: Continuous Probability Density Function (KDE vs Fitted Gaussian)."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = df["electricity_demand_mw"].values
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    norm_pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    kde = stats.gaussian_kde(data)
    kde_pdf = kde(x)
    
    ax.plot(x, kde_pdf, color="#8e44ad", lw=2.5, label="Empirical KDE (Kernel Density Estimate)")
    ax.plot(x, norm_pdf, color="#e74c3c", linestyle="--", lw=2.0, label=f"Fitted Gaussian N(mu={mu:.0f}, sigma={sigma:.0f})")
    ax.fill_between(x, kde_pdf, alpha=0.2, color="#8e44ad")
    
    # Highlight 95% Confidence Interval for Demand
    ci_lower = mu - 1.96 * sigma
    ci_upper = mu + 1.96 * sigma
    ax.axvline(ci_lower, color="gray", linestyle=":", lw=1.2, label=f"Gaussian 95% Bounds: [{ci_lower:.0f}, {ci_upper:.0f}] MW")
    ax.axvline(ci_upper, color="gray", linestyle=":", lw=1.2)
    
    ax.set_title("4. Probability Density Function (PDF) Analysis: Gaussian vs Non-Parametric KDE", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Electricity Demand (MW)", fontsize=11)
    ax.set_ylabel("Probability Density f(x)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "04_probability_density.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_05_demand_vs_temperature(df, output_dir="figures"):
    """Plot 5: Scatter Plot of Electricity Demand vs Ambient Temperature showing thermodynamic U-curve."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sc = ax.scatter(
        df["temperature_celsius"], 
        df["electricity_demand_mw"], 
        c=df["hour"], 
        cmap="viridis", 
        alpha=0.45, 
        s=16,
        edgecolors="none"
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Hour of Day (0-23)", fontsize=10)
    
    # Mark Comfort zone
    ax.axvspan(18, 22, color="gray", alpha=0.15, label="Thermal Comfort Neutral Zone (18°C - 22°C)")
    
    ax.set_title("5. Demand vs Temperature Scatter: Thermodynamic 'U-Curve' (Heating & Cooling Demands)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ambient Temperature (°C)", fontsize=11)
    ax.set_ylabel("Electricity Demand (MW)", fontsize=11)
    ax.legend(loc="upper center", frameon=True)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "05_demand_vs_temperature.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_06_polynomial_regression_curve(curve_preds, results_df, output_dir="figures"):
    """Plot 6: Polynomial Regression Curves (Degree 1, 2, 3) fitted to Temperature -> Demand."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    
    x = curve_preds["x_sorted"]
    y_act = curve_preds["y_actual"]
    
    ax.scatter(x, y_act, color="#7f8c8d", alpha=0.25, s=12, label="Actual Observations")
    ax.plot(x, curve_preds["Degree_1"], color="#e74c3c", lw=2.2, label=f"Degree 1 (Linear): R²={results_df.loc['Degree 1', 'Test R^2']:.3f}, RMSE={results_df.loc['Degree 1', 'Test RMSE']:.0f} MW")
    ax.plot(x, curve_preds["Degree_2"], color="#2ecc71", lw=2.5, label=f"Degree 2 (Quadratic): R²={results_df.loc['Degree 2', 'Test R^2']:.3f}, RMSE={results_df.loc['Degree 2', 'Test RMSE']:.0f} MW")
    ax.plot(x, curve_preds["Degree_3"], color="#2980b9", lw=2.2, linestyle="--", label=f"Degree 3 (Cubic): R²={results_df.loc['Degree 3', 'Test R^2']:.3f}, RMSE={results_df.loc['Degree 3', 'Test RMSE']:.0f} MW")
    
    ax.set_title("6. Polynomial Curve Fitting: Temperature to Electricity Demand (Unit 1 PRML Concept)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Electricity Demand (MW)", fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc="upper center")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "06_polynomial_regression_curve.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_07_discrete_pmf(pmf_series, output_dir="figures"):
    """Plot 7: Probability Mass Function (PMF) of Discrete Demand Random Variable."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    states = [0, 1, 2]
    labels = ["State 0\n(Low Demand)", "State 1\n(Normal Demand)", "State 2\n(High Demand)"]
    probs = [pmf_series[s] for s in states]
    colors = ["#27ae60", "#2980b9", "#e74c3c"]
    
    bars = ax.bar(labels, probs, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h*100:.2f}%\n(p={h:.4f})", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax.set_ylim(0, max(probs) * 1.25)
    ax.set_title("7. Probability Mass Function (PMF) of Discrete Demand States X ∈ {0, 1, 2}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Probability P(X = k)", fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "07_discrete_pmf.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_08_bayes_probability(bayes_dict, output_dir="figures"):
    """Plot 8: Bayes' Theorem: Prior vs Posterior Probability of High Demand."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    case1 = bayes_dict["Case 1 (High Temp)"]
    case2 = bayes_dict["Case 2 (Low Renewable)"]
    
    scenarios = ["Base Prior\nP(High Demand)", "Posterior Given\nHigh Temp P(A|High T)", "Posterior Given\nLow Renewable P(A|Low Ren)"]
    vals = [
        case1["Prior P(High Demand)"],
        case1["Posterior P(High Demand | High Temp) [Bayes]"],
        case2["Posterior P(High Demand | Low Renewable) [Bayes]"]
    ]
    colors = ["#7f8c8d", "#e67e22", "#c0392b"]
    
    bars = ax.bar(scenarios, vals, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h*100:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title("8. Bayes' Rule: Updating Probability of High Demand Conditioned on Environmental Evidence", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Probability", fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "08_bayes_probability.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_09_clustering_visualization(df, kmeans_dict, output_dir="figures"):
    """Plot 9: Unsupervised K-Means Clustering of Grid Operating Regimes."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    
    labels = kmeans_dict["cluster_labels"]
    regime_names = kmeans_dict["regime_names"]
    
    palette = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]
    
    for c_id in sorted(np.unique(labels)):
        mask = (labels == c_id)
        ax.scatter(
            df.loc[mask, "temperature_celsius"],
            df.loc[mask, "electricity_demand_mw"],
            s=16,
            alpha=0.5,
            label=f"Cluster {c_id}: {regime_names.get(c_id, f'Cluster {c_id}')}",
            color=palette[c_id % len(palette)]
        )
        
    ax.set_title("9. Unsupervised Learning: K-Means Clustering of Grid Operating Regimes (K=4)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ambient Temperature (°C)", fontsize=11)
    ax.set_ylabel("Electricity Demand (MW)", fontsize=11)
    ax.legend(frameon=True, fontsize=9.5, loc="upper center")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "09_clustering_visualization.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_10_actual_vs_predicted_demand(y_true, y_pred, output_dir="figures"):
    """Plot 10: Actual vs Predicted Demand on Test Set (1-week zoom)."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Parity scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.35, s=15, color="#2980b9")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax1.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Perfect Forecast Line (y = x)")
    ax1.set_title("Actual vs Predicted Demand Parity Plot", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Actual Demand (MW)", fontsize=11)
    ax1.set_ylabel("Predicted Demand (MW)", fontsize=11)
    ax1.legend(frameon=True)
    
    # Right: 168-hour (1 week) time-series tracking
    n_hours = min(168, len(y_true))
    ax2.plot(range(n_hours), y_true[:n_hours], label="Actual Demand (MW)", color="#2c3e50", lw=2.0)
    ax2.plot(range(n_hours), y_pred[:n_hours], label="Supervised Model Prediction", color="#e74c3c", lw=1.8, linestyle="--")
    ax2.set_title("10. One-Week (168-Hour) Forecast Tracking", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Test Set Hour Index", fontsize=11)
    ax2.set_ylabel("Demand (MW)", fontsize=11)
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "10_actual_vs_predicted_demand.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_11_prediction_error_distribution(residuals, uq_stats, output_dir="figures"):
    """Plot 11: Prediction Error (Residual) Distribution with Quantile Cutoffs."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(residuals, bins=45, kde=True, color="#34495e", stat="density", ax=ax, alpha=0.7, edgecolor="white")
    
    q05 = uq_stats["5th Percentile Error (e_0.05)"]
    q50 = uq_stats["50th Percentile Error (Median)"]
    q95 = uq_stats["95th Percentile Error (e_0.95)"]
    
    ax.axvline(0, color="black", linestyle="-", lw=1.5, label="Zero Bias Line (e = 0)")
    ax.axvline(q05, color="#e74c3c", linestyle="--", lw=2.0, label=f"5th Percentile: {q05:.1f} MW")
    ax.axvline(q50, color="#27ae60", linestyle=":", lw=1.8, label=f"Median Error: {q50:.1f} MW")
    ax.axvline(q95, color="#e74c3c", linestyle="--", lw=2.0, label=f"95th Percentile: {q95:.1f} MW")
    
    ax.set_title("11. Prediction Error Distribution ε = Actual - Predicted with Quantile Bounds", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Prediction Error (MW)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "11_prediction_error_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_12_prediction_interval_uncertainty(predictions_df, output_dir="figures"):
    """Plot 12: Test Segment showing Actual, Point Prediction, and Shaded 90% Prediction Interval."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # 120-hour window (5 days)
    sample_df = predictions_df.iloc[100:220].reset_index(drop=True)
    x = sample_df.index
    
    ax.fill_between(
        x, 
        sample_df["Lower_90_Empirical"], 
        sample_df["Upper_90_Empirical"], 
        color="#3498db", 
        alpha=0.3, 
        label="90% Empirical Prediction Interval [Lower Bound, Upper Bound]"
    )
    
    ax.plot(x, sample_df["Predicted_Demand_MW"], color="#2980b9", lw=2.2, label="Point Forecast (Supervised Regression)")
    ax.plot(x, sample_df["Actual_Demand_MW"], color="#e74c3c", lw=2.0, marker="o", markersize=4, linestyle="None", label="Actual Observed Demand (MW)")
    
    ax.set_title("12. Uncertainty Quantification: Point Prediction with 90% Prediction Interval", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Test Window Hours", fontsize=11)
    ax.set_ylabel("Electricity Demand (MW)", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "12_prediction_interval_uncertainty.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
