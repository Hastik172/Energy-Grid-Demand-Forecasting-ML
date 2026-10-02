"""
Interactive Dashboard for Energy Grid Demand Forecasting
=========================================================
A simple terminal-based interactive dashboard where the user can:
1. Enter/select input values (temperature, solar, wind, hour, etc.)
2. Get predicted energy demand
3. See uncertainty/prediction interval
4. See probability of high demand
5. See relevant demand category
6. See important statistical information

Run: python dashboard.py
"""

import sys
import os

# Configure UTF-8 for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Import project modules
from data.data_loader import load_energy_data
from src.preprocessing import clean_and_preprocess, split_train_test
from src.probability_analysis import analyze_discrete_random_variable


def build_dashboard_model():
    """Load data, train model, and prepare all dashboard components."""
    print("Loading and preparing data...")
    df_raw, _ = load_energy_data("data/energy_grid_dataset.csv")
    df_clean, _, _, _ = clean_and_preprocess(df_raw)
    train_df, test_df = split_train_test(df_clean, train_ratio=0.8)

    # Feature columns (same as supervised model)
    base_cols = [
        "temperature_celsius", "solar_generation_mw", "wind_generation_mw",
        "hour", "day_of_week", "month", "is_weekend"
    ]

    # Build engineered features
    def engineer_features(df):
        feat = df[base_cols].copy()
        feat["temp_squared"] = feat["temperature_celsius"] ** 2
        feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24.0)
        feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24.0)
        return feat

    X_train = engineer_features(train_df)
    y_train = train_df["electricity_demand_mw"].values
    X_test = engineer_features(test_df)
    y_test = test_df["electricity_demand_mw"].values

    model = LinearRegression()
    model.fit(X_train.values, y_train)

    # Compute residuals for uncertainty quantification
    y_pred_test = model.predict(X_test.values)
    residuals = y_test - y_pred_test

    # Residual statistics
    res_stats = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)),
        "q05": float(np.percentile(residuals, 5)),
        "q25": float(np.percentile(residuals, 25)),
        "q50": float(np.percentile(residuals, 50)),
        "q75": float(np.percentile(residuals, 75)),
        "q95": float(np.percentile(residuals, 95)),
    }

    # Discrete demand thresholds
    drv = analyze_discrete_random_variable(df_clean)
    q33 = drv["q33_cutoff"]
    q66 = drv["q66_cutoff"]
    q75_demand = df_clean["electricity_demand_mw"].quantile(0.75)

    # Dataset summary statistics
    demand_stats = {
        "mean": float(df_clean["electricity_demand_mw"].mean()),
        "std": float(df_clean["electricity_demand_mw"].std()),
        "min": float(df_clean["electricity_demand_mw"].min()),
        "max": float(df_clean["electricity_demand_mw"].max()),
        "median": float(df_clean["electricity_demand_mw"].median()),
    }

    feature_cols = list(X_train.columns)

    return model, feature_cols, res_stats, q33, q66, q75_demand, demand_stats


def classify_demand(prediction, q33, q66):
    """Classify demand into Low / Normal / High."""
    if prediction < q33:
        return "LOW DEMAND (State 0)"
    elif prediction <= q66:
        return "NORMAL DEMAND (State 1)"
    else:
        return "HIGH DEMAND (State 2)"


def estimate_high_demand_probability(prediction, std_residual, q75_threshold):
    """
    Estimate probability that actual demand exceeds the high-demand threshold.
    Uses Gaussian CDF approximation: P(Actual > threshold) = 1 - Phi((threshold - pred) / sigma)
    """
    from scipy.stats import norm
    z = (q75_threshold - prediction) / std_residual
    prob = 1.0 - norm.cdf(z)
    return prob


def get_float_input(prompt, default):
    """Get float input with a default value."""
    user_input = input(f"  {prompt} [default={default}]: ").strip()
    if user_input == "":
        return default
    try:
        return float(user_input)
    except ValueError:
        print(f"    Invalid input, using default={default}")
        return default


def get_int_input(prompt, default, min_val, max_val):
    """Get integer input with bounds and a default value."""
    user_input = input(f"  {prompt} [default={default}]: ").strip()
    if user_input == "":
        return default
    try:
        val = int(user_input)
        return max(min_val, min(max_val, val))
    except ValueError:
        print(f"    Invalid input, using default={default}")
        return default


def print_banner():
    print()
    print("=" * 78)
    print("   ENERGY GRID DEMAND FORECASTING - INTERACTIVE DASHBOARD")
    print("   with Uncertainty Quantification (Unit 1 ML & Probability Theory)")
    print("=" * 78)


def print_separator():
    print("-" * 78)


def run_dashboard():
    """Main interactive dashboard loop."""
    model, feature_cols, res_stats, q33, q66, q75_demand, demand_stats = build_dashboard_model()

    print_banner()
    print(f"\n  Model loaded successfully!")
    print(f"  Historical Demand Range: {demand_stats['min']:.0f} MW to {demand_stats['max']:.0f} MW")
    print(f"  Mean Demand: {demand_stats['mean']:.0f} MW | Median: {demand_stats['median']:.0f} MW")
    print(f"  Demand Categories: Low < {q33:.0f} MW | Normal: {q33:.0f}-{q66:.0f} MW | High > {q66:.0f} MW")
    print(f"  Model Residual Std Dev (sigma): {res_stats['std']:.1f} MW")

    while True:
        print_separator()
        print("\n  OPTIONS:")
        print("    [1] Enter custom input values for prediction")
        print("    [2] Use example scenarios (Summer Peak / Winter Night / Mild Spring)")
        print("    [3] View dataset statistics summary")
        print("    [Q] Quit dashboard")
        print()
        choice = input("  Your choice: ").strip().upper()

        if choice == "Q":
            print("\n  Exiting dashboard. Goodbye!")
            break

        elif choice == "3":
            print_separator()
            print("  DATASET STATISTICS SUMMARY")
            print_separator()
            print(f"  Electricity Demand:")
            print(f"    Mean:   {demand_stats['mean']:.2f} MW")
            print(f"    Std:    {demand_stats['std']:.2f} MW")
            print(f"    Min:    {demand_stats['min']:.2f} MW")
            print(f"    Median: {demand_stats['median']:.2f} MW")
            print(f"    Max:    {demand_stats['max']:.2f} MW")
            print(f"\n  Prediction Error Distribution:")
            print(f"    Mean Residual (Bias):   {res_stats['mean']:.2f} MW")
            print(f"    Std Dev of Residuals:   {res_stats['std']:.2f} MW")
            print(f"    5th Percentile Error:   {res_stats['q05']:.2f} MW")
            print(f"    25th Percentile Error:  {res_stats['q25']:.2f} MW")
            print(f"    Median Error:           {res_stats['q50']:.2f} MW")
            print(f"    75th Percentile Error:  {res_stats['q75']:.2f} MW")
            print(f"    95th Percentile Error:  {res_stats['q95']:.2f} MW")
            continue

        elif choice == "2":
            scenarios = {
                "A": ("Summer Afternoon Peak", 34.0, 5500.0, 300.0, 14, 2, 7, 0),
                "B": ("Winter Night Off-Peak", -2.0, 0.0, 1200.0, 3, 0, 1, 0),
                "C": ("Mild Spring Weekday Morning", 18.0, 1500.0, 800.0, 9, 3, 4, 0),
                "D": ("Autumn Weekend Evening", 12.0, 100.0, 3500.0, 19, 5, 10, 1),
            }
            print("\n  EXAMPLE SCENARIOS:")
            for key, (name, *_) in scenarios.items():
                print(f"    [{key}] {name}")
            sc_choice = input("  Select scenario: ").strip().upper()
            if sc_choice not in scenarios:
                print("  Invalid scenario. Returning to menu.")
                continue
            name, temp, solar, wind, hour, dow, month, weekend = scenarios[sc_choice]
            print(f"\n  Selected: {name}")

        elif choice == "1":
            print("\n  Enter input values (press Enter to use defaults):")
            temp = get_float_input("Temperature (C)", 20.0)
            solar = get_float_input("Solar Generation (MW)", 1000.0)
            wind = get_float_input("Wind Generation (MW)", 1500.0)
            hour = get_int_input("Hour of Day (0-23)", 12, 0, 23)
            dow = get_int_input("Day of Week (0=Mon, 6=Sun)", 2, 0, 6)
            month = get_int_input("Month (1-12)", 6, 1, 12)
            weekend = 1 if dow >= 5 else 0
        else:
            print("  Invalid choice. Please try again.")
            continue

        # Build feature vector
        features = np.array([[
            temp, solar, wind, hour, dow, month, weekend,
            temp ** 2,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0)
        ]])

        # Predict
        predicted_demand = float(model.predict(features)[0])

        # Prediction intervals
        lower_90 = predicted_demand + res_stats["q05"]
        upper_90 = predicted_demand + res_stats["q95"]
        lower_95_gauss = predicted_demand - 1.96 * res_stats["std"]
        upper_95_gauss = predicted_demand + 1.96 * res_stats["std"]

        # Demand category
        category = classify_demand(predicted_demand, q33, q66)

        # Probability of high demand
        prob_high = estimate_high_demand_probability(predicted_demand, res_stats["std"], q75_demand)

        # Predicted renewable contribution
        total_renewable = solar + wind
        renewable_pct = (total_renewable / max(predicted_demand, 1.0)) * 100

        # Display results
        print_separator()
        print("  PREDICTION RESULTS")
        print_separator()
        print(f"\n  INPUT VALUES:")
        print(f"    Temperature:        {temp:.1f} C")
        print(f"    Solar Generation:   {solar:.1f} MW")
        print(f"    Wind Generation:    {wind:.1f} MW")
        print(f"    Hour of Day:        {hour}")
        print(f"    Day of Week:        {dow} ({'Weekend' if weekend else 'Weekday'})")
        print(f"    Month:              {month}")

        print(f"\n  PREDICTED DEMAND:")
        print(f"    Point Prediction:   {predicted_demand:,.1f} MW")
        print(f"    Demand Category:    {category}")

        print(f"\n  RENEWABLE GENERATION:")
        print(f"    Total Renewable:    {total_renewable:,.1f} MW (Solar: {solar:.0f} + Wind: {wind:.0f})")
        print(f"    Renewable Share:    {renewable_pct:.1f}% of predicted demand")
        print(f"    Net Load:           {max(0, predicted_demand - total_renewable):,.1f} MW")

        print(f"\n  UNCERTAINTY QUANTIFICATION:")
        print(f"    90% Prediction Interval (Empirical Quantiles):")
        print(f"      Lower Bound (5th pct):  {lower_90:,.1f} MW")
        print(f"      Upper Bound (95th pct): {upper_90:,.1f} MW")
        print(f"      Interval Width:         {upper_90 - lower_90:,.1f} MW")
        print(f"    95% Prediction Interval (Gaussian +/- 1.96*sigma):")
        print(f"      Lower Bound:  {lower_95_gauss:,.1f} MW")
        print(f"      Upper Bound:  {upper_95_gauss:,.1f} MW")
        print(f"      Interval Width:         {upper_95_gauss - lower_95_gauss:,.1f} MW")

        print(f"\n  RISK ASSESSMENT:")
        print(f"    P(High Demand > {q75_demand:,.0f} MW): {prob_high * 100:.2f}%")
        if prob_high > 0.5:
            print(f"    WARNING: High probability of peak demand! Consider reserves.")
        elif prob_high > 0.25:
            print(f"    CAUTION: Moderate probability of elevated demand.")
        else:
            print(f"    Status: Low risk of extreme demand conditions.")

        print()


if __name__ == "__main__":
    run_dashboard()
