"""
Main Pipeline Runner for:
"Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification"
Explicitly organized around Unit 1 Machine Learning and Probability Theory.
"""

import os
import sys

# Configure UTF-8 encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from data.data_loader import load_energy_data
from src.preprocessing import inspect_dataset, clean_and_preprocess, split_train_test
from src.continuous_analysis import compute_continuous_statistics, fit_probability_density
from src.probability_analysis import (
    compute_probability_events,
    analyze_discrete_random_variable,
    compute_bayes_rule,
    analyze_independence,
    compute_expectation_and_covariance
)
from src.models import (
    fit_polynomial_curve,
    train_supervised_model,
    train_unsupervised_kmeans
)
from src.uncertainty import analyze_prediction_uncertainty
from src.visualizer import (
    plot_01_demand_timeseries,
    plot_02_renewable_generation,
    plot_03_demand_histogram,
    plot_04_probability_density,
    plot_05_demand_vs_temperature,
    plot_06_polynomial_regression_curve,
    plot_07_discrete_pmf,
    plot_08_bayes_probability,
    plot_09_clustering_visualization,
    plot_10_actual_vs_predicted_demand,
    plot_11_prediction_error_distribution,
    plot_12_prediction_interval_uncertainty
)

def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)

def print_section(number, title):
    print(f"\n[{number}] --- {title} ---")

def main():
    print_header("Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification")
    print("Designed around Unit 1: Machine Learning Foundations & Probability Theory")
    
    # -------------------------------------------------------------
    # 1. Dataset Loading & Column Adaptation
    # -------------------------------------------------------------
    print_section("1-4", "Dataset Loading & Automatic Column Adaptation")
    df_raw, _ = load_energy_data("data/energy_grid_dataset.csv")
    
    # Initial Inspection
    inspection = inspect_dataset(df_raw)
    print(f"Raw Dataset Shape: {inspection['shape'][0]} rows, {inspection['shape'][1]} columns")
    print(f"Detected Duplicates: {inspection['duplicate_count']}")
    print(f"Detected Missing Values:\n{pd.Series(inspection['missing_counts'])}")
    
    # -------------------------------------------------------------
    # 5. Data Preprocessing & Cleaning
    # -------------------------------------------------------------
    print_section("5", "Data Preprocessing, Cleaning & Temporal Feature Engineering")
    df_clean, dups_removed, missing_pre, missing_post = clean_and_preprocess(df_raw)
    print(f"Duplicate rows removed: {dups_removed}")
    print(f"Remaining Missing Values after Time-Aware Interpolation: {sum(missing_post.values())}")
    print(f"Processed Dataset Shape: {df_clean.shape}")
    print(f"Engineered Features: hour, day_of_week, month, is_weekend, season, renewable_generation_mw, net_load_mw")
    
    train_df, test_df = split_train_test(df_clean, train_ratio=0.8)
    print(f"Chronological Train Set: {len(train_df)} rows | Test Set: {len(test_df)} rows")
    
    # -------------------------------------------------------------
    # 6 & 11-12. Exploratory Data Analysis & Continuous RV Statistics
    # -------------------------------------------------------------
    print_section("6, 11-12", "Continuous Random Variables & Exploratory Data Analysis (EDA)")
    stats_df = compute_continuous_statistics(df_clean)
    print("\n--- Summary Statistics (Continuous Random Variables) ---")
    print(stats_df[["Mean (mu)", "Std Dev (sigma)", "Min", "50% (Median)", "Max", "IQR", "5th Percentile", "95th Percentile"]].to_string())
    
    # -------------------------------------------------------------
    # 7. Probability Theory & Fundamental Rules
    # -------------------------------------------------------------
    print_section("7", "Probability Theory & Kolmogorov Axioms Verification")
    prob_events = compute_probability_events(df_clean)
    print(f"Event A (High Demand > {prob_events['threshold_demand_q75']:.1f} MW): P(A) = {prob_events['P(A) [High Demand]']:.4f}")
    print(f"Event B (High Renewable > {prob_events['threshold_renewable_q75']:.1f} MW): P(B) = {prob_events['P(B) [High Renewable]']:.4f}")
    print(f"Joint Event (A ∩ B): P(A ∩ B) = {prob_events['P(A ∩ B) [Joint High Demand & High Renewable]']:.4f}")
    print(f"Empirical Union P(A U B) = {prob_events['P(A U B) [Empirical Union]']:.4f}")
    print(f"Theoretical Union P(A) + P(B) - P(A ∩ B) = {prob_events['P(A) + P(B) - P(A ∩ B) [Theoretical Union]']:.4f}")
    print(f"Kolmogorov Addition Rule Discrepancy: {prob_events['Union Rule Discrepancy']:.6e} (Axiom Verified!)")
    print(f"Conditional Probability P(A|B) = {prob_events['P(A|B) [High Demand given High Renewable]']:.4f}")
    print(f"Conditional Probability P(B|A) = {prob_events['P(B|A) [High Renewable given High Demand]']:.4f}")
    
    # -------------------------------------------------------------
    # 8. Discrete Random Variables
    # -------------------------------------------------------------
    print_section("8", "Discrete Random Variables & PMF / CDF")
    drv = analyze_discrete_random_variable(df_clean)
    print(f"States: 0 = Low Demand (< {drv['q33_cutoff']:.1f} MW), 1 = Normal Demand, 2 = High Demand (> {drv['q66_cutoff']:.1f} MW)")
    print("Probability Mass Function (PMF):")
    for k, v in drv["pmf"].items():
        print(f"  - {k}: {v:.4f}")
    print("Cumulative Distribution Function (CDF):")
    for k, v in drv["cdf"].items():
        print(f"  - {k}: {v:.4f}")
    print(f"Discrete Expectation E[X] = {drv['E[X] (Discrete Expectation)']:.4f}")
    print(f"Discrete Variance Var(X) = {drv['Var(X) (Discrete Variance)']:.4f} (Std Dev = {drv['Std(X) (Discrete Std Dev)']:.4f})")
    
    # -------------------------------------------------------------
    # 9. Bayes' Rule
    # -------------------------------------------------------------
    print_section("9", "Bayes' Theorem: Prior, Likelihood, Evidence, Posterior")
    bayes_res = compute_bayes_rule(df_clean)
    print("\n[Case 1: Probability of High Demand given Extreme High Temperature]")
    c1 = bayes_res["Case 1 (High Temp)"]
    print(f"  Prior P(High Demand):              {c1['Prior P(High Demand)']:.4f}")
    print(f"  Likelihood P(High Temp | High Dem): {c1['Likelihood P(High Temp | High Demand)']:.4f}")
    print(f"  Evidence P(High Temp):              {c1['Evidence P(High Temp)']:.4f}")
    print(f"  Posterior P(High Demand | High T):  {c1['Posterior P(High Demand | High Temp) [Bayes]']:.4f} (Direct Check: {c1['Posterior P(High Demand | High Temp) [Direct]']:.4f})")
    print(f"  Risk Multiplier (Posterior/Prior):  {c1['Risk Ratio (Posterior / Prior)']:.2f}x")
    
    print("\n[Case 2: Probability of High Demand given Low Renewable Generation]")
    c2 = bayes_res["Case 2 (Low Renewable)"]
    print(f"  Prior P(High Demand):               {c2['Prior P(High Demand)']:.4f}")
    print(f"  Likelihood P(Low Ren | High Dem):   {c2['Likelihood P(Low Renewable | High Demand)']:.4f}")
    print(f"  Evidence P(Low Renewable):          {c2['Evidence P(Low Renewable)']:.4f}")
    print(f"  Posterior P(High Dem | Low Ren):    {c2['Posterior P(High Demand | Low Renewable) [Bayes]']:.4f} (Direct Check: {c2['Posterior P(High Demand | Low Renewable) [Direct]']:.4f})")
    
    # -------------------------------------------------------------
    # 10. Independence & Conditional Independence
    # -------------------------------------------------------------
    print_section("10", "Independence and Conditional Independence Analysis")
    ind_res = analyze_independence(df_clean)
    for p in ind_res["pairwise_tests"]:
        print(f"Pair: {p['Event Pair']}")
        print(f"  P(A ∩ B) = {p['P(A ∩ B)']:.4f} vs P(A)*P(B) = {p['P(A) * P(B)']:.4f} (Diff = {p['|P(A ∩ B) - P(A)P(B)|']:.4f})")
        print(f"  Pearson Correlation r = {p['Pearson Correlation (r)']:.4f}")
        print(f"  Independent? {p['Statistically Independent?']}")
    
    cond = ind_res["conditional_test"]
    print(f"\nConditional Test: {cond['Condition']}")
    print(f"  P(Dem ∩ Sol | H=14) = {cond['P(Demand ∩ Solar | Hour=14)']:.4f} vs P(Dem|H=14)*P(Sol|H=14) = {cond['P(Demand|H=14) * P(Solar|H=14)']:.4f}")
    print(f"  Discrepancy: {cond['Discrepancy']:.4f} -> Conditionally Independent? {cond['Conditionally Independent?']}")
    
    # -------------------------------------------------------------
    # 14. Expectation & Covariance vs Correlation
    # -------------------------------------------------------------
    print_section("14", "Expectation, Covariance, and Correlation")
    exp_cov = compute_expectation_and_covariance(df_clean)
    print("Expectations E[X]:")
    for k, v in exp_cov["expectations"].items():
        print(f"  - E[{k}] = {v:.2f}")
    print("\nCovariance with Electricity Demand:")
    for k, v in exp_cov["covariances"].items():
        corr_k = k.replace("Cov", "Corr")
        print(f"  - {k} = {v:.2f} | {corr_k} = {exp_cov['correlations'][corr_k]:.4f}")
        
    # -------------------------------------------------------------
    # 15. Polynomial Curve Fitting (Temperature -> Demand)
    # -------------------------------------------------------------
    print_section("15", "Polynomial Curve Fitting (Temperature -> Demand)")
    x_train_t = train_df["temperature_celsius"].values
    y_train_d = train_df["electricity_demand_mw"].values
    x_test_t = test_df["temperature_celsius"].values
    y_test_d = test_df["electricity_demand_mw"].values
    
    poly_results, poly_models, curve_preds = fit_polynomial_curve(x_train_t, y_train_d, x_test_t, y_test_d)
    print("\nPolynomial Fitting Model Comparison:")
    print(poly_results[["Train RMSE", "Train R^2", "Test RMSE", "Test R^2", "Test MAE"]].to_string())
    
    # -------------------------------------------------------------
    # 16. Supervised Learning Model
    # -------------------------------------------------------------
    print_section("16", "Supervised Learning: Multivariate Regression")
    sup_results = train_supervised_model(train_df, test_df)
    print("\nSupervised Regression Comparison:")
    print(sup_results["comparison_df"].to_string())
    
    # -------------------------------------------------------------
    # 17. Unsupervised Learning (K-Means Clustering)
    # -------------------------------------------------------------
    print_section("17", "Unsupervised Learning: K-Means Clustering")
    kmeans_results = train_unsupervised_kmeans(df_clean, n_clusters=4)
    print("\nGrid Operational Regime Centroids:")
    print(kmeans_results["cluster_summary"].to_string())
    
    # -------------------------------------------------------------
    # 18. Uncertainty Quantification (UQ)
    # -------------------------------------------------------------
    print_section("18", "Uncertainty Quantification (Residuals & Prediction Intervals)")
    uq_stats, pred_df, residuals = analyze_prediction_uncertainty(
        sup_results["y_test"],
        sup_results["y_pred_test"],
        alpha=0.10
    )
    print("Uncertainty & Residual Metrics:")
    for k, v in uq_stats.items():
        print(f"  - {k}: {v:.2f}")
        
    # -------------------------------------------------------------
    # 19. Generate All 12 Visualizations
    # -------------------------------------------------------------
    print_section("19", "Generating All 12 Unit-1 Visualizations")
    fig_dir = "figures"
    f1 = plot_01_demand_timeseries(df_clean, fig_dir)
    f2 = plot_02_renewable_generation(df_clean, fig_dir)
    f3 = plot_03_demand_histogram(df_clean, fig_dir)
    f4 = plot_04_probability_density(df_clean, fig_dir)
    f5 = plot_05_demand_vs_temperature(df_clean, fig_dir)
    f6 = plot_06_polynomial_regression_curve(curve_preds, poly_results, fig_dir)
    f7 = plot_07_discrete_pmf(drv["pmf_series"], fig_dir)
    f8 = plot_08_bayes_probability(bayes_res, fig_dir)
    f9 = plot_09_clustering_visualization(df_clean, kmeans_results, fig_dir)
    f10 = plot_10_actual_vs_predicted_demand(sup_results["y_test"], sup_results["y_pred_test"], fig_dir)
    f11 = plot_11_prediction_error_distribution(residuals, uq_stats, fig_dir)
    f12 = plot_12_prediction_interval_uncertainty(pred_df, fig_dir)
    
    print("All 12 visualizations successfully generated and saved:")
    for i, path in enumerate([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12], 1):
        print(f"  {i}. {path}")
        
    print_header("Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    main()
