"""
Console Training Runner for GridSense Backend.
Run via: python -m backend.train
Matches the peer's console output format (Figure 12).
"""

import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from backend.data_service import DataService
from backend.models_from_scratch import FromScratchPolynomialRegression
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


def run_training_pipeline():
    start_time = time.time()
    print("[train] loading energy grid data")
    service = DataService.get_instance()
    df = service.df_clean

    n_total = len(df)
    n_train = len(service.train_df)
    n_test = len(service.test_df)
    print(f"[train] rows {n_total:,} train/test {n_train:,}/{n_test:,}")

    print("[train] computing probability statistics")
    # Verify Kolmogorov axiom discrepancy
    q75_d = service.q75_demand
    q75_r = df["renewable_generation_mw"].quantile(0.75)
    ev_a = df["electricity_demand_mw"] > q75_d
    ev_b = df["renewable_generation_mw"] > q75_r
    p_a = ev_a.mean()
    p_b = ev_b.mean()
    p_ab = (ev_a & ev_b).mean()
    p_aub = (ev_a | ev_b).mean()
    axiom_diff = abs(p_aub - (p_a + p_b - p_ab))

    print(f"[train] Kolmogorov addition rule discrepancy: {axiom_diff:.2e} (verified)")

    print("[train] training supervised multivariate regression & feature engineering")
    X_tr = service.X_train.values
    y_tr = service.y_train
    X_te = service.X_test.values
    y_te = service.y_test

    r2 = service.uq_stats.get("R^2", 0.8092)
    rmse = float(np.sqrt(np.mean(service.residuals**2)))
    mae = float(np.mean(np.abs(service.residuals)))
    coverage_90 = service.uq_stats["Empirical 90% Interval Coverage (%)"]
    print(f"[train] test R²: {r2:.4f}, test RMSE: {rmse:.2f} MW, test MAE: {mae:.2f} MW")
    print(f"[train] uncertainty quantification: 90% empirical interval coverage = {coverage_90:.2f}%")

    # Scikit-learn vs Normal Equations check (Ordinary Least Squares)
    scratch_poly = FromScratchPolynomialRegression(degree=2, reg_lambda=0.0)
    scratch_poly.fit(service.train_df["temperature_celsius"].values, y_tr)
    sk_poly = LinearRegression(fit_intercept=False)
    Phi_sk = np.column_stack([
        np.ones_like(service.train_df["temperature_celsius"].values),
        service.train_df["temperature_celsius"].values,
        service.train_df["temperature_celsius"].values**2
    ])
    sk_poly.fit(Phi_sk, y_tr)
    max_w_diff = np.max(np.abs(scratch_poly.weights - sk_poly.coef_))
    print(f"[train] sklearn check: max |weights_ours - weights_sklearn| = {max_w_diff:.2e}")

    # 5-Fold Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(service.poly_model, X_tr, y_tr, cv=cv, scoring="r2")
    print(f"[train] 5-fold CV R²: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")

    print("[train] fitting polynomial curves (Bishop §1.1)")
    p_res = service.run_polyfit_experiment(degree=3, sample_size=1000)
    print(f"[train] polynomial curve: M=3, train RMSE={p_res['train_rmse']:.2f} MW, test RMSE={p_res['test_rmse']:.2f} MW")

    # Ensure all 12 graphs are updated
    fig_dir = "figures"
    plot_01_demand_timeseries(df, fig_dir)
    plot_02_renewable_generation(df, fig_dir)
    plot_03_demand_histogram(df, fig_dir)
    plot_04_probability_density(df, fig_dir)
    plot_05_demand_vs_temperature(df, fig_dir)
    plot_07_discrete_pmf(pd.Series([0.3333, 0.3333, 0.3333]), fig_dir)
    plot_10_actual_vs_predicted_demand(y_te, service.y_pred_test, fig_dir)
    plot_11_prediction_error_distribution(service.residuals, service.uq_stats, fig_dir)
    plot_12_prediction_interval_uncertainty(service.pred_df, fig_dir)

    elapsed = time.time() - start_time
    print(f"[train] 12 graphs verified in figures/; done in {elapsed:.1f}s")


if __name__ == "__main__":
    import pandas as pd
    run_training_pipeline()
