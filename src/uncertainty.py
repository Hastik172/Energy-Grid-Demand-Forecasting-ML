"""
Uncertainty Quantification (UQ) Module
Covers Unit 1 topics:
- Residual Analysis: Error = Actual - Predicted
- Error distribution statistics: Mean, Variance, Std Dev
- Empirical Error Quantiles: 5th, 25th, 50th, 75th, 95th percentiles
- Prediction Interval Construction (Empirical Quantile vs Gaussian Parametric)
- Empirical Coverage Probability (ECP) and Prediction Sharpness
"""

import numpy as np
import pandas as pd

def analyze_prediction_uncertainty(y_true, y_pred, alpha=0.10):
    """
    Computes rigorous uncertainty quantification on test set predictions.
    
    alpha = 0.10 corresponds to a 90% Prediction Interval:
    Lower bound: 5th percentile of residuals (e_0.05)
    Upper bound: 95th percentile of residuals (e_0.95)
    """
    residuals = y_true - y_pred
    
    mean_err = float(np.mean(residuals))
    var_err = float(np.var(residuals, ddof=1))
    std_err = float(np.std(residuals, ddof=1))
    
    # Quantiles of residuals
    q05 = float(np.percentile(residuals, 5))
    q25 = float(np.percentile(residuals, 25))
    q50 = float(np.percentile(residuals, 50))
    q75 = float(np.percentile(residuals, 75))
    q95 = float(np.percentile(residuals, 95))
    
    # Construct 90% Empirical Prediction Intervals
    lower_bound_empirical = y_pred + q05
    upper_bound_empirical = y_pred + q95
    
    # Construct 95% Parametric Gaussian Prediction Intervals (+/- 1.96 * sigma)
    lower_bound_gaussian = y_pred - 1.96 * std_err
    upper_bound_gaussian = y_pred + 1.96 * std_err
    
    # Coverage probability
    inside_empirical = (y_true >= lower_bound_empirical) & (y_true <= upper_bound_empirical)
    coverage_empirical = float(np.mean(inside_empirical) * 100.0)
    
    inside_gaussian = (y_true >= lower_bound_gaussian) & (y_true <= upper_bound_gaussian)
    coverage_gaussian = float(np.mean(inside_gaussian) * 100.0)
    
    # Sharpness (average interval width)
    avg_width_empirical = float(np.mean(upper_bound_empirical - lower_bound_empirical))
    avg_width_gaussian = float(np.mean(upper_bound_gaussian - lower_bound_gaussian))
    
    stats_summary = {
        "Mean Error (Residual Bias)": mean_err,
        "Variance of Error": var_err,
        "Standard Deviation of Error (sigma_e)": std_err,
        "5th Percentile Error (e_0.05)": q05,
        "25th Percentile Error (e_0.25)": q25,
        "50th Percentile Error (Median)": q50,
        "75th Percentile Error (e_0.75)": q75,
        "95th Percentile Error (e_0.95)": q95,
        "Empirical 90% Interval Coverage (%)": coverage_empirical,
        "Gaussian 95% Interval Coverage (%)": coverage_gaussian,
        "Empirical 90% Avg Width (MW)": avg_width_empirical,
        "Gaussian 95% Avg Width (MW)": avg_width_gaussian
    }
    
    predictions_df = pd.DataFrame({
        "Actual_Demand_MW": y_true,
        "Predicted_Demand_MW": y_pred,
        "Residual_MW": residuals,
        "Lower_90_Empirical": lower_bound_empirical,
        "Upper_90_Empirical": upper_bound_empirical,
        "Lower_95_Gaussian": lower_bound_gaussian,
        "Upper_95_Gaussian": upper_bound_gaussian,
        "Inside_90_Interval": inside_empirical
    })
    
    return stats_summary, predictions_df, residuals
