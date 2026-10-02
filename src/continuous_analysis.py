"""
Continuous Random Variables and Probability Density Module
Covers Unit 1 topics:
- Continuous Random Variables definition and properties
- Summary statistics: Mean, Variance, Standard Deviation, Median, Min, Max
- Quantiles (5th, 25th, 50th, 75th, 95th) and Interquartile Range (IQR)
- Probability Density Functions (Parametric Gaussian vs Non-parametric KDE)
"""

import numpy as np
import pandas as pd
from scipy import stats

def compute_continuous_statistics(df, columns=None):
    """
    Computes rigorous continuous random variable statistics for power system features:
    Mean, Variance, Std Dev, Min, Max, Median, and Quantiles (5%, 25%, 50%, 75%, 95%).
    """
    if columns is None:
        columns = [
            "electricity_demand_mw",
            "temperature_celsius",
            "solar_generation_mw",
            "wind_generation_mw",
            "renewable_generation_mw"
        ]
        
    stats_dict = {}
    for col in columns:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        n = len(data)
        mean_val = float(np.mean(data))
        var_val = float(np.var(data, ddof=1)) # Sample variance
        std_val = float(np.std(data, ddof=1)) # Sample standard deviation
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        median_val = float(np.median(data))
        
        # Quantiles
        q05 = float(np.percentile(data, 5))
        q25 = float(np.percentile(data, 25))
        q50 = float(np.percentile(data, 50))
        q75 = float(np.percentile(data, 75))
        q95 = float(np.percentile(data, 95))
        iqr = q75 - q25
        
        # Skewness and Kurtosis
        skew_val = float(stats.skew(data))
        kurt_val = float(stats.kurtosis(data))
        
        stats_dict[col] = {
            "Count": n,
            "Mean (mu)": mean_val,
            "Variance (sigma^2)": var_val,
            "Std Dev (sigma)": std_val,
            "Min": min_val,
            "25% (Q1)": q25,
            "50% (Median)": q50,
            "75% (Q3)": q75,
            "Max": max_val,
            "IQR": iqr,
            "5th Percentile": q05,
            "95th Percentile": q95,
            "Skewness": skew_val,
            "Kurtosis": kurt_val
        }
        
    stats_df = pd.DataFrame(stats_dict).T
    return stats_df

def fit_probability_density(series, num_points=500):
    """
    Fits both a Parametric Gaussian PDF and a Non-parametric Kernel Density Estimate (KDE)
    to a continuous random variable.
    
    Returns:
    - x_eval: evaluation grid
    - normal_pdf: Gaussian probability density values f_normal(x)
    - kde_pdf: Kernel Density Estimate values f_kde(x)
    - mu, sigma: fitted Gaussian parameters
    """
    data = series.dropna().values
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    x_min = np.min(data) - 0.1 * (np.max(data) - np.min(data))
    x_max = np.max(data) + 0.1 * (np.max(data) - np.min(data))
    x_eval = np.linspace(x_min, x_max, num_points)
    
    # Parametric Gaussian PDF: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))
    normal_pdf = stats.norm.pdf(x_eval, loc=mu, scale=sigma)
    
    # Non-parametric KDE (Gaussian kernel)
    kde = stats.gaussian_kde(data)
    kde_pdf = kde(x_eval)
    
    return {
        "x_eval": x_eval,
        "normal_pdf": normal_pdf,
        "kde_pdf": kde_pdf,
        "mu": mu,
        "sigma": sigma
    }
