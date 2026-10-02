"""
Machine Learning Models Module
Covers Unit 1 topics:
- Polynomial Curve Fitting (Degrees 1, 2, 3) for Temperature -> Demand
- Evaluation Metrics: MAE, MSE, RMSE, R^2
- Supervised Learning: Multivariate Regression with calendar & meteorological features
- Unsupervised Learning: K-Means Clustering for operational grid regime identification
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_regression_metrics(y_true, y_pred):
    """
    Computes Unit-1 standard regression performance metrics:
    - MAE = (1/N) * sum(|y_i - y_hat_i|)
    - MSE = (1/N) * sum((y_i - y_hat_i)^2)
    - RMSE = sqrt(MSE)
    - R^2 = 1 - [sum((y_i - y_hat_i)^2) / sum((y_i - y_bar)^2)]
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE (MW)": mae,
        "MSE (MW^2)": mse,
        "RMSE (MW)": rmse,
        "R^2 Score": r2
    }

def fit_polynomial_curve(x_train, y_train, x_test, y_test, degrees=[1, 2, 3]):
    """
    Fits polynomial curves of varying degrees to model nonlinear Temperature -> Demand relationships.
    Compares degree 1 (linear), degree 2 (quadratic), and degree 3 (cubic).
    """
    results = {}
    models = {}
    
    # Sort for smooth plotting curves
    sort_idx = np.argsort(x_test.ravel())
    x_test_sorted = x_test[sort_idx]
    y_test_sorted = y_test[sort_idx]
    
    curve_predictions = {"x_sorted": x_test_sorted, "y_actual": y_test_sorted}
    
    for deg in degrees:
        poly = PolynomialFeatures(degree=deg, include_bias=True)
        X_poly_train = poly.fit_transform(x_train.reshape(-1, 1))
        X_poly_test = poly.transform(x_test.reshape(-1, 1))
        X_poly_sorted = poly.transform(x_test_sorted.reshape(-1, 1))
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly_train, y_train)
        
        y_pred_train = model.predict(X_poly_train)
        y_pred_test = model.predict(X_poly_test)
        y_pred_curve = model.predict(X_poly_sorted)
        
        metrics_train = compute_regression_metrics(y_train, y_pred_train)
        metrics_test = compute_regression_metrics(y_test, y_pred_test)
        
        results[f"Degree {deg}"] = {
            "Train MAE": metrics_train["MAE (MW)"],
            "Train RMSE": metrics_train["RMSE (MW)"],
            "Train R^2": metrics_train["R^2 Score"],
            "Test MAE": metrics_test["MAE (MW)"],
            "Test RMSE": metrics_test["RMSE (MW)"],
            "Test R^2": metrics_test["R^2 Score"],
            "Coefficients": model.coef_
        }
        models[deg] = (poly, model)
        curve_predictions[f"Degree_{deg}"] = y_pred_curve
        
    results_df = pd.DataFrame(results).T
    return results_df, models, curve_predictions

def train_supervised_model(train_df, test_df):
    """
    Trains a multivariate supervised regression model for Electricity Demand.
    Uses interpretable features:
    - Temperature (°C)
    - Temperature squared (capturing thermodynamic U-curve)
    - Solar Generation (MW)
    - Wind Generation (MW)
    - Calendar features: Hour, Day of Week, Month, Is Weekend
    """
    feature_cols_base = [
        "temperature_celsius",
        "solar_generation_mw",
        "wind_generation_mw",
        "hour",
        "day_of_week",
        "month",
        "is_weekend"
    ]
    
    # Model 1: Baseline Linear Regression
    X_train_base = train_df[feature_cols_base].values
    y_train = train_df["electricity_demand_mw"].values
    X_test_base = test_df[feature_cols_base].values
    y_test = test_df["electricity_demand_mw"].values
    
    linear_model = LinearRegression()
    linear_model.fit(X_train_base, y_train)
    y_pred_train_linear = linear_model.predict(X_train_base)
    y_pred_test_linear = linear_model.predict(X_test_base)
    
    # Model 2: Polynomial Feature-Engineered Regression (adds Temperature^2 & Hour interactions)
    train_feat = train_df[feature_cols_base].copy()
    test_feat = test_df[feature_cols_base].copy()
    
    # Explicit domain-guided polynomial features
    train_feat["temp_squared"] = train_feat["temperature_celsius"] ** 2
    test_feat["temp_squared"] = test_feat["temperature_celsius"] ** 2
    train_feat["hour_sin"] = np.sin(2 * np.pi * train_feat["hour"] / 24.0)
    train_feat["hour_cos"] = np.cos(2 * np.pi * train_feat["hour"] / 24.0)
    test_feat["hour_sin"] = np.sin(2 * np.pi * test_feat["hour"] / 24.0)
    test_feat["hour_cos"] = np.cos(2 * np.pi * test_feat["hour"] / 24.0)
    
    engineered_cols = list(train_feat.columns)
    
    poly_model = LinearRegression()
    poly_model.fit(train_feat.values, y_train)
    y_pred_train_poly = poly_model.predict(train_feat.values)
    y_pred_test_poly = poly_model.predict(test_feat.values)
    
    comparison = {
        "Linear Regression (Base)": {
            "Train MAE": mean_absolute_error(y_train, y_pred_train_linear),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_linear)),
            "Train R^2": r2_score(y_train, y_pred_train_linear),
            "Test MAE": mean_absolute_error(y_test, y_pred_test_linear),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_linear)),
            "Test R^2": r2_score(y_test, y_pred_test_linear)
        },
        "Feature-Engineered Polynomial Model": {
            "Train MAE": mean_absolute_error(y_train, y_pred_train_poly),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_poly)),
            "Train R^2": r2_score(y_train, y_pred_train_poly),
            "Test MAE": mean_absolute_error(y_test, y_pred_test_poly),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_poly)),
            "Test R^2": r2_score(y_test, y_pred_test_poly)
        }
    }
    
    comparison_df = pd.DataFrame(comparison).T
    
    return {
        "comparison_df": comparison_df,
        "best_model": poly_model,
        "feature_cols": engineered_cols,
        "y_test": y_test,
        "y_pred_test": y_pred_test_poly,
        "y_pred_train": y_pred_train_poly,
        "test_features_df": test_feat
    }

def train_unsupervised_kmeans(df, n_clusters=4, random_state=42):
    """
    Applies K-Means clustering to discover latent grid operating regimes.
    Features: Electricity Demand, Solar Generation, Wind Generation, Temperature.
    """
    cluster_cols = [
        "electricity_demand_mw",
        "solar_generation_mw",
        "wind_generation_mw",
        "temperature_celsius"
    ]
    
    X = df[cluster_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate unscaled cluster centroids
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels
    cluster_summary = df_clustered.groupby("cluster")[cluster_cols].mean()
    cluster_counts = df_clustered["cluster"].value_counts().sort_index()
    cluster_summary["Frequency"] = cluster_counts
    cluster_summary["Percentage (%)"] = (cluster_counts / len(df) * 100).round(2)
    
    # Qualitative Regime Labeling based on centroid values
    regime_names = {}
    for c_id, row in cluster_summary.iterrows():
        dem = row["electricity_demand_mw"]
        temp = row["temperature_celsius"]
        sol = row["solar_generation_mw"]
        wnd = row["wind_generation_mw"]
        
        if temp > 22 and dem > 23000:
            regime_names[c_id] = "Summer Cooling Peak (High Temp, High Demand)"
        elif temp < 12 and dem > 23000:
            regime_names[c_id] = "Winter Heating Peak (Low Temp, High Demand)"
        elif sol > 3000:
            regime_names[c_id] = "Solar Daytime Flush (High Solar, Mid Demand)"
        elif wnd > 4500:
            regime_names[c_id] = "Windy Transition (High Wind, Moderate Load)"
        else:
            regime_names[c_id] = "Off-Peak Baseload / Night"
            
    cluster_summary["Regime Description"] = cluster_summary.index.map(regime_names)
    
    return {
        "kmeans_model": kmeans,
        "scaler": scaler,
        "cluster_labels": cluster_labels,
        "cluster_summary": cluster_summary,
        "cluster_cols": cluster_cols,
        "regime_names": regime_names
    }
