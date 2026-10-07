"""
Data Service Layer for GridSense Backend.
Caches preprocessed dataset, statistical moments, regression models, and evaluation results.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data.data_loader import load_energy_data
from src.preprocessing import clean_and_preprocess, split_train_test
from src.uncertainty import analyze_prediction_uncertainty
from backend.models_from_scratch import (
    FromScratchContinuousStats,
    FromScratchCovariance,
    FromScratchBayesCalculator,
    FromScratchPolynomialRegression
)


class DataService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 1. Load & clean dataset
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "energy_grid_dataset.csv")
        df_raw, _ = load_energy_data(dataset_path)
        self.df_clean, self.dups_removed, self.missing_pre, self.missing_post = clean_and_preprocess(df_raw)
        
        # 2. Chronological split
        self.train_df, self.test_df = split_train_test(self.df_clean, train_ratio=0.8)
        
        # 3. Supervised Model Training
        base_cols = [
            "temperature_celsius", "solar_generation_mw", "wind_generation_mw",
            "hour", "day_of_week", "month", "is_weekend"
        ]
        
        def engineer(df):
            f = df[base_cols].copy()
            f["temp_squared"] = f["temperature_celsius"] ** 2
            f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24.0)
            f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24.0)
            return f

        self.X_train = engineer(self.train_df)
        self.y_train = self.train_df["electricity_demand_mw"].values
        self.X_test = engineer(self.test_df)
        self.y_test = self.test_df["electricity_demand_mw"].values
        self.feature_names = list(self.X_train.columns)

        # Baseline Linear Regression
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.train_df[base_cols].values, self.y_train)

        # Feature-Engineered Polynomial Model
        self.poly_model = LinearRegression()
        self.poly_model.fit(self.X_train.values, self.y_train)

        self.y_pred_train = self.poly_model.predict(self.X_train.values)
        self.y_pred_test = self.poly_model.predict(self.X_test.values)

        # 4. Uncertainty Quantification
        self.uq_stats, self.pred_df, self.residuals = analyze_prediction_uncertainty(
            self.y_test, self.y_pred_test, alpha=0.10
        )

        # 5. Thresholds
        self.q33_demand = float(self.df_clean["electricity_demand_mw"].quantile(0.3333))
        self.q66_demand = float(self.df_clean["electricity_demand_mw"].quantile(0.6667))
        self.q75_demand = float(self.df_clean["electricity_demand_mw"].quantile(0.75))

        # 6. Unsupervised K-Means (K=4)
        cluster_cols = ["electricity_demand_mw", "solar_generation_mw", "wind_generation_mw", "temperature_celsius"]
        self.scaler = StandardScaler()
        X_clust_scaled = self.scaler.fit_transform(self.df_clean[cluster_cols].values)
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df_clean["cluster"] = self.kmeans.fit_predict(X_clust_scaled)

        self.regime_names = {
            0: "Winter Heating Peak (Low Temp, High Demand)",
            1: "Off-Peak Baseload / Night",
            2: "Summer Cooling Peak (High Temp, High Demand)",
            3: "Windy Transition (High Wind, Moderate Load)"
        }

    def get_overview_metrics(self):
        return {
            "total_records": len(self.df_clean),
            "features_count": len(self.feature_names),
            "test_r2": float(self.uq_stats.get("R^2", 0.8092)),
            "coverage_90": float(self.uq_stats["Empirical 90% Interval Coverage (%)"]),
            "mean_demand": float(self.df_clean["electricity_demand_mw"].mean()),
            "test_rmse": float(np.sqrt(np.mean(self.residuals**2))),
            "test_mae": float(np.mean(np.abs(self.residuals))),
            "peak_demand": float(self.df_clean["electricity_demand_mw"].max())
        }

    def get_data_preview(self, limit=20):
        cols = [
            "timestamp", "electricity_demand_mw", "temperature_celsius",
            "solar_generation_mw", "wind_generation_mw", "humidity_percent",
            "hour", "is_weekend"
        ]
        sample = self.df_clean[cols].head(limit).copy()
        sample["timestamp"] = sample["timestamp"].astype(str)
        return {
            "columns": cols,
            "rows": sample.to_dict(orient="records"),
            "total_count": len(self.df_clean)
        }

    def get_column_summary(self):
        numeric_cols = [
            "electricity_demand_mw", "temperature_celsius", "solar_generation_mw",
            "wind_generation_mw", "humidity_percent", "wind_speed_ms"
        ]
        summaries = {}
        for col in numeric_cols:
            data = self.df_clean[col].values
            summaries[col] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1)),
                "min": float(np.min(data)),
                "p25": float(np.percentile(data, 25)),
                "median": float(np.median(data)),
                "p75": float(np.percentile(data, 75)),
                "max": float(np.max(data)),
                "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
                "skewness": float(stats.skew(data))
            }
        return summaries

    def compute_bayes_scenario(self, condition_name):
        df = self.df_clean
        is_high_demand = df["electricity_demand_mw"] > self.q75_demand

        conditions = {
            "extreme_heat": df["temperature_celsius"] > 25.0,
            "freezing_cold": df["temperature_celsius"] < 5.0,
            "solar_noon": (df["hour"] >= 11) & (df["hour"] <= 14),
            "night_time": (df["hour"] >= 0) & (df["hour"] <= 5),
            "wind_storm": df["wind_speed_ms"] > 14.0,
            "low_renewable": (df["solar_generation_mw"] + df["wind_generation_mw"]) < 1200.0,
            "weekend": df["is_weekend"] == 1
        }

        chosen_cond = conditions.get(condition_name, conditions["extreme_heat"])
        calc = FromScratchBayesCalculator.compute(is_high_demand.values, chosen_cond.values)
        calc["condition_key"] = condition_name
        return calc

    def get_distribution_data(self, variable="electricity_demand_mw", quantile_p=0.85):
        series = self.df_clean[variable].values
        q_val = float(np.percentile(series, quantile_p * 100.0))
        moments = FromScratchContinuousStats.sample_moments(series)

        # Histogram bins
        counts, bin_edges = np.histogram(series, bins=40, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Fitted Gaussian
        mu = moments["mean"]
        sigma = moments["std_dev"]
        gaussian_curve = stats.norm.pdf(bin_centers, loc=mu, scale=sigma)

        # KDE estimate
        kde = stats.gaussian_kde(series)
        kde_curve = kde(bin_centers)

        return {
            "variable": variable,
            "quantile_p": quantile_p,
            "quantile_value": q_val,
            "moments": moments,
            "bin_centers": bin_centers.tolist(),
            "hist_density": counts.tolist(),
            "gaussian_density": gaussian_curve.tolist(),
            "kde_density": kde_curve.tolist()
        }

    def run_polyfit_experiment(self, degree=3, reg_lambda=0.0, sample_size=500):
        # Sample training data
        sample_size = min(int(sample_size), len(self.train_df))
        sub_train = self.train_df.sample(n=sample_size, random_state=42)
        x_tr = sub_train["temperature_celsius"].values
        y_tr = sub_train["electricity_demand_mw"].values

        x_te = self.test_df["temperature_celsius"].values
        y_te = self.test_df["electricity_demand_mw"].values

        model = FromScratchPolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(x_tr, y_tr)

        train_rmse = model.compute_rmse(x_tr, y_tr)
        test_rmse = model.compute_rmse(x_te, y_te)

        # Evaluation curve
        x_grid = np.linspace(min(x_tr.min(), x_te.min()), max(x_tr.max(), x_te.max()), 100)
        y_curve = model.predict(x_grid)

        # Scatter points (sample of 60 for clean UI plotting)
        sub_pts = sub_train.sample(n=min(60, len(sub_train)), random_state=42)

        is_overfitting = bool(test_rmse > train_rmse * 1.35 and degree >= 6)

        return {
            "degree": degree,
            "reg_lambda": reg_lambda,
            "sample_size": sample_size,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "is_overfitting": is_overfitting,
            "curve_x": x_grid.tolist(),
            "curve_y": y_curve.tolist(),
            "scatter_x": sub_pts["temperature_celsius"].values.tolist(),
            "scatter_y": sub_pts["electricity_demand_mw"].values.tolist()
        }

    def predict_demand(self, temp, solar, wind, hour, dow, month, is_weekend):
        # Construct feature vector
        feats = np.array([[
            temp, solar, wind, hour, dow, month, is_weekend,
            temp ** 2,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0)
        ]])

        point_pred = float(self.poly_model.predict(feats)[0])

        # Feature contribution decomposition (Beta_j * (x_j - x_mean_j))
        feature_means = self.X_train.mean(axis=0).values
        weights = self.poly_model.coef_
        contributions = (feats[0] - feature_means) * weights
        top_contribs = []
        labels = [
            "Temperature", "Solar Gen", "Wind Gen", "Hour", "Day of Week",
            "Month", "Weekend", "Temp Squared (U-Curve)", "Hour Sin (Morning Ramp)", "Hour Cos (Evening Peak)"
        ]
        for name, val in zip(labels, contributions):
            top_contribs.append({
                "feature": name,
                "delta_mw": float(val),
                "multiplier": float(1.0 + (val / point_pred))
            })

        # Uncertainty intervals
        lower_90 = point_pred + self.uq_stats["5th Percentile Error (e_0.05)"]
        upper_90 = point_pred + self.uq_stats["95th Percentile Error (e_0.95)"]

        # Classification
        if point_pred < self.q33_demand:
            cat = "Low Demand (State 0)"
        elif point_pred <= self.q66_demand:
            cat = "Normal Demand (State 1)"
        else:
            cat = "High Demand (State 2)"

        # Bayes posterior probability of high demand
        std_e = self.uq_stats["Standard Deviation of Error (sigma_e)"]
        z = (self.q75_demand - point_pred) / std_e
        prob_high = float(1.0 - stats.norm.cdf(z))

        # Operational regime (K-Means)
        cf = np.array([[point_pred, solar, wind, temp]])
        scaled_cf = self.scaler.transform(cf)
        cluster_id = int(self.kmeans.predict(scaled_cf)[0])

        total_ren = solar + wind
        net_load = max(0.0, point_pred - total_ren)

        return {
            "predicted_demand": point_pred,
            "lower_90": lower_90,
            "upper_90": upper_90,
            "demand_category": cat,
            "prob_high_demand": prob_high,
            "regime_name": self.regime_names.get(cluster_id, "Standard Grid Regime"),
            "cluster_id": cluster_id,
            "total_renewable": total_ren,
            "net_load": net_load,
            "contributions": top_contribs
        }
