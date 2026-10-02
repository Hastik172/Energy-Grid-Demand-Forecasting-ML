"""
Data Preprocessing Module
Covers:
- Dataset inspection: shape, dtypes, first rows
- Missing value detection and domain-appropriate imputation (forward-fill / interpolation)
- Duplicate row detection and removal
- Datetime decomposition and calendar feature engineering
- Train-test chronological split
"""

import pandas as pd
import numpy as np

def inspect_dataset(df):
    """
    Performs initial data health check and structural inspection.
    """
    inspection = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum())
    }
    return inspection

def clean_and_preprocess(df):
    """
    Applies rigorous data cleaning:
    1. Removes duplicate rows
    2. Parses datetime column
    3. Sorts chronologically
    4. Imputes missing values using time-aware interpolation (appropriate for physical sensor series)
    5. Extracts temporal calendar features
    """
    df_clean = df.copy()
    
    # 1. Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_len - len(df_clean)
    
    # 2. Datetime conversion & chronological ordering
    if "timestamp" in df_clean.columns:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)
    
    # 3. Handle missing values
    # For physical energy systems, linear interpolation preserves the physical continuity of load and temperature
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    missing_before = df_clean[numeric_cols].isnull().sum().to_dict()
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method="linear").bfill().ffill()
    missing_after = df_clean[numeric_cols].isnull().sum().to_dict()
    
    # 4. Feature Engineering
    if "timestamp" in df_clean.columns:
        df_clean["hour"] = df_clean["timestamp"].dt.hour
        df_clean["day_of_week"] = df_clean["timestamp"].dt.dayofweek
        df_clean["month"] = df_clean["timestamp"].dt.month
        df_clean["day_of_year"] = df_clean["timestamp"].dt.dayofyear
        df_clean["is_weekend"] = (df_clean["day_of_week"] >= 5).astype(int)
        
        # Season categorical mapping (Meteorological)
        # 12, 1, 2: Winter; 3, 4, 5: Spring; 6, 7, 8: Summer; 9, 10, 11: Fall
        month = df_clean["month"]
        df_clean["season"] = np.where(month.isin([12, 1, 2]), "Winter",
                             np.where(month.isin([3, 4, 5]), "Spring",
                             np.where(month.isin([6, 7, 8]), "Summer", "Fall")))
                             
    # Net load computation: Net Load = Demand - (Solar + Wind)
    if all(col in df_clean.columns for col in ["electricity_demand_mw", "solar_generation_mw", "wind_generation_mw"]):
        df_clean["renewable_generation_mw"] = df_clean["solar_generation_mw"] + df_clean["wind_generation_mw"]
        df_clean["net_load_mw"] = df_clean["electricity_demand_mw"] - df_clean["renewable_generation_mw"]
        
    return df_clean, duplicates_removed, missing_before, missing_after

def split_train_test(df, train_ratio=0.8):
    """
    Chronological train-test split for time-series energy forecasting.
    Avoids lookahead bias (data leakage) by keeping past in train, future in test.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df
