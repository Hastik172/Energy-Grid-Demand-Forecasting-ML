"""
Intelligent Dataset Loader with Automatic Column Adaptation
Enables seamless loading of arbitrary energy datasets (OPSD, Kaggle, National Grid, custom CSVs)
by automatically mapping varying column names to standardized canonical names.
"""

import os
import re
import pandas as pd

# Canonical column definitions
CANONICAL_COLUMNS = {
    "timestamp": ["timestamp", "datetime", "date", "time", "dt", "utc_timestamp", "period"],
    "electricity_demand_mw": ["electricity_demand_mw", "demand", "load", "consumption", "power_demand", "actual_demand", "total_load"],
    "solar_generation_mw": ["solar_generation_mw", "solar", "solar_power", "pv", "solar_generation", "pv_generation", "solar_mw"],
    "wind_generation_mw": ["wind_generation_mw", "wind", "wind_power", "wind_generation", "wind_production", "wind_mw", "wind_onshore"],
    "temperature_celsius": ["temperature_celsius", "temperature", "temp", "temp_c", "ambient_temperature", "temp_air", "air_temperature"],
    "humidity_percent": ["humidity_percent", "humidity", "rh", "relative_humidity", "rel_hum"],
    "wind_speed_ms": ["wind_speed_ms", "wind_speed", "speed_wind", "wind_velocity", "windspeed"],
    "cloud_cover_percent": ["cloud_cover_percent", "cloud_cover", "cloudiness", "clouds"]
}

def auto_adapt_columns(df):
    """
    Inspects DataFrame column names and automatically maps them to canonical names.
    Uses priority-ordered exact and keyword matching.
    """
    mapped_columns = {}
    original_cols = list(df.columns)
    
    # Priority order for canonical keys
    priority_order = [
        "timestamp",
        "wind_speed_ms",
        "cloud_cover_percent",
        "humidity_percent",
        "temperature_celsius",
        "solar_generation_mw",
        "wind_generation_mw",
        "electricity_demand_mw"
    ]
    
    for col in original_cols:
        clean_col = re.sub(r"[^a-zA-Z0-9]", "_", str(col).lower()).strip("_")
        matched = False
        
        # Pass 1: Exact matches
        for canonical in priority_order:
            aliases = CANONICAL_COLUMNS[canonical]
            if clean_col in aliases:
                mapped_columns[col] = canonical
                matched = True
                break
                
        # Pass 2: Fuzzy / substring matches if not matched exactly
        if not matched:
            for canonical in priority_order:
                aliases = CANONICAL_COLUMNS[canonical]
                for alias in aliases:
                    if alias in clean_col or clean_col in alias:
                        mapped_columns[col] = canonical
                        matched = True
                        break
                if matched:
                    break
                    
        if not matched:
            mapped_columns[col] = col # Keep original if no match
            
    df_adapted = df.rename(columns=mapped_columns)
    return df_adapted, mapped_columns

def load_energy_data(file_path="data/energy_grid_dataset.csv"):
    """
    Loads dataset from CSV, performs automatic column adaptation, and returns loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    raw_df = pd.read_csv(file_path)
    adapted_df, mapping = auto_adapt_columns(raw_df)
    
    print(f"Loaded dataset from: {file_path}")
    print(f"Shape: {adapted_df.shape[0]} rows, {adapted_df.shape[1]} columns")
    print(f"Column adaptations mapped:")
    for orig, canon in mapping.items():
        if orig != canon:
            print(f"  - '{orig}' -> '{canon}'")
            
    return adapted_df, raw_df

if __name__ == "__main__":
    df, _ = load_energy_data()
    print("\nFirst 3 rows:")
    print(df.head(3))
