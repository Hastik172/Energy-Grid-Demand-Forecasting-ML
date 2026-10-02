"""
Dataset Generation Script for Energy Grid Demand & Renewable Power Forecasting
Generates realistic, physically consistent hourly power system data spanning 1 full year (8,760 hours).
Includes:
- Electricity Demand (MW) with diurnal, weekly, and thermodynamic U-curve temperature dependence
- Solar Power Generation (MW) following solar elevation geometry and cloud attenuation
- Wind Power Generation (MW) following Weibull distribution and standard turbine power curves
- Meteorological features: Temperature (°C), Humidity (%), Wind Speed (m/s), Cloud Cover (%)
- Realistic missing values and duplicate rows for practical preprocessing demonstration
"""

import os
import numpy as np
import pandas as pd

def generate_energy_dataset(output_path="data/energy_grid_dataset.csv", seed=42):
    np.random.seed(seed)
    
    # 1 full year of hourly records (8,760 hours for 2023)
    date_range = pd.date_range(start="2023-01-01 00:00:00", end="2023-12-31 23:00:00", freq="h")
    n = len(date_range)
    
    day_of_year = date_range.dayofyear.values
    hour_of_day = date_range.hour.values
    day_of_week = date_range.dayofweek.values # 0=Monday, 6=Sunday
    is_weekend = (day_of_week >= 5).astype(int)
    
    # --- 1. Ambient Temperature (°C) ---
    # Seasonal cycle (coldest ~Jan/Feb day 25, hottest ~July/Aug day 205)
    t_season = 16.0 + 12.5 * np.cos(2 * np.pi * (day_of_year - 205) / 365)
    # Diurnal cycle (coolest at 05:00, warmest at 15:00)
    t_diurnal = 4.8 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    # Autocorrelated synoptic weather noise AR(1)
    weather_noise = np.zeros(n)
    for i in range(1, n):
        weather_noise[i] = 0.94 * weather_noise[i-1] + np.random.normal(0, 1.1)
    temperature = np.round(t_season + t_diurnal + weather_noise, 2)
    
    # --- 2. Cloud Cover (%) and Relative Humidity (%) ---
    cloud_noise = np.zeros(n)
    for i in range(1, n):
        cloud_noise[i] = 0.88 * cloud_noise[i-1] + np.random.normal(0, 12.0)
    cloud_cover = np.clip(np.round(45.0 + 15.0 * np.sin(2 * np.pi * day_of_year / 365) + cloud_noise, 1), 0, 100)
    
    # Humidity negatively correlated with temperature, positively with cloud cover
    humidity = np.clip(np.round(82.0 - 1.4 * (temperature - 10) + 0.25 * cloud_cover + np.random.normal(0, 4.0, n), 1), 15, 99)
    
    # --- 3. Wind Speed (m/s) and Wind Power Generation (MW) ---
    # Wind speed modeled via Weibull distribution with temporal autocorrelation
    base_wind = np.random.weibull(2.1, n) * 6.8 # average ~ 6.0 m/s
    wind_diurnal = 1.2 * np.cos(2 * np.pi * (hour_of_day - 3) / 24) # higher wind early morning
    wind_speed = np.clip(np.round(base_wind + wind_diurnal, 2), 0.5, 26.0)
    
    # Wind Turbine Power Curve (Installed capacity = 10,000 MW)
    # Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s
    wind_capacity = 10000.0
    wind_power = np.zeros(n)
    for i in range(n):
        v = wind_speed[i]
        if v < 3.0 or v > 25.0:
            wind_power[i] = 0.0
        elif v <= 12.0:
            # Cubic power relationship between cut-in and rated speed
            wind_power[i] = wind_capacity * ((v - 3.0) / (12.0 - 3.0))**3
        else:
            wind_power[i] = wind_capacity
    # Add slight operational variability
    wind_power = np.clip(np.round(wind_power * np.random.uniform(0.92, 1.02, n), 2), 0, wind_capacity)
    
    # --- 4. Solar Power Generation (MW) ---
    # Installed capacity = 8,000 MW
    solar_capacity = 8000.0
    solar_power = np.zeros(n)
    for i in range(n):
        h = hour_of_day[i]
        d = day_of_year[i]
        if 6 <= h <= 19: # Daylight hours
            # Solar elevation profile peaking at 12.5 (solar noon)
            elevation = np.sin(np.pi * (h - 5.5) / 14.0)
            # Seasonal factor (summer days have higher solar irradiance)
            season_factor = 0.65 + 0.35 * np.cos(2 * np.pi * (d - 172) / 365)
            # Cloud attenuation (Beer-Lambert law approximation)
            cloud_transmittance = 1.0 - 0.75 * (cloud_cover[i] / 100.0)
            ideal_solar = solar_capacity * (elevation ** 1.3) * season_factor * cloud_transmittance
            solar_power[i] = max(0.0, ideal_solar + np.random.normal(0, 45.0))
        else:
            solar_power[i] = 0.0
    solar_power = np.clip(np.round(solar_power, 2), 0, solar_capacity)
    
    # --- 5. Electricity Demand (MW) ---
    # Baseload
    base_demand = 22000.0
    
    # Diurnal profile: Morning ramp (07:00-09:00), midday plateau, Evening peak (18:00-21:00), night trough (01:00-05:00)
    hourly_factors = np.array([
        -3800, -4600, -5100, -5200, -4800, -3200,   # 00:00 - 05:00 (Night trough)
         -800,  1800,  3200,  3600,  3800,  3900,   # 06:00 - 11:00 (Morning ramp & noon)
         3700,  3500,  3400,  3700,  4300,  5500,   # 12:00 - 17:00 (Afternoon ramp)
         6200,  5800,  4800,  3100,   600, -1800    # 18:00 - 23:00 (Evening peak & night drop)
    ])
    diurnal_demand = hourly_factors[hour_of_day]
    
    # Weekend reduction (commercial/industrial load drop)
    weekend_factor = np.where(is_weekend == 1, -3400.0, 0.0)
    
    # Thermodynamic U-Curve (Temperature effect):
    # Heating demand below 18°C (space heaters, heat pumps)
    heating_demand = 28.0 * np.maximum(0, 18.0 - temperature) ** 2
    # Cooling demand above 22°C (air conditioning, chillers)
    cooling_demand = 42.0 * np.maximum(0, temperature - 22.0) ** 2
    
    # Combined demand with Gaussian noise
    demand_noise = np.random.normal(0, 450.0, n)
    electricity_demand = np.round(
        base_demand + diurnal_demand + weekend_factor + heating_demand + cooling_demand + demand_noise,
        2
    )
    
    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": date_range.strftime("%Y-%m-%d %H:%M:%S"),
        "electricity_demand_mw": electricity_demand,
        "solar_generation_mw": solar_power,
        "wind_generation_mw": wind_power,
        "temperature_celsius": temperature,
        "humidity_percent": humidity,
        "wind_speed_ms": wind_speed,
        "cloud_cover_percent": cloud_cover
    })
    
    # Inject ~25 realistic missing values (NaN) to demonstrate real-world cleaning
    missing_indices = np.random.choice(n, size=25, replace=False)
    for idx in missing_indices[:8]:
        df.loc[idx, "temperature_celsius"] = np.nan
    for idx in missing_indices[8:15]:
        df.loc[idx, "humidity_percent"] = np.nan
    for idx in missing_indices[15:20]:
        df.loc[idx, "solar_generation_mw"] = np.nan
    for idx in missing_indices[20:]:
        df.loc[idx, "electricity_demand_mw"] = np.nan
        
    # Inject 5 duplicate rows to demonstrate duplicate detection and removal
    dup_rows = df.iloc[[100, 500, 1200, 3400, 5600]].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Shuffle slightly at the end to keep duplicate handling authentic
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total rows (with duplicates): {len(df)}, Columns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    generate_energy_dataset()
