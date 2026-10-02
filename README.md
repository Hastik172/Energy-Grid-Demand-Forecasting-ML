# Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification

## Unit 1: Machine Learning & Probability Theory Project

A complete beginner-to-intermediate Machine Learning project designed around **Unit 1 ML and Probability Theory** topics. This system analyzes historical electricity demand and renewable energy generation data, predicts future energy demand, and quantifies the uncertainty of predictions using probability distributions, variance, quantiles, and prediction intervals.

---

## Project Structure

```
ML 1/
├── main.py                          # Full pipeline runner (all 21 sections)
├── dashboard.py                     # Interactive terminal dashboard
├── generate_notebook.py             # Jupyter notebook generator
├── Energy_Grid_Forecasting_Unit1_ML.ipynb  # Complete Jupyter Notebook
├── README.md                        # This file
│
├── data/
│   ├── generate_dataset.py          # Realistic energy dataset generator
│   ├── data_loader.py               # Adaptive column-mapping data loader
│   └── energy_grid_dataset.csv      # Generated dataset (8,760 hourly records)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Data cleaning, feature engineering, train/test split
│   ├── continuous_analysis.py       # Continuous RV statistics, PDF fitting
│   ├── probability_analysis.py      # Probability events, Bayes, independence, covariance
│   ├── models.py                    # Polynomial fitting, supervised regression, K-Means
│   ├── uncertainty.py               # Residual analysis, prediction intervals
│   └── visualizer.py                # All 12 required visualizations
│
└── figures/                         # Generated visualization PNGs
    ├── 01_demand_timeseries.png
    ├── 02_renewable_generation.png
    ├── 03_demand_histogram.png
    ├── 04_probability_density.png
    ├── 05_demand_vs_temperature.png
    ├── 06_polynomial_regression_curve.png
    ├── 07_discrete_pmf.png
    ├── 08_bayes_probability.png
    ├── 09_clustering_visualization.png
    ├── 10_actual_vs_predicted_demand.png
    ├── 11_prediction_error_distribution.png
    └── 12_prediction_interval_uncertainty.png
```

---

## Unit 1 Topics Covered

| # | Topic | Project Section |
|---|-------|----------------|
| 1 | What is Machine Learning | Introduction & Problem Statement |
| 2 | Supervised Learning | Linear & Polynomial Regression (Section 16) |
| 3 | Unsupervised Learning | K-Means Clustering (Section 17) |
| 4 | Polynomial Curve Fitting | Temperature → Demand fitting (Section 15) |
| 5 | Probability Theory | Fundamental rules with energy events (Section 7) |
| 6 | Discrete Random Variables | 3-state demand PMF/CDF (Section 8) |
| 7 | Fundamental Rules of Probability | P(A), P(B), P(A∩B), P(A∪B), Addition Rule (Section 7) |
| 8 | Bayes' Rule | P(High Demand \| High Temp) (Section 9) |
| 9 | Independence | Pairwise & conditional independence tests (Section 10) |
| 10 | Continuous Random Variables | Demand/Temperature/Solar/Wind analysis (Section 11) |
| 11 | Quantiles | 5th, 25th, 50th, 75th, 95th percentiles (Section 12) |
| 12 | Mean and Variance | Sample statistics for all variables (Section 12) |
| 13 | Probability Density Functions | Gaussian PDF vs KDE (Section 13) |
| 14 | Expectation and Covariance | E[X], Cov(X,Y), Corr(X,Y) (Section 14) |

---

## How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### 1. Generate Dataset (if not present)
```bash
python data/generate_dataset.py
```

### 2. Run Full Pipeline
```bash
python main.py
```

### 3. Interactive Dashboard
```bash
python dashboard.py
```

### 4. Jupyter Notebook
```bash
# Install Jupyter if needed:
pip install jupyter
jupyter notebook Energy_Grid_Forecasting_Unit1_ML.ipynb
```

---

## Key Results

- **Supervised Model R² = 0.809** (Feature-Engineered Polynomial Regression)
- **90% Prediction Interval Coverage = 89.95%** (close to nominal 90%)
- **Bayes' Rule**: High temperature increases high-demand probability by **1.79x**
- **K-Means identifies 4 grid regimes**: Winter Heating, Summer Cooling, Solar Daytime, Windy Transition
- **Kolmogorov Addition Rule verified with 0.0 discrepancy**

---

## Technology Stack

- Python 3.12
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- SciPy

**No deep learning or advanced algorithms used.** All methods are within Unit 1 scope.

---

## Dataset

Synthetically generated with physical equations modeling:
- **8,760 hourly records** (full year 2023)
- **Thermodynamic U-curve** temperature-demand relationship (heating + cooling)
- **Solar generation** following solar elevation geometry and cloud attenuation
- **Wind power** following Weibull distribution and standard turbine power curves
- **Weather features**: Temperature, Humidity, Wind Speed, Cloud Cover
- **Realistic imperfections**: 25 missing values, 5 duplicate rows

---

## Author

B.Tech CSE — Unit 1 Machine Learning & Probability Theory Project
