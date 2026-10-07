"""
REST API Routes for GridSense Backend using FastAPI.
Provides comprehensive endpoints matching the peer's Swagger UI structure.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from backend.data_service import DataService

router = APIRouter(prefix="/api", tags=["GridSense API"])


class PredictionRequest(BaseModel):
    temperature: float = Field(24.0, description="Ambient air temperature in Celsius (°C)")
    solar_generation: float = Field(3500.0, description="Solar power output in MW")
    wind_generation: float = Field(1200.0, description="Wind power output in MW")
    hour: int = Field(14, ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(7, ge=1, le=12, description="Month of year (1-12)")
    is_weekend: Optional[int] = Field(0, ge=0, le=1, description="1 if weekend, else 0")


@router.get("/overview", summary="High-level model metrics & headline statistics")
def get_overview():
    service = DataService.get_instance()
    return service.get_overview_metrics()


@router.get("/data/preview", summary="Raw dataset preview rows")
def get_data_preview(limit: int = Query(20, ge=1, le=100)):
    service = DataService.get_instance()
    return service.get_data_preview(limit=limit)


@router.get("/data/summary", summary="Per-column descriptive statistics")
def get_data_summary():
    service = DataService.get_instance()
    return service.get_column_summary()


@router.get("/stats/probability", summary="Bayes rule and conditional probability calculation")
def get_probability_stats(condition: str = Query("extreme_heat", description="Condition name")):
    service = DataService.get_instance()
    return service.compute_bayes_scenario(condition)


@router.get("/stats/distributions", summary="Continuous random variable density & quantile analysis")
def get_distributions(
    variable: str = Query("electricity_demand_mw"),
    quantile_p: float = Query(0.85, ge=0.01, le=0.99)
):
    service = DataService.get_instance()
    return service.get_distribution_data(variable=variable, quantile_p=quantile_p)


@router.get("/polyfit", summary="Bishop §1.1 Polynomial curve fitting & overfitting laboratory")
def get_polyfit(
    degree: int = Query(3, ge=1, le=9),
    regularization: float = Query(0.0, ge=0.0),
    sample_size: int = Query(500, ge=50, le=7000)
):
    service = DataService.get_instance()
    return service.run_polyfit_experiment(
        degree=degree,
        reg_lambda=regularization,
        sample_size=sample_size
    )


@router.post("/predict", summary="Predict electricity demand with uncertainty intervals")
def predict_demand(req: PredictionRequest):
    service = DataService.get_instance()
    weekend = req.is_weekend if req.is_weekend is not None else (1 if req.day_of_week >= 5 else 0)
    return service.predict_demand(
        temp=req.temperature,
        solar=req.solar_generation,
        wind=req.wind_generation,
        hour=req.hour,
        dow=req.day_of_week,
        month=req.month,
        is_weekend=weekend
    )


@router.get("/graphs", summary="Catalogue of all training graphs with detailed explanations")
def list_graphs():
    graphs = [
        {
            "id": "01_demand_timeseries",
            "title": "Demand Time-Series",
            "topic": "Time-Series & Diurnal Cycles",
            "image": "/figures/01_demand_timeseries.png",
            "what_it_shows": "Full 8,760-hour annual electricity demand curve alongside a 2-week diurnal cycle zoom.",
            "how_computed": "Plotted directly from time-indexed hourly load measurements.",
            "how_to_read": "The top panel shows seasonal shifts (winter and summer peaks); the bottom reveals 24-hour cycles with evening ramps.",
            "where_used": "Used by grid operators for medium-term capacity scheduling and fuel procurement."
        },
        {
            "id": "02_renewable_generation",
            "title": "Renewable Generation Dynamics",
            "topic": "Solar vs Wind Intermittency",
            "image": "/figures/02_renewable_generation.png",
            "what_it_shows": "Comparison of solar photovoltaic generation vs stochastic wind turbine power output.",
            "how_computed": "Aggregated hourly MW generation from utility solar farms and wind turbine parks.",
            "how_to_read": "Solar follows a smooth daytime bell curve, while wind exhibits rapid stochastic fluctuations.",
            "where_used": "Informs dispatchers how much flexible thermal backup is needed to balance renewable intermittency."
        },
        {
            "id": "03_demand_histogram",
            "title": "Demand Histogram & Quantiles",
            "topic": "Empirical Frequency Distribution",
            "image": "/figures/03_demand_histogram.png",
            "what_it_shows": "Empirical distribution of demand with Q1 (25%), Median (50%), and Q3 (75%) cutoffs.",
            "how_computed": "Histogram binning over 8,760 demand observations normalized to probability density.",
            "how_to_read": "The horizontal axis shows demand in MW; vertical lines show quartile thresholds.",
            "where_used": "Used by regulatory commissions to establish baseline tariff structures and capacity brackets."
        },
        {
            "id": "04_probability_density",
            "title": "Probability Density: Gaussian vs KDE",
            "topic": "Parametric vs Non-Parametric PDF",
            "image": "/figures/04_probability_density.png",
            "what_it_shows": "Continuous PDF comparison: fitted parametric Gaussian N(mu, sigma) vs non-parametric KDE.",
            "how_computed": "KDE computed via Gaussian kernels; parametric curve fitted with sample mean and sample variance.",
            "how_to_read": "Divergence between the curves highlights non-Gaussian multi-modal behavior driven by day/night cycles.",
            "where_used": "Probabilistic loss estimation and Value-at-Risk (VaR) calculations in energy trading."
        },
        {
            "id": "05_demand_vs_temperature",
            "title": "Demand vs Temperature U-Curve",
            "topic": "Nonlinear Physical Thermodynamics",
            "image": "/figures/05_demand_vs_temperature.png",
            "what_it_shows": "Scatter plot of electricity demand vs ambient temperature showing the thermodynamic U-curve.",
            "how_computed": "Hourly load plotted against ambient station temperature, shaded by hour of day.",
            "how_to_read": "Low temps (<18°C) trigger heating load; high temps (>22°C) trigger cooling; middle zone is comfort neutral.",
            "where_used": "Guides weather-sensitive demand response programs and peaker plant dispatch triggers."
        },
        {
            "id": "06_polynomial_regression_curve",
            "title": "Polynomial Curve Fitting (Bishop §1.1)",
            "topic": "Polynomial Regression & Bias-Variance",
            "image": "/figures/06_polynomial_regression_curve.png",
            "what_it_shows": "Degrees 1, 2, and 3 regression curves fitted to Temperature -> Demand.",
            "how_computed": "Ordinary least squares Normal Equations minimizing sum-of-squares error.",
            "how_to_read": "Degree 1 underfits (straight line); Degree 2 captures the U-curve inflection with lower error.",
            "where_used": "Used in building thermal energy modeling and degree-day energy demand estimation."
        },
        {
            "id": "07_discrete_pmf",
            "title": "Discrete Demand State PMF",
            "topic": "Discrete Random Variables & PMF",
            "image": "/figures/07_discrete_pmf.png",
            "what_it_shows": "Probability Mass Function of 3 discrete demand states: Low (0), Normal (1), High (2).",
            "how_computed": "Continuous load partitioned into empirical terciles, calculating P(X = k) = N_k / N.",
            "how_to_read": "Equal 33.3% heights reflect balanced categorization; discrete expectation E[X] = 1.0.",
            "where_used": "Discrete grid operational alert levels (Green / Amber / Red capacity alerts)."
        },
        {
            "id": "08_bayes_probability",
            "title": "Bayes' Rule: Updating Risk",
            "topic": "Bayesian Conditional Inference",
            "image": "/figures/08_bayes_probability.png",
            "what_it_shows": "Prior probability (25%) updated to posterior probability (44.8%) conditioned on extreme heat.",
            "how_computed": "Bayes' formula: P(High Demand | High Temp) = P(High Temp | High Demand) * P(High Demand) / P(High Temp).",
            "how_to_read": "Bar heights show how evidence multiplies baseline odds by 1.79x.",
            "where_used": "Transmission System Operator (TSO) emergency capacity warnings based on weather forecasts."
        },
        {
            "id": "09_clustering_visualization",
            "title": "Unsupervised Regime Clustering",
            "topic": "K-Means Unsupervised Learning",
            "image": "/figures/09_clustering_visualization.png",
            "what_it_shows": "4 latent operational regimes discovered on the Temperature-Demand plane by K-Means.",
            "how_computed": "Standardized K-Means clustering on [Demand, Solar, Wind, Temperature] with K = 4.",
            "how_to_read": "Color-coded clusters separate winter heating, summer cooling, off-peak baseload, and windy periods.",
            "where_used": "Automated power plant dispatch scheduling based on operational regime classification."
        },
        {
            "id": "10_actual_vs_predicted_demand",
            "title": "Actual vs Predicted Demand",
            "topic": "Supervised Model Performance",
            "image": "/figures/10_actual_vs_predicted_demand.png",
            "what_it_shows": "Parity plot (y = x) alongside 168-hour (1-week) continuous tracking on unseen test data.",
            "how_computed": "Predictions from feature-engineered polynomial regression plotted against actual meter data.",
            "how_to_read": "Tight alignment along the red 45-degree diagonal confirms high explanatory power (R² = 0.809).",
            "where_used": "Day-ahead electricity market dispatch and bidding algorithms."
        },
        {
            "id": "11_prediction_error_distribution",
            "title": "Prediction Error Distribution",
            "topic": "Residual Analysis & Error Quantiles",
            "image": "/figures/11_prediction_error_distribution.png",
            "what_it_shows": "Distribution of prediction errors (residuals) with e_0.05 and e_0.95 quantile cutoffs.",
            "how_computed": "Residual = Actual - Predicted on test partition, evaluated via sample quantiles.",
            "how_to_read": "Centered near zero (mean error = -244 MW); 90% of errors lie between -2,760 MW and +2,075 MW.",
            "where_used": "Sizing spinning reserve capacity and battery storage buffers."
        },
        {
            "id": "12_prediction_interval_uncertainty",
            "title": "Uncertainty Prediction Interval",
            "topic": "Empirical Prediction Intervals",
            "image": "/figures/12_prediction_interval_uncertainty.png",
            "what_it_shows": "Point forecast with shaded 90% uncertainty envelope enclosing observed ground truth.",
            "how_computed": "Lower Bound = y_hat + e_0.05; Upper Bound = y_hat + e_0.95.",
            "how_to_read": "The blue shaded band represents uncertainty; exactly 89.95% of test points fall inside the band.",
            "where_used": "Reliability-driven security-constrained unit commitment (SCUC) in power systems."
        }
    ]
    return graphs


@router.get("/graphs/{gid}", summary="Detailed view of a single graph with metadata")
def get_graph_by_id(gid: str):
    graphs = list_graphs()
    for g in graphs:
        if g["id"] == gid:
            return g
    raise HTTPException(status_code=404, detail="Graph not found")
