# Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification
## Unit 1: Machine Learning & Probability Theory — Comprehensive Academic Project Report

**Project Title:** Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification  
**Curriculum Scope:** Unit 1 — Machine Learning Foundations & Probability Theory  
**Academic Target:** 3rd Year B.Tech Computer Science & Engineering  
**Technology Stack:** Python 3.12, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn, Flask  

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Description & Architecture](#4-dataset-description--architecture)
5. [Data Preprocessing & Cleaning](#5-data-preprocessing--cleaning)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Probability Theory & Kolmogorov Axioms](#7-probability-theory--kolmogorov-axioms)
8. [Discrete Random Variables & PMF / CDF](#8-discrete-random-variables--pmf--cdf)
9. [Bayes' Rule & Posterior Updating](#9-bayes-rule--posterior-updating)
10. [Independence & Conditional Independence](#10-independence--conditional-independence)
11. [Continuous Random Variables](#11-continuous-random-variables)
12. [Mean, Variance, and Quantiles](#12-mean-variance-and-quantiles)
13. [Probability Density Functions (Gaussian vs KDE)](#13-probability-density-functions-gaussian-vs-kde)
14. [Expectation, Covariance, and Correlation](#14-expectation-covariance-and-correlation)
15. [Polynomial Curve Fitting (PRML Bias-Variance Tradeoff)](#15-polynomial-curve-fitting-prml-bias-variance-tradeoff)
16. [Supervised Learning (Multivariate Regression)](#16-supervised-learning-multivariate-regression)
17. [Unsupervised Learning (K-Means Clustering)](#17-unsupervised-learning-k-means-clustering)
18. [Uncertainty Quantification (UQ & Prediction Intervals)](#18-uncertainty-quantification-uq--prediction-intervals)
19. [Experimental Results & Visualizations](#19-experimental-results--visualizations)
20. [Conclusion](#20-conclusion)
21. [Future Scope](#21-future-scope)

---

## 1. Introduction

### 1.1 What is Machine Learning and Why is it Needed?
Traditional software engineering relies on explicit, deterministic rules:
$$\text{Data} + \text{Rules} \longrightarrow \text{Answers}$$
However, complex physical systems such as electrical power grids cannot be modeled solely by hand-crafted if-else rules. Power demand fluctuates continuously with weather patterns, human social activity, industrial schedules, and thermodynamic heating/cooling requirements.

**Machine Learning (ML)** inverts this paradigm:
$$\text{Data} + \text{Answers} \longrightarrow \text{Rules (Model)}$$

As formally defined by Tom Mitchell (1997):
> *"A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."*

In this project:
* **Task ($T$):** Predict hourly electricity demand (MW) and evaluate uncertainty bounds.
* **Experience ($E$):** 8,760 hourly historical measurements of load, temperature, solar radiation, wind speed, and calendar variables.
* **Performance ($P$):** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).

### 1.2 Supervised vs. Unsupervised Learning
* **Supervised Learning:** Learns a mapping function $f: \mathcal{X} \to \mathcal{Y}$ from labeled training pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$. Here, environmental features $\mathbf{x}$ predict labeled continuous load $y$ (regression).
* **Unsupervised Learning:** Discovers latent statistical structure, clusters, or distributions $p(\mathbf{x})$ without target labels. In this project, K-Means discovers hidden operational grid regimes (e.g., summer cooling peaks, baseload night).

### 1.3 Why Uncertainty Quantification (UQ) is Critical in Power Grids
A single point forecast ($\hat{y} = 28,500\text{ MW}$) provides zero insight into operational risk. Power grids must maintain instantaneous equilibrium:
$$\sum P_{\text{generation}}(t) = \sum P_{\text{load}}(t)$$
An unexpected demand spike can trigger frequency drops and rolling blackouts; unnecessary peaker dispatch wastes millions of dollars. UQ equips grid operators with calibrated confidence bounds:
$$[\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}] \quad \text{such that} \quad P(y \in [\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}]) \approx 1 - \alpha$$

---

## 2. Problem Statement

Given 8,760 hours of multi-variate environmental and electrical measurements, design, formulate, and evaluate an interpretable Machine Learning system grounded strictly within **Unit 1 principles** (linear/polynomial regression, basic probability, Kolmogorov axioms, Bayes' rule, random variables, expectation, covariance, and quantiles). The system must:
1. Predict electricity demand with high accuracy without relying on deep neural networks or complex ensembles.
2. Quantify forecast uncertainty via empirical residual quantiles and parametric Gaussian bounds.
3. Automatically adapt to varying column names in public datasets.
4. Provide both terminal and interactive web-based operator dashboards.

---

## 3. Objectives

* **Objective 1:** Formulate data-cleaning pipelines handling missing values and duplicates via physically consistent time-series interpolation.
* **Objective 2:** Verify Kolmogorov probability axioms and addition rules empirically.
* **Objective 3:** Implement Bayes' Theorem to calculate posterior probabilities of high-demand states conditioned on extreme weather.
* **Objective 4:** Model nonlinear thermodynamic load characteristics using polynomial curve fitting (Degrees 1, 2, and 3).
* **Objective 5:** Train multivariate supervised regression models comparing baseline linear vs. domain-engineered polynomial features.
* **Objective 6:** Apply unsupervised K-Means clustering to classify grid operational states.
* **Objective 7:** Formulate empirical and parametric 90% and 95% prediction intervals with coverage probability evaluation.

---

## 4. Dataset Description & Architecture

The system uses 1 full year (8,760 hours, 2023) of hourly power and meteorological records:

| Variable | Symbol | Unit | Physical Range | Description |
|---|---|---|---|---|
| Timestamp | $t$ | YYYY-MM-DD HH:MM | 2023-01-01 to 2023-12-31 | Chronological hourly index |
| Electricity Demand | $D_t$ | Megawatts (MW) | 12,330 – 40,031 MW | Target grid electrical load |
| Solar Generation | $S_t$ | Megawatts (MW) | 0 – 7,844 MW | Solar PV farm output (diurnal) |
| Wind Generation | $W_t$ | Megawatts (MW) | 0 – 10,000 MW | Wind turbine farm output (Weibull) |
| Ambient Temperature | $T_t$ | Celsius (°C) | -7.56 – 39.15 °C | Air temperature at weather station |
| Relative Humidity | $H_t$ | Percentage (%) | 15.0 – 99.0 % | Atmospheric humidity |
| Wind Speed | $v_t$ | Meters/second (m/s) | 0.5 – 26.0 m/s | Anemometer speed at hub height |
| Cloud Cover | $C_t$ | Percentage (%) | 0.0 – 100.0 % | Sky cloud fraction (Beer-Lambert) |

### Automatic Column Name Adaptation
The module [`data/data_loader.py`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/data/data_loader.py) uses priority-ordered fuzzy matching so that datasets containing names like `load`, `consumption`, `temp`, `pv`, or `wind_speed` automatically map to canonical names without code modifications.

---

## 5. Data Preprocessing & Cleaning

### 5.1 Duplicate Detection
Exact duplicate timestamps represent sensor re-transmissions:
$$\text{Duplicates removed} = 5 \implies N = 8,765 \to 8,760$$

### 5.2 Time-Aware Missing Value Imputation
Sensors intermittently drop packets (25 missing values injected across columns). For physical continuous time series, mean imputation breaks continuity. We apply **linear time-aware interpolation**:
$$x_t = x_{t_0} + \frac{x_{t_1} - x_{t_0}}{t_1 - t_0} (t - t_0)$$
followed by backward/forward fill for boundary points. Remaining missing values = $0$.

### 5.3 Feature Engineering & Chronological Split
* $H_t = \text{hour} \in \{0, \dots, 23\}$
* $D_{\text{week}} = \text{dayofweek} \in \{0, \dots, 6\}$
* $M_t = \text{month} \in \{1, \dots, 12\}$
* $I_{\text{weekend}} \in \{0, 1\}$
* $\text{Renewable}_t = S_t + W_t$
* $\text{Net Load}_t = D_t - \text{Renewable}_t$
* Cyclic diurnal encodings: $\sin\left(\frac{2\pi H_t}{24}\right)$, $\cos\left(\frac{2\pi H_t}{24}\right)$

**Chronological Split (80/20):**
* Training set: First 7,008 hours (Jan 1 to Oct 20)
* Test set: Last 1,752 hours (Oct 21 to Dec 31)  
*Preserves temporal causality and prevents data leakage.*

---

## 6. Exploratory Data Analysis (EDA)

Summary statistics computed on the continuous variables:

| Variable | Mean ($\mu$) | Std Dev ($\sigma$) | Median | Min | Max | IQR | 5th Pct | 95th Pct |
|---|---|---|---|---|---|---|---|---|
| Electricity Demand (MW) | 24,611.16 | 4,535.11 | 25,395.92 | 12,330.86 | 40,031.69 | 5,899.93 | 16,599.38 | 31,322.15 |
| Temperature (°C) | 15.98 | 9.46 | 15.78 | -7.56 | 39.15 | 15.32 | 1.00 | 30.97 |
| Solar Generation (MW) | 1,147.36 | 1,525.89 | 346.88 | 0.00 | 7,844.57 | 1,884.19 | 0.00 | 4,456.73 |
| Wind Generation (MW) | 1,460.36 | 2,470.59 | 263.16 | 0.00 | 10,000.00 | 1,676.68 | 0.00 | 8,287.64 |
| Renewable Power (MW) | 2,607.72 | 2,754.04 | 1,747.84 | 0.00 | 16,195.75 | 3,294.49 | 0.10 | 9,284.89 |

---

## 7. Probability Theory & Kolmogorov Axioms

### 7.1 Mathematical Foundation
Let sample space $\Omega$ be all recorded hourly states. Kolmogorov's probability axioms state:
1. **Axiom 1:** $P(E) \ge 0 \quad \forall E \subseteq \Omega$
2. **Axiom 2:** $P(\Omega) = 1$
3. **Axiom 3:** If $A \cap B = \emptyset$, then $P(A \cup B) = P(A) + P(B)$

For non-disjoint events, the general addition rule is:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

### 7.2 Empirical Verification on Energy Data
Define events:
* **Event $A$:** "High Demand" ($D_t > Q_{0.75} = 27,714.0\text{ MW}$)
* **Event $B$:** "High Renewable" ($R_t > Q_{0.75} = 3,760.2\text{ MW}$)

Empirical probabilities:
* $P(A) = \frac{2,190}{8,760} = 0.2500$
* $P(B) = \frac{2,190}{8,760} = 0.2500$
* $P(A \cap B) = 0.0626$
* Empirical Union: $P(A \cup B) = 0.4374$
* Theoretical Union: $P(A) + P(B) - P(A \cap B) = 0.2500 + 0.2500 - 0.0626 = 0.4374$
* **Discrepancy:** $|0.4374 - 0.4374| = 0.000000 \times 10^0$ (Exact Axiomatic Verification)
* Conditional probabilities: $P(A|B) = \frac{0.0626}{0.2500} = 0.2502$, $P(B|A) = 0.2502$

---

## 8. Discrete Random Variables & PMF / CDF

Continuous demand is mapped to a 3-state discrete random variable $X$:
$$X = \begin{cases} 0 \quad (\text{Low Demand}) & D_t < 23,061.4\text{ MW } (Q_{0.33}) \\ 1 \quad (\text{Normal Demand}) & 23,061.4\text{ MW} \le D_t \le 26,866.9\text{ MW } (Q_{0.67}) \\ 2 \quad (\text{High Demand}) & D_t > 26,866.9\text{ MW } (Q_{0.67}) \end{cases}$$

### Probability Mass Function (PMF) and CDF
$$p(k) = P(X = k) = \frac{N_k}{N}, \quad F(k) = P(X \le k) = \sum_{j \le k} p(j)$$

| State $k$ | Label | $p(k) = P(X = k)$ | $F(k) = P(X \le k)$ |
|---|---|---|---|
| 0 | Low Demand | 0.3333 | 0.3333 |
| 1 | Normal Demand | 0.3333 | 0.6667 |
| 2 | High Demand | 0.3333 | 1.0000 |

* **Expectation:** $E[X] = \sum_{k=0}^2 k \cdot p(k) = 0(0.3333) + 1(0.3333) + 2(0.3333) = 1.0000$
* **Variance:** $\text{Var}(X) = \sum_{k=0}^2 (k - 1.0)^2 p(k) = (1)(0.3333) + (0)(0.3333) + (1)(0.3333) = 0.6667$
* **Standard Deviation:** $\sigma_X = \sqrt{0.6667} = 0.8165$

---

## 9. Bayes' Rule & Posterior Updating

### 9.1 Mathematical Formula
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
* $P(A)$: **Prior** (base probability of high demand before checking weather)
* $P(B|A)$: **Likelihood** (probability of extreme weather given high demand)
* $P(B)$: **Evidence** (marginal probability of extreme weather)
* $P(A|B)$: **Posterior** (updated risk of high demand given weather observation)

### 9.2 Case 1: High Demand Conditioned on High Temperature
* Condition: $T > 80^{\text{th}}$ percentile ($T > 25.1^\circ\text{C}$)
* Prior $P(\text{High Demand}) = 0.2500$
* Likelihood $P(\text{High Temp} | \text{High Demand}) = 0.3575$
* Evidence $P(\text{High Temp}) = 0.1997$
* Posterior:
  $$P(\text{High Demand} | \text{High Temp}) = \frac{0.3575 \times 0.2500}{0.1997} = 0.4477 \quad (44.77\%)$$
* **Risk Multiplier:** $\frac{0.4477}{0.2500} = 1.79\times$ (Hot temperatures nearly double peak demand risk!)

---

## 10. Independence & Conditional Independence

### 10.1 Statistical Independence
Two events are statistically independent if and only if:
$$P(A \cap B) = P(A) \cdot P(B) \iff P(A|B) = P(A)$$

| Pair | $P(A \cap B)$ | $P(A) \cdot P(B)$ | Absolute Diff | Pearson Corr ($r$) | Statistical Verdict |
|---|---|---|---|---|---|
| Demand & Solar | 0.0680 | 0.0625 | 0.0055 | +0.3404 | Marginally Weak Independent |
| Demand & Wind | 0.0508 | 0.0625 | 0.0117 | -0.1239 | Dependent |
| Demand & Temperature | 0.0938 | 0.0625 | 0.0313 | -0.0484 | Strongly Dependent |

> **Critical Viva Distinction:** Notice that Temperature and Demand have a Pearson correlation near zero ($r = -0.0484$), yet they are **strongly dependent** ($|P(A \cap B) - P(A)P(B)| = 0.0313$). Why? Pearson correlation measures *linear* association. Because the relationship is a symmetric U-curve (heating below 18°C, cooling above 22°C), positive and negative slopes cancel each other out linearly, while probabilistic dependence remains high!

### 10.2 Conditional Independence
Conditioned on solar noon ($\text{Hour} = 14$):
* $P(\text{Dem} \cap \text{Sol} \mid H=14) = 0.1151$
* $P(\text{Dem} \mid H=14) \times P(\text{Sol} \mid H=14) = 0.0622$
* Discrepancy $= 0.0529 \implies$ **Conditionally Dependent**

---

## 11. Continuous Random Variables

Continuous variables have an infinite number of possible outcomes within any interval:
$$P(X = x) = 0 \quad \text{for any specific value } x$$
Probabilities exist solely over continuous intervals:
$$P(a \le X \le b) = \int_a^b f_X(x) \, dx$$
Properties:
1. $f_X(x) \ge 0 \quad \forall x \in \mathbb{R}$
2. $\int_{-\infty}^\infty f_X(x) \, dx = 1$

---

## 12. Mean, Variance, and Quantiles

* Sample Mean: $\bar{x} = \frac{1}{N}\sum_{i=1}^N x_i$
* Sample Variance: $s^2 = \frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2$ (Bessel's correction)
* Quantile function: $Q(p) = \inf\{x \in \mathbb{R} : F_X(x) \ge p\}$

For electricity demand:
* 5th percentile ($Q_{0.05}$) = $16,599.38\text{ MW}$ (Base load margin)
* 50th percentile (Median) = $25,395.92\text{ MW}$
* 95th percentile ($Q_{0.95}$) = $31,322.15\text{ MW}$ (Peaker plant trigger threshold)

---

## 13. Probability Density Functions (Gaussian vs KDE)

1. **Parametric Gaussian PDF:**
   $$f_{\text{norm}}(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
   Fitted parameters: $\mu = 24,611.16\text{ MW}$, $\sigma = 4,535.11\text{ MW}$.
2. **Non-parametric Kernel Density Estimation (KDE):**
   $$\hat{f}_h(x) = \frac{1}{Nh} \sum_{i=1}^N K\left(\frac{x - x_i}{h}\right)$$
   Captures the empirical bimodal peaks caused by the diurnal day/night human activity transition.

---

## 14. Expectation, Covariance, and Correlation

* $E[X] = \bar{X}$
* $\text{Cov}(X, Y) = \frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$
* $\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$

| Variable Pair | Covariance | Correlation ($r$) | Physical Interpretation |
|---|---|---|---|
| Cov(Solar, Demand) | $+2,355,624.74\text{ MW}^2$ | $+0.3404$ | Positive co-movement: solar generation peaks midday alongside commercial activity |
| Cov(Wind, Demand) | $-1,388,500.92\text{ MW}^2$ | $-0.1239$ | Weak negative association: night breezes occur during lower demand |
| Cov(Temp, Demand) | $-2,076.74\text{ MW}\cdot^\circ\text{C}$ | $-0.0484$ | Linear cancellation of U-curve (heating vs cooling demands) |

---

## 15. Polynomial Curve Fitting (PRML Bias-Variance Tradeoff)

To model the thermodynamic relationship between Temperature and Demand:
$$y(x, \mathbf{w}) = \sum_{j=0}^M w_j x^j$$
We fit Degrees $M \in \{1, 2, 3\}$ using Ordinary Least Squares minimizing:
$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N (y(x_n, \mathbf{w}) - t_n)^2$$

### Numerical Comparison:

| Degree | Train RMSE | Train $R^2$ | Test RMSE | Test $R^2$ | Test MAE | Under/Overfitting Analysis |
|---|---|---|---|---|---|---|
| **Degree 1 (Linear)** | 4,735.27 MW | 0.0014 | 3,593.32 MW | 0.0007 | 2,885.46 MW | **High Bias (Underfitting):** Cannot capture the curvature |
| **Degree 2 (Quadratic)** | 3,928.05 MW | 0.3128 | 4,016.19 MW | -0.2483 | 3,488.41 MW | Captures the U-curve on train, but temperature alone lacks diurnal factors |
| **Degree 3 (Cubic)** | 3,856.21 MW | 0.3377 | 3,946.30 MW | -0.2052 | 3,386.35 MW | Slight improvement in inflection |

**Key Finding:** Single-variable polynomial fitting captures the U-curve shape ($R^2 = 0.338$ on train), but demand cannot be predicted accurately by temperature alone without calendar and hour features.

---

## 16. Supervised Learning (Multivariate Regression)

We formulate a multivariate regression model with domain-engineered features:
$$\hat{D}_t = \beta_0 + \beta_1 T_t + \beta_2 T_t^2 + \beta_3 S_t + \beta_4 W_t + \beta_5 H_t + \beta_6 \sin\left(\frac{2\pi H_t}{24}\right) + \beta_7 \cos\left(\frac{2\pi H_t}{24}\right) + \dots$$

### Performance Metrics:
* $\text{MAE} = \frac{1}{N}\sum |y_i - \hat{y}_i|$
* $\text{MSE} = \frac{1}{N}\sum (y_i - \hat{y}_i)^2$
* $\text{RMSE} = \sqrt{\text{MSE}}$
* $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

| Model Architecture | Train MAE | Train RMSE | Train $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
|---|---|---|---|---|---|---|
| Linear Regression (Raw Features) | 2,430.43 MW | 3,110.39 MW | 0.5691 | 3,797.97 MW | 4,506.56 MW | -0.5717 |
| **Feature-Engineered Polynomial Model** | **1,246.30 MW** | **1,466.59 MW** | **0.9042** | **1,332.72 MW** | **1,570.08 MW** | **0.8092** |

> **Conclusion:** The feature-engineered model achieves **$R^2 = 0.8092$** and **$\text{MAE} = 1,332.72\text{ MW}$** on unseen test data, dramatically outperforming raw linear regression. Domain feature engineering (cyclic hour encoding and thermodynamic temperature squaring) solves the forecasting problem within linear algebra constraints.

---

## 17. Unsupervised Learning (K-Means Clustering)

Using Lloyd's algorithm on standardized features $[D_t, S_t, W_t, T_t]$ with $K = 4$:

| Cluster ID | Mean Demand | Mean Solar | Mean Wind | Mean Temp | Sample Count | Share (%) | Grid Operational Regime |
|---|---|---|---|---|---|---|---|
| **0** | 27,063 MW | 692 MW | 687 MW | 8.1 °C | 3,236 | 36.94% | **Winter Heating Peak (Low Temp, High Demand)** |
| **1** | 19,575 MW | 147 MW | 867 MW | 18.0 °C | 2,365 | 27.00% | **Off-Peak Baseload / Night** |
| **2** | 26,932 MW | 3,060 MW | 696 MW | 25.8 °C | 2,261 | 25.81% | **Summer Cooling Peak (High Temp, High Demand)** |
| **3** | 23,197 MW | 604 MW | 7,736 MW | 14.3 °C | 898 | 10.25% | **Windy Transition (High Wind, Moderate Load)** |

The unsupervised clustering discovers distinct grid dispatch regimes without any manual labels!

---

## 18. Uncertainty Quantification (UQ & Prediction Intervals)

### 18.1 Residual Analysis
$$\varepsilon_i = y_i^{\text{actual}} - \hat{y}_i^{\text{predicted}}$$
* **Mean Residual Error:** $\mu_\varepsilon = -244.08\text{ MW}$ (negligible bias relative to 25,000 MW mean)
* **Residual Variance:** $\sigma_\varepsilon^2 = 2,406,963.70\text{ MW}^2$
* **Residual Standard Deviation:** $\sigma_\varepsilon = 1,551.44\text{ MW}$

### 18.2 Empirical Error Quantiles
* $e_{0.05} = -2,760.31\text{ MW}$ (5th percentile)
* $e_{0.25} = -1,475.29\text{ MW}$ (25th percentile)
* $e_{0.50} = -224.66\text{ MW}$ (Median error)
* $e_{0.75} = +1,078.76\text{ MW}$ (75th percentile)
* $e_{0.95} = +2,075.07\text{ MW}$ (95th percentile)

### 18.3 Prediction Interval Construction
1. **90% Empirical Quantile Interval:**
   $$\hat{y}_i + e_{0.05} \le y_i \le \hat{y}_i + e_{0.95}$$
   * **Empirical Coverage Probability (ECP):** **89.95%** (target 90.00% achieved!)
   * **Average Interval Width:** $4,835.38\text{ MW}$
2. **95% Gaussian Parametric Interval:**
   $$\hat{y}_i - 1.96 \sigma_\varepsilon \le y_i \le \hat{y}_i + 1.96 \sigma_\varepsilon$$
   * **Coverage:** **97.32%**
   * **Average Width:** $6,081.64\text{ MW}$

---

## 19. Experimental Results & Visualizations

All 12 required figures were generated and saved to the `figures/` directory:

| # | Figure File | Topic / Description |
|---|---|---|
| 1 | [`figures/01_demand_timeseries.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/01_demand_timeseries.png) | Annual demand profile + 2-week diurnal cycle zoom |
| 2 | [`figures/02_renewable_generation.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/02_renewable_generation.png) | Solar elevation curve vs. Weibull wind volatility |
| 3 | [`figures/03_demand_histogram.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/03_demand_histogram.png) | Empirical histogram with $Q_1$, Median, and $Q_3$ cutoffs |
| 4 | [`figures/04_probability_density.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/04_probability_density.png) | Gaussian PDF vs. non-parametric Kernel Density Estimation |
| 5 | [`figures/05_demand_vs_temperature.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/05_demand_vs_temperature.png) | Thermodynamic U-curve with comfort zone overlay |
| 6 | [`figures/06_polynomial_regression_curve.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/06_polynomial_regression_curve.png) | Degrees 1, 2, and 3 regression curves on scatter data |
| 7 | [`figures/07_discrete_pmf.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/07_discrete_pmf.png) | PMF bar chart of discrete demand states $X \in \{0, 1, 2\}$ |
| 8 | [`figures/08_bayes_probability.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/08_bayes_probability.png) | Prior vs. Posterior probabilities via Bayes' Rule |
| 9 | [`figures/09_clustering_visualization.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/09_clustering_visualization.png) | K-Means 4-regime clusters on Temperature-Demand plane |
| 10 | [`figures/10_actual_vs_predicted_demand.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/10_actual_vs_predicted_demand.png) | Parity scatter plot + 168-hour time series tracking |
| 11 | [`figures/11_prediction_error_distribution.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/11_prediction_error_distribution.png) | Residual error histogram with $e_{0.05}$ and $e_{0.95}$ lines |
| 12 | [`figures/12_prediction_interval_uncertainty.png`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/figures/12_prediction_interval_uncertainty.png) | Forecast curve with shaded 90% prediction envelope |

---

## 20. Conclusion

This project successfully proves that **Unit 1 Machine Learning and Probability Theory principles** are fully capable of solving practical, mission-critical energy grid forecasting problems without requiring opaque deep neural networks.

1. **Foundational Rigor:** Verified Kolmogorov's addition rule with exact zero discrepancy and demonstrated Bayesian risk updating (1.79× risk increase during heatwaves).
2. **Engineering Excellence:** Feature engineering (thermodynamic temperature squaring and trigonometric hour encoding) boosted test $R^2$ from negative to **0.8092**.
3. **Calibrated Uncertainty:** Constructed 90% empirical prediction intervals that achieved an **89.95% empirical coverage**, giving grid operators actionable safety envelopes.
4. **Interpretability:** K-Means clustering naturally identified the four thermodynamic operating regimes of the power system.

---

## 21. Future Scope

1. **Autoregressive Feature Extensions:** Incorporate lag features $D_{t-1}, D_{t-24}$ to capture Markovian time-series dependence (Unit 2).
2. **Regularization Techniques:** Implement Ridge ($L_2$) and Lasso ($L_1$) regression to penalize high polynomial degrees and prevent overfitting.
3. **Bayesian Linear Regression:** Transition from empirical residual quantiles to a full analytical posterior distribution over regression weights:
   $$p(\mathbf{w} \mid \mathcal{D}) = \mathcal{N}(\mathbf{m}_N, \mathbf{S}_N)$$
4. **Dynamic Quantile Regression:** Train pinball-loss quantile regression models to estimate heteroscedastic prediction intervals whose width adapts dynamically to severe storms.
