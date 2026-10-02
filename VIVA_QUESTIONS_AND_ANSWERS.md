# Unit 1 Machine Learning & Probability Theory — Comprehensive Viva Voce Guide
## 25+ Essential Viva Questions & Answers for B.Tech CSE Students

This guide is specifically designed for defending the project:  
**"Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification"**

---

### Section 1: Machine Learning Foundations

#### Q1. What is Machine Learning, and how does it differ from traditional programming?
**Answer:**  
In **traditional programming**, a human developer writes explicit deterministic rules, which the computer executes on input data to generate answers:
$$\text{Data} + \text{Rules} \longrightarrow \text{Answers}$$
In **Machine Learning**, the computer takes historical input data and observed answers (labels) to automatically discover the underlying mathematical mapping or rules:
$$\text{Data} + \text{Answers} \longrightarrow \text{Rules (Model)}$$
*Energy Grid Context:* Instead of hardcoding static rules like "if temperature is 35°C, dispatch 30,000 MW", an ML model learns the continuous non-linear relationship between temperature, time of day, wind speed, solar radiation, and electricity demand from 8,760 hours of historical measurements.

#### Q2. State Tom Mitchell's formal definition of Machine Learning. Identify $T$, $P$, and $E$ in this project.
**Answer:**  
*"A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."*
In our project:
* **Task ($T$):** Forecasting future hourly electricity demand (MW) and predicting uncertainty intervals.
* **Experience ($E$):** 8,760 hours of historical power demand, solar/wind generation, and weather records.
* **Performance Measure ($P$):** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ score evaluated on unseen test data.

#### Q3. What is the fundamental difference between Supervised and Unsupervised Learning?
**Answer:**  
* **Supervised Learning:** The training data contains both input features $\mathbf{x}$ and ground-truth target labels $y$ ($\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$). The goal is to learn a mapping function $f(\mathbf{x}) \approx y$. In our project, multivariate regression uses weather and calendar features to predict electricity demand (MW).
* **Unsupervised Learning:** The training data consists only of input features $\mathbf{x}$ with no target labels ($\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$). The goal is to discover latent patterns, clustering structures, or density distributions $p(\mathbf{x})$. In our project, K-Means clustering groups hours into 4 distinct grid operational regimes (e.g., summer peak, winter heating).

---

### Section 2: Polynomial Curve Fitting & Supervised Regression

#### Q4. Why did you use Polynomial Curve Fitting instead of simple linear regression for Temperature vs. Demand?
**Answer:**  
Electricity demand exhibits a **thermodynamic U-curve** with temperature:
* At cold temperatures ($T < 15^\circ\text{C}$), demand rises due to space heating and heat pumps.
* In the thermal comfort zone ($18^\circ\text{C} - 22^\circ\text{C}$), demand reaches an annual minimum.
* At hot temperatures ($T > 24^\circ\text{C}$), demand spikes steeply due to air conditioning and chillers.

A linear regression line ($\hat{y} = w_0 + w_1 x$) has constant derivative $\frac{dy}{dx} = w_1$ and cannot change direction; it resulted in $R^2 \approx 0.001$. A degree 2 or 3 polynomial ($\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3$) has variable curvature ($\frac{d^2 y}{dx^2} \ne 0$) and successfully models the parabolic heating/cooling inflection.

#### Q5. What is the Bias-Variance Tradeoff in Polynomial Regression? What happens if degree $M$ is too high?
**Answer:**  
* **High Bias (Underfitting):** When degree $M = 1$ (linear), the model is overly simplistic and cannot capture the underlying U-shape. Training and test errors are both large.
* **High Variance (Overfitting):** If degree $M$ is made excessively large (e.g., $M = 15$), the polynomial passes through random noise and outliers (Runge's phenomenon). It achieves near-zero training error but oscillates wildly, resulting in catastrophic generalization errors on test data.
* **Optimal Model:** Degree 2–3 captures the physical curvature with low variance.

#### Q6. Define MAE, MSE, RMSE, and $R^2$. Why do we report RMSE alongside MAE?
**Answer:**  
1. $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$: Average absolute magnitude of errors; robust to outliers; expressed in target units (MW).
2. $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$: Penalizes large errors heavily; units are squared ($\text{MW}^2$).
3. $\text{RMSE} = \sqrt{\text{MSE}}$: In target units (MW); penalizes large forecasting mistakes much more than MAE.
   * *Viva Insight:* If $\text{RMSE} \gg \text{MAE}$, it indicates that the model occasionally makes very large forecasting errors (spikes), which is critical for grid operators to know.
4. $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$: Proportion of total variance in demand explained by the model features.

#### Q7. Why did raw linear regression yield a negative test $R^2$, while feature-engineered polynomial regression reached $R^2 = 0.809$?
**Answer:**  
Raw linear regression assumed straight-line linear dependence across seasons. Because the test set was chronologically partitioned (winter months Oct–Dec), the raw model failed completely when winter heating loads set in.  
By engineering explicit domain features—specifically $\text{temp}^2$ to capture the thermodynamic U-curve, and $\sin(2\pi H/24), \cos(2\pi H/24)$ to capture the 24-hour cyclic diurnal load ramp—the model learned the physical invariants of the power grid, achieving $R^2 = 0.8092$ and reducing RMSE from 4,506 MW to 1,570 MW.

---

### Section 3: Probability Theory & Random Variables

#### Q8. State the three Kolmogorov Axioms of Probability. How did you verify the Addition Rule on the dataset?
**Answer:**  
For a sample space $\Omega$ and event $E$:
1. **Axiom 1 (Non-negativity):** $P(E) \ge 0$
2. **Axiom 2 (Normalization):** $P(\Omega) = 1$
3. **Axiom 3 (Countable Additivity):** For mutually disjoint events $A \cap B = \emptyset$, $P(A \cup B) = P(A) + P(B)$.

For non-disjoint events, the general addition rule is:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
*Dataset Verification:*
* Defined Event $A$ = High Demand ($D > Q_{0.75} = 27,714\text{ MW}$), $P(A) = 0.2500$
* Defined Event $B$ = High Renewable ($R > Q_{0.75} = 3,760\text{ MW}$), $P(B) = 0.2500$
* Joint probability $P(A \cap B) = 0.0626$
* Empirical union $P(A \cup B) = 0.4374$
* Theoretical calculation: $0.2500 + 0.2500 - 0.0626 = 0.4374$
* **Discrepancy:** $|0.4374 - 0.4374| = 0.000000$, exactly verifying the axiom.

#### Q9. What is a Discrete Random Variable? Explain how you constructed one from continuous electricity demand.
**Answer:**  
A **Discrete Random Variable** is a variable that takes values from a countable set of distinct states.
Continuous electricity demand was partitioned using empirical quantiles into 3 operational states:
$$X = \begin{cases} 0 & \text{Low Demand } (D < Q_{0.33} = 23,061\text{ MW}) \\ 1 & \text{Normal Demand } (23,061\text{ MW} \le D \le 26,867\text{ MW}) \\ 2 & \text{High Demand } (D > Q_{0.67} = 26,867\text{ MW}) \end{cases}$$
* Probability Mass Function (PMF): $P(X = 0) = 0.3333$, $P(X = 1) = 0.3333$, $P(X = 2) = 0.3333$
* Cumulative Distribution Function (CDF): $F(0) = 0.3333$, $F(1) = 0.6667$, $F(2) = 1.0000$
* Expectation: $E[X] = \sum k \cdot p(k) = 1.0000$
* Variance: $\text{Var}(X) = \sum (k - 1)^2 p(k) = 0.6667$

#### Q10. State Bayes' Rule. Explain its terms and how you applied it to energy risk forecasting.
**Answer:**  
Bayes' Theorem updates the probability of a hypothesis $A$ given observed evidence $B$:
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
* $P(A)$: **Prior Probability** — base probability of high demand ($P(A) = 0.2500$).
* $P(B|A)$: **Likelihood** — probability that temperature is extremely high given demand is high ($P(B|A) = 0.3575$).
* $P(B)$: **Evidence** — unconditional probability of extreme temperature across the year ($P(B) = 0.1997$).
* $P(A|B)$: **Posterior Probability** — updated probability of high demand upon observing high temperature:
  $$P(\text{High Demand} \mid \text{High Temp}) = \frac{0.3575 \times 0.2500}{0.1997} = 0.4477 \quad (44.77\%)$$
*Energy Grid Significance:* Observing extreme ambient heat increases the risk of high demand by **1.79×** (from 25% to 44.77%), alerting the transmission operator to prepare spinning reserves.

#### Q11. Can two variables have zero correlation but still be statistically dependent? Prove this from your project.
**Answer:**  
**Yes, absolutely.** This is a critical mathematical distinction.
* **Statistical Independence** requires that the joint distribution factors into the product of marginals:
  $$P(A \cap B) = P(A) \cdot P(B) \quad \forall A, B$$
* **Pearson Correlation** ($r$) measures only **linear** association.
* *Project Proof:* Temperature and Electricity Demand have a Pearson correlation of $r = -0.0484$ (near zero). A naive analyst might claim they are independent. However, testing event independence yields:
  $$P(A \cap B) = 0.0938 \quad \text{vs.} \quad P(A) \cdot P(B) = 0.0625 \implies \Delta = 0.0313 \ne 0$$
  They are strongly dependent because of the U-shaped physical relationship: cold temps increase demand (negative slope) and hot temps increase demand (positive slope), canceling each other out linearly while maintaining strong non-linear dependence!

#### Q12. For a Continuous Random Variable $X$, why is $P(X = x) = 0$ for any specific value $x$?
**Answer:**  
For a continuous random variable, the sample space contains an uncountably infinite number of points. Probability is defined as the integral of the Probability Density Function (PDF) $f(x)$ over an interval:
$$P(a \le X \le b) = \int_a^b f(x) \, dx$$
For a single exact point $x$, the interval width is zero ($[x, x]$):
$$P(X = x) = \int_x^x f(t) \, dt = 0$$
Therefore, in continuous energy data, it is meaningless to ask "what is the probability demand will be exactly 25,000.0000 MW?". We must ask "what is the probability demand will lie between 24,500 MW and 25,500 MW?".

#### Q13. Distinguish between Expectation, Covariance, and Correlation. Include their units.
**Answer:**  
* **Expectation ($E[X]$):** The probability-weighted average (mean) value. Units match $X$ ($\text{MW}$).
* **Covariance ($\text{Cov}(X, Y)$):** Measures the direction of joint linear variability:
  $$\text{Cov}(X, Y) = \frac{1}{N-1}\sum (x_i - \bar{x})(y_i - \bar{y})$$
  Units are the product of the units of $X$ and $Y$ ($\text{MW} \cdot ^\circ\text{C}$ or $\text{MW}^2$). Because covariance is scale-dependent, its magnitude cannot be easily compared across different physical domains.
* **Correlation ($\text{Corr}(X, Y)$):** Dimensionless, normalized covariance bounded in $[-1, +1]$:
  $$\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

---

### Section 4: Uncertainty Quantification (UQ) & Grid Operations

#### Q14. What is a Prediction Residual, and what properties should it ideally possess?
**Answer:**  
The prediction residual (forecast error) is defined as:
$$\varepsilon_i = y_i^{\text{actual}} - \hat{y}_i^{\text{predicted}}$$
In an optimal, unbiased regression model:
1. **Mean error $\mu_\varepsilon \approx 0$:** Unbiased forecasts (in our project, $\mu_\varepsilon = -244\text{ MW}$, small relative to $25,000\text{ MW}$).
2. **Homoscedasticity:** Constant error variance across all hours.
3. **No autocorrelation:** Residuals should resemble white noise with no predictable pattern left unmodeled.

#### Q15. How did you construct the 90% Empirical Prediction Interval?
**Answer:**  
Rather than blindly assuming errors follow a perfect Gaussian distribution, we compute the empirical quantiles of test residuals $\varepsilon$:
* $e_{0.05} = -2,760.31\text{ MW}$ (5th percentile of errors)
* $e_{0.95} = +2,075.07\text{ MW}$ (95th percentile of errors)

For any future point forecast $\hat{y}_i$, the 90% empirical prediction interval is:
$$\left[\hat{y}_i + e_{0.05}, \; \hat{y}_i + e_{0.95}\right]$$
*Verification:* When tested across all 1,752 test hours, exactly **89.95%** of actual demand values fell inside this interval, confirming that our uncertainty bounds are accurately calibrated.

#### Q16. Does a 90% prediction interval guarantee that actual demand will never fall outside it?
**Answer:**  
**No.** A prediction interval is a probabilistic statement, not a deterministic guarantee. An empirical 90% interval explicitly implies that approximately **10% of future hours** will fall outside the bounds (5% above the upper bound and 5% below the lower bound). Grid operators budget "spinning reserve capacity" specifically to cover these 5% tail risk events.

#### Q17. How did K-Means clustering assist grid operators?
**Answer:**  
K-Means ($K=4$) partitioned multivariate operational vectors $[D_t, S_t, W_t, T_t]$ into four physically meaningful grid dispatch regimes:
1. **Winter Heating Peak (36.9%):** Low temperature (8.1°C), high demand (27,063 MW), low solar. Demands maximum thermal/hydro generation.
2. **Off-Peak Baseload (27.0%):** Night hours, lowest demand (19,575 MW). Nuclear/baseload plants run efficiently.
3. **Summer Cooling Peak (25.8%):** High temperature (25.8°C), high demand (26,932 MW), peak solar generation (3,060 MW). Solar generation partially offsets AC peak.
4. **Windy Transition (10.3%):** Moderate demand (23,197 MW), massive wind power (7,736 MW). Allows conventional fossil generators to throttle down.

---

### Section 5: Practical & Coding Viva Questions

#### Q18. How does your code adapt automatically if a user provides a dataset with different column names?
**Answer:**  
In [`data/data_loader.py`](file:///c:/Users/Hastik%20Pangi/Downloads/ML%201/data/data_loader.py), we implemented an intelligent column adaptation function `auto_adapt_columns()`. It maintains a priority-ordered dictionary of known aliases (e.g., `load`, `consumption`, `actual_demand` for electricity demand; `temp`, `ambient_temp` for temperature). It performs exact matching first, followed by substring keyword matching, automatically renaming any incoming CSV columns to canonical names.

#### Q19. Why did you use chronological train/test splitting instead of `train_test_split(shuffle=True)`?
**Answer:**  
Electricity demand is a continuous time series. Randomly shuffling rows causes **lookahead bias (data leakage)**—the model would train on hour $t+1$ to predict hour $t$, artificially inflating accuracy scores. Chronological splitting (training on Jan–Oct, testing on Oct–Dec) simulates real-world deployment where the model only ever has access to past records.

#### Q20. If an examiner asks: "Can you run your interactive dashboard right now?", how do you demonstrate it?
**Answer:**  
We have two working dashboard implementations:
1. **Terminal Dashboard:** Run `python dashboard.py` — prompts for temperature, solar, wind, hour, or lets the user choose preset scenarios (Summer Peak, Winter Night) and displays predictions, bounds, and Bayes probabilities.
2. **Web Dashboard:** Run `python web_dashboard.py` and open `http://127.0.0.1:5000` in any browser — provides interactive real-time sliders, visual gauges, and uncertainty intervals.
