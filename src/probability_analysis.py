"""
Probability Theory and Random Variables Module
Covers Unit 1 topics:
- Fundamental Rules of Probability (Kolmogorov Axioms, Joint, Union, Conditional)
- Discrete Random Variables (PMF, CDF, Expectation, Variance)
- Bayes' Rule (Prior, Likelihood, Evidence, Posterior)
- Statistical Independence and Conditional Independence
- Expectation and Covariance vs Correlation
"""

import numpy as np
import pandas as pd

def compute_probability_events(df):
    """
    Defines real-world grid events and computes fundamental probabilities:
    Event A: Electricity Demand is High (> 75th percentile)
    Event B: Renewable Generation is High (> 75th percentile)
    
    Verifies Kolmogorov's Axioms and the Addition Rule:
    P(A U B) = P(A) + P(B) - P(A ∩ B)
    """
    N = len(df)
    q75_demand = df["electricity_demand_mw"].quantile(0.75)
    q75_renewable = df["renewable_generation_mw"].quantile(0.75)
    
    event_A = df["electricity_demand_mw"] > q75_demand
    event_B = df["renewable_generation_mw"] > q75_renewable
    
    # Counts
    n_A = int(event_A.sum())
    n_B = int(event_B.sum())
    n_A_and_B = int((event_A & event_B).sum())
    n_A_or_B = int((event_A | event_B).sum())
    
    # Probabilities
    p_A = n_A / N
    p_B = n_B / N
    p_A_and_B = n_A_and_B / N
    p_A_or_B = n_A_or_B / N
    
    # Conditional Probabilities
    p_A_given_B = p_A_and_B / p_B if p_B > 0 else 0.0
    p_B_given_A = p_A_and_B / p_A if p_A > 0 else 0.0
    
    # Addition Rule Verification: P(A U B) == P(A) + P(B) - P(A ∩ B)
    theoretical_union = p_A + p_B - p_A_and_B
    union_check_error = abs(p_A_or_B - theoretical_union)
    
    return {
        "threshold_demand_q75": q75_demand,
        "threshold_renewable_q75": q75_renewable,
        "P(A) [High Demand]": p_A,
        "P(B) [High Renewable]": p_B,
        "P(A ∩ B) [Joint High Demand & High Renewable]": p_A_and_B,
        "P(A U B) [Empirical Union]": p_A_or_B,
        "P(A) + P(B) - P(A ∩ B) [Theoretical Union]": theoretical_union,
        "Union Rule Discrepancy": union_check_error,
        "P(A|B) [High Demand given High Renewable]": p_A_given_B,
        "P(B|A) [High Renewable given High Demand]": p_B_given_A
    }

def analyze_discrete_random_variable(df):
    """
    Transforms continuous electricity demand into a 3-state Discrete Random Variable X:
    0 = Low Demand (Demand < 33rd percentile)
    1 = Normal Demand (33rd percentile <= Demand <= 66th percentile)
    2 = High Demand (Demand > 66th percentile)
    
    Calculates PMF, CDF, Expected Value E[X], and Variance Var(X).
    """
    q33 = df["electricity_demand_mw"].quantile(0.3333)
    q66 = df["electricity_demand_mw"].quantile(0.6667)
    
    # Assign states
    states = np.zeros(len(df), dtype=int)
    states[(df["electricity_demand_mw"] >= q33) & (df["electricity_demand_mw"] <= q66)] = 1
    states[df["electricity_demand_mw"] > q66] = 2
    
    # Probability Mass Function (PMF)
    state_counts = pd.Series(states).value_counts().sort_index()
    pmf = state_counts / len(df)
    
    # Cumulative Distribution Function (CDF)
    cdf = pmf.cumsum()
    
    # Expectation: E[X] = sum(k * p(k))
    expected_value = sum(k * p for k, p in pmf.items())
    
    # Variance: Var(X) = sum((k - E[X])^2 * p(k))
    variance = sum(((k - expected_value) ** 2) * p for k, p in pmf.items())
    std_dev = np.sqrt(variance)
    
    state_names = {0: "Low Demand", 1: "Normal Demand", 2: "High Demand"}
    pmf_dict = {f"P(X={k}) [{state_names[k]}]": float(pmf[k]) for k in range(3)}
    cdf_dict = {f"F({k}) = P(X<={k})": float(cdf[k]) for k in range(3)}
    
    return {
        "states": states,
        "q33_cutoff": q33,
        "q66_cutoff": q66,
        "pmf": pmf_dict,
        "cdf": cdf_dict,
        "pmf_series": pmf,
        "E[X] (Discrete Expectation)": expected_value,
        "Var(X) (Discrete Variance)": variance,
        "Std(X) (Discrete Std Dev)": std_dev
    }

def compute_bayes_rule(df):
    """
    Applies Bayes' Theorem:
    P(A|B) = [P(B|A) * P(A)] / P(B)
    
    Case 1: Probability of High Demand (A) given Extreme High Temperature (B)
    - Prior P(A): Base probability of high demand
    - Likelihood P(B|A): Probability of high temperature when demand is already high
    - Evidence P(B): Overall probability of high temperature
    - Posterior P(A|B): Updated probability of high demand after observing high temperature
    
    Case 2: Probability of High Demand (A) given Low Renewable Generation (B_low)
    """
    N = len(df)
    q75_demand = df["electricity_demand_mw"].quantile(0.75)
    q80_temp = df["temperature_celsius"].quantile(0.80)
    q25_renewable = df["renewable_generation_mw"].quantile(0.25)
    
    is_high_demand = df["electricity_demand_mw"] > q75_demand
    is_high_temp = df["temperature_celsius"] > q80_temp
    is_low_renewable = df["renewable_generation_mw"] < q25_renewable
    
    # --- Case 1: High Demand given High Temperature ---
    p_prior = is_high_demand.mean() # P(A)
    p_evidence_temp = is_high_temp.mean() # P(B)
    p_joint_temp = (is_high_temp & is_high_demand).mean() # P(A ∩ B)
    p_likelihood_temp = p_joint_temp / p_prior if p_prior > 0 else 0.0 # P(B|A)
    
    # Bayes formula: P(A|B) = [P(B|A) * P(A)] / P(B)
    p_posterior_temp_bayes = (p_likelihood_temp * p_prior) / p_evidence_temp if p_evidence_temp > 0 else 0.0
    p_posterior_temp_direct = (is_high_demand[is_high_temp]).mean() # Direct empirical conditional
    
    # --- Case 2: High Demand given Low Renewable Generation ---
    p_evidence_low_ren = is_low_renewable.mean()
    p_joint_ren = (is_low_renewable & is_high_demand).mean()
    p_likelihood_ren = p_joint_ren / p_prior if p_prior > 0 else 0.0
    p_posterior_ren_bayes = (p_likelihood_ren * p_prior) / p_evidence_low_ren if p_evidence_low_ren > 0 else 0.0
    p_posterior_ren_direct = (is_high_demand[is_low_renewable]).mean()
    
    return {
        "Case 1 (High Temp)": {
            "Prior P(High Demand)": p_prior,
            "Likelihood P(High Temp | High Demand)": p_likelihood_temp,
            "Evidence P(High Temp)": p_evidence_temp,
            "Posterior P(High Demand | High Temp) [Bayes]": p_posterior_temp_bayes,
            "Posterior P(High Demand | High Temp) [Direct]": p_posterior_temp_direct,
            "Risk Ratio (Posterior / Prior)": p_posterior_temp_bayes / p_prior if p_prior > 0 else 0.0
        },
        "Case 2 (Low Renewable)": {
            "Prior P(High Demand)": p_prior,
            "Likelihood P(Low Renewable | High Demand)": p_likelihood_ren,
            "Evidence P(Low Renewable)": p_evidence_low_ren,
            "Posterior P(High Demand | Low Renewable) [Bayes]": p_posterior_ren_bayes,
            "Posterior P(High Demand | Low Renewable) [Direct]": p_posterior_ren_direct,
            "Risk Ratio (Posterior / Prior)": p_posterior_ren_bayes / p_prior if p_prior > 0 else 0.0
        }
    }

def analyze_independence(df):
    """
    Tests statistical independence between grid events and variables.
    
    Definition: A and B are statistically independent iff:
    P(A ∩ B) = P(A) * P(B)  <=>  P(A|B) = P(A)
    
    Tests:
    1. Demand High & Solar High
    2. Demand High & Wind High
    3. Demand High & High Temperature
    
    Also tests Conditional Independence:
    P(A ∩ B | C) vs P(A | C) * P(B | C) conditioned on peak daytime (Hour = 14)
    """
    N = len(df)
    q75_demand = df["electricity_demand_mw"].quantile(0.75)
    q75_solar = df["solar_generation_mw"].quantile(0.75)
    q75_wind = df["wind_generation_mw"].quantile(0.75)
    q75_temp = df["temperature_celsius"].quantile(0.75)
    
    A_demand = df["electricity_demand_mw"] > q75_demand
    B_solar = df["solar_generation_mw"] > q75_solar
    B_wind = df["wind_generation_mw"] > q75_wind
    B_temp = df["temperature_celsius"] > q75_temp
    
    def test_pair(name, ev1, ev2, var1, var2):
        p1 = ev1.mean()
        p2 = ev2.mean()
        p_joint = (ev1 & ev2).mean()
        p_prod = p1 * p2
        diff = abs(p_joint - p_prod)
        corr = np.corrcoef(var1, var2)[0, 1]
        
        # Heuristic check for practical independence
        is_independent = diff < 0.01
        
        return {
            "Event Pair": name,
            "P(A)": p1,
            "P(B)": p2,
            "P(A ∩ B)": p_joint,
            "P(A) * P(B)": p_prod,
            "|P(A ∩ B) - P(A)P(B)|": diff,
            "Pearson Correlation (r)": corr,
            "Statistically Independent?": is_independent
        }
        
    pair_solar = test_pair("Demand & Solar", A_demand, B_solar, df["electricity_demand_mw"], df["solar_generation_mw"])
    pair_wind = test_pair("Demand & Wind", A_demand, B_wind, df["electricity_demand_mw"], df["wind_generation_mw"])
    pair_temp = test_pair("Demand & Temperature", A_demand, B_temp, df["electricity_demand_mw"], df["temperature_celsius"])
    
    # Conditional Independence Test: Demand & Solar conditioned on Hour == 14 (Daytime peak)
    subset_h14 = df[df["hour"] == 14]
    q75_dem_14 = subset_h14["electricity_demand_mw"].quantile(0.75)
    q75_sol_14 = subset_h14["solar_generation_mw"].quantile(0.75)
    
    c_dem = subset_h14["electricity_demand_mw"] > q75_dem_14
    c_sol = subset_h14["solar_generation_mw"] > q75_sol_14
    
    p_dem_given_C = c_dem.mean()
    p_sol_given_C = c_sol.mean()
    p_joint_given_C = (c_dem & c_sol).mean()
    p_prod_given_C = p_dem_given_C * p_sol_given_C
    diff_cond = abs(p_joint_given_C - p_prod_given_C)
    
    cond_test = {
        "Condition": "Hour == 14 (Solar Noon)",
        "P(Demand High | Hour=14)": p_dem_given_C,
        "P(Solar High | Hour=14)": p_sol_given_C,
        "P(Demand ∩ Solar | Hour=14)": p_joint_given_C,
        "P(Demand|H=14) * P(Solar|H=14)": p_prod_given_C,
        "Discrepancy": diff_cond,
        "Conditionally Independent?": diff_cond < 0.02
    }
    
    return {
        "pairwise_tests": [pair_solar, pair_wind, pair_temp],
        "conditional_test": cond_test
    }

def compute_expectation_and_covariance(df):
    """
    Computes Expectation E[X], Covariance Cov(X, Y), and Pearson Correlation:
    E[X] = (1/N) * sum(X_i)
    Cov(X, Y) = (1 / (N - 1)) * sum((X_i - E[X]) * (Y_i - E[Y]))
    Corr(X, Y) = Cov(X, Y) / (std_X * std_Y)
    """
    vars_to_analyze = [
        "electricity_demand_mw",
        "temperature_celsius",
        "solar_generation_mw",
        "wind_generation_mw",
        "renewable_generation_mw"
    ]
    
    # Expectations
    expectations = {col: float(df[col].mean()) for col in vars_to_analyze}
    
    # Covariances and Correlations with Electricity Demand
    target = df["electricity_demand_mw"]
    covariances = {}
    correlations = {}
    
    for col in vars_to_analyze:
        cov = float(np.cov(df[col], target)[0, 1])
        corr = float(np.corrcoef(df[col], target)[0, 1])
        covariances[f"Cov({col}, Demand)"] = cov
        correlations[f"Corr({col}, Demand)"] = corr
        
    return {
        "expectations": expectations,
        "covariances": covariances,
        "correlations": correlations
    }
