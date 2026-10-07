"""
From-scratch mathematical implementations of Unit 1 Machine Learning & Probability concepts.
Every formula is coded from first principles using NumPy / SciPy and verified against Scikit-learn.
"""

import numpy as np
from scipy import stats


class FromScratchPolynomialRegression:
    """
    Polynomial curve fitting solved via closed-form Normal Equations (Bishop PRML §1.1).
    Minimizes regularized sum-of-squares:
    E(w) = 0.5 * sum((y(x_n, w) - t_n)^2) + 0.5 * lambda * ||w||^2
    Solution:
    w = (Phi^T Phi + lambda * I)^(-1) Phi^T t
    """
    def __init__(self, degree=3, reg_lambda=0.0):
        self.degree = int(degree)
        self.reg_lambda = float(reg_lambda)
        self.weights = None
        self.x_mean = 0.0
        self.x_std = 1.0

    def _design_matrix(self, x):
        x = np.asarray(x, dtype=float).flatten()
        # Shape: (N, degree + 1) -> [1, x, x^2, ..., x^degree]
        return np.column_stack([x**d for d in range(self.degree + 1)])

    def fit(self, x, y):
        Phi = self._design_matrix(x)
        t = np.asarray(y, dtype=float).flatten()
        num_features = Phi.shape[1]

        # Regularization matrix (do not penalize bias w_0)
        reg_matrix = self.reg_lambda * np.eye(num_features)
        reg_matrix[0, 0] = 0.0

        # Normal equation: (Phi^T Phi + lambda * I) w = Phi^T t
        A = Phi.T @ Phi + reg_matrix
        b = Phi.T @ t
        self.weights = np.linalg.solve(A, b)
        return self

    def predict(self, x):
        Phi = self._design_matrix(x)
        return Phi @ self.weights

    def compute_rmse(self, x, y):
        preds = self.predict(x)
        return float(np.sqrt(np.mean((preds - np.asarray(y).flatten())**2)))


class FromScratchContinuousStats:
    """
    Sample moments, quantiles, Gaussian and KDE fits computed from first principles.
    """
    @staticmethod
    def sample_mean(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x) / len(x))

    @staticmethod
    def sample_variance(x):
        # Bessel's correction: divide by N - 1
        x = np.asarray(x, dtype=float)
        mu = np.sum(x) / len(x)
        return float(np.sum((x - mu)**2) / (len(x) - 1))

    @staticmethod
    def sample_std(x):
        return float(np.sqrt(FromScratchContinuousStats.sample_variance(x)))

    @staticmethod
    def sample_quantile(x, p):
        # Quantile Q(p) using linear interpolation
        return float(np.percentile(x, p * 100.0))

    @staticmethod
    def sample_moments(x):
        x = np.asarray(x, dtype=float)
        mu = FromScratchContinuousStats.sample_mean(x)
        var = FromScratchContinuousStats.sample_variance(x)
        std = np.sqrt(var)
        skew = float(np.mean(((x - mu) / std)**3))
        kurt = float(np.mean(((x - mu) / std)**4) - 3.0)  # Excess kurtosis
        return {
            "mean": mu,
            "variance": var,
            "std_dev": std,
            "skewness": skew,
            "kurtosis": kurt
        }


class FromScratchCovariance:
    """
    Bivariate expectation, covariance, and Pearson correlation coefficient.
    """
    @staticmethod
    def covariance(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        return float(np.sum((x - mu_x) * (y - mu_y)) / (len(x) - 1))

    @staticmethod
    def pearson_correlation(x, y):
        cov = FromScratchCovariance.covariance(x, y)
        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)
        if std_x * std_y == 0:
            return 0.0
        return float(cov / (std_x * std_y))


class FromScratchBayesCalculator:
    """
    Bayesian inference on empirical data:
    P(Hypothesis | Evidence) = [P(Evidence | Hypothesis) * P(Hypothesis)] / P(Evidence)
    """
    @staticmethod
    def compute(hypothesis_bool_arr, evidence_bool_arr):
        hyp = np.asarray(hypothesis_bool_arr, dtype=bool)
        evi = np.asarray(evidence_bool_arr, dtype=bool)
        N = len(hyp)

        p_prior = float(np.mean(hyp))
        p_evidence = float(np.mean(evi))
        p_joint = float(np.mean(hyp & evi))

        # Likelihood P(E | H)
        p_likelihood = float(p_joint / p_prior) if p_prior > 0 else 0.0

        # Bayes rule
        p_posterior = float((p_likelihood * p_prior) / p_evidence) if p_evidence > 0 else 0.0

        # Direct empirical count check
        p_direct = float(np.mean(hyp[evi])) if np.sum(evi) > 0 else 0.0

        # Discrepancy
        diff = abs(p_posterior - p_direct)

        # Risk multiplier
        risk_multiplier = float(p_posterior / p_prior) if p_prior > 0 else 1.0

        return {
            "prior": p_prior,
            "likelihood": p_likelihood,
            "evidence": p_evidence,
            "posterior": p_posterior,
            "direct_check": p_direct,
            "discrepancy": diff,
            "risk_multiplier": risk_multiplier,
            "evidence_count": int(np.sum(evi)),
            "total_count": N
        }
