"""
kalman_fast.py — High-performance 1D scalar Kalman Filter (Numba JIT).

Notation: z_k, x̂_k, P_k, Q, R, K_k, F=1, H=1.
"""

import numpy as np
from numba import njit

from packages.indicators.rolling_mean_stddev import rolling_mean_stddev


@njit
def kalman_1d_step(
    measurement: float,
    prev_estimate: float,
    prev_error_cov: float,
    process_variance: float,
    measurement_variance: float,
) -> tuple:
    """Single stateless predict-update-correct step for the 1D scalar filter.

    Returns (current_estimate, current_error_cov) as (float, float).
    """
    # Predict: x̂_k^- = x̂_{k-1}; P_k^- = P_{k-1} + Q
    prior_estimate = prev_estimate
    prior_error_cov = prev_error_cov + process_variance

    # Gain: K_k = P_k^- / (P_k^- + R)
    kalman_gain = prior_error_cov / (prior_error_cov + measurement_variance)

    # Correct: x̂_k = x̂_k^- + K_k(z_k - x̂_k^-); P_k = (1 - K_k) P_k^-
    current_estimate = prior_estimate + kalman_gain * (measurement - prior_estimate)
    current_error_cov = (1.0 - kalman_gain) * prior_error_cov

    return current_estimate, current_error_cov


@njit
def kalman_1d_batch(
    measurements: np.ndarray,
    initial_estimate: float,
    initial_error_cov: float,
    process_variance: float,
    measurement_variance: float,
) -> tuple:
    """Batch 1D Kalman filter over a full measurement sequence.

    Returns (estimates, error_covariances) — float64 arrays of shape (N,).
    """
    n = len(measurements)
    estimates = np.empty(n, dtype=np.float64)
    error_covariances = np.empty(n, dtype=np.float64)

    est = initial_estimate
    cov = initial_error_cov

    for k in range(n):
        est, cov = kalman_1d_step(
            measurements[k], est, cov, process_variance, measurement_variance
        )
        estimates[k] = est
        error_covariances[k] = cov

    return estimates, error_covariances


@njit
def kalman_1d_batch_adaptive(
    measurements: np.ndarray,
    process_variance: float,
    window: int,
) -> tuple:
    """Batch 1D Kalman filter with per-index adaptive measurement variance.

    measurement_variance[k] = rolling_mean_stddev(measurements, window)[k, 1] ** 2.
    process_variance is fixed across all indices. initial_estimate = measurements[0];
    initial_error_cov = measurement_variance[0].

    Returns (estimates, error_covariances) — float64 arrays of shape (N,).
    """
    n = len(measurements)
    mean_std = rolling_mean_stddev(measurements, window)
    variance = mean_std[:, 1] ** 2

    estimates = np.empty(n, dtype=np.float64)
    error_covariances = np.empty(n, dtype=np.float64)

    est = measurements[0]
    cov = variance[0]

    for k in range(n):
        est, cov = kalman_1d_step(
            measurements[k], est, cov, process_variance, variance[k]
        )
        estimates[k] = est
        error_covariances[k] = cov

    return estimates, error_covariances
