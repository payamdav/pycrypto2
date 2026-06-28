"""
kalman_2d.py — 2D constant-velocity Kalman Filter (Numba JIT).

State x̂ = [value, speed]ᵀ, shape (2,1).
F = [[1, dt],[0, 1]], H = [[1, 0]].
Notation: z_k, x̂_k, P_k, Q, R, K_k, S.
"""

import numpy as np
from numba import njit


@njit
def kalman_2d_step(
    measurement: float,
    prev_state: np.ndarray,
    prev_covariance: np.ndarray,
    process_noise: np.ndarray,
    measurement_variance: float,
    dt: float,
) -> tuple:
    """Single stateless predict-update-correct step for the 2D filter.

    Returns (state (2,1), covariance (2,2)).
    """
    # Build F and H
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)

    # Predict: x̂_k^- = F x̂; P_k^- = F P Fᵀ + Q
    x_prior = F @ prev_state
    P_prior = F @ prev_covariance @ F.T + process_noise

    # Gain: S = H P_k^- Hᵀ + R (1×1); K_k = P_k^- Hᵀ S⁻¹
    S = (H @ P_prior @ H.T)[0, 0] + measurement_variance
    K = P_prior @ H.T / S

    # Correct: x̂_k = x̂_k^- + K_k(z_k - H x̂_k^-); P_k = (I - K_k H) P_k^-
    z_k = np.array([[measurement]], dtype=np.float64)
    x = x_prior + K * (z_k[0, 0] - (H @ x_prior)[0, 0])
    P = (I - K @ H) @ P_prior

    return x, P


@njit
def kalman_2d_batch(
    measurements: np.ndarray,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    process_noise: np.ndarray,
    measurement_variance: float,
    dt: float,
) -> tuple:
    """Batch 2D Kalman filter over a full measurement sequence.

    Returns (states (N,2,1), covariances (N,2,2)) as float64 arrays.
    """
    n = len(measurements)
    states = np.empty((n, 2, 1), dtype=np.float64)
    covariances = np.empty((n, 2, 2), dtype=np.float64)

    x = initial_state.copy()
    P = initial_covariance.copy()

    for k in range(n):
        x, P = kalman_2d_step(
            measurements[k], x, P, process_noise, measurement_variance, dt
        )
        states[k] = x
        covariances[k] = P

    return states, covariances
