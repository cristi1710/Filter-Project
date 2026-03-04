import numpy as np
from .utils import prewarping, bilinear_transform


def _compute_order(Omega_p, Omega_s, delta_p, delta_s):
    """
    Compute the minimum Butterworth filter order.

    Uses the closed-form order formula derived from the passband and
    stopband magnitude constraints.

    Parameters

    Omega_p : float
        Prewarped passband edge frequency in rad/s.
    Omega_s : float
        Prewarped stopband edge frequency in rad/s.
    delta_p : float
        Maximum passband ripple (linear).
    delta_s : float
        Maximum stopband magnitude (linear).

    Returns

    int
        Minimum filter order satisfying the specifications.
    """
    Mp = 1.0 - delta_p
    numerator = np.log((Mp**2 / delta_s**2) * ((1.0 - delta_s**2) / (1.0 - Mp**2)))
    denominator = 2.0 * np.log(Omega_s / Omega_p)
    return int(np.ceil(numerator / denominator))


def _compute_cutoff(Omega_p, M, delta_p):
    """
    Compute the Butterworth cutoff frequency that meets the passband edge.

    Parameters

    Omega_p : float
        Prewarped passband edge frequency in rad/s.
    M : int
        Filter order.
    delta_p : float
        Maximum passband ripple (linear).

    Returns

    float
        Cutoff frequency Omega_c in rad/s.
    """
    Mp = 1.0 - delta_p
    return Omega_p / ((1.0 - Mp**2) / Mp**2) ** (1.0 / (2.0 * M))


def _compute_stable_poles(Omega_c, M):
    """
    Compute the left-half-plane poles of an analog Butterworth filter.

    Parameters

    Omega_c : float
        Cutoff frequency in rad/s.
    M : int
        Filter order.

    Returns

    ndarray of complex
        Stable analog poles (negative real part only).
    """
    m = np.arange(1, M + 1)
    angles = (M + (2 * m - 1)) * np.pi / (2.0 * M)
    poles = Omega_c * np.exp(1j * angles)
    return poles[np.real(poles) < 0]


def design_butterworth(w_p, w_s, delta_p, delta_s, Ts):
    """
    Design a digital Butterworth low-pass filter from scratch.

    Computes the minimum order, analog prototype poles, and maps them
    to the digital domain via frequency prewarping and the bilinear transform.

    Parameters

    w_p : float
        Normalized passband edge frequency in [0, 1] (1 = Nyquist).
    w_s : float
        Normalized stopband edge frequency in [0, 1].
    delta_p : float
        Maximum passband ripple (linear, e.g. 0.05).
    delta_s : float
        Maximum stopband magnitude (linear).
    Ts : float
        Sampling period in seconds.

    Returns
    
    B : ndarray
        Numerator polynomial coefficients of the digital filter.
    A : ndarray
        Denominator polynomial coefficients of the digital filter.
    M : int
        Filter order.
    """
    Omega_p = prewarping(w_p, Ts)
    Omega_s = prewarping(w_s, Ts)
    M = _compute_order(Omega_p, Omega_s, delta_p, delta_s)
    Omega_c = _compute_cutoff(Omega_p, M, delta_p)
    poles = _compute_stable_poles(Omega_c, M)
    B, A = bilinear_transform(poles, Ts)
    return B, A, M