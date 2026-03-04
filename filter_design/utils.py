import numpy as np
from scipy.signal import freqz, group_delay


def prewarping(w_normalized, Ts):
    """
    Apply frequency prewarping for the bilinear transform.

    Converts a normalized digital frequency to an analog frequency,
    compensating for the nonlinear frequency warping introduced by
    the bilinear transform.

    Parameters
    ----------
    w_normalized : float
        Normalized digital frequency in [0, 1] (where 1 = Nyquist = pi rad/sample).
    Ts : float
        Sampling period in seconds.

    Returns
    -------
    float
        Prewarped analog frequency in rad/s.
    """
    return (2.0 / Ts) * np.tan(w_normalized * np.pi / 2.0)


def bilinear_transform(poles_analog, Ts):
    """
    Convert analog filter poles to a digital filter via the bilinear transform.

    Maps analog (s-domain) poles to the z-domain and places zeros at z = -1
    (Nyquist), then computes the numerator and denominator polynomial
    coefficients of the resulting digital filter.

    Parameters
    ----------
    poles_analog : array_like of complex
        Stable analog poles (must all have negative real part).
    Ts : float
        Sampling period in seconds.

    Returns
    -------
    B : ndarray
        Numerator polynomial coefficients (length M+1).
    A : ndarray
        Denominator polynomial coefficients (length M+1), real-valued.
    """
    M = len(poles_analog)
    z_poles = (1.0 + (Ts / 2.0) * poles_analog) / (1.0 - (Ts / 2.0) * poles_analog)
    z_zeros = -np.ones(M)
    gain_num = np.real(np.prod(-poles_analog * Ts / (2.0 - poles_analog * Ts)))
    B = gain_num * np.poly(z_zeros)
    A = np.real(np.poly(z_poles))
    B = np.real(B)
    return B, A


def compute_frequency_response(B, A, n_points=5000):
    """
    Compute the complex frequency response of a digital filter.

    Parameters
    ----------
    B : array_like
        Numerator polynomial coefficients.
    A : array_like
        Denominator polynomial coefficients.
    n_points : int, optional
        Number of frequency points (default: 5000).

    Returns
    -------
    H : ndarray of complex
        Complex frequency response values.
    w : ndarray
        Frequency axis in rad/sample, ranging from 0 to pi.
    """
    w, H = freqz(B, A, worN=n_points)
    return H, w


def compute_group_delay(B, A, n_points=5000):
    """
    Compute the group delay of a digital filter.

    Parameters
    ----------
    B : array_like
        Numerator polynomial coefficients.
    A : array_like
        Denominator polynomial coefficients.
    n_points : int, optional
        Number of frequency points (default: 5000).

    Returns
    -------
    gd : ndarray
        Group delay in samples.
    w : ndarray
        Frequency axis in rad/sample, ranging from 0 to pi.
    """
    # Evaluate up to 99 % of pi to avoid the singularity at Nyquist
    w_eval = np.linspace(0, np.pi * 0.99, n_points)
    w, gd = group_delay((B, A), w=w_eval)
    return gd, w


def check_specs(H, w, w_p, w_s, delta_p, delta_s):
    """
    Check whether a filter satisfies passband and stopband specifications.

    Both ``w`` (from ``compute_frequency_response``) and the thresholds
    ``w_p`` / ``w_s`` must be expressed in **rad/sample** (0 to pi).
    Pass ``w_p * np.pi`` and ``w_s * np.pi`` when your design uses
    normalized frequencies in [0, 1].

    Parameters
    ----------
    H : ndarray of complex
        Complex frequency response (same length as ``w``).
    w : ndarray
        Frequency axis in rad/sample (0 to pi).
    w_p : float
        Passband edge frequency in rad/sample.
    w_s : float
        Stopband edge frequency in rad/sample.
    delta_p : float
        Maximum passband ripple (linear, e.g. 0.05 means ±5 %).
    delta_s : float
        Maximum stopband magnitude (linear).

    Returns
    -------
    ok : bool
        True if both passband and stopband constraints are satisfied.
    val_pass : float
        Minimum magnitude in the passband.
    val_stop : float
        Maximum magnitude in the stopband.
    """
    mag = np.abs(H)
    idx_pass = w <= w_p
    idx_stop = w >= w_s
    val_pass = np.min(mag[idx_pass]) if np.any(idx_pass) else 0.0
    val_stop = np.max(mag[idx_stop]) if np.any(idx_stop) else 1.0
    pass_ok = val_pass >= (1.0 - delta_p)
    stop_ok = val_stop <= delta_s
    return pass_ok and stop_ok, val_pass, val_stop
