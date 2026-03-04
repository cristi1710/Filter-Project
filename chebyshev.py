import numpy as np
from scipy.signal import cheby1, cheby2, cheb1ord, cheb2ord
from .utils import prewarping


def _todb(delta_p, delta_s):
    """
    Convert linear ripple/attenuation values to decibels.

    Parameters
    ----------
    delta_p : float
        Maximum passband ripple (linear).
    delta_s : float
        Maximum stopband magnitude (linear).

    Returns
    -------
    Rp : float
        Passband ripple in dB.
    Rs : float
        Stopband attenuation in dB.
    """
    Rp = -20.0 * np.log10(1.0 - delta_p)
    Rs = -20.0 * np.log10(delta_s)
    return Rp, Rs


def design_chebyshev1(w_p, w_s, delta_p, delta_s, Ts):
    """
    Design a digital Chebyshev Type I low-pass filter via scipy.

    Type I filters have equiripple in the passband and monotonic stopband.

    Parameters
    ----------
    w_p : float
        Normalized passband edge frequency in [0, 1] (1 = Nyquist).
    w_s : float
        Normalized stopband edge frequency in [0, 1].
    delta_p : float
        Maximum passband ripple (linear).
    delta_s : float
        Maximum stopband magnitude (linear).
    Ts : float
        Sampling period in seconds (not used by scipy — kept for API consistency).

    Returns
    -------
    B : ndarray
        Numerator polynomial coefficients.
    A : ndarray
        Denominator polynomial coefficients.
    M : int
        Filter order.
    """
    Rp, Rs = _todb(delta_p, delta_s)
    M, Wn = cheb1ord(w_p, w_s, Rp, Rs)
    B, A = cheby1(M, Rp, Wn)
    return B, A, M


def design_chebyshev2(w_p, w_s, delta_p, delta_s, Ts):
    """
    Design a digital Chebyshev Type II low-pass filter via scipy.

    Type II filters have monotonic passband and equiripple stopband.

    Parameters
    ----------
    w_p : float
        Normalized passband edge frequency in [0, 1] (1 = Nyquist).
    w_s : float
        Normalized stopband edge frequency in [0, 1].
    delta_p : float
        Maximum passband ripple (linear).
    delta_s : float
        Maximum stopband magnitude (linear).
    Ts : float
        Sampling period in seconds (not used by scipy — kept for API consistency).

    Returns
    -------
    B : ndarray
        Numerator polynomial coefficients.
    A : ndarray
        Denominator polynomial coefficients.
    M : int
        Filter order.
    """
    Rp, Rs = _todb(delta_p, delta_s)
    M, Wn = cheb2ord(w_p, w_s, Rp, Rs)
    B, A = cheby2(M, Rs, Wn)
    return B, A, M