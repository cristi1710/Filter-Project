import numpy as np
from scipy.signal import ellip, ellipord
from .utils import prewarping


def design_cauer(w_p, w_s, delta_p, delta_s, Ts):
    """
    Design a digital Cauer (elliptic) low-pass filter via scipy.

    Cauer filters achieve the minimum order for given passband/stopband
    specs by allowing equiripple in both passband and stopband.

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
    Rp = -20.0 * np.log10(1.0 - delta_p)
    Rs = -20.0 * np.log10(delta_s)
    M, Wn = ellipord(w_p, w_s, Rp, Rs)
    B, A = ellip(M, Rp, Rs, Wn)
    return B, A, M