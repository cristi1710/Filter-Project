import numpy as np
from scipy.signal import freqz, group_delay, lfilter
from .utils import compute_frequency_response, compute_group_delay

# Ponderile criteriului J — identice cu proiectul (Faza 3, sectiunea 4.9.2)
WEIGHTS = {
    "order":     2.0,   # W_M
    "freq":      50.0,  # W_freq
    "phase":     20.0,  # W_phase
    "stability": 10.0,  # W_stab
    "transient": 20.0,  # W_trans
}

# Epsilon pentru stabilitate numerica in J_stab = W / (1 - rho + eps)
_EPS = 1e-12


def compute_J(B, A, w_p, w_s, weights=None):
    """
    Compute the composite performance cost J for a digital filter.

    Implements the exact criterion from the project report (Faza 3, sec. 4.9.2):

        J = J_ord + J_freq + J_phase + J_stab + J_tran

    where:
        J_ord   = W_M * M
        J_freq  = W_freq * (RMSE_pass + 0.1 * RMSE_stop)
        J_phase = W_phase * std(gd) / mean(gd)   [in passband]
        J_stab  = W_stab / (1 - rho + eps)        [continuous penalty]
        J_tran  = W_trans * max(0, max(step) - 1)

    Parameters
    ----------
    B : array_like
        Numerator polynomial coefficients.
    A : array_like
        Denominator polynomial coefficients.
    w_p : float
        Passband edge frequency in rad/sample (0 to pi).
    w_s : float
        Stopband edge frequency in rad/sample (0 to pi).
    weights : dict, optional
        Override default WEIGHTS. Keys: order, freq, phase, stability, transient.

    Returns
    -------
    dict with keys: J, J_ord, J_freq, J_phase, J_stab, J_tran, M, rho.
    """
    if weights is None:
        weights = WEIGHTS
    W = weights

    # Frequency response on a dense grid (exclude Nyquist to avoid singularity)
    n_points = 5000
    w_eval = np.linspace(0, np.pi, n_points, endpoint=False)
    _, H = freqz(B, A, worN=w_eval)
    mag = np.abs(H)

    idx_pass = w_eval <= w_p
    idx_stop = w_eval >= w_s

    rmse_pass = np.sqrt(np.mean((mag[idx_pass] - 1.0) ** 2))
    rmse_stop = np.sqrt(np.mean(mag[idx_stop] ** 2))
    J_freq = W["freq"] * (rmse_pass + 0.1 * rmse_stop)

    # Group delay — clamp to 99 % of pi to avoid Nyquist singularity
    gd, w_gd = compute_group_delay(B, A)
    gd_pass = gd[w_gd <= w_p]
    mean_gd = np.mean(gd_pass)
    J_phase = W["phase"] * np.std(gd_pass) / mean_gd if mean_gd != 0 else 0.0

    # Stability — continuous penalty (exact formula from report sec. 4.9.2)
    poles = np.roots(A)
    rho = np.max(np.abs(poles))
    J_stab = W["stability"] / (1.0 - rho + _EPS)

    # Step-response overshoot
    step_resp = lfilter(B, A, np.ones(500))
    overshoot = max(0.0, np.max(step_resp) - 1.0)
    J_tran = W["transient"] * overshoot

    # Order penalty
    M = len(A) - 1
    J_ord = W["order"] * M

    J_total = J_ord + J_freq + J_phase + J_stab + J_tran

    return {
        "J":       J_total,
        "J_ord":   J_ord,
        "J_freq":  J_freq,
        "J_phase": J_phase,
        "J_stab":  J_stab,
        "J_tran":  J_tran,
        "M":       M,
        "rho":     rho,
    }


def rank_filters(filters_dict, w_p, w_s, weights=None):
    """
    Rank filters by cost J ascending (lower = better).

    Parameters
    ----------
    filters_dict : dict
        name -> (B, A) pairs.
    w_p, w_s : float
        Band edges in rad/sample (0 to pi).
    weights : dict, optional

    Returns
    -------
    list of dict sorted by J (ascending).
    """
    results = []
    for name, (B, A) in filters_dict.items():
        scores = compute_J(B, A, w_p, w_s, weights)
        scores["name"] = name
        results.append(scores)
    results.sort(key=lambda x: x["J"])
    return results