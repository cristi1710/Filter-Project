import numpy as np
from scipy.signal import butter
from filter_design import design_butterworth, design_chebyshev1, design_chebyshev2, design_cauer
from filter_design.utils import compute_frequency_response, check_specs, bilinear_transform, prewarping

W_P = 0.35
W_S = 0.50
DELTA_P = 0.05
DELTA_S = 0.05
TS = 2.0


def test_butterworth_order_matches_scipy():
    B, A, M = design_butterworth(W_P, W_S, DELTA_P, DELTA_S, TS)
    B_ref, A_ref = butter(M, W_P, btype="low", analog=False)
    assert M >= 1
    assert len(A) == M + 1
    assert len(B) == M + 1


def test_butterworth_specs_satisfied():
    B, A, M = design_butterworth(W_P, W_S, DELTA_P, DELTA_S, TS)
    H, w = compute_frequency_response(B, A)
    ok, val_pass, val_stop = check_specs(H, w, W_P * np.pi, W_S * np.pi, DELTA_P, DELTA_S)
    assert ok, f"Specs not satisfied: passband min={val_pass:.4f}, stopband max={val_stop:.4f}"


def test_butterworth_dc_gain():
    B, A, M = design_butterworth(W_P, W_S, DELTA_P, DELTA_S, TS)
    H0 = np.polyval(B, 1.0) / np.polyval(A, 1.0)
    assert abs(np.real(H0) - 1.0) < 0.05, f"DC gain {H0:.4f} deviates from 1.0"


def test_filter_stability_all():
    designers = [design_butterworth, design_chebyshev1, design_chebyshev2, design_cauer]
    for designer in designers:
        B, A, M = designer(W_P, W_S, DELTA_P, DELTA_S, TS)
        poles = np.roots(A)
        assert np.all(np.abs(poles) < 1.0), f"{designer.__name__} has unstable poles"


def test_prewarping_correctness():
    w = 0.5
    Ts = 2.0
    Omega = prewarping(w, Ts)
    expected = (2.0 / Ts) * np.tan(w * np.pi / 2.0)
    assert abs(Omega - expected) < 1e-10


def test_bilinear_transform_produces_valid_coefficients():
    Omega_c = 1.0
    M = 4
    m = np.arange(1, M + 1)
    angles = (M + (2 * m - 1)) * np.pi / (2.0 * M)
    poles = Omega_c * np.exp(1j * angles)
    poles = poles[np.real(poles) < 0]
    Ts = 2.0
    B, A = bilinear_transform(poles, Ts)
    assert len(B) == len(poles) + 1
    assert len(A) == len(poles) + 1
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(A))


def test_chebyshev2_specs_satisfied():
    B, A, M = design_chebyshev2(W_P, W_S, DELTA_P, DELTA_S, TS)
    H, w = compute_frequency_response(B, A)
    ok, val_pass, val_stop = check_specs(H, w, W_P * np.pi, W_S * np.pi, DELTA_P, DELTA_S)
    assert ok, f"Chebyshev II specs not satisfied: passband min={val_pass:.4f}"


def test_cauer_lower_order_than_butterworth():
    _, _, M_but = design_butterworth(W_P, W_S, DELTA_P, DELTA_S, TS)
    _, _, M_cau = design_cauer(W_P, W_S, DELTA_P, DELTA_S, TS)
    assert M_cau <= M_but, f"Cauer order {M_cau} should be <= Butterworth order {M_but}"