# IIR Filter Design via Bilinear Transform

This project solves the **IIR Filter Tolerance Design Problem (PPFTI)** by implementing the bilinear transform method from scratch in Python. Four classical IIR filter families — Butterworth, Chebyshev I, Chebyshev II, and Cauer (Elliptic) — are designed, analyzed, and ranked against strict frequency-domain specifications across three progressive phases.

---

## Quickstart (copy–paste)

```bash
# 1) Clone the repository
git clone https://github.com/cristi1710/Filter-Project.git
cd Filter-Project

# 2) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3) Install dependencies
pip install numpy scipy matplotlib pytest

# 4) Create output directory for plots
mkdir plots

# 5) Run any phase
python phase1_butterworth.py
python phase2_comparison.py
python phase3_design_contest.py

# 6) Run unit tests
pytest test_filters.py -v
```

---

## Description

The pipeline designs low-pass IIR filters from analog prototypes and maps them to the digital domain via the bilinear transform. All four classical approximation families are compared against the same frequency-domain specifications using a composite 5-dimensional performance criterion.

Main stages:

1. **Prewarping** — map digital band edges to analog frequencies, compensating for the nonlinear frequency compression of the bilinear transform
2. **Analog prototype design** — compute minimum order, cutoff frequency, and stable left-half-plane poles (Butterworth from scratch; Chebyshev I/II and Cauer via scipy)
3. **Bilinear mapping** — convert analog poles to digital (B, A) coefficients; zeros placed at z = −1 (Nyquist)
4. **Specification verification** — check passband and stopband constraints numerically on 5000-point frequency grid
5. **Performance scoring** — Phase 2 uses a compromise score (order + ripple penalty); Phase 3 uses a composite criterion J covering order, magnitude error, group delay variation, pole stability, and step overshoot
6. **Ranking and plots** — filters sorted by score; frequency response and phase plots saved to `plots/`

---

## Project Structure

```
Filter-Project/
│
├── filter_design/                # Core library package
│   ├── __init__.py               # Package exports
│   ├── butterworth.py            # From-scratch Butterworth: order, poles, bilinear transform
│   ├── chebyshev.py              # Chebyshev Type I & II via scipy (cheb1ord, cheb2ord)
│   ├── cauer.py                  # Cauer (elliptic) filter via scipy (ellipord, ellip)
│   ├── utils.py                  # Prewarping, bilinear transform, freq. response, group delay, spec check
│   └── cost_function.py          # Composite cost J and filter ranking (Phase 3)
│
├── phase1_butterworth.py         # Phase 1: Butterworth PPFTI solution + plots
├── phase2_comparison.py          # Phase 2: Four-filter comparison with compromise score
├── phase3_design_contest.py      # Phase 3: Design contest with full criterion J
├── test_filters.py               # Unit tests (pytest) — 8 tests
├── plots/                        # Output directory for generated figures (PNG)
└── README.md
```

---

## Input / Output

### Inputs (hardcoded specifications per phase)

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Passband edge ωp | 1.1887 rad/sample | 1.1887 rad/sample | 0.5189π rad/sample |
| Stopband edge ωs | 1.4985 rad/sample | 1.4985 rad/sample | ωp + π/33 |
| Passband ripple Δp | 0.0708 | 0.0708 | 0.05 |
| Stopband attenuation Δs | 0.0708 | 2·Δp = 0.1416 | 10^(−35/20) ≈ 0.0178 |
| Sampling period Ts | 2.1892 s | 2.1892 s | 2.1892 s |

### Outputs

| File | Description |
|------|-------------|
| `plots/phase1_butterworth.png` | Magnitude response + passband zoom + phase |
| `plots/phase2_comparison.png` | Frequency responses + bar chart ranking by score |
| `plots/phase3_design_contest.png` | 2×4 matrix: magnitude and phase for all 4 filters ordered best-to-worst |

---

## Mathematical Background

### Bilinear Transform

Maps the analog s-domain to the digital z-domain, preserving stability:

$$s = \frac{2}{T_s} \cdot \frac{1 - z^{-1}}{1 + z^{-1}}$$

### Frequency Prewarping

Compensates for the nonlinear frequency warping:

$$\Omega_{p,s} = \frac{2}{T_s} \tan\!\left(\frac{\omega_{p,s}}{2}\right)$$

**Key invariance property:** the ratio Ωs/Ωp is independent of Ts (and of the bilinear scaling constant C = 2/Ts), so the filter order M and the digital transfer function H(z) are invariant to changes in sampling period. Verified analytically and numerically (errors at machine precision ~10⁻¹³) in Phase 1b and 1c.

### Butterworth Order Formula

$$M \geq \frac{\log\!\left(\dfrac{M_p^2}{\Delta_s^2} \cdot \dfrac{1 - \Delta_s^2}{1 - M_p^2}\right)}{2 \log(\Omega_s / \Omega_p)}, \qquad M_p = 1 - \Delta_p$$

### Filter Family Comparison

| Filter | Passband | Stopband | Order vs Butterworth |
|--------|----------|----------|----------------------|
| Butterworth | Monotone | Monotone | Reference (highest) |
| Chebyshev I | Equiripple | Monotone | Lower |
| Chebyshev II | Monotone | Equiripple | Lower |
| Cauer (Elliptic) | Equiripple | Equiripple | Minimum (lowest possible) |

### Phase 3 Composite Performance Criterion

$$J = J_\text{ord} + J_\text{freq} + J_\text{phase} + J_\text{stab} + J_\text{tran}$$

| Component | Formula | Weight | Measures |
|-----------|---------|--------|----------|
| J_ord | W_M · M | W_M = 2 | Filter complexity (order) |
| J_freq | W_f · (RMSE_pass + 0.1·RMSE_stop) | W_f = 50 | Distance from ideal magnitude |
| J_phase | W_φ · std(τg) / mean(τg) \|ω≤ωp | W_φ = 20 | Group delay variation in passband |
| J_stab | W_s / (1 − ρ + ε) | W_s = 10 | Pole proximity to unit circle (ρ = max\|poles\|) |
| J_tran | W_t · max(0, max(y_step) − 1) | W_t = 20 | Step response overshoot |

### Phase 2 Compromise Score

```
Score = M + 1.2·rip_pass + 0.1·rip_stop
```

Passband ripple is penalized more heavily (1.2 vs 0.1) because it directly distorts the useful signal. Ripple flags are assigned theoretically based on filter family properties, not by numerical detection.

---

## Results

### Phase 1 — Butterworth

| Metric | Value |
|--------|-------|
| Filter order M | **12** |
| Min passband magnitude | 0.9301 ≥ 0.9292 ✓ |
| Max stopband magnitude | 0.0543 ≤ 0.0708 ✓ |
| Specs satisfied | Yes |
| Equivalent FIR order (Window / Hamming) | **89** — 7.4× higher than IIR |
| Equivalent FIR order (Least Squares) | **30** — 2.5× higher than IIR |

Phase 1 sub-investigations:

- **1b — Bilinear constant invariance:** Tustin (2/Ts) and Pseudo-Tustin (1/Ts) produce identical H(z); spectral error norm = 0
- **1c — Ts independence:** H(z) invariant for Ts ∈ [0.1·Tref, 3.0·Tref]; errors at machine precision (~10⁻¹³)
- **1d — Order sensitivity:** 16 combinations of (Δp, Δs) studied; M varies from 8 (tolerances doubled) to 15 (tolerances halved)

### Phase 2 — Filter Comparison (Δs = 2Δp)

| Rank | Filter | M | rip_pass | rip_stop | Score |
|------|--------|---|----------|----------|-------|
| 1 | Cauer (Elliptic) | 3 | Yes | Yes | 4.3 |
| 2 | **Chebyshev II** | 5 | No | Yes | **5.1 — Best Buy** |
| 3 | Chebyshev I | 5 | Yes | No | 6.2 |
| 4 | Butterworth | 9 | No | No | 9.0 |

Chebyshev II is the best practical choice: it keeps the passband clean (no ripple affecting the useful signal) while reducing order from 9 to 5.

### Phase 3 — Design Contest (Δp = 5%, As = 35 dB, Δω = π/33)

| Rank | Filter | M | J_ord | J_freq | J_phase | J_stab | J_tran | J total |
|------|--------|---|-------|--------|---------|--------|--------|---------|
| 1 | **Chebyshev II** | 14 | 28.0 | 0.1 | 20.6 | 279.6 | 3.8 | **332.18** |
| 2 | Cauer | 6 | 12.0 | 1.6 | 26.0 | 369.8 | 3.0 | 412.46 |
| 3 | Butterworth | 54 | 108.0 | 0.1 | 9.6 | 348.8 | 4.8 | 471.43 |
| 4 | Chebyshev I | 14 | 28.0 | 1.6 | 15.4 | 688.5 | 4.1 | 737.57 |

Key insight: Butterworth requires M = 54 for the narrow band (Δω = π/33), pushing poles very close to the unit circle (ρ ≈ 0.971) and causing a large J_stab penalty. Chebyshev II wins because it combines moderate order with excellent magnitude accuracy and favorable pole placement.

---

## API Reference

### `filter_design.utils`

| Function | Signature | Description |
|----------|-----------|-------------|
| `prewarping` | `(w_norm, Ts) → float` | Analog prewarped frequency: (2/Ts)·tan(w·π/2) |
| `bilinear_transform` | `(poles_analog, Ts) → B, A` | Maps analog poles to digital B, A coefficients; zeros at z = −1 |
| `compute_frequency_response` | `(B, A, n=5000) → H, w` | Complex H(e^jω) on [0, π] |
| `compute_group_delay` | `(B, A, n=5000) → gd, w` | Group delay on [0, 0.99π] (avoids Nyquist singularity) |
| `check_specs` | `(H, w, w_p, w_s, Δp, Δs) → ok, val_pass, val_stop` | Verify passband/stopband constraints |

### `filter_design.butterworth`

| Function | Signature | Description |
|----------|-----------|-------------|
| `design_butterworth` | `(w_p, w_s, Δp, Δs, Ts) → B, A, M` | Full from-scratch Butterworth design |

### `filter_design.chebyshev`

| Function | Signature | Description |
|----------|-----------|-------------|
| `design_chebyshev1` | `(w_p, w_s, Δp, Δs, Ts) → B, A, M` | Chebyshev Type I via scipy |
| `design_chebyshev2` | `(w_p, w_s, Δp, Δs, Ts) → B, A, M` | Chebyshev Type II via scipy |

### `filter_design.cauer`

| Function | Signature | Description |
|----------|-----------|-------------|
| `design_cauer` | `(w_p, w_s, Δp, Δs, Ts) → B, A, M` | Cauer (elliptic) filter via scipy |

### `filter_design.cost_function`

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_J` | `(B, A, w_p, w_s, weights=None) → dict` | Compute all J components + total |
| `rank_filters` | `(filters_dict, w_p, w_s) → list[dict]` | Rank filters by J ascending |

> **Frequency convention:** design functions expect normalized `[0, 1]` frequencies (scipy convention). `check_specs()`, `rank_filters()`, and `compute_J()` expect rad/sample `[0, π]`. Convert with `w_rad = w_norm × π`.

---

## Unit Tests

```
test_butterworth_order_matches_scipy            PASSED  — M ≥ 1, len(B) = len(A) = M+1
test_butterworth_specs_satisfied                PASSED  — passband min ≥ 1−Δp, stopband max ≤ Δs
test_butterworth_dc_gain                        PASSED  — H(0) within 5% of 1.0
test_filter_stability_all                       PASSED  — all poles inside unit circle (all 4 types)
test_prewarping_correctness                     PASSED  — matches formula to 10⁻¹⁰ tolerance
test_bilinear_transform_produces_valid_coeff    PASSED  — B, A finite, correct length
test_chebyshev2_specs_satisfied                 PASSED  — passband and stopband constraints met
test_cauer_lower_order_than_butterworth         PASSED  — M_cauer ≤ M_butterworth for same specs
```

---

## Known Issues & Fixes

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `UserWarning: denominator extremely small at Nyquist` | `scipy.group_delay()` numerical singularity at ω = π for high-order filters | `compute_group_delay()` evaluates on `[0, 0.99π]` instead of `[0, π]` |
| Phase 2 ranking used Phase 3 criterion | `rank_filters()` (composite J) was called in phase2 script | Phase 2 now uses `Score = M + 1.2·rip_pass + 0.1·rip_stop` with theoretical ripple flags |
| scipy reference visually diverges from manual filter | Different DC gain normalization: scipy sets H(ωp) = 1−Δp; manual design sets H(0) = 1.0 | Both are mathematically correct; passband zoom subplot added to Phase 1 to show tolerance bounds clearly |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, polynomial arithmetic |
| `scipy` | Filter design (cheby1/2, ellip), frequency response, group delay |
| `matplotlib` | Frequency response and phase plots |
| `pytest` | Unit testing |

---

## Notes

- **Butterworth is implemented from scratch** — order formula, cutoff computation, stable pole placement, and bilinear mapping are all done manually without calling `scipy.butter`
- **Chebyshev and Cauer use scipy** — `cheb1ord` / `cheby1`, `cheb2ord` / `cheby2`, `ellipord` / `ellip` with tolerances converted to dB
- **Group delay clamped to 0.99π** — avoids the numerical singularity that scipy's `group_delay()` produces near Nyquist for high-order filters
- **Phase 3 cost criterion** — implements the exact formula from the project report (Faza 3, section 4.9.2); weights are fixed at W_M=2, W_freq=50, W_phase=20, W_stab=10, W_trans=20
