import numpy as np
import matplotlib.pyplot as plt
from filter_design import (
    design_butterworth,
    design_chebyshev1,
    design_chebyshev2,
    design_cauer,
)
from filter_design.utils import compute_frequency_response
from filter_design.cost_function import rank_filters

# Design-contest specifications
# Narrow transition band: w_s - w_p = 1/33 (normalized).
# Stopband attenuation fixed at 35 dB → delta_s = 10^(-35/20).
# Ts is inherited from the lab sampling rate used in previous phases.
W_P       = 0.5189                       # passband edge, normalized [0,1]
W_S       = W_P + 1.0 / 33.0            # stopband edge (narrow transition)
DELTA_P   = 0.05                         # max passband ripple (linear)
DELTA_S   = 10 ** (-35.0 / 20.0)        # stopband attenuation: -35 dB
TS        = 2.1891891892                 # sampling period in seconds

# Convert to rad/sample for functions that require it (0 to pi scale)
W_P_RAD = W_P * np.pi
W_S_RAD = W_S * np.pi

print("Design contest specifications:")
print(f"  w_p     = {W_P:.4f} (normalized 0-1)")
print(f"  w_s     = {W_S:.4f} (normalized 0-1)")
print(f"  delta_p = {DELTA_P:.4f}")
print(f"  delta_s = {DELTA_S:.6f} ({-20*np.log10(DELTA_S):.1f} dB attenuation)")
print(f"  Ts      = {TS:.4f} s")
print()

B_but, A_but, M_but = design_butterworth(W_P, W_S, DELTA_P, DELTA_S, TS)
B_c1,  A_c1,  M_c1  = design_chebyshev1(W_P, W_S, DELTA_P, DELTA_S, TS)
B_c2,  A_c2,  M_c2  = design_chebyshev2(W_P, W_S, DELTA_P, DELTA_S, TS)
B_cau, A_cau, M_cau = design_cauer(W_P, W_S, DELTA_P, DELTA_S, TS)

filters_named = {
    "Butterworth": (B_but, A_but, M_but),
    "Chebyshev I":  (B_c1,  A_c1,  M_c1),
    "Chebyshev II": (B_c2,  A_c2,  M_c2),
    "Cauer":        (B_cau, A_cau, M_cau),
}

# rank_filters expects thresholds in rad/sample (0 to pi)
ranking = rank_filters(
    {k: (v[0], v[1]) for k, v in filters_named.items()},
    W_P_RAD,
    W_S_RAD,
)

print(f"{'Rank':<5} {'Filter':<14} {'M':<5} {'J_ord':<8} {'J_freq':<8} {'J_phase':<9} {'J_stab':<8} {'J_tran':<8} {'J':<10}")
print("-" * 75)
for i, r in enumerate(ranking, 1):
    print(f"{i:<5} {r['name']:<14} {r['M']:<5} {r['J_ord']:<8.1f} {r['J_freq']:<8.1f} {r['J_phase']:<9.1f} {r['J_stab']:<8.1f} {r['J_tran']:<8.1f} {r['J']:<10.2f}")

ordered_names   = [r["name"] for r in ranking]
ordered_filters = [(n, *filters_named[n]) for n in ordered_names]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
tol_up_db   = 20 * np.log10(1 + DELTA_P)
tol_low_db  = 20 * np.log10(1 - DELTA_P)
tol_stop_db = 20 * np.log10(DELTA_S)

for col, (name, B, A, M) in enumerate(ordered_filters):
    H, w = compute_frequency_response(B, A)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    phase  = np.unwrap(np.angle(H))
    J_val  = ranking[col]["J"]

    ax_mag = axes[0, col]
    ax_mag.plot(w / np.pi, mag_db, linewidth=1.5, color="#1f77b4")
    ax_mag.axvline(W_P, color="g", linestyle="--", linewidth=1)
    ax_mag.axvline(W_S, color="r", linestyle="--", linewidth=1)
    ax_mag.axhline(tol_up_db,   color="k", linestyle=":", linewidth=0.8)
    ax_mag.axhline(tol_low_db,  color="k", linestyle=":", linewidth=0.8)
    ax_mag.axhline(tol_stop_db, color="b", linestyle=":", linewidth=0.8)
    ax_mag.set_xlim([0, 1])
    ax_mag.set_ylim([-80, 5])
    ax_mag.set_title(f"{col + 1}. {name}\nM={M}  J={J_val:.2f}", fontsize=10)
    ax_mag.set_xlabel("Normalized frequency")
    if col == 0:
        ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True)

    ax_ph = axes[1, col]
    ax_ph.plot(w / np.pi, phase, linewidth=1.5, color="k")
    ax_ph.axvline(W_P, color="g", linestyle="--", linewidth=1)
    ax_ph.axvline(W_S, color="r", linestyle="--", linewidth=1)
    ax_ph.set_xlim([0, 1])
    ax_ph.set_title(f"Phase {name}", fontsize=9)
    ax_ph.set_xlabel("Normalized frequency")
    if col == 0:
        ax_ph.set_ylabel("Phase (rad)")
    ax_ph.grid(True)

plt.suptitle("Design Contest — Filters ranked by criterion J (left = best)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/phase3_design_contest.png", dpi=150)
plt.show()
print("Plot saved to plots/phase3_design_contest.png")