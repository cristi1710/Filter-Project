import numpy as np
import matplotlib.pyplot as plt
from filter_design import (
    design_butterworth,
    design_chebyshev1,
    design_chebyshev2,
    design_cauer,
)
from filter_design.utils import compute_frequency_response

# Same band edges as Phase 1, stopband attenuation relaxed to 2*delta_p.
W_P_RAD = 1.1887107338          # passband edge in rad/sample
W_S_RAD = 1.4985168813          # stopband edge in rad/sample
DELTA_P = 0.0708108108          # max passband ripple (linear)
DELTA_S = 2 * DELTA_P           # relaxed stopband attenuation
TS      = 2.1891891892          # sampling period in seconds

w_p = W_P_RAD / np.pi          # normalized [0,1] for scipy
w_s = W_S_RAD / np.pi

B_but, A_but, M_but = design_butterworth(w_p, w_s, DELTA_P, DELTA_S, TS)
B_c1,  A_c1,  M_c1  = design_chebyshev1(w_p, w_s, DELTA_P, DELTA_S, TS)
B_c2,  A_c2,  M_c2  = design_chebyshev2(w_p, w_s, DELTA_P, DELTA_S, TS)
B_cau, A_cau, M_cau = design_cauer(w_p, w_s, DELTA_P, DELTA_S, TS)

# Score = M + 1.2*rip_pass + 0.1*rip_stop
# Ripple profile is determined by filter type (theoretical property):
#   Butterworth — monotone in both bands - (0, 0)
#   Chebyshev I — equiripple passband - (1, 0)
#   Chebyshev II — equiripple stopband - (0, 1)
#   Cauer — equiripple in both bands - (1, 1)
# Passband ripple is penalised more (1.2) than stopband ripple (0.1)
# because passband distortion directly affects the useful signal.
RIPPLE_PROFILE = {
    "Butterworth":     (0, 0),
    "Chebyshev I":     (1, 0),
    "Chebyshev II":    (0, 1),
    "Cauer (Eliptic)": (1, 1),
}

filter_data = {
    "Butterworth":     (B_but, A_but, M_but),
    "Chebyshev I":     (B_c1,  A_c1,  M_c1),
    "Chebyshev II":    (B_c2,  A_c2,  M_c2),
    "Cauer (Eliptic)": (B_cau, A_cau, M_cau),
}

ranking = []
for name, (B, A, M) in filter_data.items():
    rp, rs = RIPPLE_PROFILE[name]
    score  = M + 1.2 * rp + 0.1 * rs
    ranking.append({"name": name, "B": B, "A": A, "M": M,
                    "rip_pass": rp, "rip_stop": rs, "score": score})
ranking.sort(key=lambda x: x["score"])

print(f"{'Rank':<5} {'Filter':<18} {'M':<5} {'rip_pass':<10} {'rip_stop':<10} {'Score':<8}")
print("-" * 60)
for i, r in enumerate(ranking, 1):
    print(f"{i:<5} {r['name']:<18} {r['M']:<5} {r['rip_pass']:<10} {r['rip_stop']:<10} {r['score']:<8.1f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = {"Butterworth": "#1f77b4", "Chebyshev I": "#d62728",
          "Chebyshev II": "#2ca02c", "Cauer (Eliptic)": "#ff7f0e"}

for name, (B, A, M) in filter_data.items():
    H, w = compute_frequency_response(B, A)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    axes[0].plot(w / np.pi, mag_db, label=f"{name} (M={M})",
                 linewidth=1.8, color=colors[name])

axes[0].axvline(w_p, color="gray", linestyle="--", linewidth=1, label="ωp / ωs")
axes[0].axvline(w_s, color="gray", linestyle="--", linewidth=1)
axes[0].axhline(20 * np.log10(1 - DELTA_P), color="k", linestyle=":", linewidth=1, label="Tolerance limits")
axes[0].axhline(20 * np.log10(DELTA_S),      color="k", linestyle=":", linewidth=1)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([-80, 5])
axes[0].set_xlabel("Normalized frequency (× π rad/sample)")
axes[0].set_ylabel("Magnitude (dB)")
axes[0].set_title("Phase 2 — Frequency Response Comparison")
axes[0].legend(fontsize=9)
axes[0].grid(True)

names  = [r["name"]  for r in ranking]
scores = [r["score"] for r in ranking]
bar_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"]
bars = axes[1].barh(names, scores, color=bar_colors)
axes[1].set_xlabel("Score = M + 1.2·rip_pass + 0.1·rip_stop  (lower is better)")
axes[1].set_title("Phase 2 — Compromise Score Ranking")
axes[1].grid(True, axis="x")
for bar, score in zip(bars, scores):
    axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{score:.1f}", va="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("plots/phase2_comparison.png", dpi=150)
plt.show()
print("Plot saved to plots/phase2_comparison.png")