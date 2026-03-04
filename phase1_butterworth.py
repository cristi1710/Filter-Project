import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import butter, freqz
from filter_design import design_butterworth
from filter_design.utils import compute_frequency_response, check_specs

# ── Specifications ────────────────────────────────────────────────────────────
W_P_RAD = 1.1887107338          # passband edge in rad/sample
W_S_RAD = 1.4985168813          # stopband edge in rad/sample
DELTA_P = 0.0708108108          # max passband ripple (linear)
DELTA_S = DELTA_P               # symmetric stopband attenuation
TS      = 2.1891891892          # sampling period in seconds

w_p = W_P_RAD / np.pi          # normalized [0,1] for scipy
w_s = W_S_RAD / np.pi

B, A, M = design_butterworth(w_p, w_s, DELTA_P, DELTA_S, TS)
H_manual, w = compute_frequency_response(B, A)

B_scipy, A_scipy = butter(M, w_p)
_, H_scipy = freqz(B_scipy, A_scipy, worN=w)

spec_ok, val_pass, val_stop = check_specs(H_manual, w, W_P_RAD, W_S_RAD, DELTA_P, DELTA_S)
error_norm = np.linalg.norm(np.abs(H_manual) - np.abs(H_scipy))

print(f"Filter order M        : {M}")
print(f"Specs satisfied       : {spec_ok}")
print(f"Min passband magnitude: {val_pass:.6f} (>= {1 - DELTA_P:.6f})")
print(f"Max stopband magnitude: {val_stop:.6f} (<= {DELTA_S:.6f})")
print(f"Error norm vs scipy   : {error_norm:.6e}")

# ── Tolerance levels in dB ────────────────────────────────────────────────────
TOL_UP_DB   = 20 * np.log10(1 + DELTA_P)   # +0.59 dB
TOL_LOW_DB  = 20 * np.log10(1 - DELTA_P)   # -0.64 dB
TOL_STOP_DB = 20 * np.log10(DELTA_S)        # -23.0 dB

mag_manual = np.abs(H_manual).astype(float)
mag_scipy  = np.abs(H_scipy).astype(float)
mag_manual_db = 20 * np.log10(mag_manual + 1e-12)
mag_scipy_db  = 20 * np.log10(mag_scipy  + 1e-12)

w_norm = w / np.pi

fig = plt.figure(figsize=(12, 9))
gs  = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.4, wspace=0.35)

# ── Main magnitude plot ───────────────────────────────────────────────────────
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(w_norm, mag_manual_db, label=f"Manual (M={M})",          color="#1f77b4", linewidth=2)
ax_main.plot(w_norm, mag_scipy_db,  label="scipy.butter reference",   color="#ff7f0e", linewidth=1.5, linestyle="--")
ax_main.axvline(w_p, color="red",  linestyle="--", linewidth=1, label="ωp / ωs")
ax_main.axvline(w_s, color="red",  linestyle="--", linewidth=1)
ax_main.axhline(TOL_UP_DB,   color="#2ca02c", linestyle=":",  linewidth=1.2, label=f"1+Δp = {1 + DELTA_P}")
ax_main.axhline(TOL_LOW_DB,  color="#d62728", linestyle=":",  linewidth=1.2, label=f"1−Δp = {1 - DELTA_P}")
ax_main.axhline(TOL_STOP_DB, color="#9467bd", linestyle="-.", linewidth=1.2, label=f"Δs  = {DELTA_S}")
ax_main.set_xlim([0, 1])
ax_main.set_ylim([-80, 5])
ax_main.set_xlabel("Normalized frequency (× π rad/sample)", fontsize=11)
ax_main.set_ylabel("Magnitude (dB)", fontsize=11)
ax_main.set_title(f"Butterworth Low-Pass Filter — Order M={M}", fontsize=13, fontweight="bold")
ax_main.legend(fontsize=9, loc="lower left")
ax_main.grid(True, alpha=0.4)

# ── Zoom inset: passband detail (0 to wp) ────────────────────────────────────
ax_zoom = fig.add_subplot(gs[1, 0])
mask = w_norm <= (w_p + 0.05)
ax_zoom.plot(w_norm[mask], mag_manual_db[mask], color="#1f77b4", linewidth=2)
ax_zoom.plot(w_norm[mask], mag_scipy_db[mask],  color="#ff7f0e", linewidth=1.5, linestyle="--")
ax_zoom.axvline(w_p, color="red",    linestyle="--", linewidth=1)
ax_zoom.axhline(TOL_UP_DB,  color="#2ca02c", linestyle=":", linewidth=1.2)
ax_zoom.axhline(TOL_LOW_DB, color="#d62728", linestyle=":", linewidth=1.2)
ax_zoom.set_xlim([0, w_p + 0.05])
ax_zoom.set_ylim([TOL_LOW_DB - 1, TOL_UP_DB + 1])
ax_zoom.set_xlabel("Normalized frequency", fontsize=10)
ax_zoom.set_ylabel("Magnitude (dB)", fontsize=10)
ax_zoom.set_title("Zoom — Passband detail", fontsize=10)
ax_zoom.grid(True, alpha=0.4)

# ── Phase response ────────────────────────────────────────────────────────────
ax_phase = fig.add_subplot(gs[1, 1])
phase_manual = np.unwrap(np.angle(H_manual))
ax_phase.plot(w_norm, phase_manual, color="#1f77b4", linewidth=2)
ax_phase.axvline(w_p, color="red", linestyle="--", linewidth=1, label="ωp / ωs")
ax_phase.axvline(w_s, color="red", linestyle="--", linewidth=1)
ax_phase.set_xlim([0, 1])
ax_phase.set_xlabel("Normalized frequency (× π rad/sample)", fontsize=10)
ax_phase.set_ylabel("Phase (rad)", fontsize=10)
ax_phase.set_title("Phase Response", fontsize=10)
ax_phase.grid(True, alpha=0.4)

plt.savefig("plots/phase1_butterworth.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to plots/phase1_butterworth.png")