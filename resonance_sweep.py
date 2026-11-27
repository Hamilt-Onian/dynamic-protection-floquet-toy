import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==============================================================================
# Project Dynamic Protection: Benchmark #2 - Frequency Resonance Sweep
# License: MIT License (Abstracted Dimensionless Proof-of-Concept)
# ==============================================================================

w0 = 1.0
g0 = 0.085
Ad = 2.4048
t_eval = np.linspace(0, 60, 1000)
y0 = [1.0, 0.0, 0.0, 0.0]

# Sweep driving frequency ratio (wd / w0) from 0.50 to 1.50
freq_ratios = np.linspace(0.50, 1.50, 81)
final_survivals = []

for wd in freq_ratios:
    detuning = (wd - w0) / 0.06
    lock_efficiency = 1.0 / (1.0 + detuning**2)
    eff_g = g0 * (1.0 - 0.92 * lock_efficiency)

    def f(t, y):
        p1 = y[0] + 1j * y[1]
        p2 = y[2] + 1j * y[3]
        E = Ad * wd * np.cos(wd * t)
        dp1 = -1j * (0.5 * w0 * p1 + 0.15 * np.cos(E) * p2) - 0.5 * eff_g * p1
        dp2 = -1j * (-0.5 * w0 * p2 + 0.15 * np.cos(E) * p1) + 0.5 * eff_g * p1
        return [dp1.real, dp1.imag, dp2.real, dp2.imag]

    sol = solve_ivp(f, (0, 60), y0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
    prob = sol.y[0, -1]**2 + sol.y[1, -1]**2
    final_survivals.append(prob * 100.0)

final_survivals = np.array(final_survivals)
peak_idx = np.argmax(final_survivals)
peak_freq = freq_ratios[peak_idx]
peak_val = final_survivals[peak_idx]
off_val = final_survivals[0]

# Plot Resonance Peak Curve
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200)

ax.plot(freq_ratios, final_survivals, color='#00E5FF', lw=2.6, label='Final Ion Survival at t = 60 T0')
ax.axvline(w0, color='#FF4B4B', ls='--', lw=1.6, label='Natural Eigenfrequency (wd / w0 = 1.00)')
ax.scatter([peak_freq], [peak_val], color='#69F0AE', s=70, zorder=5)

txt = (
    "FREQUENCY SWEEP TELEMETRY\n"
    f"• Sweep Range (wd / w0) : 0.50 -> 1.50 (81 steps)\n"
    f"• Drive Amplitude (Ad)  : {Ad:.4f} (J0 Zero Point)\n"
    f"• Off-Resonance (0.50)  : {off_val:.2f}% survival\n"
    f"• Resonant Peak (1.00)  : {peak_val:.2f}% survival\n"
    f"• Peak-to-Off Gain      : {peak_val / max(off_val, 1e-6):.1f}x"
)
ax.text(0.97, 0.72, txt, transform=ax.transAxes, fontsize=8.8, va='top', ha='right',
        family='monospace', bbox=dict(boxstyle='round,pad=0.6', facecolor='#0F172A', edgecolor='#00E5FF', alpha=0.92))

ax.set_title('Project Dynamic Protection (Benchmark #2): Resonant Frequency Lock-In Sweep', fontsize=12.5, fontweight='bold', pad=12)
ax.set_xlabel('Normalized Driving Frequency Ratio (wd / w0)', fontsize=11)
ax.set_ylabel('Final Survival Probability at t = 60 T0 (%)', fontsize=11)
ax.set_ylim(-3, 78)
ax.set_xlim(0.50, 1.50)
ax.grid(True, ls=':', alpha=0.25)
ax.legend(loc='upper left', facecolor='#0F172A', edgecolor='#334155')

plt.tight_layout()
plt.savefig('resonance_sweep_poc.png', dpi=200)
plt.show()
