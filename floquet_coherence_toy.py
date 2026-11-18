"""
Project Dynamic Protection - Dimensionless Open Quantum System Toy Model
Demonstrates coherence preservation under periodic Floquet modulation vs. unmodulated decay
using a dimensionless Lindblad master equation (RK4 integrator).
"""

import numpy as np

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)

def lindblad_rhs(rho, t, omega_0, drive_amp, drive_freq, gamma_relax, gamma_dephase):
    """Dimensionless Lindblad master equation right-hand side."""
    H = 0.5 * omega_0 * sz + drive_amp * np.cos(drive_freq * t) * sx
    commutator = -1j * (H @ rho - rho @ H)

    d_relax = gamma_relax * (sm @ rho @ sp - 0.5 * (sp @ sm @ rho + rho @ sp @ sm))
    d_dephase = gamma_dephase * (sz @ rho @ sz - rho)

    return commutator + d_relax + d_dephase

def run_benchmark(drive_amp=0.0, steps=1000, dt=0.02):
    """Runs a dimensionless time-evolution benchmark and returns final coherence |rho_01|."""
    omega_0 = 1.0
    drive_freq = 1.0
    gamma_relax = 0.05
    gamma_dephase = 0.15

    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

    for step in range(steps):
        t = step * dt
        k1 = lindblad_rhs(rho, t, omega_0, drive_amp, drive_freq, gamma_relax, gamma_dephase)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, t + 0.5 * dt, omega_0, drive_amp, drive_freq, gamma_relax, gamma_dephase)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, t + 0.5 * dt, omega_0, drive_amp, drive_freq, gamma_relax, gamma_dephase)
        k4 = lindblad_rhs(rho + dt * k3, t + dt, omega_0, drive_amp, drive_freq, gamma_relax, gamma_dephase)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    coherence = np.abs(rho[0, 1])
    return float(coherence)

if __name__ == "__main__":
    c_off = run_benchmark(drive_amp=0.0)
    c_on = run_benchmark(drive_amp=0.8)
    print(f"Final Coherence (Control OFF): {c_off:.4f}")
    print(f"Final Coherence (Floquet Control ON): {c_on:.4f}")
