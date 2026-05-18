"""
Synthetic SRS-like channel generator for AoA validation.

Generates frequency-domain channel estimates as observed at a ULA receiver,
mimicking the shape of OAI's:
    srs_estimated_channel_freq[rx_ant][tx_ant][subcarrier]

For a UE transmitting from angle theta with K subcarriers, the channel seen
at rx antenna n on subcarrier k is:

    H[n, k] = sum_{p}  alpha_p * exp(-j 2pi n d/lambda sin(theta_p))
                       * exp(-j 2pi k Delta_f tau_p)

We start with a single LoS path; multi-path is added optionally for stress
tests.
"""

import numpy as np


def generate_srs_channel(
    theta_deg,
    n_rx=2,
    n_sc=600,
    d_over_lambda=0.5,
    snr_db=20.0,
    delay_samples=0.0,
    n_sc_full=1536,
    alpha=1.0,
    rng=None,
):
    """
    Generate a single-path SRS-like channel estimate at a ULA receiver.

    Parameters
    ----------
    theta_deg : float
        True angle of arrival (degrees, from array broadside).
    n_rx : int
        Number of RX antennas (ULA).
    n_sc : int
        Number of active SRS subcarriers (occupied bandwidth).
    d_over_lambda : float
        Antenna spacing in wavelengths.
    snr_db : float
        Per-antenna, per-subcarrier SNR in dB.
    delay_samples : float
        Propagation delay in samples (creates frequency-domain phase ramp).
        Doesn't affect AoA but mimics real OAI channel estimates.
    n_sc_full : int
        Full FFT size (default matches Table III: 1536).
    alpha : complex
        Path gain.
    rng : np.random.Generator

    Returns
    -------
    H : ndarray (n_rx, n_sc), complex
        Frequency-domain channel estimate.
    info : dict with ground-truth metadata.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Spatial signature
    n = np.arange(n_rx).reshape(-1, 1)                          # (N, 1)
    a = np.exp(-2j * np.pi * d_over_lambda * n * np.sin(np.deg2rad(theta_deg)))

    # Frequency-domain delay (phase ramp across subcarriers)
    k = np.arange(n_sc).reshape(1, -1)                          # (1, K)
    delay_phase = np.exp(-2j * np.pi * k * delay_samples / n_sc_full)

    # Noiseless channel
    H_clean = alpha * a * delay_phase                           # (N, K)

    # Add complex Gaussian noise calibrated to requested SNR
    sig_power = np.mean(np.abs(H_clean) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = (rng.standard_normal(H_clean.shape) +
             1j * rng.standard_normal(H_clean.shape)) * np.sqrt(noise_power / 2)

    H = H_clean + noise

    info = {
        "theta_true_deg": theta_deg,
        "n_rx": n_rx,
        "n_sc": n_sc,
        "d_over_lambda": d_over_lambda,
        "snr_db": snr_db,
        "delay_samples": delay_samples,
    }
    return H, info


def generate_multipath_channel(
    angles_deg,
    powers_db,
    n_rx=4,
    n_sc=600,
    d_over_lambda=0.5,
    snr_db=20.0,
    rng=None,
):
    """
    Generate a multi-path channel for testing MUSIC's resolution capability.

    Parameters
    ----------
    angles_deg : list of float
        True AoAs of each path.
    powers_db : list of float
        Relative powers of each path in dB.
    """
    if rng is None:
        rng = np.random.default_rng(1)

    n_sc_full = 1536
    H_clean = np.zeros((n_rx, n_sc), dtype=complex)

    for theta, P_db in zip(angles_deg, powers_db):
        alpha_mag = 10 ** (P_db / 20)
        # random phase per path, random small delay
        phi = rng.uniform(0, 2 * np.pi)
        tau = rng.uniform(0, 10)
        alpha = alpha_mag * np.exp(1j * phi)

        n = np.arange(n_rx).reshape(-1, 1)
        a = np.exp(-2j * np.pi * d_over_lambda * n * np.sin(np.deg2rad(theta)))
        k = np.arange(n_sc).reshape(1, -1)
        delay_phase = np.exp(-2j * np.pi * k * tau / n_sc_full)

        H_clean += alpha * a * delay_phase

    sig_power = np.mean(np.abs(H_clean) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = (rng.standard_normal(H_clean.shape) +
             1j * rng.standard_normal(H_clean.shape)) * np.sqrt(noise_power / 2)

    H = H_clean + noise
    return H, {"angles_true_deg": angles_deg, "snr_db": snr_db}
