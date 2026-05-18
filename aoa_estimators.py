"""
AoA Estimation for Uniform Linear Array (ULA)
==============================================
Implements two classical algorithms:
  1. Bartlett (Conventional Beamforming) - baseline
  2. MUSIC (Multiple Signal Classification) - high resolution

Designed to work with SRS channel estimates from OAI of shape:
    H[rx_ant, subcarrier]  (single TX antenna assumed)

References:
  - Schmidt, R. "Multiple Emitter Location and Signal Parameter Estimation"
    IEEE Trans. Antennas Propag., 1986.
  - Van Trees, "Optimum Array Processing", Ch. 9.
"""

import numpy as np


# ---------- Array geometry ----------------------------------------------------

def steering_vector_ula(theta_deg, n_ant, d_over_lambda=0.5):
    """
    Steering vector for a Uniform Linear Array (ULA).

    a(theta) = [1, e^{-j 2pi d/lambda sin(theta)}, ..., e^{-j 2pi (N-1) d/lambda sin(theta)}]^T

    Parameters
    ----------
    theta_deg : float or array
        Angle(s) of arrival in degrees, measured from array broadside.
    n_ant : int
        Number of antenna elements.
    d_over_lambda : float
        Element spacing in wavelengths. Default 0.5 (half-wavelength) avoids
        spatial aliasing within +/- 90 degrees.

    Returns
    -------
    a : ndarray, shape (n_ant,) if scalar input, else (n_ant, len(theta_deg))
    """
    theta_rad = np.deg2rad(np.atleast_1d(theta_deg))
    n = np.arange(n_ant).reshape(-1, 1)            # (N, 1)
    phase = -2j * np.pi * d_over_lambda * n * np.sin(theta_rad).reshape(1, -1)
    a = np.exp(phase)                              # (N, len(theta))
    return a.squeeze()


# ---------- Covariance estimation --------------------------------------------

def sample_covariance(H):
    """
    Estimate spatial covariance matrix from channel snapshots.

    For OAI SRS channel estimates of shape (N_rx, K_sc), each subcarrier is
    treated as one snapshot since the channel observed across frequency is
    independent realisations of the spatial signature (narrowband-per-sc
    assumption).

    Parameters
    ----------
    H : ndarray, shape (N_rx, K)
        Channel estimates across antennas and snapshots (subcarriers / time).

    Returns
    -------
    R : ndarray, shape (N_rx, N_rx)
        Sample spatial covariance.
    """
    H = np.atleast_2d(H)
    K = H.shape[1]
    R = (H @ H.conj().T) / K
    return R


# ---------- Bartlett beamformer ----------------------------------------------

def bartlett_spectrum(R, theta_grid_deg, d_over_lambda=0.5):
    """
    Bartlett (conventional) spatial spectrum.

        P_B(theta) = a^H(theta) R a(theta) / (a^H(theta) a(theta))

    Parameters
    ----------
    R : ndarray, shape (N, N)
        Spatial covariance matrix.
    theta_grid_deg : array
        Angles (deg) to evaluate the spectrum at.
    d_over_lambda : float
        Element spacing in wavelengths.

    Returns
    -------
    P : ndarray, shape (len(theta_grid_deg),)
        Spectrum values (real, non-negative).
    """
    n_ant = R.shape[0]
    A = steering_vector_ula(theta_grid_deg, n_ant, d_over_lambda)  # (N, G)
    # P(theta) = a^H R a  -- normalise by a^H a = N (constant for ULA)
    P = np.einsum('ig,ij,jg->g', A.conj(), R, A) / n_ant
    return np.real(P)


# ---------- MUSIC ------------------------------------------------------------

def music_spectrum(R, theta_grid_deg, n_sources=1, d_over_lambda=0.5):
    """
    MUSIC pseudo-spectrum.

        P_M(theta) = 1 / (a^H(theta) E_n E_n^H a(theta))

    where E_n is the noise subspace (eigenvectors associated with the smallest
    N - n_sources eigenvalues of R).

    Parameters
    ----------
    R : ndarray, shape (N, N)
        Spatial covariance matrix.
    theta_grid_deg : array
        Angles (deg) to evaluate the spectrum at.
    n_sources : int
        Assumed number of sources (model order). Must be < N.
    d_over_lambda : float
        Element spacing in wavelengths.

    Returns
    -------
    P : ndarray, shape (len(theta_grid_deg),)
        Pseudo-spectrum values.
    """
    n_ant = R.shape[0]
    if n_sources >= n_ant:
        raise ValueError(f"n_sources ({n_sources}) must be < n_ant ({n_ant})")

    # Eigendecomposition (R is Hermitian PSD -> use eigh)
    eigvals, eigvecs = np.linalg.eigh(R)
    # eigh returns ascending order -> smallest are noise subspace
    En = eigvecs[:, : n_ant - n_sources]           # (N, N - n_sources)

    A = steering_vector_ula(theta_grid_deg, n_ant, d_over_lambda)  # (N, G)
    # a^H E_n E_n^H a  -> for each angle
    Z = En.conj().T @ A                            # (N - n_sources, G)
    denom = np.sum(np.abs(Z) ** 2, axis=0)         # (G,)
    P = 1.0 / (denom + 1e-12)
    return P


# ---------- Peak finding ------------------------------------------------------

def find_aoa_peaks(spectrum, theta_grid_deg, n_peaks=1):
    """
    Return the angles of the top-`n_peaks` peaks of a spectrum.

    Simple argmax-based search on a dense grid. Adequate for ULA spectra which
    are smooth in theta.
    """
    from scipy.signal import find_peaks
    # log-domain helps when MUSIC pseudo-spectrum has huge dynamic range
    log_P = 10 * np.log10(spectrum / spectrum.max())
    peaks, _ = find_peaks(log_P)
    if len(peaks) == 0:
        # fall back to global max
        return np.array([theta_grid_deg[np.argmax(spectrum)]])
    # sort peaks by height, descending
    peak_heights = log_P[peaks]
    order = np.argsort(peak_heights)[::-1]
    top = peaks[order[:n_peaks]]
    return theta_grid_deg[top]
