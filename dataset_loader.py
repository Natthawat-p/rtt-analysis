"""
Dataset loader for the OAI RTT dataset (Mundlamuri et al., 2024).
Dataset DOI: https://hdl.handle.net/2047/D20666241

This module is a Python port of the authors' reference MATLAB script
(`main.m`), so its assumptions match the dataset exactly:

  - Every .raw file is Q15 complex (int16 I, int16 Q interleaved).
  - Every file reshapes to (N=1536, n_frames) -- i.e. N samples per frame
    regardless of whether the file is freq-domain or time-domain.
  - The first 499 frames are invalid and dropped (frames 500..end kept).
  - Frequency-domain files (`srs_chF.raw`, `srsrxdataF.raw`) require an
    fftshift along the subcarrier axis after loading.
  - `srs_chF_lin_interp.raw` and `srs_chT.raw` are NOT fftshifted.

Folder names in the dataset look like:
    scenario_ssb_<distance>m_ue_att_<X>_<timestamp>
e.g. scenario_ssb_10m_ue_att_20_20240405T135125

System parameters (from main.m, matching Table III of the paper):
  - PRB:                106
  - subcarrier spacing: 30 kHz
  - centre frequency:   3.6192 GHz   (note: paper text rounds to 3.69 GHz)
  - sampling rate:      46.08 MHz
  - FFT size N:         1536
  - total subcarriers:  106 * 12 = 1272
  - guard subcarriers:  1536 - 1272 = 264   (132 each side)
  - DC subcarriers skipped: middle 12 (one PRB worth) around DC
"""

from __future__ import annotations

import os
import re
import numpy as np
from dataclasses import dataclass


# --- System constants from main.m ---------------------------------------------
SYS_PARAMS = {
    "centre_freq_hz":     3.6192e9,
    "speed_of_light":     3e8,                  # main.m uses 3e8, not 2.998e8
    "PRB":                106,
    "subcarrier_spacing": 30e3,
    "fft_size":           1536,
    "usrp_max_gain_db":   89.5,
    "invalid_frames":     499,                  # main.m: srs_chT(:, 500:end)
}

# Derived
TOTAL_CARRIERS = SYS_PARAMS["PRB"] * 12                          # 1272
GUARD_CARRIERS = SYS_PARAMS["fft_size"] - TOTAL_CARRIERS         # 264
SYS_PARAMS["total_carriers"]   = TOTAL_CARRIERS
SYS_PARAMS["guard_carriers"]   = GUARD_CARRIERS
SYS_PARAMS["bandwidth_hz"]     = TOTAL_CARRIERS * SYS_PARAMS["subcarrier_spacing"]
SYS_PARAMS["sampling_rate_hz"] = SYS_PARAMS["fft_size"] * SYS_PARAMS["subcarrier_spacing"]


# --- SRS / noise subcarrier index sets ----------------------------------------
# Mirrors main.m. SRS uses comb=2 -> every other subcarrier in the allocation
# carries a pilot; the in-between subcarriers carry no SRS energy and are
# used as a "noise" reference for SNR. The 12 DC subcarriers (one PRB around
# DC) are skipped in both sets.

def _compute_srs_and_noise_indices() -> tuple[np.ndarray, np.ndarray]:
    """
    Return 0-based numpy arrays of SRS subcarrier indices and noise
    subcarrier indices, mirroring main.m's `srs_subcarrier_idx` and
    `noise_subcarrier_idx`.

    MATLAB (1-based) logic from main.m:
        ind_skip_dc          = guard/2 + (PRB/2)*12+1 : ((PRB/2)+1)*12;
        srs_subcarrier_idx   = [guard/2 + ind(1)     : 2 : ind_skip_dc(1)-1,
                                ind_skip_dc(end)+1   : 2 : guard/2 + ind(end)];
        noise_subcarrier_idx = [guard/2 + ind(1) + 1 : 2 : ind_skip_dc(1)-1,
                                ind_skip_dc(end)+2   : 2 : guard/2 + ind(end)];
    where ind(1)=1 and ind(end)=1272 (for CSRS=25, full bandwidth).
    """
    g_half = GUARD_CARRIERS // 2                      # 132
    sc_first_1 = g_half + 1                           # 133
    sc_last_1  = g_half + TOTAL_CARRIERS              # 1404

    half_prb = SYS_PARAMS["PRB"] // 2
    dc_first_1 = g_half + half_prb * 12 + 1           # 769
    dc_last_1  = g_half + (half_prb + 1) * 12         # 780

    srs_left_1   = np.arange(sc_first_1,         dc_first_1,     2)
    srs_right_1  = np.arange(dc_last_1 + 1,      sc_last_1 + 1,  2)
    srs_1based   = np.concatenate([srs_left_1, srs_right_1])

    noise_left_1  = np.arange(sc_first_1 + 1,    dc_first_1,     2)
    noise_right_1 = np.arange(dc_last_1 + 2,     sc_last_1 + 1,  2)
    noise_1based  = np.concatenate([noise_left_1, noise_right_1])

    return srs_1based - 1, noise_1based - 1


SRS_SUBCARRIER_IDX, NOISE_SUBCARRIER_IDX = _compute_srs_and_noise_indices()


# --- Q15 binary reader (fixed shape: (N, n_frames)) ---------------------------

def read_q15_file(path: str, n_per_frame: int = None) -> np.ndarray:
    """
    Read a .raw file and reshape to (n_per_frame, n_frames), column-major
    (matching MATLAB `reshape(IQ, N, [])`).

    Returns
    -------
    X : np.ndarray, shape (n_per_frame, n_frames), complex128
    """
    if n_per_frame is None:
        n_per_frame = SYS_PARAMS["fft_size"]

    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2 != 0:
        raise ValueError(f"{path}: odd int16 count ({raw.size})")
    iq = raw.astype(np.float64)
    z = iq[0::2] + 1j * iq[1::2]
    if z.size % n_per_frame != 0:
        raise ValueError(
            f"{path}: {z.size} complex samples not divisible by "
            f"n_per_frame={n_per_frame}"
        )
    return z.reshape(n_per_frame, -1, order="F")     # MATLAB-style reshape


# --- Folder-name parsing ------------------------------------------------------

@dataclass
class FolderMeta:
    folder_name:       str
    distance_m:        float | None
    tx_attenuation_db: float | None
    tx_gain_db:        float | None
    timestamp:         str   | None
    has_ssb_prefix:    bool

    @property
    def is_high_snr(self) -> bool:
        return self.tx_gain_db is not None and abs(self.tx_gain_db - 89.5) < 0.1


def parse_folder_name(name: str) -> FolderMeta:
    """
    Parse folder names like:
        scenario_ssb_10m_ue_att_20_20240405T135125
        10m_ue_att_0
        scenario_8m_ue_att_50_20240101T000000
    """
    has_ssb = "_ssb_" in name or name.startswith("ssb_")

    m = re.search(r"(?:^|[_/\s])(\d+(?:\.\d+)?)\s*m(?:[_/\s]|$)", name)
    dist = float(m.group(1)) if m else None

    m = re.search(r"ue_att_(\d+(?:\.\d+)?)", name)
    att = float(m.group(1)) if m else None

    m = re.search(r"(\d{8}T\d{6})", name)
    ts = m.group(1) if m else None

    gain = (SYS_PARAMS["usrp_max_gain_db"] - att) if att is not None else None
    return FolderMeta(folder_name=name, distance_m=dist,
                      tx_attenuation_db=att, tx_gain_db=gain,
                      timestamp=ts, has_ssb_prefix=has_ssb)


# --- High-level: load one measurement folder ----------------------------------

@dataclass
class Measurement:
    """
    All .raw files from one folder, post-processed exactly as main.m does:
      - reshape to (N=1536, n_frames), column-major.
      - drop first 499 frames.
      - fftshift srs_chF and srsrxdataF along the subcarrier axis.
      - srs_chT and srs_chF_lin_interp NOT fftshifted.
      - Q15 -> float scaling NOT applied here; do `/2**15` only at plot time
        if you want unit-normalised magnitudes (main.m does it inline in stem).
    """
    meta: FolderMeta
    srs_chT:            np.ndarray | None
    srs_chF:            np.ndarray | None
    srsrxdataF:         np.ndarray | None
    srs_chF_lin_interp: np.ndarray | None


def _try_load(path: str) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    try:
        return read_q15_file(path)
    except ValueError as e:
        print(f"  [warn] {path}: {e}")
        return None


def load_measurement(folder: str, drop_invalid: bool = True) -> Measurement:
    """
    Load all .raw files from one folder, replicating main.m's preprocessing.

    Parameters
    ----------
    folder : str
        Path to a measurement folder.
    drop_invalid : bool
        If True (default), drop the first 499 frames as main.m does.
    """
    meta = parse_folder_name(os.path.basename(folder.rstrip("/")))

    srs_chT    = _try_load(os.path.join(folder, "srs_chT.raw"))
    srs_chF    = _try_load(os.path.join(folder, "srs_chF.raw"))
    srsrxdataF = _try_load(os.path.join(folder, "srsrxdataF.raw"))
    srs_lin    = _try_load(os.path.join(folder, "srs_chF_lin_interp.raw"))

    if drop_invalid:
        s = SYS_PARAMS["invalid_frames"]
        if srs_chT    is not None and srs_chT.shape[1]    > s: srs_chT    = srs_chT[:,    s:]
        if srs_chF    is not None and srs_chF.shape[1]    > s: srs_chF    = srs_chF[:,    s:]
        if srsrxdataF is not None and srsrxdataF.shape[1] > s: srsrxdataF = srsrxdataF[:, s:]
        if srs_lin    is not None and srs_lin.shape[1]    > s: srs_lin    = srs_lin[:,    s:]

    if srs_chF    is not None: srs_chF    = np.fft.fftshift(srs_chF,    axes=0)
    if srsrxdataF is not None: srsrxdataF = np.fft.fftshift(srsrxdataF, axes=0)

    return Measurement(meta=meta, srs_chT=srs_chT, srs_chF=srs_chF,
                       srsrxdataF=srsrxdataF, srs_chF_lin_interp=srs_lin)


def discover_folders(root: str) -> list[str]:
    """
    Walk root, return subfolders containing at least one expected .raw file,
    sorted by (distance, tx_attenuation).
    """
    expected = {"srs_chT.raw", "srs_chF.raw",
                "srsrxdataF.raw", "srs_chF_lin_interp.raw"}
    found = []
    for dirpath, _, files in os.walk(root):
        if any(f in expected for f in files):
            found.append(dirpath)

    def sort_key(p):
        m = parse_folder_name(os.path.basename(p))
        return (m.distance_m        if m.distance_m        is not None else 1e9,
                m.tx_attenuation_db if m.tx_attenuation_db is not None else 1e9)

    return sorted(found, key=sort_key)
