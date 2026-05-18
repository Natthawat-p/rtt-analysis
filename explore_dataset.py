"""
Main driver to explore the OAI RTT dataset.

Usage:
    Edit DATASET_ROOT below, then run:
        python3 explore_dataset.py

What it does:
  1. Walk DATASET_ROOT to find all measurement folders.
  2. Print a sanity table of parsed metadata.
  3. Compute per-folder summaries (frame counts, power, SNR, range).
  4. Deep-dive on one folder: stem plots that match main.m, plus extensions.
  5. Cross-folder views: SNR-vs-gain, combining gain.

NOTE: I'm leaving DATASET_ROOT as a placeholder -- you said you'd run it.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import (
    discover_folders, load_measurement, parse_folder_name,
    SYS_PARAMS, SRS_SUBCARRIER_IDX, NOISE_SUBCARRIER_IDX,
)
from dataset_analysis import (
    summarise_measurement, estimate_snr_db,
)
from dataset_plots import (
    plot_impulse_response, plot_channel_freq_response,
    plot_impulse_around_peak, plot_srs_vs_noise_subcarriers,
    plot_snr_per_frame, plot_iq_constellation,
    plot_snr_vs_gain, plot_combining_gain_M,
)


# ============================================================================
# CONFIG -- edit these paths to match your local dataset location
# ============================================================================
DATASET_ROOT   = "./ssb_measurements"
OUT_DIR        = "./analysis_output"
INSPECT_FOLDER = None      # e.g. "scenario_ssb_10m_ue_att_20_20240405T135125"


# ============================================================================
# Step 1: Inventory
# ============================================================================

def step1_inventory(root: str) -> list[str]:
    print(f"\n=== Step 1: Inventory of {root} ===")
    folders = discover_folders(root)
    if not folders:
        print(f"  No folders with .raw files found under {root}.")
        print("  Set DATASET_ROOT at the top of this script.")
        return []

    print(f"  Found {len(folders)} folder(s).\n")
    print(f"  {'folder':<55}  {'dist (m)':>8}  {'att (dB)':>8}  {'gain (dB)':>9}  {'timestamp':>15}")
    print("  " + "-" * 105)
    for f in folders:
        m = parse_folder_name(os.path.basename(f))
        d = "-" if m.distance_m        is None else f"{m.distance_m:>8.1f}"
        a = "-" if m.tx_attenuation_db is None else f"{m.tx_attenuation_db:>8.1f}"
        g = "-" if m.tx_gain_db        is None else f"{m.tx_gain_db:>9.1f}"
        t = "-" if m.timestamp         is None else m.timestamp
        print(f"  {os.path.basename(f):<55}  {d}  {a}  {g}  {t:>15}")
    return folders


# ============================================================================
# Step 2: Per-folder numeric summaries
# ============================================================================

def step2_summaries(folders: list[str]) -> list[dict]:
    print(f"\n=== Step 2: Per-folder summaries ===")
    summaries = []
    for f in folders:
        try:
            meas = load_measurement(f)
            summaries.append(summarise_measurement(meas))
        except Exception as e:
            print(f"  [error] {f}: {e}")

    keys = ["folder", "distance_m", "tx_gain_db",
            "n_frames_srs_chT",
            "snr_db_est",
            "range_m_frame0_chT",
            "range_m_PD_all",
            "range_m_MF_all"]
    print()
    print("  " + "  ".join(f"{k[:20]:>20}" for k in keys))
    print("  " + "-" * (22 * len(keys)))
    for s in summaries:
        cells = []
        for k in keys:
            v = s.get(k)
            if isinstance(v, float):
                cells.append(f"{v:>20.3f}")
            elif v is None:
                cells.append(f"{'-':>20}")
            else:
                cells.append(f"{str(v)[:20]:>20}")
        print("  " + "  ".join(cells))
    return summaries


# ============================================================================
# Step 3: Deep-dive on one folder
# ============================================================================

def step3_inspect(folder: str, out_dir: str):
    print(f"\n=== Step 3: Inspect {os.path.basename(folder)} ===")
    meas = load_measurement(folder)
    print(f"  meta: {meas.meta}")
    for name, arr in [("srs_chT",            meas.srs_chT),
                      ("srs_chF",            meas.srs_chF),
                      ("srsrxdataF",         meas.srsrxdataF),
                      ("srs_chF_lin_interp", meas.srs_chF_lin_interp)]:
        if arr is None:
            print(f"  {name:<22}: <missing>")
        else:
            print(f"  {name:<22}: shape={arr.shape}  |x|_avg={np.mean(np.abs(arr)):.3e}")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(folder)

    # 1. main.m's stem plot of impulse response (frame 0)
    if meas.srs_chT is not None and meas.srs_chT.shape[1] > 0:
        fig = plot_impulse_response(meas.srs_chT[:, 0])
        fig.savefig(os.path.join(out_dir, f"{base}__01_impulse.png"), dpi=110)
        plt.close(fig)

        fig = plot_impulse_around_peak(meas.srs_chT[:, 0])
        fig.savefig(os.path.join(out_dir, f"{base}__02_impulse_zoom.png"), dpi=110)
        plt.close(fig)

    # 2. main.m's stem plot of frequency response (frame 0)
    if meas.srs_chF is not None and meas.srs_chF.shape[1] > 0:
        fig = plot_channel_freq_response(meas.srs_chF[:, 0])
        fig.savefig(os.path.join(out_dir, f"{base}__03_chF.png"), dpi=110)
        plt.close(fig)

    # 3. Signal vs noise subcarriers (visual basis of SNR estimation)
    if meas.srsrxdataF is not None and meas.srsrxdataF.shape[1] > 0:
        fig = plot_srs_vs_noise_subcarriers(meas.srsrxdataF, frame=0)
        fig.savefig(os.path.join(out_dir, f"{base}__04_srs_vs_noise.png"), dpi=110)
        plt.close(fig)

        fig = plot_snr_per_frame(meas.srsrxdataF)
        fig.savefig(os.path.join(out_dir, f"{base}__05_snr_per_frame.png"), dpi=110)
        plt.close(fig)

        snr = estimate_snr_db(meas.srsrxdataF)
        print(f"  Estimated SNR: {snr['snr_db']:.2f} dB  "
              f"(P_signal={snr['signal_power_mean']:.3e}, "
              f"P_noise={snr['noise_power_mean']:.3e})")

    # 4. IQ scatter of srs_chF_lin_interp
    if meas.srs_chF_lin_interp is not None and meas.srs_chF_lin_interp.shape[1] > 0:
        fig = plot_iq_constellation(meas.srs_chF_lin_interp[:, :5],
                                    title=f"IQ scatter -- {base}")
        fig.savefig(os.path.join(out_dir, f"{base}__06_iq.png"), dpi=110)
        plt.close(fig)

    # 5. Combining gain (PD + MF) vs M
    if meas.srs_chT is not None and meas.srs_chT.shape[1] >= 2:
        fig = plot_combining_gain_M(
            meas.srs_chT,
            srs_chF_lin_interp=meas.srs_chF_lin_interp,
            true_distance_m=meas.meta.distance_m,
        )
        fig.savefig(os.path.join(out_dir, f"{base}__07_combining.png"), dpi=110)
        plt.close(fig)

    print(f"  Plots saved as {out_dir}/{base}__*.png")


# ============================================================================
# Step 4: Cross-folder
# ============================================================================

def step4_cross(summaries: list[dict], out_dir: str):
    print(f"\n=== Step 4: Cross-folder analysis ===")
    os.makedirs(out_dir, exist_ok=True)

    fig = plot_snr_vs_gain(summaries)
    fig.savefig(os.path.join(out_dir, "_snr_vs_gain.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(out_dir, "_summaries.json"), "w") as fp:
        json.dump(
            [{k: (float(v) if isinstance(v, np.floating) else v)
              for k, v in s.items() if not isinstance(v, np.ndarray)}
             for s in summaries],
            fp, indent=2, default=str,
        )
    print(f"  Saved _snr_vs_gain.png and _summaries.json under {out_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=== OAI RTT dataset explorer ===")
    print(f"  DATASET_ROOT = {DATASET_ROOT}")
    print(f"  OUT_DIR      = {OUT_DIR}")
    print(f"  PRB={SYS_PARAMS['PRB']}, FFT={SYS_PARAMS['fft_size']}, "
          f"fs={SYS_PARAMS['sampling_rate_hz']/1e6:.2f} MHz")
    print(f"  SRS pilot subcarriers : {len(SRS_SUBCARRIER_IDX)}")
    print(f"  Noise subcarriers     : {len(NOISE_SUBCARRIER_IDX)}")

    folders = step1_inventory(DATASET_ROOT)
    if not folders:
        return

    summaries = step2_summaries(folders)

    target = None
    if INSPECT_FOLDER is not None:
        for f in folders:
            if os.path.basename(f) == INSPECT_FOLDER:
                target = f
                break
        if target is None:
            print(f"\n[warn] INSPECT_FOLDER={INSPECT_FOLDER} not found.")
    if target is None:
        target = folders[0]

    step3_inspect(target, OUT_DIR)
    step4_cross(summaries, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
