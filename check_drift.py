"""
Diagnostic: track peak index over frames to spot clock drift.

Clock drift between gNB and UE causes the impulse response peak to slowly
move over time. If we see a monotonic trend in peak_index vs frame_index,
that's the signature. The paper's scheme (Section V of arXiv:2407.20463)
is designed to be robust to this; here we just visualise it.

Run this after the main explore_dataset.py has succeeded.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_loader import load_measurement, SYS_PARAMS
from dataset_analysis import REF_INDEX, parabolic_refine, C, FS

# ---- Edit if needed ----
TARGETS = [
    "./ssb_measurements/scenario_ssb_10m_ue_att_0_20240405T133530",
    "./ssb_measurements/scenario_ssb_10m_ue_att_0_20240405T133333",
    "./ssb_measurements/scenario_ssb_11m_ue_att_0_20240405T141916",
]

fig, axes = plt.subplots(len(TARGETS), 2, figsize=(14, 4 * len(TARGETS)))
if len(TARGETS) == 1:
    axes = axes.reshape(1, -1)

for ax_row, folder in zip(axes, TARGETS):
    if not os.path.isdir(folder):
        ax_row[0].set_title(f"NOT FOUND: {folder}")
        continue

    meas = load_measurement(folder)
    if meas.srs_chT is None:
        continue

    H = meas.srs_chT
    N_frames = H.shape[1]
    name = os.path.basename(folder)

    # Compute refined peak index per frame
    peaks = np.empty(N_frames, dtype=float)
    for m in range(N_frames):
        mag = np.abs(H[:, m])
        p_int = int(np.argmax(mag))
        peaks[m] = parabolic_refine(mag, p_int)

    peaks_rel = peaks - REF_INDEX                     # samples from time-zero
    ranges = peaks_rel * C / (2.0 * FS)               # metres

    # Time axis: 1 SRS measurement per slot. 30 kHz SCS -> 0.5 ms per slot
    # The paper says SRS sent every 20 slots, so ~ 10 ms per measurement.
    # We can't be 100% sure without the timestamp metadata, so just plot
    # vs frame index.
    t = np.arange(N_frames)

    # Left: peak index over frames
    ax_row[0].plot(t, peaks_rel, ".", markersize=1, alpha=0.5)
    # Running mean to see the trend
    window = max(50, N_frames // 50)
    running = np.convolve(peaks_rel,
                          np.ones(window)/window, mode="valid")
    ax_row[0].plot(np.arange(len(running)) + window//2, running,
                   "r-", lw=2, label=f"running mean (w={window})")
    ax_row[0].set_xlabel("Frame index (post-drop)")
    ax_row[0].set_ylabel("Peak index − N/2 (samples)")
    ax_row[0].set_title(f"{name[:50]}\nPeak position over time")
    ax_row[0].grid(True, alpha=0.3)
    ax_row[0].legend(fontsize=8)

    # Right: histogram of peak positions
    ax_row[1].hist(peaks_rel, bins=50, alpha=0.7, edgecolor="k")
    ax_row[1].axvline(np.mean(peaks_rel), color="r", linestyle="--",
                      label=f"mean = {np.mean(peaks_rel):.3f}")
    ax_row[1].set_xlabel("Peak index − N/2 (samples)")
    ax_row[1].set_ylabel("Count")
    ax_row[1].set_title(f"Histogram\n"
                        f"mean range = {np.mean(ranges):.3f} m, "
                        f"std = {np.std(ranges):.3f} m")
    ax_row[1].grid(True, alpha=0.3)
    ax_row[1].legend(fontsize=8)

    # Print summary
    print(f"\n=== {name} ===")
    print(f"  N frames: {N_frames}")
    print(f"  Peak (samples from N/2): "
          f"mean={np.mean(peaks_rel):.3f}, "
          f"std={np.std(peaks_rel):.3f}, "
          f"min={np.min(peaks_rel):.2f}, "
          f"max={np.max(peaks_rel):.2f}")
    print(f"  Range (m): mean={np.mean(ranges):.3f}, "
          f"std={np.std(ranges):.3f}")
    # Linear regression to quantify drift
    slope, intercept = np.polyfit(t, peaks_rel, 1)
    drift_per_1000_frames = slope * 1000
    print(f"  Linear drift: slope={slope:.6f} samples/frame  "
          f"→ {drift_per_1000_frames:.2f} samples per 1000 frames")
    if abs(drift_per_1000_frames) > 0.5:
        print(f"  *** SIGNIFICANT DRIFT DETECTED ***")

out_path = "./analysis_output/_clock_drift_diagnostic.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, dpi=110)
print(f"\nPlot saved: {out_path}")
