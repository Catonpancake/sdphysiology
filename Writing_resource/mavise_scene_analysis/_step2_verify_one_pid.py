"""
Step 2 verify run: run the FIXED behavior_features pipeline on PID 001 only.

Goal:
  1. Confirm process_one_behavior_pid_ts() runs end-to-end on fixed code.
  2. Inspect dist_min and count_personal per scene — must have variance in
     Outside / Hall (which collapsed to 0 in buggy version).
  3. Save output to a temporary folder for inspection.

NO leakage by construction:
  - process_one_behavior_pid_ts() drops `anxiety`, `y_cont`, `y_label`, ...
    in (D)-(F) before computing relations.
  - Output features never include the target.
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, r'c:/Users/user/code/SDPhysiology')

import numpy as np
import pandas as pd

from behavior_features import ColumnMapping
from ml_dataloader import process_one_behavior_pid_ts

DATA_DIR = r'D:\Labroom\SDPhysiology\Data\Processed\processed_individual_anonymized'
PID = '001'
TARGET_SCENES = ['Elevator1', 'Outside', 'Hallway', 'Elevator2', 'Hall']

cols = ColumnMapping()

t0 = time.time()
print(f"Running fixed pipeline on PID {PID}...")
out = process_one_behavior_pid_ts(
    pid_str=PID,
    cols=cols,
    data_dir=DATA_DIR,
    target_scenes=TARGET_SCENES,
    fs_beh=60.0,
    window_seconds=5.0,
    stride_seconds=2.0,
)
print(f"Elapsed: {time.time()-t0:.1f}s\n")

if out is None:
    print("FATAL: out is None"); sys.exit(1)

X_beh, y_win, pid_arr, scene_arr, widx_arr, feat_names = out
feat_names = list(feat_names)

print(f"X_beh shape : {X_beh.shape}   (N_windows, T=300, C={X_beh.shape[2]})")
print(f"y_win shape : {y_win.shape}")
print(f"PIDs        : {np.unique(pid_arr)}")
print(f"Scenes      : {dict(zip(*np.unique(scene_arr, return_counts=True)))}")
print(f"n_features  : {len(feat_names)}")
print()

# Confirm no leak columns
leak_substr = ('anxiety', 'y_cont', 'y_label', 'target')
leak_found = [f for f in feat_names if any(s in str(f).lower() for s in leak_substr)]
if leak_found:
    print(f"  WARNING leak-like feature names: {leak_found}")
else:
    print(f"  No leak feature names detected (checked {len(feat_names)} features)")
print()

# Check the critical features per scene
KEY_FEATS = ['dist_min', 'dist_mean', 'count_personal', 'count_social',
             'count_intimate', 'count_public', 'count_fov', 'count_approach',
             'backward_flag', 'speed']

print("=== Per-scene stats for KEY features ===")
print(f"{'feature':<20} {'scene':<11} {'mean':>9} {'std':>9} {'min':>8} {'max':>8} {'n_uniq':>8}")
for fname in KEY_FEATS:
    if fname not in feat_names:
        print(f"  {fname:<20}  NOT IN feature list")
        continue
    fi = feat_names.index(fname)
    for s in TARGET_SCENES:
        m = scene_arr == s
        if m.sum() == 0: continue
        Xs = X_beh[m, :, fi].flatten()
        Xs = Xs[np.isfinite(Xs)]
        if len(Xs) == 0:
            print(f"  {fname:<20} {s:<11}   ALL NaN")
        else:
            print(f"  {fname:<20} {s:<11} {Xs.mean():+.3f} {Xs.std():>9.4f} "
                  f"{Xs.min():>+8.3f} {Xs.max():>+8.3f} {len(np.unique(Xs)):>8}")
    print()

# Save to verify_out folder
OUT = Path(r'c:/Users/user/code/SDPhysiology/Writing_resource/mavise_scene_analysis/_step2_verify_out')
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / 'X_beh.npy', X_beh)
np.save(OUT / 'y_win.npy', y_win)
np.save(OUT / 'scene_arr.npy', scene_arr)
np.save(OUT / 'feat_names.npy', np.array(feat_names, dtype=object))
print(f"Saved to: {OUT}")
print(f"  feat_names ({len(feat_names)}): {feat_names}")
