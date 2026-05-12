"""
Step 3 cross-check: compare freshly computed raw behavior features (before
z-score) against the ISMAR NPZ (MOMENT_ready_v2/{pid}_{scene}.npz), which
were regenerated 2026-03-12 with the same fixed behavior_features.py.

If the fix is correctly applied, the freshly-computed dist_min / dist_mean /
count_* values per frame must MATCH the ISMAR NPZ values, within rounding /
windowing alignment.

This isolates the bug fix verification: we don't trust the buggy preprocessed
data; we trust the freshly-fixed ISMAR NPZ as ground truth for the 9 channels
they share.
"""
import sys
sys.path.insert(0, r'c:/Users/user/code/SDPhysiology')

import numpy as np
import pandas as pd
from pathlib import Path

from behavior_features import ColumnMapping, compute_agent_player_relations
from ml_dataloader import downsample_120_to_60_scenewise

DATA_DIR = r'D:\Labroom\SDPhysiology\Data\Processed\processed_individual_anonymized'
NPZ_DIR  = Path(r'D:\Labroom\SDPhysiology\Data\MOMENT_ready_v2')
PID = '001'
SCENES = ['Elevator1', 'Outside', 'Hallway', 'Elevator2', 'Hall']

cols = ColumnMapping()

print(f"Loading PID {PID} raw...")
main_df = pd.read_pickle(f"{DATA_DIR}/{PID}_Main.pkl")
agent_df = pd.read_pickle(f"{DATA_DIR}/{PID}_Agent.pkl")

# Downsample 120→60
main_60 = downsample_120_to_60_scenewise(main_df, cols=cols, factor=2)
agent_60 = downsample_120_to_60_scenewise(agent_df, cols=cols, factor=2)

# Strip leak cols (anxiety etc) before compute
main_60_feat = main_60.drop(columns=[c for c in main_60.columns
                                     if c.lower() in {'anxiety','y_cont','y_label','target'}],
                            errors='ignore')

print("Running compute_agent_player_relations (FIXED)...")
df_ts = compute_agent_player_relations(
    main_60_feat, agent_60, cols=cols, dt=1.0/60.0
)
print(f"df_ts shape: {df_ts.shape}, cols sample: {list(df_ts.columns)[:15]}")
print()

# Compare with ISMAR NPZ for each scene
SHARED_CHANNELS = ['speed', 'head_rot_speed', 'dist_min', 'dist_mean',
                   'count_agents', 'count_personal', 'count_social',
                   'count_fov', 'count_approach']

print("=== Cross-check: our raw (60Hz framewise) vs ISMAR NPZ ===")
print(f"{'scene':<10} {'channel':<18} | {'ours mean':>10} {'NPZ mean':>10} {'diff':>9} "
      f"| {'ours std':>9} {'NPZ std':>9}")
print('-'*100)

for sc in SCENES:
    npz_path = NPZ_DIR / f"{PID}_{sc}.npz"
    if not npz_path.exists():
        print(f"{sc}: NPZ not found"); continue
    d = np.load(npz_path, allow_pickle=True)
    npz_beh = d['behavior']
    npz_cols = list(d['behavior_cols'])
    T_npz = npz_beh.shape[0]

    ours_sc = df_ts[df_ts[cols.scene] == sc].sort_values(cols.frame).reset_index(drop=True)
    T_ours = len(ours_sc)
    if T_ours == 0:
        print(f"{sc}: ours has 0 rows"); continue

    # Align by truncating to the shorter
    Tmin = min(T_ours, T_npz)
    print(f"\n--- {sc}  (T_ours={T_ours}, T_npz={T_npz}, comparing first {Tmin}) ---")
    for ch in SHARED_CHANNELS:
        if ch not in ours_sc.columns:
            print(f"  {sc:<10} {ch:<18}: missing in ours"); continue
        if ch not in npz_cols:
            print(f"  {sc:<10} {ch:<18}: missing in NPZ"); continue
        ours_v = ours_sc[ch].to_numpy(dtype=float)[:Tmin]
        npz_v = npz_beh[:Tmin, npz_cols.index(ch)].astype(float)

        # Skip NaN frames
        m = np.isfinite(ours_v) & np.isfinite(npz_v)
        if m.sum() < 10:
            print(f"  {sc:<10} {ch:<18}: too few finite frames"); continue
        diff = np.abs(ours_v[m] - npz_v[m])
        match_pct = (diff < 1e-4).mean() * 100
        max_diff = diff.max()
        print(f"  {sc:<10} {ch:<18} | {ours_v[m].mean():>+10.3f} {npz_v[m].mean():>+10.3f} "
              f"{(ours_v[m]-npz_v[m]).mean():>+9.4f} "
              f"| {ours_v[m].std():>9.3f} {npz_v[m].std():>9.3f}  | "
              f"max_diff={max_diff:.4e} match%={match_pct:.1f}")
