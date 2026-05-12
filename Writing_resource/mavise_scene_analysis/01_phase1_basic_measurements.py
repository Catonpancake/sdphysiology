"""
Phase 1: foundational descriptive measurements for MAVISE scene-dependent
interpretation.

Outputs (this single script writes 4 result CSVs + 1 paired-test CSV +
1 per-PID trajectory CSV):
  01_y_variance.csv               per-scene Y variance (window + per-PID-mean)
  02_anxiety_trajectory_perpid.csv  per-PID per-scene anxiety mean (wide)
  02_anxiety_trajectory_summary.csv per-scene anxiety mean summary
  02_elev1_vs_elev2_paired.csv      paired t-test Elev1 vs Elev2
  03_backward_flag_rate.csv       per-scene backward_flag activation rate
  04_proximity_variance.csv       per-scene dist_min variance

Data:
  ml_processed_behavior_all/ — all 5 protocol scenes preserved:
    Elevator1 / Outside / Hallway / Elevator2 / Hall
  X: (N, 300, 67)  — 67ch features × 300 timesteps per 20-s window
  y: (N,)          — per-PID z-scored anxiety target
  pid, scene       — per-window labels

Leakage check:
  - NO model training, NO train/test split.
  - Pure descriptive statistics on all 106 PIDs.
  - Output is independent of any model's predictions or feature selection.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import sys

ROOT = Path('c:/Users/user/code/SDPhysiology/ml_processed_behavior_all')
OUT  = Path('c:/Users/user/code/SDPhysiology/Writing_resource/mavise_scene_analysis')
OUT.mkdir(parents=True, exist_ok=True)

SCENES_ORDER = ['Elevator1', 'Outside', 'Hallway', 'Elevator2', 'Hall']

# ── Load ───────────────────────────────────────────────────────────────────
X     = np.load(ROOT / 'X_array.npy')
y     = np.load(ROOT / 'y_array.npy')
pid   = np.load(ROOT / 'pid_array.npy',   allow_pickle=True)
scene = np.load(ROOT / 'scene_array.npy', allow_pickle=True)
feats = list(np.load(ROOT / 'feature_tag_list.npy', allow_pickle=True))

IDX_BW   = feats.index('beh::backward_flag')
IDX_DIST = feats.index('beh::dist_min')

print(f"X={X.shape}  y={y.shape}  PIDs={len(np.unique(pid))}")
print(f"feature idx: backward_flag={IDX_BW}, dist_min={IDX_DIST}")
print(f"backward_flag unique values (sample): "
      f"{np.unique(X[:1000, :, IDX_BW].flatten())[:10]}")

# Helper: report value as median (Q1, Q3)
def fmt_iqr(med, q1, q3):
    return f"{med:.3f} ({q1:.3f}, {q3:.3f})"

# ── 01: per-scene Y variance ───────────────────────────────────────────────
print("\n=== 01: Per-scene Y variance ===")
rows = []
for s in SCENES_ORDER:
    m = scene == s
    y_s   = y[m]
    pid_s = pid[m]
    pid_stds, pid_means = [], []
    for p in np.unique(pid_s):
        y_p = y_s[pid_s == p]
        if len(y_p) >= 2:
            pid_stds.append(np.std(y_p))
            pid_means.append(np.mean(y_p))
    pid_stds  = np.array(pid_stds)
    pid_means = np.array(pid_means)
    rows.append(dict(
        scene=s,
        protocol_order=SCENES_ORDER.index(s) + 1,
        n_windows=int(m.sum()),
        n_pids=int(len(np.unique(pid_s))),
        y_std_window=float(np.std(y_s)),
        y_mean_window=float(np.mean(y_s)),
        y_std_pidmean_median=float(np.median(pid_stds)),
        y_std_pidmean_q25=float(np.quantile(pid_stds, 0.25)),
        y_std_pidmean_q75=float(np.quantile(pid_stds, 0.75)),
        y_mean_pidmean_median=float(np.median(pid_means)),
        y_mean_pidmean_q25=float(np.quantile(pid_means, 0.25)),
        y_mean_pidmean_q75=float(np.quantile(pid_means, 0.75)),
    ))
df01 = pd.DataFrame(rows)
df01.to_csv(OUT / '01_y_variance.csv', index=False)
print(df01.to_string(index=False, float_format='%.4f'))

# ── 02: Per-PID per-scene anxiety mean + Elev1 vs Elev2 paired ─────────────
print("\n=== 02: Anxiety trajectory ===")
all_pids = sorted(np.unique(pid))
trajectory_rows = []
for p in all_pids:
    row = {'pid': p}
    for s in SCENES_ORDER:
        y_ps = y[(pid == p) & (scene == s)]
        row[s] = float(np.mean(y_ps)) if len(y_ps) else np.nan
        row[s + '_n_windows'] = int(len(y_ps))
    trajectory_rows.append(row)
df02_traj = pd.DataFrame(trajectory_rows)
df02_traj.to_csv(OUT / '02_anxiety_trajectory_perpid.csv', index=False)

# Per-scene summary
sum_rows = []
for s in SCENES_ORDER:
    vals = df02_traj[s].dropna().values
    sum_rows.append(dict(
        scene=s,
        protocol_order=SCENES_ORDER.index(s) + 1,
        n_pids=int(len(vals)),
        y_pidmean_mean=float(np.mean(vals)),
        y_pidmean_std=float(np.std(vals)),
        y_pidmean_median=float(np.median(vals)),
        y_pidmean_q25=float(np.quantile(vals, 0.25)),
        y_pidmean_q75=float(np.quantile(vals, 0.75)),
        y_pidmean_sem=float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
    ))
df02_sum = pd.DataFrame(sum_rows)
df02_sum.to_csv(OUT / '02_anxiety_trajectory_summary.csv', index=False)
print(df02_sum.to_string(index=False, float_format='%.4f'))

# Paired Elev1 vs Elev2
paired = df02_traj[['pid', 'Elevator1', 'Elevator2']].dropna()
diff = paired['Elevator2'].values - paired['Elevator1'].values
t_stat, p_val = stats.ttest_rel(paired['Elevator2'], paired['Elevator1'])
mean_diff = float(np.mean(diff))
std_diff  = float(np.std(diff, ddof=1))
cohens_d  = mean_diff / std_diff if std_diff > 0 else np.nan
# Wilcoxon as robust alternative
w_stat, w_p = stats.wilcoxon(paired['Elevator2'], paired['Elevator1'])
paired_summary = pd.DataFrame([dict(
    comparison='Elevator2 minus Elevator1 (per-PID anxiety mean)',
    n_pids_paired=int(len(paired)),
    mean_diff=mean_diff,
    std_diff=std_diff,
    median_diff=float(np.median(diff)),
    cohens_d=float(cohens_d),
    t_stat=float(t_stat),
    p_value_t=float(p_val),
    wilcoxon_W=float(w_stat),
    p_value_wilcoxon=float(w_p),
)])
paired_summary.to_csv(OUT / '02_elev1_vs_elev2_paired.csv', index=False)
print(f"\n02b paired Elev2 - Elev1: n={len(paired)}, "
      f"mean_diff={mean_diff:+.4f}, t={t_stat:.3f}, p={p_val:.4g}, "
      f"Cohen's d={cohens_d:+.3f}, Wilcoxon p={w_p:.4g}")

# ── 03: backward_flag activation rate ──────────────────────────────────────
print("\n=== 03: backward_flag activation rate ===")
bw_per_window = X[:, :, IDX_BW].mean(axis=1)  # fraction of 300 timesteps with flag=1
rows_bw = []
for s in SCENES_ORDER:
    m = scene == s
    bw_s   = bw_per_window[m]
    pid_s  = pid[m]
    pid_rates = []
    for p in np.unique(pid_s):
        bw_p = bw_s[pid_s == p]
        pid_rates.append(np.mean(bw_p))
    pid_rates = np.array(pid_rates)
    rows_bw.append(dict(
        scene=s,
        protocol_order=SCENES_ORDER.index(s) + 1,
        n_windows=int(m.sum()),
        n_pids=int(len(np.unique(pid_s))),
        rate_window=float(np.mean(bw_s)),
        rate_pid_mean=float(np.mean(pid_rates)),
        rate_pid_median=float(np.median(pid_rates)),
        rate_pid_std=float(np.std(pid_rates)),
        rate_pid_q25=float(np.quantile(pid_rates, 0.25)),
        rate_pid_q75=float(np.quantile(pid_rates, 0.75)),
        rate_pid_sem=float(np.std(pid_rates, ddof=1) / np.sqrt(len(pid_rates))),
    ))
df03 = pd.DataFrame(rows_bw)
df03.to_csv(OUT / '03_backward_flag_rate.csv', index=False)
print(df03.to_string(index=False, float_format='%.4f'))

# ── 04: dist_min variance ──────────────────────────────────────────────────
print("\n=== 04: dist_min variance ===")
rows_d = []
for s in SCENES_ORDER:
    m = scene == s
    pid_s = pid[m]
    X_s = X[m, :, IDX_DIST]  # (n_win_s, 300)
    pid_stds, pid_means = [], []
    for p in np.unique(pid_s):
        Xp = X_s[pid_s == p].flatten()
        Xp = Xp[np.isfinite(Xp)]
        if len(Xp) >= 2:
            pid_stds.append(np.std(Xp))
            pid_means.append(np.mean(Xp))
    pid_stds  = np.array(pid_stds)
    pid_means = np.array(pid_means)
    rows_d.append(dict(
        scene=s,
        protocol_order=SCENES_ORDER.index(s) + 1,
        dist_std_window=float(np.std(X_s[np.isfinite(X_s)])),
        dist_mean_window=float(np.mean(X_s[np.isfinite(X_s)])),
        dist_std_pidmean_median=float(np.median(pid_stds)),
        dist_std_pidmean_q25=float(np.quantile(pid_stds, 0.25)),
        dist_std_pidmean_q75=float(np.quantile(pid_stds, 0.75)),
        dist_mean_pidmean_median=float(np.median(pid_means)),
    ))
df04 = pd.DataFrame(rows_d)
df04.to_csv(OUT / '04_proximity_variance.csv', index=False)
print(df04.to_string(index=False, float_format='%.4f'))

print(f"\nDone. CSVs in: {OUT}")
