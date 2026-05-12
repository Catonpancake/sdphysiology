"""
_autonomous_master.py — MAVISE clean-data regeneration + baseline + analysis.

Runs unattended for ~6 hours, writes everything to disk, logs all events,
generates a final consolidated report.

Phases:
  A. Re-extract 56ch behavior with FIXED behavior_features.py for all 106 PIDs,
     merge with existing 11ch physio → ml_processed_behavior_{scene}/ (67ch).
  B. Re-run all 6 MAVISE baseline scripts.
  C. Re-run Phase 1 measurements + Phase 2 three-factor + Phase 3 4-panel figure
     on clean data.

Safety:
  - Each phase wrapped in try/except. Hard errors stop that phase only.
  - Each subprocess gets a 90-min hard cap.
  - All stdout/stderr captured to per-step log files.
  - Final report aggregates everything.

Output:
  Writing_resource/mavise_scene_analysis/autonomous_run/
    AUTONOMOUS_RUN_LOG.md          ← timestamped event log
    FINAL_REPORT.md                ← consolidated results & anomaly summary
    phase_A/                       ← extraction & merge artifacts
    phase_B/                       ← baseline CSV outputs (copies)
    phase_C/                       ← analysis CSV/PNG outputs
"""
from __future__ import annotations
import os, sys, time, json, shutil, subprocess, traceback
from pathlib import Path

# Force UTF-8 for stdout so em-dashes etc. don't crash on cp949 console (Windows-KR)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = Path(r'c:/Users/user/code/SDPhysiology')
sys.path.insert(0, str(ROOT))

OUT_BASE = ROOT / 'Writing_resource/mavise_scene_analysis/autonomous_run'
PHASE_A_DIR = OUT_BASE / 'phase_A'
PHASE_B_DIR = OUT_BASE / 'phase_B'
PHASE_C_DIR = OUT_BASE / 'phase_C'
LOG_FILE = OUT_BASE / 'AUTONOMOUS_RUN_LOG.md'
FINAL_REPORT = OUT_BASE / 'FINAL_REPORT.md'

for d in [OUT_BASE, PHASE_A_DIR, PHASE_B_DIR, PHASE_C_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PYTHON = r'C:/Users/user/anaconda3/envs/ml_env/python.exe'

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RAW_DATA_DIR = r'D:/Labroom/SDPhysiology/Data/Processed/processed_individual_anonymized'
PHYSIO_BASE = ROOT  # ml_processed_{scene}/  is at project root
SCENE_GROUPS = {
    # Group name : list of fine-grained scene tags that fall under it
    'Elevator': ['Elevator1', 'Elevator2'],
    'Hallway':  ['Hallway'],
    'Hall':     ['Hall'],
    'Outside':  ['Outside'],
}
ALL_SCENES_FINE = ['Elevator1', 'Outside', 'Hallway', 'Elevator2', 'Hall']

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.0
FS_BEHAV = 60.0
N_JOBS_EXTRACT = 4

PHASE_TIMEOUT_S = 90 * 60   # 90 min hard cap per subprocess

BASELINES = [
    # (name, script)
    ('classical_baseline', 'eval_mavise_classical_baseline.py'),
    ('classical_v2',        'eval_mavise_classical_v2.py'),
    ('hv',                  'eval_mavise_hv.py'),
    ('directions',          'eval_mavise_directions.py'),
    ('cnn',                 'eval_mavise_cnn_baseline.py'),
    ('rnn',                 'eval_mavise_rnn_baseline.py'),
]

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
STATE = {'errors': [], 'warnings': [], 'anomalies': [], 'phases': {}}


def log(msg: str, level: str = 'info') -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] [{level.upper()}] {msg}'
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Fallback for Windows-KR cp949 console
        print(line.encode('ascii', errors='replace').decode('ascii'), flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    if level == 'error':
        STATE['errors'].append((ts, msg))
    elif level == 'warning':
        STATE['warnings'].append((ts, msg))
    elif level == 'anomaly':
        STATE['anomalies'].append((ts, msg))


def section(title: str):
    bar = '=' * 78
    log('\n' + bar, level='info')
    log(title, level='info')
    log(bar, level='info')


# --------------------------------------------------------------------------
# Phase A — extraction & merge
# --------------------------------------------------------------------------
def phase_a():
    section('PHASE A — Re-extract behavior 56ch & merge with physio → 67ch')
    t_start = time.time()
    a_result = {}

    # ---- A.1 Generate behavior_all/ (56ch behavior for all 5 scenes) ----
    log('[A.1] Running build_behavior_windows_ts_60hz for all 5 scenes ...')
    from behavior_features import ColumnMapping
    from ml_dataloader import build_behavior_windows_ts_60hz

    out_beh_all = ROOT / 'behavior_all'
    out_beh_all.mkdir(exist_ok=True)
    t_a1 = time.time()
    try:
        bh = build_behavior_windows_ts_60hz(
            data_dir=RAW_DATA_DIR,
            out_dir=str(out_beh_all),
            target_scenes=ALL_SCENES_FINE,
            fs_beh=FS_BEHAV,
            window_seconds=WINDOW_SECONDS,
            stride_seconds=STRIDE_SECONDS,
            n_jobs=N_JOBS_EXTRACT,
            cols=ColumnMapping(),
        )
        elapsed = time.time() - t_a1
        log(f'[A.1] behavior_all/ saved. X={bh["X"].shape}, '
            f'PIDs={len(np.unique(bh["pid"]))}, skipped={len(bh.get("skipped", []))}, '
            f'elapsed={elapsed:.0f}s')
        a_result['A1'] = {
            'X_shape': tuple(bh['X'].shape),
            'n_pids': int(len(np.unique(bh['pid']))),
            'skipped': bh.get('skipped', []),
            'elapsed': elapsed,
        }
        if len(bh.get('skipped', [])) > 5:
            log(f'[A.1] WARNING: {len(bh["skipped"])} PIDs skipped during extraction',
                level='warning')
    except Exception as e:
        log(f'[A.1] FATAL: extraction failed: {e}\n{traceback.format_exc()}',
            level='error')
        a_result['A1'] = {'status': 'failed', 'error': str(e)}
        return a_result

    # ---- A.2 Split behavior_all → behavior_{scene}/ ----
    log('[A.2] Splitting behavior_all → per-scene behavior_{scene}/ ...')
    X_all = np.load(out_beh_all / 'X_array.npy')
    y_all = np.load(out_beh_all / 'y_array.npy')
    pid_all = np.load(out_beh_all / 'pid_array.npy', allow_pickle=True)
    scene_all = np.load(out_beh_all / 'scene_array.npy', allow_pickle=True)
    widx_all = np.load(out_beh_all / 'windex_array.npy', allow_pickle=True)
    feat_b = np.load(out_beh_all / 'feature_tag_list.npy', allow_pickle=True)

    for group_name, scene_tags in SCENE_GROUPS.items():
        mask = np.isin(scene_all, scene_tags)
        out_dir = ROOT / f'behavior_{group_name}'
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / 'X_array.npy', X_all[mask])
        np.save(out_dir / 'y_array.npy', y_all[mask])
        np.save(out_dir / 'pid_array.npy', pid_all[mask])
        np.save(out_dir / 'scene_array.npy', scene_all[mask])
        np.save(out_dir / 'windex_array.npy', widx_all[mask])
        np.save(out_dir / 'feature_tag_list.npy', feat_b)
        log(f'   [A.2] behavior_{group_name}: N={mask.sum()}, scenes={scene_tags}')

    # ---- A.3 Verify physio ml_processed_{scene}/ exist and look sane ----
    log('[A.3] Verifying physio data (ml_processed_{scene}/) ...')
    physio_summary = {}
    for group_name in SCENE_GROUPS:
        d = PHYSIO_BASE / f'ml_processed_{group_name}'
        if not d.exists():
            log(f'   [A.3] WARNING: {d} missing!', level='warning')
            physio_summary[group_name] = {'exists': False}
            continue
        Xp = np.load(d / 'X_array.npy', mmap_mode='r')
        physio_summary[group_name] = {
            'exists': True,
            'X_shape': tuple(Xp.shape),
            'n_pids': int(len(np.unique(np.load(d / 'pid_array.npy', allow_pickle=True)))),
        }
        log(f'   [A.3] ml_processed_{group_name}: X={Xp.shape}')
        # Sanity: no all-NaN
        sample = Xp[: min(10, Xp.shape[0])]
        if not np.isfinite(sample).any():
            log(f'   [A.3] ANOMALY: physio {group_name} sample all NaN', level='anomaly')
    a_result['A3_physio'] = physio_summary

    # Check if physio is (N,T,C) or (N,C,T) — should be (N,T,C) for merge
    sample_physio = np.load(PHYSIO_BASE / f'ml_processed_{list(SCENE_GROUPS)[0]}' / 'X_array.npy',
                            mmap_mode='r')
    if sample_physio.shape[1] < sample_physio.shape[2]:
        log(f'[A.3] Physio shape {sample_physio.shape}: appears (N,C,T). Will transpose.',
            level='warning')
        physio_transpose = True
    else:
        log(f'[A.3] Physio shape {sample_physio.shape}: appears (N,T,C). OK.')
        physio_transpose = False
    del sample_physio

    # ---- A.4 Merge per scene → ml_processed_behavior_{scene}/ ----
    log('[A.4] Merging physio + behavior → 67ch per scene ...')
    merge_results = {}
    feat_b_list = list(feat_b)
    for group_name in SCENE_GROUPS:
        log(f'   [A.4] Merging {group_name} ...')
        try:
            phys_dir = PHYSIO_BASE / f'ml_processed_{group_name}'
            beh_dir = ROOT / f'behavior_{group_name}'
            out_dir = ROOT / f'ml_processed_behavior_{group_name}'
            out_dir.mkdir(exist_ok=True)

            X_p = np.load(phys_dir / 'X_array.npy')
            if physio_transpose:
                X_p = np.transpose(X_p, (0, 2, 1))
            y_p = np.load(phys_dir / 'y_array.npy')
            pid_p = np.load(phys_dir / 'pid_array.npy', allow_pickle=True)
            scene_p = np.load(phys_dir / 'scene_array.npy', allow_pickle=True)
            widx_p = np.load(phys_dir / 'windex_array.npy', allow_pickle=True)
            feat_p = list(np.load(phys_dir / 'feature_tag_list.npy', allow_pickle=True))

            X_b = np.load(beh_dir / 'X_array.npy')
            y_b = np.load(beh_dir / 'y_array.npy')
            pid_b = np.load(beh_dir / 'pid_array.npy', allow_pickle=True)
            scene_b = np.load(beh_dir / 'scene_array.npy', allow_pickle=True)
            widx_b = np.load(beh_dir / 'windex_array.npy', allow_pickle=True)

            # Key-based intersection
            key_p = pd.MultiIndex.from_arrays(
                [pid_p.astype(str), scene_p.astype(str), widx_p.astype(int)],
                names=['pid', 'scene', 'widx'])
            key_b = pd.MultiIndex.from_arrays(
                [pid_b.astype(str), scene_b.astype(str), widx_b.astype(int)],
                names=['pid', 'scene', 'widx'])
            common = key_p.intersection(key_b)
            idx_p = key_p.get_indexer(common)
            idx_b = key_b.get_indexer(common)

            if X_p.shape[1] != X_b.shape[1]:
                log(f'      [A.4] T mismatch: physio {X_p.shape}, beh {X_b.shape}',
                    level='error')
                continue

            X_p_sel = X_p[idx_p]; y_p_sel = y_p[idx_p]
            X_b_sel = X_b[idx_b]; y_b_sel = y_b[idx_b]
            pid_sel = pid_p[idx_p]; scene_sel = scene_p[idx_p]; widx_sel = widx_p[idx_p]

            # y check
            if y_p_sel.shape == y_b_sel.shape:
                d = float(np.nanmax(np.abs(y_p_sel - y_b_sel)))
                if d > 1e-3:
                    log(f'      [A.4] y_p vs y_b differ max={d:.4f}, using physio y',
                        level='warning')

            X_merged = np.concatenate([X_p_sel, X_b_sel], axis=2)
            feat_merged = [f'physio::{f}' for f in feat_p] + [f'beh::{f}' for f in feat_b_list]

            np.save(out_dir / 'X_array.npy', X_merged.astype(np.float32))
            np.save(out_dir / 'y_array.npy', y_p_sel.astype(np.float32))
            np.save(out_dir / 'pid_array.npy', pid_sel)
            np.save(out_dir / 'scene_array.npy', scene_sel)
            np.save(out_dir / 'windex_array.npy', widx_sel)
            np.save(out_dir / 'feature_tag_list.npy', np.array(feat_merged, dtype=object))

            log(f'      [A.4] {group_name}: X_merged={X_merged.shape}, '
                f'common={len(common)}/(p={len(key_p)}/b={len(key_b)})')
            merge_results[group_name] = {
                'X_shape': tuple(X_merged.shape),
                'n_common': int(len(common)),
                'n_physio': int(len(key_p)),
                'n_behavior': int(len(key_b)),
            }
        except Exception as e:
            log(f'      [A.4] {group_name} merge FAILED: {e}', level='error')
            merge_results[group_name] = {'status': 'failed', 'error': str(e)}
    a_result['A4_merge'] = merge_results

    # ---- A.5 Combine into ml_processed_behavior_all/ ----
    log('[A.5] Combining per-scene merged → ml_processed_behavior_all/ ...')
    try:
        X_all_m, y_all_m, pid_all_m, scene_all_m, widx_all_m = [], [], [], [], []
        feat_ref = None
        for group_name in SCENE_GROUPS:
            d = ROOT / f'ml_processed_behavior_{group_name}'
            if not (d / 'X_array.npy').exists():
                continue
            X_all_m.append(np.load(d / 'X_array.npy'))
            y_all_m.append(np.load(d / 'y_array.npy'))
            pid_all_m.append(np.load(d / 'pid_array.npy', allow_pickle=True))
            scene_all_m.append(np.load(d / 'scene_array.npy', allow_pickle=True))
            widx_all_m.append(np.load(d / 'windex_array.npy', allow_pickle=True))
            if feat_ref is None:
                feat_ref = np.load(d / 'feature_tag_list.npy', allow_pickle=True)
        Xa = np.concatenate(X_all_m, axis=0)
        out_all = ROOT / 'ml_processed_behavior_all'
        out_all.mkdir(exist_ok=True)
        np.save(out_all / 'X_array.npy', Xa.astype(np.float32))
        np.save(out_all / 'y_array.npy', np.concatenate(y_all_m).astype(np.float32))
        np.save(out_all / 'pid_array.npy', np.concatenate(pid_all_m))
        np.save(out_all / 'scene_array.npy', np.concatenate(scene_all_m))
        np.save(out_all / 'windex_array.npy', np.concatenate(widx_all_m))
        np.save(out_all / 'feature_tag_list.npy', feat_ref)
        log(f'   [A.5] ml_processed_behavior_all/ saved: X={Xa.shape}')
        a_result['A5_all'] = {'X_shape': tuple(Xa.shape)}
    except Exception as e:
        log(f'   [A.5] Combine failed: {e}', level='error')
        a_result['A5_all'] = {'status': 'failed', 'error': str(e)}

    # ---- A.6 Final verification stats on regenerated data ----
    log('[A.6] Verification stats on regenerated data ...')
    verify_rows = []
    try:
        d_all = ROOT / 'ml_processed_behavior_all'
        X = np.load(d_all / 'X_array.npy')
        y = np.load(d_all / 'y_array.npy')
        scene = np.load(d_all / 'scene_array.npy', allow_pickle=True)
        feats = list(np.load(d_all / 'feature_tag_list.npy', allow_pickle=True))
        idx_dist_min = feats.index('beh::dist_min')
        idx_count_personal = feats.index('beh::count_personal')
        idx_backward = feats.index('beh::backward_flag')
        idx_speed = feats.index('beh::speed')

        for sc in ALL_SCENES_FINE:
            m = scene == sc
            row = {'scene': sc, 'n_windows': int(m.sum())}
            for name, ix in [('dist_min', idx_dist_min),
                             ('count_personal', idx_count_personal),
                             ('backward_flag', idx_backward),
                             ('speed', idx_speed)]:
                vals = X[m, :, ix].flatten()
                vals = vals[np.isfinite(vals)]
                row[f'{name}_mean'] = float(np.mean(vals)) if len(vals) else np.nan
                row[f'{name}_std']  = float(np.std(vals)) if len(vals) else np.nan
                row[f'{name}_n_uniq'] = int(len(np.unique(vals)))
            verify_rows.append(row)
        verify_df = pd.DataFrame(verify_rows)
        verify_df.to_csv(PHASE_A_DIR / 'verification_stats.csv', index=False)
        log('   [A.6] verification_stats.csv written')
        log('\n' + verify_df.to_string(index=False, float_format='%.4f'))

        # Anomaly: dist_min still all-zero in any scene?
        for r in verify_rows:
            if r['dist_min_std'] < 1e-3 or r['dist_min_n_uniq'] < 5:
                log(f"   [A.6] ANOMALY: {r['scene']} dist_min still degenerate "
                    f"(std={r['dist_min_std']:.4f}, n_uniq={r['dist_min_n_uniq']})",
                    level='anomaly')
            if r['count_personal_std'] < 1e-3 or r['count_personal_n_uniq'] < 5:
                log(f"   [A.6] ANOMALY: {r['scene']} count_personal still degenerate",
                    level='anomaly')
    except Exception as e:
        log(f'   [A.6] Verification failed: {e}', level='error')
        a_result['A6_verify'] = {'status': 'failed', 'error': str(e)}

    a_result['elapsed_total'] = time.time() - t_start
    a_result['verify_stats'] = verify_rows
    return a_result


# --------------------------------------------------------------------------
# Phase B — re-run baselines via subprocess
# --------------------------------------------------------------------------
def phase_b():
    section('PHASE B — Re-run baselines on clean data')
    t_start = time.time()
    b_result = {}
    for name, script in BASELINES:
        log(f'[B] Running {name} ({script}) ...')
        t0 = time.time()
        logfile = PHASE_B_DIR / f'{name}.log'
        try:
            with open(logfile, 'w', encoding='utf-8') as lf:
                proc = subprocess.run(
                    [PYTHON, str(ROOT / script)],
                    stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(ROOT), timeout=PHASE_TIMEOUT_S, encoding='utf-8',
                    errors='replace',
                )
            elapsed = time.time() - t0
            status = 'ok' if proc.returncode == 0 else f'rc={proc.returncode}'
            b_result[name] = {'status': status, 'elapsed': elapsed,
                              'log': str(logfile)}
            log(f'[B] {name} → {status} ({elapsed:.0f}s)')
            if proc.returncode != 0:
                log(f'[B] WARNING: {name} returned {proc.returncode}', level='warning')
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            b_result[name] = {'status': 'timeout', 'elapsed': elapsed,
                              'log': str(logfile)}
            log(f'[B] {name} TIMEOUT at {elapsed:.0f}s', level='error')
        except Exception as e:
            b_result[name] = {'status': 'exception', 'error': str(e)}
            log(f'[B] {name} EXCEPTION: {e}', level='error')

    # Snapshot result CSVs to phase_B/
    log('[B] Snapshotting result CSVs ...')
    wr = ROOT / 'Writing_resource'
    for fname in ['mavise_classical_baseline.csv', 'mavise_classical_v2.csv',
                  'mavise_hv_results.csv', 'mavise_directions.csv',
                  'mavise_cnn_baseline.csv', 'mavise_rnn_baseline.csv']:
        src = wr / fname
        if src.exists():
            shutil.copy2(src, PHASE_B_DIR / fname)
            log(f'   copied {fname}')
        else:
            log(f'   missing {fname} (script may have failed)', level='warning')

    b_result['elapsed_total'] = time.time() - t_start
    return b_result


# --------------------------------------------------------------------------
# Phase C — analyses on clean data
# --------------------------------------------------------------------------
def phase_c_phase1_measurements():
    """Re-run Phase 1 (Y_var, anxiety trajectory, bw_flag rate, prox_var)."""
    from scipy import stats
    d_all = ROOT / 'ml_processed_behavior_all'
    X = np.load(d_all / 'X_array.npy')
    y = np.load(d_all / 'y_array.npy')
    pid = np.load(d_all / 'pid_array.npy', allow_pickle=True)
    scene = np.load(d_all / 'scene_array.npy', allow_pickle=True)
    feats = list(np.load(d_all / 'feature_tag_list.npy', allow_pickle=True))
    IDX_BW = feats.index('beh::backward_flag')
    IDX_DIST = feats.index('beh::dist_min')

    # 01: Y variance
    rows1 = []
    for s in ALL_SCENES_FINE:
        m = scene == s
        y_s = y[m]; pid_s = pid[m]
        pid_stds, pid_means = [], []
        for p in np.unique(pid_s):
            yp = y_s[pid_s == p]
            if len(yp) >= 2:
                pid_stds.append(np.std(yp)); pid_means.append(np.mean(yp))
        rows1.append(dict(
            scene=s, protocol_order=ALL_SCENES_FINE.index(s)+1,
            n_windows=int(m.sum()), n_pids=int(len(np.unique(pid_s))),
            y_std_window=float(np.std(y_s)),
            y_std_pidmean_median=float(np.median(pid_stds)),
            y_std_pidmean_q25=float(np.quantile(pid_stds,0.25)),
            y_std_pidmean_q75=float(np.quantile(pid_stds,0.75)),
            y_mean_pidmean_median=float(np.median(pid_means)),
        ))
    pd.DataFrame(rows1).to_csv(PHASE_C_DIR / '01_y_variance.csv', index=False)

    # 02: anxiety trajectory + Elev1/Elev2 paired
    all_pids = sorted(np.unique(pid))
    traj = []
    for p in all_pids:
        row = {'pid': p}
        for s in ALL_SCENES_FINE:
            yp = y[(pid == p) & (scene == s)]
            row[s] = float(np.mean(yp)) if len(yp) else np.nan
        traj.append(row)
    df_traj = pd.DataFrame(traj)
    df_traj.to_csv(PHASE_C_DIR / '02_anxiety_trajectory_perpid.csv', index=False)
    rows2 = []
    for s in ALL_SCENES_FINE:
        v = df_traj[s].dropna().values
        rows2.append(dict(scene=s, protocol_order=ALL_SCENES_FINE.index(s)+1,
                          n_pids=int(len(v)),
                          y_pidmean_mean=float(np.mean(v)),
                          y_pidmean_std=float(np.std(v)),
                          y_pidmean_sem=float(np.std(v, ddof=1)/np.sqrt(len(v)))))
    pd.DataFrame(rows2).to_csv(PHASE_C_DIR / '02_anxiety_trajectory_summary.csv', index=False)
    paired = df_traj[['pid', 'Elevator1', 'Elevator2']].dropna()
    diff = paired['Elevator2'].values - paired['Elevator1'].values
    t_s, p_v = stats.ttest_rel(paired['Elevator2'], paired['Elevator1'])
    w_s, p_w = stats.wilcoxon(paired['Elevator2'], paired['Elevator1'])
    pd.DataFrame([dict(
        n_pids_paired=int(len(paired)),
        mean_diff=float(diff.mean()), std_diff=float(diff.std(ddof=1)),
        cohens_d=float(diff.mean()/diff.std(ddof=1)) if diff.std(ddof=1)>0 else np.nan,
        t_stat=float(t_s), p_value_t=float(p_v),
        wilcoxon_W=float(w_s), p_value_wilcoxon=float(p_w),
    )]).to_csv(PHASE_C_DIR / '02_elev1_vs_elev2_paired.csv', index=False)

    # 03: backward_flag rate
    bw_per_window = X[:, :, IDX_BW].mean(axis=1)
    rows3 = []
    for s in ALL_SCENES_FINE:
        m = scene == s
        bw_s = bw_per_window[m]; pid_s = pid[m]
        pid_rates = [bw_s[pid_s == p].mean() for p in np.unique(pid_s)]
        rows3.append(dict(scene=s, protocol_order=ALL_SCENES_FINE.index(s)+1,
                          n_windows=int(m.sum()), n_pids=int(len(np.unique(pid_s))),
                          rate_window=float(np.mean(bw_s)),
                          rate_pid_mean=float(np.mean(pid_rates)),
                          rate_pid_std=float(np.std(pid_rates))))
    pd.DataFrame(rows3).to_csv(PHASE_C_DIR / '03_backward_flag_rate.csv', index=False)

    # 04: proximity variance (dist_min)
    rows4 = []
    for s in ALL_SCENES_FINE:
        m = scene == s; pid_s = pid[m]
        X_s = X[m, :, IDX_DIST]
        pid_stds = []
        for p in np.unique(pid_s):
            Xp = X_s[pid_s == p].flatten()
            Xp = Xp[np.isfinite(Xp)]
            if len(Xp) >= 2:
                pid_stds.append(np.std(Xp))
        rows4.append(dict(scene=s, protocol_order=ALL_SCENES_FINE.index(s)+1,
                          dist_std_window=float(np.std(X_s[np.isfinite(X_s)])),
                          dist_std_pidmean_median=float(np.median(pid_stds)) if pid_stds else np.nan))
    pd.DataFrame(rows4).to_csv(PHASE_C_DIR / '04_proximity_variance.csv', index=False)
    return rows1, rows2, rows3, rows4


def phase_c_three_factor():
    """Per-scene three-factor (Y_var, X_var, Coupling) with Fisher-z avg."""
    d_all = ROOT / 'ml_processed_behavior_all'
    X = np.load(d_all / 'X_array.npy')
    y = np.load(d_all / 'y_array.npy')
    pid = np.load(d_all / 'pid_array.npy', allow_pickle=True)
    scene = np.load(d_all / 'scene_array.npy', allow_pickle=True)
    feats = list(np.load(d_all / 'feature_tag_list.npy', allow_pickle=True))
    IDX_BW = feats.index('beh::backward_flag')
    IDX_DIST_DIFF = feats.index('beh::dist_min_diff') if 'beh::dist_min_diff' in feats \
                    else feats.index('beh::dist_min')

    def fisher_z_mean(rs):
        rs = np.clip(np.array(rs), -0.999, 0.999)
        if len(rs) == 0:
            return np.nan
        return float(np.tanh(np.mean(np.arctanh(rs))))

    rows = []
    for s in ALL_SCENES_FINE:
        m = scene == s
        y_s = y[m]; pid_s = pid[m]
        bw_per_win = X[m, :, IDX_BW].mean(axis=1)
        dist_per_win = X[m, :, IDX_DIST_DIFF].mean(axis=1)

        # Y_var: per-PID std of y (median across PIDs)
        y_pid_stds = []
        # X_var: per-PID std of bw_flag rate (median)
        bw_pid_stds = []
        # Coupling: per-PID corr(bw, y); Fisher-z mean; only PIDs with n>=20
        bw_corrs, dist_corrs = [], []
        n_kept = 0
        for p in np.unique(pid_s):
            mask_p = pid_s == p
            yp = y_s[mask_p]; bwp = bw_per_win[mask_p]; dp = dist_per_win[mask_p]
            if len(yp) < 20:
                continue
            n_kept += 1
            y_pid_stds.append(np.std(yp))
            bw_pid_stds.append(np.std(bwp))
            if np.std(yp) > 1e-6 and np.std(bwp) > 1e-6:
                bw_corrs.append(np.corrcoef(bwp, yp)[0, 1])
            if np.std(yp) > 1e-6 and np.std(dp) > 1e-6:
                dist_corrs.append(np.corrcoef(dp, yp)[0, 1])

        rows.append(dict(
            scene=s,
            n_pids_kept=n_kept,
            Y_var=float(np.median(y_pid_stds)) if y_pid_stds else np.nan,
            X_var_bw=float(np.median(bw_pid_stds)) if bw_pid_stds else np.nan,
            coupling_bw_fisher=fisher_z_mean(bw_corrs),
            coupling_dist_fisher=fisher_z_mean(dist_corrs),
            n_bw_corrs=len(bw_corrs),
            n_dist_corrs=len(dist_corrs),
        ))
    df = pd.DataFrame(rows)

    # Normalize each factor (z-score across the 4 main scenes, exclude Elev1/Elev2 separate)
    # Use all 5 scenes for normalization but flag bottleneck per scene
    for col in ['Y_var', 'X_var_bw', 'coupling_bw_fisher']:
        mu, sd = df[col].mean(), df[col].std(ddof=1)
        df[col + '_z'] = (df[col] - mu) / (sd if sd > 0 else 1.0)

    # Bottleneck rule: factor with smallest z (most negative). If all >= -0.5, "none".
    def bottleneck(row):
        zs = {'Y_var (target)': row['Y_var_z'],
              'X_var (feature)': row['X_var_bw_z'],
              'Coupling': row['coupling_bw_fisher_z']}
        lo_name, lo_z = min(zs.items(), key=lambda kv: kv[1])
        if lo_z >= -0.5:
            return 'none (all sufficient)'
        return f'{lo_name} (z={lo_z:+.2f})'
    df['Bottleneck'] = df.apply(bottleneck, axis=1)

    df.to_csv(PHASE_C_DIR / '05_three_factor_bottleneck.csv', index=False)
    return df


def phase_c_4panel_figure(rows1, rows2, df_three_factor):
    """4-panel figure: Y_var×R², three-factor, trajectory, bw_flag×R²."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Read baseline R² (XGB per-scene) for x-axis of scatter
    # Use mavise_hv_results.csv if available
    hv_path = ROOT / 'Writing_resource/mavise_hv_results.csv'
    if hv_path.exists():
        hv = pd.read_csv(hv_path)
    else:
        hv = None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Y_var × R² scatter (per-scene)
    ax = axes[0, 0]
    if hv is not None:
        # extract XGB R² per scene (grouped scenes)
        scene_groups = ['Hallway', 'Hall', 'Elevator', 'Outside']
        # Y_var: take Y_var by mapping Elev1/Elev2 → Elevator
        yvar_map = {r['scene']: r['Y_var'] for r in df_three_factor.to_dict('records')}
        elev_yvar = np.mean([yvar_map.get('Elevator1', np.nan),
                             yvar_map.get('Elevator2', np.nan)])
        # try to extract XGB column from hv
        xs, ys, labels = [], [], []
        for sg in scene_groups:
            if sg == 'Elevator':
                yv = elev_yvar
            else:
                yv = yvar_map.get(sg, np.nan)
            sub = hv[hv['scene'] == sg] if 'scene' in hv.columns else None
            r2 = np.nan
            if sub is not None and not sub.empty:
                for col in ['xgb_r2', 'XGB', 'test_r2']:
                    if col in sub.columns:
                        r2 = float(sub[col].iloc[0]); break
            xs.append(yv); ys.append(r2); labels.append(sg)
        ax.scatter(xs, ys, s=120, c=['tab:red','tab:orange','tab:blue','tab:green'])
        for x, y_v, l in zip(xs, ys, labels):
            ax.annotate(l, (x, y_v), fontsize=10, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Y variance (per-PID std median)')
    ax.set_ylabel('Test R² (XGB, HV)')
    ax.set_title('(A) Y_var × R² — predictability vs target variance')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.grid(alpha=0.3)

    # (B) three-factor bar
    ax = axes[0, 1]
    factors = ['Y_var', 'X_var_bw', 'coupling_bw_fisher']
    factor_labels = ['Y_var', 'X_var (bw)', 'Coupling (bw,y)']
    width = 0.25
    xpos = np.arange(len(ALL_SCENES_FINE))
    for i, (f, fl) in enumerate(zip(factors, factor_labels)):
        vals = [float(df_three_factor[df_three_factor.scene == s][f].iloc[0])
                if not df_three_factor[df_three_factor.scene == s].empty else 0
                for s in ALL_SCENES_FINE]
        ax.bar(xpos + (i - 1) * width, vals, width=width, label=fl)
    ax.set_xticks(xpos); ax.set_xticklabels(ALL_SCENES_FINE, rotation=30)
    ax.set_title('(B) Three-factor decomposition per scene')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (C) anxiety trajectory across protocol order
    ax = axes[1, 0]
    means = [r['y_pidmean_mean'] for r in rows2]
    sems = [r['y_pidmean_sem'] for r in rows2]
    ax.errorbar(range(1, len(ALL_SCENES_FINE) + 1), means, yerr=sems,
                marker='o', capsize=5, linewidth=2)
    ax.set_xticks(range(1, len(ALL_SCENES_FINE) + 1))
    ax.set_xticklabels(ALL_SCENES_FINE, rotation=30)
    ax.set_ylabel('Anxiety mean (per-PID z-scored)')
    ax.set_title('(C) Anxiety trajectory across protocol order')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.grid(alpha=0.3)

    # (D) backward_flag × R² scatter
    ax = axes[1, 1]
    bw_path = PHASE_C_DIR / '03_backward_flag_rate.csv'
    if bw_path.exists() and hv is not None:
        bw_df = pd.read_csv(bw_path)
        # group Elev1+Elev2 → Elevator for matching with hv
        bw_map = dict(zip(bw_df['scene'], bw_df['rate_pid_mean']))
        elev_bw = np.mean([bw_map.get('Elevator1', np.nan), bw_map.get('Elevator2', np.nan)])
        xs, ys, labels = [], [], []
        for sg in scene_groups:
            bw = elev_bw if sg == 'Elevator' else bw_map.get(sg, np.nan)
            sub = hv[hv['scene'] == sg] if 'scene' in hv.columns else None
            r2 = np.nan
            if sub is not None and not sub.empty:
                for col in ['xgb_r2', 'XGB', 'test_r2']:
                    if col in sub.columns:
                        r2 = float(sub[col].iloc[0]); break
            xs.append(bw); ys.append(r2); labels.append(sg)
        ax.scatter(xs, ys, s=120, c=['tab:red','tab:orange','tab:blue','tab:green'])
        for x, y_v, l in zip(xs, ys, labels):
            ax.annotate(l, (x, y_v), fontsize=10, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('backward_flag activation rate (per-PID mean)')
    ax.set_ylabel('Test R² (XGB, HV)')
    ax.set_title('(D) backward_flag rate × R²')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = PHASE_C_DIR / '06_4panel_figure.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    return out_png


def phase_c():
    section('PHASE C — analyses on clean data (Phase 1 + Phase 2 + Phase 3)')
    t_start = time.time()
    c_result = {}

    try:
        log('[C.1] Phase-1 measurements ...')
        rows1, rows2, rows3, rows4 = phase_c_phase1_measurements()
        log('   [C.1] done. 4 CSVs written.')
        c_result['phase1'] = {'rows1': rows1, 'rows2': rows2,
                              'rows3': rows3, 'rows4': rows4}
    except Exception as e:
        log(f'[C.1] FAIL: {e}\n{traceback.format_exc()}', level='error')
        c_result['phase1'] = {'status': 'failed', 'error': str(e)}
        rows1 = rows2 = rows3 = rows4 = None

    try:
        log('[C.2] Phase-2 three-factor ...')
        df_tf = phase_c_three_factor()
        log('   [C.2] done. CSV written.')
        log('\n' + df_tf.to_string(index=False, float_format='%.4f'))
        c_result['phase2'] = df_tf.to_dict('records')
    except Exception as e:
        log(f'[C.2] FAIL: {e}\n{traceback.format_exc()}', level='error')
        c_result['phase2'] = {'status': 'failed', 'error': str(e)}
        df_tf = None

    try:
        if rows1 is not None and df_tf is not None:
            log('[C.3] Phase-3 4-panel figure ...')
            png = phase_c_4panel_figure(rows1, rows2, df_tf)
            log(f'   [C.3] done. {png}')
            c_result['phase3'] = {'png': str(png)}
        else:
            log('[C.3] Skipped (dependencies failed)', level='warning')
            c_result['phase3'] = {'status': 'skipped'}
    except Exception as e:
        log(f'[C.3] FAIL: {e}\n{traceback.format_exc()}', level='error')
        c_result['phase3'] = {'status': 'failed', 'error': str(e)}

    c_result['elapsed_total'] = time.time() - t_start
    return c_result


# --------------------------------------------------------------------------
# Final report
# --------------------------------------------------------------------------
def write_final_report(a_result, b_result, c_result, total_elapsed):
    lines = []
    lines.append('# Autonomous run — final report\n')
    lines.append(f'**Total elapsed**: {total_elapsed/60:.1f} min  '
                 f'({total_elapsed/3600:.2f} h)\n')
    lines.append(f'**Errors**: {len(STATE["errors"])}, '
                 f'**Warnings**: {len(STATE["warnings"])}, '
                 f'**Anomalies**: {len(STATE["anomalies"])}\n')

    lines.append('## Phase A — extraction & merge\n')
    if a_result.get('A1', {}).get('X_shape'):
        lines.append(f'- behavior_all/ X_shape: {a_result["A1"]["X_shape"]}, '
                     f'PIDs={a_result["A1"]["n_pids"]}, '
                     f'skipped={len(a_result["A1"]["skipped"])}, '
                     f'elapsed={a_result["A1"]["elapsed"]:.0f}s\n')
        if a_result['A1']['skipped']:
            lines.append('  - Skipped PIDs:\n')
            for sk in a_result['A1']['skipped'][:10]:
                lines.append(f'    - {sk}\n')
    if a_result.get('A4_merge'):
        lines.append('- Per-scene merge results:\n')
        for sg, r in a_result['A4_merge'].items():
            lines.append(f'  - {sg}: {r}\n')
    if a_result.get('A5_all'):
        lines.append(f'- ml_processed_behavior_all/ X: {a_result["A5_all"].get("X_shape")}\n')
    if a_result.get('verify_stats'):
        lines.append('\n### Verification stats (Phase A.6)\n')
        df = pd.DataFrame(a_result['verify_stats'])
        lines.append(df.to_markdown(index=False, floatfmt='.4f') + '\n')

    lines.append('\n## Phase B — baselines\n')
    for name, res in b_result.items():
        if name == 'elapsed_total':
            continue
        lines.append(f'- **{name}**: {res.get("status")} '
                     f'({res.get("elapsed", 0):.0f}s)\n')

    # Try to summarize key baseline results
    try:
        if (PHASE_B_DIR / 'mavise_hv_results.csv').exists():
            hv = pd.read_csv(PHASE_B_DIR / 'mavise_hv_results.csv')
            lines.append('\n### MAVISE HV (XGB) per-scene R²\n')
            lines.append(hv.to_markdown(index=False) + '\n')
    except Exception as e:
        lines.append(f'\n[hv summarize failed: {e}]\n')

    lines.append('\n## Phase C — analyses\n')
    if c_result.get('phase2'):
        if isinstance(c_result['phase2'], list):
            lines.append('\n### Three-factor decomposition (Phase 2)\n')
            lines.append(pd.DataFrame(c_result['phase2'])
                          .to_markdown(index=False, floatfmt='.4f') + '\n')
    if c_result.get('phase3', {}).get('png'):
        lines.append(f'\n4-panel figure: `{c_result["phase3"]["png"]}`\n')

    lines.append('\n## Anomalies & warnings (sorted)\n')
    for ts, msg in STATE['anomalies']:
        lines.append(f'- **ANOMALY** [{ts}] {msg}\n')
    for ts, msg in STATE['errors']:
        lines.append(f'- **ERROR** [{ts}] {msg}\n')
    for ts, msg in STATE['warnings']:
        lines.append(f'- WARNING [{ts}] {msg}\n')

    FINAL_REPORT.write_text(''.join(lines), encoding='utf-8')
    log(f'Final report written: {FINAL_REPORT}')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    log(f'=== Autonomous run START ===')
    log(f'CWD: {os.getcwd()}')
    log(f'OUT_BASE: {OUT_BASE}')

    a_result, b_result, c_result = {}, {}, {}

    # Phase A
    try:
        a_result = phase_a()
        STATE['phases']['A'] = 'ok'
    except Exception as e:
        log(f'PHASE A unrecoverable: {e}\n{traceback.format_exc()}', level='error')
        STATE['phases']['A'] = 'fatal'

    # Phase B requires Phase A's outputs
    if STATE['phases'].get('A') == 'ok' and a_result.get('A5_all', {}).get('X_shape'):
        try:
            b_result = phase_b()
            STATE['phases']['B'] = 'ok'
        except Exception as e:
            log(f'PHASE B unrecoverable: {e}\n{traceback.format_exc()}', level='error')
            STATE['phases']['B'] = 'fatal'
    else:
        log('Phase B SKIPPED (Phase A did not complete merge)', level='warning')
        STATE['phases']['B'] = 'skipped'

    # Phase C
    if STATE['phases'].get('A') == 'ok' and a_result.get('A5_all', {}).get('X_shape'):
        try:
            c_result = phase_c()
            STATE['phases']['C'] = 'ok'
        except Exception as e:
            log(f'PHASE C unrecoverable: {e}\n{traceback.format_exc()}', level='error')
            STATE['phases']['C'] = 'fatal'
    else:
        log('Phase C SKIPPED (Phase A did not complete merge)', level='warning')
        STATE['phases']['C'] = 'skipped'

    total = time.time() - t0
    log(f'=== Autonomous run END ({total/60:.1f} min) ===')

    # Save state json (for inspection)
    state_path = OUT_BASE / 'state.json'
    state_path.write_text(json.dumps({
        'phases': STATE['phases'],
        'errors': STATE['errors'],
        'warnings': STATE['warnings'],
        'anomalies': STATE['anomalies'],
        'total_elapsed_s': total,
    }, indent=2, default=str), encoding='utf-8')

    write_final_report(a_result, b_result, c_result, total)


if __name__ == '__main__':
    main()
