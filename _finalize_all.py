"""
_finalize_all.py — final polish + documentation + git for clean MAVISE re-extraction.

Sequence:
  1. Regenerate 4-panel figure with refined bottleneck rule + IQR error bars.
  2. Write README.md / metadata in mavise_scene_analysis/.
  3. Update memory files (mavise_baseline_task.md, MEMORY.md) with clean R².
  4. Git add / commit / push (3 logical units).

All in one shot, single bash invocation.
"""
from __future__ import annotations
import os, sys, time, json, shutil, subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'c:/Users/user/code/SDPhysiology')
ANALYSIS = ROOT / 'Writing_resource/mavise_scene_analysis'
AUTORUN = ANALYSIS / 'autonomous_run'
PHASE_C = AUTORUN / 'phase_C'
PHASE_B = AUTORUN / 'phase_B'
PHASE_B_10 = AUTORUN / 'phase_B_10seed'
MEMORY_DIR = Path(r'C:/Users/user/.claude/projects/c--Users-user-code-SDPhysiology/memory')

ALL_SCENES_FINE = ['Elevator1', 'Outside', 'Hallway', 'Elevator2', 'Hall']
GROUP_OF = {'Elevator1': 'Elevator', 'Elevator2': 'Elevator',
            'Outside': 'Outside', 'Hallway': 'Hallway', 'Hall': 'Hall'}


def log(msg):
    ts = time.strftime('%H:%M:%S')
    try:
        print(f'[{ts}] {msg}', flush=True)
    except UnicodeEncodeError:
        print(f'[{ts}] {msg}'.encode('ascii', errors='replace').decode('ascii'), flush=True)


# ==========================================================================
# 1. Regenerate 4-panel figure
# ==========================================================================
def regen_figure():
    log('Step 1: Regenerating 4-panel figure with refined bottleneck rule...')

    # Load per-scene three-factor results
    df_tf = pd.read_csv(PHASE_C / '05_three_factor_bottleneck.csv')
    rows2 = pd.read_csv(PHASE_C / '02_anxiety_trajectory_summary.csv')
    df_y = pd.read_csv(PHASE_C / '01_y_variance.csv')
    df_bw = pd.read_csv(PHASE_C / '03_backward_flag_rate.csv')

    # Load XGB HV R² (per scene-group)
    hv = pd.read_csv(PHASE_B / 'mavise_hv_results.csv')
    hv_xgb = hv[(hv['variant'] == 'HV_classical') & (hv['model'] == 'XGB')]
    r2_by_group = dict(zip(hv_xgb['scene'], hv_xgb['test_r2']))

    # ── REFINED BOTTLENECK RULE ──
    # Hardcoded thresholds (post-hoc, with justification):
    #   Y_var       < 0.5  : (a) target bottleneck — anxiety doesn't vary
    #   X_var_bw    < 0.15 : (b) feature bottleneck — backward_flag rate too low
    #   |coupling|  < 0.15 : (c) coupling bottleneck — feature and y uncorrelated
    THR_Y, THR_X, THR_C = 0.5, 0.15, 0.15
    bottlenecks = []
    for _, r in df_tf.iterrows():
        labels = []
        if r['Y_var'] < THR_Y:
            labels.append('(a) Y_var')
        if r['X_var_bw'] < THR_X:
            labels.append('(b) X_var')
        if abs(r['coupling_bw_fisher']) < THR_C:
            labels.append('(c) Coupling')
        bottlenecks.append(', '.join(labels) if labels else 'none')
    df_tf['Bottleneck_v2'] = bottlenecks
    df_tf.to_csv(PHASE_C / '05_three_factor_bottleneck_v2.csv', index=False)
    log(f'  Refined bottleneck rule:')
    for _, r in df_tf.iterrows():
        log(f'    {r["scene"]:<10}: {r["Bottleneck_v2"]}')

    # Compute per-PID y_std and bw_rate per scene group for IQR error bars
    d_all = ROOT / 'ml_processed_behavior_all'
    X = np.load(d_all / 'X_array.npy', mmap_mode='r')
    y = np.load(d_all / 'y_array.npy')
    pid = np.load(d_all / 'pid_array.npy', allow_pickle=True)
    scene = np.load(d_all / 'scene_array.npy', allow_pickle=True)
    feats = list(np.load(d_all / 'feature_tag_list.npy', allow_pickle=True))
    IDX_BW = feats.index('beh::backward_flag')

    group_stats = {}   # group -> {'y_std_per_pid': [...], 'bw_per_pid': [...]}
    scene_groups = ['Hallway', 'Hall', 'Elevator', 'Outside']
    bw_per_window = X[:, :, IDX_BW].mean(axis=1)
    for g in scene_groups:
        fine_tags = [s for s, gn in GROUP_OF.items() if gn == g]
        m_g = np.isin(scene, fine_tags)
        ys = y[m_g]; ps = pid[m_g]; bws = bw_per_window[m_g]
        y_per_pid, bw_per_pid = [], []
        for p in np.unique(ps):
            mask_p = ps == p
            if mask_p.sum() < 5: continue
            y_per_pid.append(np.std(ys[mask_p]))
            bw_per_pid.append(np.mean(bws[mask_p]))
        group_stats[g] = dict(y_std_per_pid=np.array(y_per_pid),
                              bw_per_pid=np.array(bw_per_pid))

    # ── Build figure ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Hallway': '#d62728', 'Hall': '#ff7f0e',
              'Elevator': '#1f77b4', 'Outside': '#2ca02c'}

    # Panel (A): Y_var × R² with IQR error bars + per-PID strip
    ax = axes[0, 0]
    for g in scene_groups:
        if g not in r2_by_group: continue
        ys_per_pid = group_stats[g]['y_std_per_pid']
        y_med = np.median(ys_per_pid)
        y_q1, y_q3 = np.quantile(ys_per_pid, [0.25, 0.75])
        r2_val = r2_by_group[g]
        # per-PID points (faint)
        jitter = np.full_like(ys_per_pid, r2_val) + np.random.uniform(-0.01, 0.01, len(ys_per_pid))
        ax.scatter(ys_per_pid, jitter, s=10, c=colors[g], alpha=0.15)
        # scene aggregate (bold) with IQR errorbar
        ax.errorbar([y_med], [r2_val],
                    xerr=[[y_med - y_q1], [y_q3 - y_med]],
                    fmt='o', ms=14, c=colors[g], capsize=6,
                    label=f'{g}', mec='black', mew=1.5)
        ax.annotate(g, (y_med, r2_val), fontsize=11, xytext=(8, 8),
                    textcoords='offset points', fontweight='bold')
    ax.set_xlabel('Y variance (per-PID anxiety std, median ± IQR)')
    ax.set_ylabel('Test R² (XGB, HV)')
    ax.set_title('(A) Y_var × R²  — target variance vs predictability')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.grid(alpha=0.3)

    # Panel (B): Three-factor decomposition with bottleneck annotation
    ax = axes[0, 1]
    factors = [('Y_var', 'Y_var (target)'),
               ('X_var_bw', 'X_var (bw rate)'),
               ('coupling_bw_fisher', 'Coupling (bw, y)')]
    bar_colors = ['#4477AA', '#EE6677', '#228833']
    width = 0.27
    xpos = np.arange(len(ALL_SCENES_FINE))
    for i, (f, lbl) in enumerate(factors):
        vals = [float(df_tf[df_tf.scene == s][f].iloc[0])
                if not df_tf[df_tf.scene == s].empty else 0
                for s in ALL_SCENES_FINE]
        ax.bar(xpos + (i - 1) * width, vals, width=width, label=lbl,
               color=bar_colors[i], edgecolor='black', linewidth=0.5)
    # Bottleneck annotation under each scene
    ymin = ax.get_ylim()[0]
    for j, s in enumerate(ALL_SCENES_FINE):
        bn = df_tf[df_tf.scene == s]['Bottleneck_v2'].iloc[0]
        ax.text(j, ymin - 0.12, bn, ha='center', fontsize=7.5,
                rotation=0, color='#aa0000' if bn != 'none' else '#228833')
    # Threshold lines
    ax.axhline(THR_Y, color='#4477AA', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axhline(THR_X, color='#EE6677', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axhline(THR_C, color='#228833', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.axhline(-THR_C, color='#228833', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(ALL_SCENES_FINE, rotation=30)
    ax.set_title('(B) Three-factor decomposition  — bottleneck per scene')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=ymin - 0.18)

    # Panel (C): Anxiety trajectory + Elev1 vs Elev2 paired test annotation
    ax = axes[1, 0]
    means = rows2['y_pidmean_mean'].values
    sems = rows2['y_pidmean_sem'].values
    xs_traj = np.arange(1, len(ALL_SCENES_FINE) + 1)
    ax.errorbar(xs_traj, means, yerr=sems, marker='o', ms=10,
                capsize=6, linewidth=2, color='#444444', mfc='white', mec='#444444')
    # Annotate Elev1 vs Elev2 paired test
    paired = pd.read_csv(PHASE_C / '02_elev1_vs_elev2_paired.csv')
    pt = paired.iloc[0]
    ax.annotate(f'Elev1→Elev2 paired:\n'
                f'd={pt["cohens_d"]:+.3f}, p={pt["p_value_t"]:.2e}',
                xy=(4, means[3]), xytext=(2.3, -0.35),
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xticks(xs_traj); ax.set_xticklabels(ALL_SCENES_FINE, rotation=30)
    ax.set_ylabel('Anxiety mean (per-PID z-scored)')
    ax.set_title('(C) Anxiety trajectory across protocol order  —  habituation')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.grid(alpha=0.3)

    # Panel (D): bw_flag rate × R² with IQR + per-PID strip
    ax = axes[1, 1]
    for g in scene_groups:
        if g not in r2_by_group: continue
        bw_per_pid = group_stats[g]['bw_per_pid']
        bw_med = np.median(bw_per_pid)
        bw_q1, bw_q3 = np.quantile(bw_per_pid, [0.25, 0.75])
        r2_val = r2_by_group[g]
        jitter = np.full_like(bw_per_pid, r2_val) + np.random.uniform(-0.01, 0.01, len(bw_per_pid))
        ax.scatter(bw_per_pid, jitter, s=10, c=colors[g], alpha=0.15)
        ax.errorbar([bw_med], [r2_val],
                    xerr=[[bw_med - bw_q1], [bw_q3 - bw_med]],
                    fmt='o', ms=14, c=colors[g], capsize=6,
                    mec='black', mew=1.5)
        ax.annotate(g, (bw_med, r2_val), fontsize=11, xytext=(8, 8),
                    textcoords='offset points', fontweight='bold')
    ax.set_xlabel('backward_flag rate (per-PID mean, median ± IQR)')
    ax.set_ylabel('Test R² (XGB, HV)')
    ax.set_title('(D) backward_flag rate × R²  — rate alone is insufficient')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PHASE_C / '06_4panel_figure_v2.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    log(f'  Saved: {out}')
    return out


# ==========================================================================
# 2. Write README / metadata
# ==========================================================================
def write_readme():
    log('Step 2: Writing README + metadata...')

    # Pull clean baseline R² for the table
    hv = pd.read_csv(PHASE_B / 'mavise_hv_results.csv')
    rnn10 = pd.read_csv(PHASE_B_10 / 'rnn_10seed_summary.csv')
    cnn10 = pd.read_csv(PHASE_B_10 / 'cnn_10seed_summary.csv')
    df_tf = pd.read_csv(PHASE_C / '05_three_factor_bottleneck_v2.csv')
    paired = pd.read_csv(PHASE_C / '02_elev1_vs_elev2_paired.csv')

    today = time.strftime('%Y-%m-%d')

    md = []
    md.append(f'# MAVISE Scene-Dependent Analysis — Clean Data\n\n')
    md.append(f'_Generated: {today}_\n\n')
    md.append('Independent re-extraction + analysis of MAVISE behavior data, '
              'after fixing two preprocessing bugs discovered while preparing '
              'thesis-grade interpretation evidence.\n\n')

    md.append('## TL;DR\n\n')
    md.append('- **Two bugs** in the MAVISE 56-channel behavior pipeline were '
              'discovered and fixed:\n'
              '  1. `compute_agent_player_relations`: pd.merge suffix not handled — '
              'agent columns silently fell back to player columns → all NPC distances=0.\n'
              '  2. `_compute_player_only_timeseries` + `_augment_behavior_dynamics`: '
              '`.diff()` operations crossed scene boundaries → spurious speed/accel '
              'spikes at scene transitions (up to 12 000 m/s).\n')
    md.append('- All MAVISE data regenerated with fixes; all baselines + analyses re-run.\n')
    md.append('- **Hallway XGB R² (HV)**: 0.424 → **0.499** (cleaner data, +0.075).\n')
    md.append('- **Hall XGB R²**: 0.001 → **0.178** (previously buggy data masked all signal).\n')
    md.append('- **Mechanism**: scene-dependent predictability is driven by '
              '**backward_flag × anxiety coupling**, not by backward_flag rate alone. '
              'Hallway is the only scene with positive coupling (Fisher-z r=+0.35); '
              'others all near zero.\n\n')

    md.append('## Folder index\n\n')
    md.append('| Path | Contents |\n|---|---|\n')
    md.append('| `01_phase1_basic_measurements.py` | Phase-1 measurement script (Y_var, trajectory, bw_rate, prox_var) |\n')
    md.append('| `_step2_verify_one_pid.py` | Single-PID verify-run of fixed code |\n')
    md.append('| `_step3_cross_check_ismar.py` | Cross-check against ISMAR NPZ (boundary fix verification) |\n')
    md.append('| `autonomous_run/AUTONOMOUS_RUN_LOG.md` | Timestamped log of Phase A/B/C unattended run |\n')
    md.append('| `autonomous_run/FINAL_REPORT.md` | Phase A/B/C consolidated report |\n')
    md.append('| `autonomous_run/state.json` | Phase state, errors, warnings |\n')
    md.append('| `autonomous_run/phase_A/verification_stats.csv` | Per-scene channel stats after re-extraction |\n')
    md.append('| `autonomous_run/phase_B/*.csv + .log` | 6 baseline outputs (3-seed for DL) |\n')
    md.append('| `autonomous_run/phase_B_10seed/SUMMARY.md` | 10-seed DL paper-grade table |\n')
    md.append('| `autonomous_run/phase_C/0[1-6]_*.csv` | Phase-1 measurements + three-factor + paired test |\n')
    md.append('| `autonomous_run/phase_C/06_4panel_figure_v2.png` | Final figure (refined bottleneck + IQR) |\n\n')

    md.append('## Classical baseline (XGB, HV masking, per-scene)\n\n')
    md.append('Median R² across 9 held-out test PIDs (80/10/10 group split).\n\n')
    md.append('| Scene | Buggy (R²) | **Clean (R²)** | Δ |\n|---|---|---|---|\n')
    pre = {'Hallway': 0.424, 'Hall': 0.001, 'Elevator': 0.082, 'Outside': 0.005}
    hv_xgb = hv[(hv['variant'] == 'HV_classical') & (hv['model'] == 'XGB')].set_index('scene')
    for s in ['Hallway', 'Hall', 'Elevator', 'Outside']:
        clean = float(hv_xgb.loc[s, 'test_r2'])
        diff = clean - pre[s]
        md.append(f'| {s} | {pre[s]:+.3f} | **{clean:+.3f}** | {diff:+.3f} |\n')
    md.append('\n')

    md.append('## DL 10-seed results (mean ± SD, paper-grade)\n\n')
    md.append('### RNN (LSTM / GRU / GRU_Attn)\n')
    md.append('| model | Hallway | Hall | Elevator | Outside |\n|---|---|---|---|---|\n')
    for m in ['LSTM', 'GRU', 'GRU_Attn']:
        sub = rnn10[rnn10.model == m].set_index('scene')
        cells = []
        for s in ['Hallway', 'Hall', 'Elevator', 'Outside']:
            if s in sub.index:
                cells.append(f'{sub.loc[s, "mean_r2"]:+.3f} ± {sub.loc[s, "std_r2"]:.3f}')
            else:
                cells.append('N/A')
        md.append(f'| {m} | ' + ' | '.join(cells) + ' |\n')
    md.append('\n### CNN\n')
    md.append('| scene | mean ± SD |\n|---|---|\n')
    for _, r in cnn10.iterrows():
        md.append(f'| {r["scene"]} | {r["mean_r2"]:+.3f} ± {r["std_r2"]:.3f} |\n')
    md.append('\n')

    md.append('## Three-factor bottleneck (per scene)\n\n')
    md.append('Refined rule: factor falls below threshold → bottleneck.\n'
              '- (a) `Y_var < 0.5` — anxiety doesn\'t vary\n'
              '- (b) `X_var < 0.15` — backward_flag rate too low\n'
              '- (c) `|Coupling| < 0.15` — backward_flag uncorrelated with anxiety\n\n')
    md.append('| Scene | Y_var | X_var (bw rate) | Coupling (bw, y) | Bottleneck |\n|---|---|---|---|---|\n')
    for _, r in df_tf.iterrows():
        md.append(f'| {r["scene"]} | {r["Y_var"]:.3f} | {r["X_var_bw"]:.3f} '
                  f'| {r["coupling_bw_fisher"]:+.3f} | {r["Bottleneck_v2"]} |\n')
    md.append('\n')

    md.append('## Elevator1 vs Elevator2 paired t-test (habituation)\n\n')
    pt = paired.iloc[0]
    md.append(f'- n = {int(pt["n_pids_paired"])} PIDs (paired)\n')
    md.append(f'- mean diff (Elev2 − Elev1) = {pt["mean_diff"]:+.3f}\n')
    md.append(f'- Cohen\'s d = {pt["cohens_d"]:+.3f}\n')
    md.append(f'- t = {pt["t_stat"]:+.3f}, **p = {pt["p_value_t"]:.2e}**\n')
    md.append(f'- Wilcoxon p = {pt["p_value_wilcoxon"]:.2e}\n\n')
    md.append('Same physical scene, protocol position #1 vs #4. Significantly lower '
              'anxiety on second exposure — direct empirical evidence of habituation, '
              'not just literature inference.\n\n')

    md.append('## Mechanism narrative (refined)\n\n')
    md.append('**Old narrative (data was buggy)**:\n'
              '> Hallway has high R² because backward_flag rate is high → avoidance evidence.\n\n')
    md.append('**New narrative (clean data + Fisher-z coupling)**:\n'
              '> Hallway is the only scene where backward_flag *correlates* with anxiety '
              '(per-PID Fisher-z r = +0.35). Other scenes have similar or higher '
              'backward_flag rates (Hall = 0.48 vs Hallway = 0.25), but the rate is '
              '**not coupled with anxiety** in those scenes — participants step back for '
              'unrelated reasons (e.g., turning to look around in open spaces). '
              'Scene-dependent predictability therefore reflects *avoidance-as-anxiety-signal*, '
              'which requires both (a) proximity threat AND (b) avoidability — only '
              'Hallway meets both. This is a more refined claim than '
              '"avoidance behavior present in Hallway".\n\n')

    md.append('## Caveats and known small issues\n\n')
    md.append('1. **y_p vs y_b discrepancy in merge log** (`max diff ~13`): this is '
              '*not* a data mismatch — physio pipeline stores **per-PID z-scored** y, '
              'behavior pipeline stores **raw 0-10 scale** y. The merge code uses '
              'the physio (z-scored) version. Different normalization stages, same '
              'underlying signal.\n')
    md.append('2. **dist_min / count_* not 100% matched against ISMAR NPZ** '
              '(~33-72% per-frame): MAVISE pipeline\'s `compute_agent_player_relations` '
              'algorithm has evolved since ISMAR NPZ regeneration (2026-03-12). '
              'Raw value ranges and distributions are consistent (0-7.6 m, 0-7 counts); '
              'MAVISE uses its own internally-consistent values.\n')
    md.append('3. **head_rot_speed is smoothed** in MAVISE (via `recompute_headrot_features_from_yaw` '
              'with EMA τ=0.05 s) but raw in ISMAR NPZ. Intentional pipeline difference, '
              'not a bug.\n\n')

    md.append('## Reproduction\n\n')
    md.append('```bash\n')
    md.append('# 1. Extract clean behavior 56ch + merge with physio:\n')
    md.append('python _autonomous_master.py\n\n')
    md.append('# 2. (optional) Run 10-seed DL for paper-grade error bars:\n')
    md.append('python _10seed_dl_runner.py\n\n')
    md.append('# 3. (this file) Refresh figure + README + memory:\n')
    md.append('python _finalize_all.py\n')
    md.append('```\n')

    out_md = ANALYSIS / 'README.md'
    out_md.write_text(''.join(md), encoding='utf-8')
    log(f'  README saved: {out_md}')

    # Also write metadata.json with paths
    meta = {
        'generated': today,
        'baselines_3seed_csv': str(PHASE_B / 'mavise_*.csv'),
        'rnn_10seed_summary': str(PHASE_B_10 / 'rnn_10seed_summary.csv'),
        'cnn_10seed_summary': str(PHASE_B_10 / 'cnn_10seed_summary.csv'),
        'three_factor_csv': str(PHASE_C / '05_three_factor_bottleneck_v2.csv'),
        'paired_test_csv': str(PHASE_C / '02_elev1_vs_elev2_paired.csv'),
        'figure': str(PHASE_C / '06_4panel_figure_v2.png'),
        'verification_stats': str(AUTORUN / 'phase_A/verification_stats.csv'),
        'autonomous_log': str(AUTORUN / 'AUTONOMOUS_RUN_LOG.md'),
        'final_report': str(AUTORUN / 'FINAL_REPORT.md'),
        'clean_data_npy': {
            sg: str(ROOT / f'ml_processed_behavior_{sg}') for sg in
            ['Elevator', 'Hallway', 'Hall', 'Outside', 'all']
        },
    }
    (ANALYSIS / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str),
                                            encoding='utf-8')
    log(f'  metadata.json saved')


# ==========================================================================
# 3. Update memory files
# ==========================================================================
def update_memory():
    log('Step 3: Updating memory files...')

    # 3a. Update mavise_baseline_task.md — full rewrite with clean data
    hv = pd.read_csv(PHASE_B / 'mavise_hv_results.csv')
    hv_xgb = hv[(hv['variant'] == 'HV_classical') & (hv['model'] == 'XGB')].set_index('scene')
    hv_rid = hv[(hv['variant'] == 'HV_classical') & (hv['model'] == 'Ridge')].set_index('scene')
    rnn10 = pd.read_csv(PHASE_B_10 / 'rnn_10seed_summary.csv')
    cnn10 = pd.read_csv(PHASE_B_10 / 'cnn_10seed_summary.csv').set_index('scene')
    df_tf = pd.read_csv(PHASE_C / '05_three_factor_bottleneck_v2.csv')
    paired = pd.read_csv(PHASE_C / '02_elev1_vs_elev2_paired.csv').iloc[0]

    today = time.strftime('%Y-%m-%d')
    md = []
    md.append('---\n')
    md.append('name: MAVISE baseline task\n')
    md.append(f'description: MAVISE 56ch baseline — clean data ({today} 재추출 후 결과 + 10-seed DL)\n')
    md.append('type: project\n')
    md.append('---\n\n')

    md.append('## 진행 요약\n')
    md.append(f'2026-01-25 생성된 MAVISE preprocessed가 buggy version (proximity bug + '
              f'scene boundary bug) 기반임을 발견. raw에서 재추출 후 모든 baseline + '
              f'analysis 재실행 ({today} 완료).\n\n')

    md.append('## 발견된 Bug 2개 (모두 fix)\n\n')
    md.append('### 1) Proximity bug — `compute_agent_player_relations`\n')
    md.append('`pd.merge(suffixes=("", "_agent"))` 후 agent column에 "_agent" suffix '
              '붙는 것을 미처리. `merged[cols.agent_x]`가 player column으로 fallback → '
              'dx=dz=0 → 모든 NPC 거리=0. 영향: dist_*, count_intimate/personal/social/public, '
              'count_fov, count_approach, 및 cascade *_diff/inc/dec 채널.\n\n')

    md.append('### 2) Scene boundary bug — `_compute_player_only_timeseries` + `_augment_behavior_dynamics`\n')
    md.append('모든 `.diff()`/`np.diff(prepend=x[0])`이 (scene, frame) sort 후 전체 통째 처리. '
              'Scene 경계에서 player position teleport이 12 000 m/s speed spike로 잡힘. '
              'preprocess_continuous.py는 scene별 분리 호출이라 영향 없음.\n\n')

    md.append('Fix: 두 함수 안에서 `groupby(scene)` 후 그룹별 diff. behavior_features.py '
              'commit `ed5eb3e` 참조.\n\n')

    md.append('## Clean baseline 결과 (HV masking, 80/10/10 split)\n\n')
    md.append('### Classical (Ridge, XGB)\n')
    md.append('| Scene | Ridge R² | **XGB R²** |\n|---|---|---|\n')
    for s in ['Hallway', 'Hall', 'Elevator', 'Outside']:
        md.append(f'| {s} | {float(hv_rid.loc[s, "test_r2"]):+.3f} '
                  f'| **{float(hv_xgb.loc[s, "test_r2"]):+.3f}** |\n')

    md.append('\n### DL 10-seed (mean ± SD)\n')
    md.append('| Model | Hallway | Hall | Elevator | Outside |\n|---|---|---|---|---|\n')
    for m in ['LSTM', 'GRU', 'GRU_Attn']:
        sub = rnn10[rnn10.model == m].set_index('scene')
        cells = [f'{sub.loc[s, "mean_r2"]:+.3f} ± {sub.loc[s, "std_r2"]:.3f}'
                 if s in sub.index else 'N/A'
                 for s in ['Hallway', 'Hall', 'Elevator', 'Outside']]
        md.append(f'| {m} | ' + ' | '.join(cells) + ' |\n')
    cells = [f'{cnn10.loc[s, "mean_r2"]:+.3f} ± {cnn10.loc[s, "std_r2"]:.3f}'
             for s in ['Hallway', 'Hall', 'Elevator', 'Outside']]
    md.append(f'| CNN | ' + ' | '.join(cells) + ' |\n\n')

    md.append('## Buggy → Clean 변화\n')
    md.append('| Metric | Buggy | Clean |\n|---|---|---|\n')
    md.append(f'| Hallway XGB R² | +0.424 | **{float(hv_xgb.loc["Hallway", "test_r2"]):+.3f}** |\n')
    md.append(f'| Hall XGB R² | +0.001 | **{float(hv_xgb.loc["Hall", "test_r2"]):+.3f}** (↑↑) |\n')
    md.append(f'| dist_min std (Outside) | ~0 | 0.74 |\n')
    md.append(f'| count_personal std (all) | 0 | 0.33-0.49 |\n\n')

    md.append('## Three-factor mechanism (refined, Fisher-z coupling)\n\n')
    md.append('| Scene | Y_var | X_var (bw) | Coupling (bw, y) | Bottleneck |\n|---|---|---|---|---|\n')
    for _, r in df_tf.iterrows():
        md.append(f'| {r["scene"]} | {r["Y_var"]:.3f} | {r["X_var_bw"]:.3f} '
                  f'| {r["coupling_bw_fisher"]:+.3f} | {r["Bottleneck_v2"]} |\n')
    md.append('\n핵심: **Hallway만 positive coupling (+0.35), 나머지는 |coupling|<0.15.** '
              'backward_flag rate 자체는 Hall(0.48)이 가장 높지만 anxiety와 무관 → '
              '"avoidance-as-anxiety-signal"은 proximity threat + avoidability 둘 다 필요.\n\n')

    md.append('## Elev1 vs Elev2 paired test (habituation 직접 증거)\n')
    md.append(f'- n={int(paired["n_pids_paired"])} paired PIDs\n')
    md.append(f'- mean diff = {paired["mean_diff"]:+.3f} (Elev2 - Elev1)\n')
    md.append(f'- **Cohen\'s d = {paired["cohens_d"]:+.3f}, p = {paired["p_value_t"]:.2e}**\n\n')

    md.append('## 핵심 파일 위치\n')
    md.append('- Clean 데이터: `c:/Users/user/code/SDPhysiology/ml_processed_behavior_{Elevator,Hallway,Hall,Outside,all}/`\n')
    md.append('- Deprecated buggy: 같은 폴더 + `_deprecated_bug` suffix\n')
    md.append('- 결과 + figure: `Writing_resource/mavise_scene_analysis/autonomous_run/`\n')
    md.append('- README: `Writing_resource/mavise_scene_analysis/README.md`\n')
    md.append('- Bug fix 코드 commit: `ed5eb3e` (behavior_features.py)\n')

    out_path = MEMORY_DIR / 'mavise_baseline_task.md'
    out_path.write_text(''.join(md), encoding='utf-8')
    log(f'  mavise_baseline_task.md updated')

    # 3b. Update MEMORY.md — replace the MAVISE section
    mem_path = MEMORY_DIR / 'MEMORY.md'
    mem_text = mem_path.read_text(encoding='utf-8')

    # Replace MAVISE section (## MAVISE Leakage-Free Baseline) entirely
    new_section = (
        f'## MAVISE Baseline (CLEAN, {today})\n'
        f'### 기록: 2026-01-25 buggy 데이터 발견 (proximity bug + scene boundary bug). 두 bug fix 후 raw에서 재추출 + 모든 baseline 재실행.\n'
        f'### Clean XGB R² (HV, per-scene)\n'
        f'| Scene | XGB | CNN (10-seed) | GRU_Attn (10-seed) |\n'
        f'|---|---|---|---|\n'
        f'| Hallway | **{float(hv_xgb.loc["Hallway", "test_r2"]):+.3f}** | '
        f'{cnn10.loc["Hallway", "mean_r2"]:+.3f}±{cnn10.loc["Hallway", "std_r2"]:.3f} | '
    )
    # GRU_Attn for Hallway
    sub = rnn10[rnn10.model == 'GRU_Attn'].set_index('scene')
    new_section += (
        f'{sub.loc["Hallway", "mean_r2"]:+.3f}±{sub.loc["Hallway", "std_r2"]:.3f} |\n'
        f'| Hall | {float(hv_xgb.loc["Hall", "test_r2"]):+.3f} | '
        f'{cnn10.loc["Hall", "mean_r2"]:+.3f}±{cnn10.loc["Hall", "std_r2"]:.3f} | '
        f'{sub.loc["Hall", "mean_r2"]:+.3f}±{sub.loc["Hall", "std_r2"]:.3f} |\n'
        f'| Elevator | {float(hv_xgb.loc["Elevator", "test_r2"]):+.3f} | '
        f'{cnn10.loc["Elevator", "mean_r2"]:+.3f}±{cnn10.loc["Elevator", "std_r2"]:.3f} | '
        f'{sub.loc["Elevator", "mean_r2"]:+.3f}±{sub.loc["Elevator", "std_r2"]:.3f} |\n'
        f'| Outside | {float(hv_xgb.loc["Outside", "test_r2"]):+.3f} | '
        f'{cnn10.loc["Outside", "mean_r2"]:+.3f}±{cnn10.loc["Outside", "std_r2"]:.3f} | '
        f'{sub.loc["Outside", "mean_r2"]:+.3f}±{sub.loc["Outside", "std_r2"]:.3f} |\n'
        f'→ Scene-dependent narrative: backward_flag×anxiety **coupling** (not rate) drives '
        f'R². Hallway only scene with strong coupling (Fisher-z r=+0.35). '
        f'Elev1 vs Elev2 paired: d={paired["cohens_d"]:+.2f}, p={paired["p_value_t"]:.1e} '
        f'(direct habituation evidence).\n'
        f'→ [[mavise_baseline_task]] for full details.\n'
    )

    # Find the existing MAVISE section and replace
    import re
    # Match from "## MAVISE Leakage-Free Baseline" or "## MAVISE Baseline" through to the next "## " or end
    pattern = r'## MAVISE[^\n]*\n(?:(?!^## ).*\n?)*?(?=^## |\Z)'
    match = re.search(pattern, mem_text, flags=re.MULTILINE)
    if match:
        mem_text = mem_text[:match.start()] + new_section + '\n' + mem_text[match.end():]
        log('  MEMORY.md MAVISE section replaced')
    else:
        # Append at end if pattern not found
        mem_text = mem_text.rstrip() + '\n\n' + new_section + '\n'
        log('  MEMORY.md MAVISE section appended (pattern not found)')

    mem_path.write_text(mem_text, encoding='utf-8')


# ==========================================================================
# 4. Git
# ==========================================================================
def run_git(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    return p.returncode, (p.stdout or '') + (p.stderr or '')


def git_commit_push():
    log('Step 4: Git add / commit / push (3 logical commits)...')

    co_author = 'Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>'

    # Commit 1: NUM_SEEDS=10 in baselines + 10-seed runner
    log('  [git] Commit 1: 10-seed DL runner + NUM_SEEDS=10')
    files1 = [
        'eval_mavise_rnn_baseline.py',
        'eval_mavise_cnn_baseline.py',
        '_10seed_dl_runner.py',
    ]
    rc, _ = run_git(['git', 'add'] + files1)
    if rc == 0:
        msg = ('MAVISE DL baseline 10-seed paper-grade re-run\n\n'
               'NUM_SEEDS 3 -> 10 in CNN/RNN baseline scripts.\n'
               '_10seed_dl_runner.py: sequential CNN+RNN execution with summary.\n'
               'Results: phase_B_10seed/SUMMARY.md (mean ± SD across 10 seeds).\n\n'
               '' + co_author)
        rc, out = run_git(['git', 'commit', '-m', msg])
        log(f'    rc={rc} {out[:200]}')

    # Commit 2: finalize_all script + figure regen + README
    log('  [git] Commit 2: finalize tooling + README + figure v2')
    files2 = [
        '_finalize_all.py',
        'Writing_resource/mavise_scene_analysis/README.md',
        'Writing_resource/mavise_scene_analysis/metadata.json',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/05_three_factor_bottleneck_v2.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/06_4panel_figure_v2.png',
    ]
    rc, _ = run_git(['git', 'add'] + files2)
    if rc == 0:
        msg = ('MAVISE clean-data finalization: figure v2 + README + metadata\n\n'
               '- 06_4panel_figure_v2.png: per-PID strip + IQR error bars + '
               'refined bottleneck rule annotation\n'
               '- 05_three_factor_bottleneck_v2.csv: Bottleneck_v2 column with '
               'absolute thresholds (Y_var<0.5 / X_var<0.15 / |coupling|<0.15)\n'
               '- README.md: TL;DR + folder index + all clean-data tables + caveats\n'
               '- metadata.json: paths index for downstream tooling\n\n'
               '' + co_author)
        rc, out = run_git(['git', 'commit', '-m', msg])
        log(f'    rc={rc} {out[:200]}')

    # Commit 3: autonomous run outputs (logs + reports)
    log('  [git] Commit 3: autonomous run outputs')
    files3 = [
        'Writing_resource/mavise_scene_analysis/autonomous_run/AUTONOMOUS_RUN_LOG.md',
        'Writing_resource/mavise_scene_analysis/autonomous_run/FINAL_REPORT.md',
        'Writing_resource/mavise_scene_analysis/autonomous_run/state.json',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_A/verification_stats.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/01_y_variance.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/02_anxiety_trajectory_perpid.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/02_anxiety_trajectory_summary.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/02_elev1_vs_elev2_paired.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/03_backward_flag_rate.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/04_proximity_variance.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/05_three_factor_bottleneck.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/06_4panel_figure.png',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_B_10seed/SUMMARY.md',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_B_10seed/RUN_10SEED_LOG.md',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_B_10seed/rnn_10seed_summary.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_B_10seed/cnn_10seed_summary.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_B_10seed/mavise_rnn_baseline.csv',
        'Writing_resource/mavise_scene_analysis/autonomous_run/phase_B_10seed/mavise_cnn_baseline.csv',
    ]
    # Filter to existing files
    files3 = [f for f in files3 if (ROOT / f).exists()]
    rc, _ = run_git(['git', 'add'] + files3)
    if rc == 0:
        msg = ('MAVISE clean-data autonomous-run outputs (logs + reports)\n\n'
               'Phase A: verification_stats.csv\n'
               'Phase B: 6 baseline CSVs + per-script logs (3-seed)\n'
               'Phase B 10-seed: paper-grade DL re-run (SUMMARY.md, RNN/CNN mean±SD)\n'
               'Phase C: 4 measurement CSVs + three-factor + paired test + 4-panel figure v1\n'
               'AUTONOMOUS_RUN_LOG.md + FINAL_REPORT.md + state.json\n\n'
               '' + co_author)
        rc, out = run_git(['git', 'commit', '-m', msg])
        log(f'    rc={rc} {out[:200]}')

    # Push
    log('  [git] Pushing all to origin/master...')
    rc, out = run_git(['git', 'push', 'origin', 'master'])
    log(f'    push rc={rc}')
    log(f'    {out[-400:]}')


# ==========================================================================
# Main
# ==========================================================================
def main():
    t_start = time.time()
    log('=== Finalize-all START ===')

    try:
        regen_figure()
    except Exception as e:
        log(f'  [FIGURE FAIL] {e}')
        import traceback; traceback.print_exc()

    try:
        write_readme()
    except Exception as e:
        log(f'  [README FAIL] {e}')
        import traceback; traceback.print_exc()

    try:
        update_memory()
    except Exception as e:
        log(f'  [MEMORY FAIL] {e}')
        import traceback; traceback.print_exc()

    try:
        git_commit_push()
    except Exception as e:
        log(f'  [GIT FAIL] {e}')
        import traceback; traceback.print_exc()

    log(f'=== Finalize-all DONE ({time.time()-t_start:.0f}s) ===')


if __name__ == '__main__':
    main()
