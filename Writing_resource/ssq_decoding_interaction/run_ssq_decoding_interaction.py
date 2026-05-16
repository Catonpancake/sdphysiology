"""
SSQ × Decoding interaction analysis.

Question (#1 priority from SSQ_Analysis_Handoff.md):
  Does cybersickness (SSQ_Total) moderate decoding/prediction performance?

  - If High-SSQ PIDs have higher R²/AUC: model might be picking up
    sickness-related behavior rather than pure anxiety signal (concerning).
  - If no relationship: prediction is independent of cybersickness.

Tests per (task, model, modality/condition):
  1. Mann-Whitney U: SSQ-Bad (Post SSQ_Total ≥ 52) vs SSQ-OK
  2. Spearman ρ(performance, Post SSQ_Total) — continuous moderator
  3. Spearman ρ(performance, Δ SSQ = Post − Pre) — change-score moderator

All tests BH-FDR corrected within each task family.

Inputs:
  - D:\\Labroom\\SDPhysiology\\Data\\MOMENT_ready_v2\\trait_scores.csv
      Post SSQ_Total + Pre_STAI/Post_STAI + IPQ per PID
  - D:\\Labroom\\SDPhysiology\\Data\\processed_survey\\pre_survey.csv
      → compute Pre SSQ_Total (sum of items × 3.74), same formula as Post
  - Writing_resource per-PID CSVs:
      * T2A_raw_perpid.csv      — T2-A (binary AUC) × {RF, Ridge, XGB} × {behavior, physio, full}
      * T3A_xgb_lopo_per_pid.csv — T3-A (causal R²) × XGB × 7 conditions
      * T3A_rf_lopo_per_pid.csv  — T3-A × RF × 7 conditions
      * T3W_ar_xgb_per_pid.csv   — T3-W × {Ridge, XGB} × {AR_only, AR+Physio, AR+Full} × {30,60,120,180,300} s
      * beh_ablation_perpid.csv  — T1-A (dec) + T2-A (bin) + T3-A (cau_ar) × full_beh ablation

Outputs (in this folder):
  per_pid_joined.csv         — PID × Post SSQ × Pre SSQ × Δ SSQ × {all decoding metrics}
  test_results.csv           — long table: task, model, modality, test_type, statistic, p, p_fdr
  test_results_compact.md    — readable markdown table per family
  SUMMARY.md                 — paper-ready summary + interpretation
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
except ImportError as e:
    sys.exit(f'statsmodels required: {e}')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ─── Paths ───────────────────────────────────────────────────────────────
ROOT      = Path(r'c:/Users/user/code/SDPhysiology')
WR        = ROOT / 'Writing_resource'
OUT_DIR   = WR / 'ssq_decoding_interaction'
TRAIT_CSV = Path(r'D:/Labroom/SDPhysiology/Data/MOMENT_ready_v2/trait_scores.csv')
PRE_CSV   = Path(r'D:/Labroom/SDPhysiology/Data/processed_survey/pre_survey.csv')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SSQ_BAD_THRESH = 52   # Kennedy 1993 "moderate" cutoff
EXCLUDED_PIDS  = {'064', '086'}

# ─── 1. Load + compute Pre SSQ ───────────────────────────────────────────
print('[1] Loading SSQ scores...')
trait = pd.read_csv(TRAIT_CSV)
trait['PID_str'] = trait['PID_str'].astype(str).str.zfill(3)
trait = trait[~trait['PID_str'].isin(EXCLUDED_PIDS)].copy()
print(f'    trait_scores n_PIDs: {len(trait)} (after excluding {EXCLUDED_PIDS})')

# Pre SSQ from item-level data — same formula as Post (sum × 3.74)
try:
    pre = pd.read_csv(PRE_CSV, encoding='utf-8')
except UnicodeDecodeError:
    pre = pd.read_csv(PRE_CSV, encoding='cp949')
# Drop docstring row (rows where ID is non-numeric / equals 'ID')
pre['ID'] = pd.to_numeric(pre['ID'], errors='coerce')
pre = pre.dropna(subset=['ID']).copy()
pre['ID'] = pre['ID'].astype(int)
pre['PID_str'] = pre['ID'].astype(str).str.zfill(3)
ssq_cols = [c for c in pre.columns if c.startswith('SSQ_')]
pre[ssq_cols] = pre[ssq_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
pre['Pre_SSQ_Total'] = pre[ssq_cols].sum(axis=1) * 3.74

ssq = trait[['PID_str', 'SSQ_Total']].rename(columns={'SSQ_Total': 'Post_SSQ_Total'})
ssq = ssq.merge(pre[['PID_str', 'Pre_SSQ_Total']], on='PID_str', how='left')
ssq['Delta_SSQ'] = ssq['Post_SSQ_Total'] - ssq['Pre_SSQ_Total']
ssq['SSQ_Bad']   = (ssq['Post_SSQ_Total'] >= SSQ_BAD_THRESH).astype(int)

print(f'    Pre SSQ computed:    mean={ssq["Pre_SSQ_Total"].mean():.2f}, '
      f'std={ssq["Pre_SSQ_Total"].std():.2f}')
print(f'    Post SSQ:            mean={ssq["Post_SSQ_Total"].mean():.2f}, '
      f'std={ssq["Post_SSQ_Total"].std():.2f}')
print(f'    Δ SSQ (Post − Pre):  mean={ssq["Delta_SSQ"].mean():.2f}, '
      f'std={ssq["Delta_SSQ"].std():.2f}')
print(f'    SSQ-Bad (≥ {SSQ_BAD_THRESH}):     n = {int(ssq["SSQ_Bad"].sum())}/{len(ssq)}')

# ─── 2. Build per-PID decoding metrics matrix ────────────────────────────
print('\n[2] Loading per-PID decoding results...')
metrics = []   # rows: PID × (task, model, modality_or_condition) → metric

# 2a. T1-A / T2-A / T3-A from beh_ablation_perpid (full_beh ablation only)
df_ba = pd.read_csv(WR / 'beh_ablation_perpid.csv')
df_ba['pid'] = df_ba['pid'].astype(str).str.zfill(3)
task_map = {'dec': 'T1-A', 'bin': 'T2-A', 'cau_ar': 'T3-A_AR'}
metric_map = {'dec': 'R2', 'bin': 'AUC', 'cau_ar': 'R2'}
for task_raw in ['dec', 'bin', 'cau_ar']:
    sub = df_ba[(df_ba.ablation == 'full_beh') & (df_ba.task == task_raw)]
    for _, r in sub.iterrows():
        metrics.append(dict(
            pid=r['pid'], task=task_map[task_raw],
            model='RF_behavior_only_9ch', condition='full_beh',
            metric_name=metric_map[task_raw], value=float(r['value']),
        ))

# 2b. T2-A from T2A_raw_perpid (3 models × 3 modalities)
df_t2 = pd.read_csv(WR / 'T2A_raw_perpid.csv')
df_t2['pid'] = df_t2['pid'].astype(str).str.zfill(3)
for _, r in df_t2.iterrows():
    metrics.append(dict(
        pid=r['pid'], task='T2-A', model=str(r['model']),
        condition=str(r['modality']),
        metric_name='AUC', value=float(r['auc']),
    ))

# 2c. T3-A from XGB and RF lopo per_pid (focus on per-condition full feature sets)
for model_csv, model_label in [
    ('T3A_xgb_lopo_per_pid.csv', 'XGB'),
    ('T3A_rf_lopo_per_pid.csv', 'RF'),
]:
    df = pd.read_csv(WR / model_csv)
    df['pid'] = df['pid'].astype(str).str.zfill(3)
    for _, r in df.iterrows():
        metrics.append(dict(
            pid=r['pid'], task='T3-A',
            model=model_label, condition=str(r['condition']),
            metric_name='R2', value=float(r['r2']),
        ))

# 2d. T3-W from T3W_ar_xgb_per_pid (focus on best L = 300s = 5min)
df_t3w = pd.read_csv(WR / 'T3W_ar_xgb_per_pid.csv')
df_t3w['pid'] = df_t3w['pid'].astype(str).str.zfill(3)
df_t3w_300 = df_t3w[df_t3w.L_sec == 300]
for _, r in df_t3w_300.iterrows():
    metrics.append(dict(
        pid=r['pid'], task='T3-W',
        model=str(r['condition']).split('_')[0],
        condition=str(r['condition']),
        metric_name='R2', value=float(r['r2']),
    ))

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df[~metrics_df['pid'].isin(EXCLUDED_PIDS)]
print(f'    Built metrics matrix: {len(metrics_df)} rows')
print(metrics_df.groupby('task').size().to_string())

# ─── 3. Run statistical tests per (task, model, condition) ───────────────
print('\n[3] Running tests...')
test_rows = []
for (task, model, condition), grp in metrics_df.groupby(['task', 'model', 'condition']):
    merged = grp.merge(ssq, left_on='pid', right_on='PID_str', how='inner')
    merged = merged.dropna(subset=['value', 'Post_SSQ_Total'])
    if len(merged) < 10:
        continue
    n = len(merged)
    bad = merged[merged.SSQ_Bad == 1]['value'].values
    ok  = merged[merged.SSQ_Bad == 0]['value'].values

    # (a) Mann-Whitney U: Bad vs OK
    if len(bad) >= 3 and len(ok) >= 3:
        try:
            u_stat, p_mw = stats.mannwhitneyu(bad, ok, alternative='two-sided')
            # rank-biserial effect size
            rbs = 1 - (2 * u_stat) / (len(bad) * len(ok))
        except Exception:
            u_stat, p_mw, rbs = np.nan, np.nan, np.nan
    else:
        u_stat, p_mw, rbs = np.nan, np.nan, np.nan
    test_rows.append(dict(
        task=task, model=model, condition=condition,
        metric_name=grp['metric_name'].iloc[0], n=n,
        n_bad=len(bad), n_ok=len(ok),
        test='MannWhitney_Bad_vs_OK',
        statistic=float(u_stat) if np.isfinite(u_stat) else np.nan,
        effect_size=float(rbs) if np.isfinite(rbs) else np.nan,
        effect_size_kind='rank_biserial_r',
        bad_median=float(np.median(bad)) if len(bad) else np.nan,
        ok_median=float(np.median(ok)) if len(ok) else np.nan,
        p_raw=float(p_mw) if np.isfinite(p_mw) else np.nan,
    ))

    # (b) Spearman vs continuous Post SSQ_Total
    rho, p = stats.spearmanr(merged['value'], merged['Post_SSQ_Total'])
    test_rows.append(dict(
        task=task, model=model, condition=condition,
        metric_name=grp['metric_name'].iloc[0], n=n,
        n_bad=len(bad), n_ok=len(ok),
        test='Spearman_vs_Post_SSQ',
        statistic=float(rho), effect_size=float(rho),
        effect_size_kind='spearman_rho',
        bad_median=np.nan, ok_median=np.nan,
        p_raw=float(p),
    ))

    # (c) Spearman vs Δ SSQ
    if merged['Delta_SSQ'].notna().sum() >= 10:
        rho_d, p_d = stats.spearmanr(merged['value'], merged['Delta_SSQ'])
        test_rows.append(dict(
            task=task, model=model, condition=condition,
            metric_name=grp['metric_name'].iloc[0], n=n,
            n_bad=len(bad), n_ok=len(ok),
            test='Spearman_vs_Delta_SSQ',
            statistic=float(rho_d), effect_size=float(rho_d),
            effect_size_kind='spearman_rho',
            bad_median=np.nan, ok_median=np.nan,
            p_raw=float(p_d),
        ))

tests_df = pd.DataFrame(test_rows)

# BH-FDR within each family = (task, test). Each family is one set of related tests.
print('    BH-FDR within family (task × test)...')
tests_df['p_fdr'] = np.nan
tests_df['significant_05'] = False
for (task, test), grp in tests_df.groupby(['task', 'test']):
    mask = grp['p_raw'].notna()
    if mask.sum() == 0:
        continue
    p_raw = grp.loc[mask, 'p_raw'].values
    rej, p_fdr, _, _ = multipletests(p_raw, alpha=0.05, method='fdr_bh')
    tests_df.loc[grp.index[mask], 'p_fdr'] = p_fdr
    tests_df.loc[grp.index[mask], 'significant_05'] = rej

tests_df.to_csv(OUT_DIR / 'test_results.csv', index=False)
print(f'    {len(tests_df)} tests run, '
      f'{int(tests_df["significant_05"].sum())} significant after FDR.')

# ─── 4. Build per-PID joined table for downstream re-use ─────────────────
print('\n[4] Building per-PID joined table ...')
joined = ssq.copy()
# Pivot metrics_df so each (model, condition, task, metric) is one column
metrics_df['col'] = (metrics_df['task'] + '__' + metrics_df['model'] + '__' +
                     metrics_df['condition'] + '__' + metrics_df['metric_name'])
wide = metrics_df.pivot_table(index='pid', columns='col', values='value', aggfunc='mean')
wide = wide.reset_index().rename(columns={'pid': 'PID_str'})
joined = joined.merge(wide, on='PID_str', how='left')
joined.to_csv(OUT_DIR / 'per_pid_joined.csv', index=False)
print(f'    {len(joined)} PIDs × {len(joined.columns)} columns saved.')

# ─── 5. Compact markdown per-family summary ─────────────────────────────
print('\n[5] Writing markdown summaries ...')
md_compact = []
md_compact.append('# Test results — compact per-family tables\n\n')
md_compact.append('FDR within (task × test) family. Sorted by absolute effect size.\n\n')
for (task, test), grp in tests_df.groupby(['task', 'test']):
    md_compact.append(f'## {task} — {test}\n\n')
    g = grp.copy().sort_values('effect_size', key=lambda s: s.abs(), ascending=False)
    md_compact.append('| model | condition | n | effect | p_raw | p_fdr | sig |\n')
    md_compact.append('|---|---|---|---|---|---|---|\n')
    for _, r in g.iterrows():
        es = f'{r["effect_size"]:+.3f}'
        p_r = f'{r["p_raw"]:.3g}' if pd.notna(r['p_raw']) else 'N/A'
        p_f = f'{r["p_fdr"]:.3g}' if pd.notna(r['p_fdr']) else 'N/A'
        sig = '✅' if r['significant_05'] else ''
        md_compact.append(f'| {r["model"]} | {r["condition"]} | {r["n"]} '
                          f'| {es} | {p_r} | {p_f} | {sig} |\n')
    md_compact.append('\n')
(OUT_DIR / 'test_results_compact.md').write_text(''.join(md_compact), encoding='utf-8')

# ─── 6. SUMMARY.md ───────────────────────────────────────────────────────
print('\n[6] Writing SUMMARY.md ...')
md = []
md.append('# SSQ × Decoding interaction — analysis summary\n\n')
md.append(f'**Question**: Does cybersickness (SSQ_Total) moderate decoding/'
          f'prediction performance? Higher R²/AUC for High-SSQ PIDs would '
          f'suggest the model picks up sickness-related signal instead of '
          f'pure anxiety.\n\n')

md.append('## Sample\n\n')
md.append(f'- N = {len(ssq)} PIDs (excludes {sorted(EXCLUDED_PIDS)})\n')
md.append(f'- SSQ-Bad (Post SSQ_Total ≥ {SSQ_BAD_THRESH}, Kennedy moderate cutoff): '
          f'n = {int(ssq["SSQ_Bad"].sum())}\n')
md.append(f'- SSQ-OK: n = {int((1 - ssq["SSQ_Bad"]).sum())}\n')
md.append(f'- Post SSQ_Total: mean = {ssq["Post_SSQ_Total"].mean():.2f}, '
          f'std = {ssq["Post_SSQ_Total"].std():.2f}, '
          f'range = [{ssq["Post_SSQ_Total"].min():.1f}, {ssq["Post_SSQ_Total"].max():.1f}]\n')
md.append(f'- Pre SSQ_Total:  mean = {ssq["Pre_SSQ_Total"].mean():.2f}, '
          f'std = {ssq["Pre_SSQ_Total"].std():.2f}\n')
md.append(f'- Δ SSQ (Post − Pre): mean = {ssq["Delta_SSQ"].mean():.2f}, '
          f'std = {ssq["Delta_SSQ"].std():.2f}\n\n')

# Significant tests
sig_tests = tests_df[tests_df['significant_05']].copy()
md.append(f'## Significant interactions after BH-FDR (n = {len(sig_tests)} / '
          f'{len(tests_df)})\n\n')
if len(sig_tests) == 0:
    md.append('**No significant SSQ moderation of decoding performance survives '
              'BH-FDR correction in any task family.**\n\n')
    md.append('Interpretation: anxiety prediction performance is independent of '
              'cybersickness level across all tested tasks (T1-A, T2-A, T3-A, '
              'T3-W) and models. This supports the validity of the decoding '
              'pipeline — performance is not driven by sickness-related '
              'physiological/behavioral changes.\n\n')
else:
    md.append('| task | model | condition | test | effect | p_fdr |\n')
    md.append('|---|---|---|---|---|---|\n')
    for _, r in sig_tests.iterrows():
        md.append(f'| {r["task"]} | {r["model"]} | {r["condition"]} | '
                  f'{r["test"]} | {r["effect_size"]:+.3f} | {r["p_fdr"]:.3g} |\n')
    md.append('\n')

# Per-family counts
md.append('## Tests by family (task × test type)\n\n')
md.append('| task | test | n_tests | n_sig (FDR 0.05) | min p_raw |\n')
md.append('|---|---|---|---|---|\n')
for (task, test), grp in tests_df.groupby(['task', 'test']):
    md.append(f'| {task} | {test} | {len(grp)} | '
              f'{int(grp["significant_05"].sum())} | '
              f'{grp["p_raw"].min():.3g} |\n')
md.append('\n')

# Files
md.append('## Files\n\n')
md.append('| File | Description |\n|---|---|\n')
md.append('| `run_ssq_decoding_interaction.py` | This analysis script |\n')
md.append('| `per_pid_joined.csv` | PID × (Pre/Post/Δ SSQ + all decoding metrics) for re-use |\n')
md.append('| `test_results.csv` | Long table: every test with raw + FDR p, effect size |\n')
md.append('| `test_results_compact.md` | Per-family markdown tables |\n')
md.append('| `SUMMARY.md` | This file |\n\n')

# Caveats
md.append('## Caveats\n\n')
md.append('1. **SSQ_Total uses simple sum × 3.74**, matching `compute_trait_scores.py` '
          '(not the standard Kennedy subscale weighting). Subscale-level '
          'analyses (N/O/D) are still pending — see Handoff doc #4.\n')
md.append('2. **N=106 baseline; ~100 for T2-A** (6 PIDs lost a single-class fold '
          'during LOPO and were dropped per `T2A_raw_perpid.csv`).\n')
md.append('3. **Δ SSQ assumes Pre vs Post comparable**: same 16 items, same scoring. '
          'For 4 PIDs missing Pre, Δ is NaN and they are excluded from the Δ test.\n')
md.append('4. **FDR is within (task × test) family**, not across the whole table; '
          'a global Bonferroni correction would be more conservative.\n')

(OUT_DIR / 'SUMMARY.md').write_text(''.join(md), encoding='utf-8')
print(f'\nAll outputs in {OUT_DIR}')
print('  - SUMMARY.md       ← read first')
print('  - test_results.csv ← full results')
print('  - per_pid_joined.csv ← for downstream re-use')
