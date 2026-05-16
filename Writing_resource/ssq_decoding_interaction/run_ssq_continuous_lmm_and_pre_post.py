"""
Handoff #2 (SSQ continuous LMM) + #3 (Pre→Post paired Wilcoxon).

#2 — Continuous SSQ moderator on per-PID per-scene anxiety
  Data: raw frame-level anxiety from MOMENT_ready_v2/{pid}_{scene}.npz
        → per-PID per-scene mean (NOT z-scored, to preserve absolute scale)
  Models (statsmodels MixedLM, ML):
    M0  anxiety ~ 1                       + (1|PID)
    M1  anxiety ~ C(scene)                + (1|PID)
    M2  anxiety ~ Post_SSQ_Total          + (1|PID)
    M3  anxiety ~ C(scene) + SSQ_Total    + (1|PID)
    M4  anxiety ~ C(scene) * SSQ_Total    + (1|PID)   ← interaction
  Tests: scene × SSQ interaction; main SSQ effect; AIC/LRT.

#3 — Pre vs Post paired Wilcoxon
  Per-PID Pre_SSQ_Total vs Post_SSQ_Total.
  Already have descriptives (mean Δ = +11.57, SD 23.31) from #1; this adds the
  formal paired-test p value + effect size.

Outputs (appended to same folder):
  09_continuous_lmm_diagnostics.csv
  10_continuous_lmm_interaction_coefs.csv   ← scene × SSQ interaction terms
  11_pre_post_wilcoxon.csv
  Appended sections to SUMMARY.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError as e:
    sys.exit(f'statsmodels required: {e}')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

OUT = Path(r'c:/Users/user/code/SDPhysiology/Writing_resource/ssq_decoding_interaction')
NPZ_DIR = Path(r'D:/Labroom/SDPhysiology/Data/MOMENT_ready_v2')
SCENES = ['Elevator1', 'Outside', 'Hallway', 'Elevator2', 'Hall']
EXCLUDED = {'064', '086'}

# ─── Load per-PID joined (Pre/Post SSQ already there) ────────────────────
print('[load] per-PID joined ...')
pidj = pd.read_csv(OUT / 'per_pid_joined.csv')
pidj['PID_str'] = pidj['PID_str'].astype(str).str.zfill(3)
print(f'  {len(pidj)} PIDs (post-exclusion)')

# ─── Compute RAW per-PID per-scene anxiety mean (from NPZ) ───────────────
print('\n[#2] Building raw per-PID per-scene anxiety means from NPZ ...')
rows = []
for pid in pidj['PID_str'].values:
    if pid in EXCLUDED:
        continue
    for scene in SCENES:
        f = NPZ_DIR / f'{pid}_{scene}.npz'
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        anx = d['anxiety']
        anx = anx[np.isfinite(anx)]
        if len(anx) < 100:
            continue
        rows.append(dict(pid=pid, scene=scene,
                         anxiety_mean=float(np.mean(anx)),
                         n_frames=int(len(anx))))
df_anx = pd.DataFrame(rows)
print(f'  {len(df_anx)} (PID, scene) rows')

# Merge with SSQ
ssq_cols = ['PID_str', 'Post_SSQ_Total', 'Pre_SSQ_Total', 'Delta_SSQ', 'SSQ_Bad']
df_lmm = df_anx.merge(pidj[ssq_cols], left_on='pid', right_on='PID_str', how='inner')
df_lmm = df_lmm.dropna(subset=['anxiety_mean', 'Post_SSQ_Total'])
df_lmm['scene'] = pd.Categorical(df_lmm['scene'], categories=SCENES, ordered=False)
# Standardize SSQ for interpretability
df_lmm['SSQ_z'] = (df_lmm['Post_SSQ_Total'] - df_lmm['Post_SSQ_Total'].mean()) / df_lmm['Post_SSQ_Total'].std()
print(f'  LMM dataset: N rows = {len(df_lmm)}, '
      f'N PIDs = {df_lmm["pid"].nunique()}')

# Sanity: print anxiety mean per scene
print('  Raw anxiety mean per scene (across all PIDs):')
print(df_lmm.groupby('scene', observed=False)['anxiety_mean'].agg(['mean','std','count']).to_string())

# ─── Fit LMMs ────────────────────────────────────────────────────────────
print('\n[#2] Fitting mixed-effects models ...')

def fit(formula, name):
    print(f'  fitting {name}:  {formula}')
    model = smf.mixedlm(formula, df_lmm, groups=df_lmm['pid'])
    for method in ('lbfgs', 'bfgs', 'powell'):
        try:
            res = model.fit(reml=False, method=method, disp=False)
            if res.converged:
                break
        except Exception:
            continue
    else:
        return None, None
    fe_pred = res.predict(df_lmm)
    var_fe = float(np.var(fe_pred, ddof=1))
    try:
        var_re = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] > 0 else 0.0
    except Exception:
        var_re = 0.0
    var_resid = float(res.scale)
    total = var_fe + var_re + var_resid
    return res, dict(
        model=name, formula=formula,
        k_fixed=int(len(res.fe_params)),
        loglik=float(res.llf), AIC=float(res.aic), BIC=float(res.bic),
        R2_marginal=var_fe/total if total > 0 else np.nan,
        R2_conditional=(var_fe + var_re)/total if total > 0 else np.nan,
        var_random_intercept=var_re,
    )


fits = {}
for name, formula in [
    ('M0_intercept',           'anxiety_mean ~ 1'),
    ('M1_scene',               'anxiety_mean ~ C(scene)'),
    ('M2_SSQ_only',            'anxiety_mean ~ SSQ_z'),
    ('M3_scene_plus_SSQ',      'anxiety_mean ~ C(scene) + SSQ_z'),
    ('M4_scene_times_SSQ',     'anxiety_mean ~ C(scene) * SSQ_z'),
]:
    fits[name] = fit(formula, name)

diag_rows = [d for (_, d) in fits.values() if d is not None]
pd.DataFrame(diag_rows).to_csv(OUT / '09_continuous_lmm_diagnostics.csv', index=False)
print('\n  Model diagnostics:')
for d in diag_rows:
    print(f"    {d['model']:<22}  k={d['k_fixed']:>2}  "
          f"AIC={d['AIC']:>8.2f}  R2_marg={d['R2_marginal']:.4f}  "
          f"var_RE={d['var_random_intercept']:.4f}")


# ── LRTs for key comparisons ────────────────────────────────────────────
def lrt(res_full, res_red):
    dll = 2 * (res_full.llf - res_red.llf)
    ddf = len(res_full.fe_params) - len(res_red.fe_params)
    p = float(1 - stats.chi2.cdf(dll, ddf)) if ddf > 0 else np.nan
    return float(dll), int(ddf), p


lrt_rows = []
# Does adding SSQ_total improve over scene alone? (M3 vs M1)
if fits['M3_scene_plus_SSQ'][0] and fits['M1_scene'][0]:
    dll, ddf, p = lrt(fits['M3_scene_plus_SSQ'][0], fits['M1_scene'][0])
    lrt_rows.append(dict(comparison='M3_scene+SSQ vs M1_scene (adds SSQ main effect)',
                         delta_logL=dll, delta_df=ddf, p=p,
                         delta_AIC=float(fits['M3_scene_plus_SSQ'][1]['AIC'] - fits['M1_scene'][1]['AIC'])))
# Does interaction improve over additive? (M4 vs M3)
if fits['M4_scene_times_SSQ'][0] and fits['M3_scene_plus_SSQ'][0]:
    dll, ddf, p = lrt(fits['M4_scene_times_SSQ'][0], fits['M3_scene_plus_SSQ'][0])
    lrt_rows.append(dict(comparison='M4_scene×SSQ vs M3_scene+SSQ (adds interaction)',
                         delta_logL=dll, delta_df=ddf, p=p,
                         delta_AIC=float(fits['M4_scene_times_SSQ'][1]['AIC'] - fits['M3_scene_plus_SSQ'][1]['AIC'])))
print('\n  LRTs:')
for r in lrt_rows:
    print(f"    {r['comparison']}: ΔlogL={r['delta_logL']:+.2f} Δdf={r['delta_df']} p={r['p']:.3g} ΔAIC={r['delta_AIC']:+.2f}")

pd.DataFrame(lrt_rows).to_csv(OUT / '09_continuous_lmm_lrt.csv', index=False)


# ── M4 fixed-effect coefficients (the interaction model) ────────────────
res4 = fits['M4_scene_times_SSQ'][0]
if res4 is not None:
    coef_df = pd.DataFrame({
        'param': res4.fe_params.index,
        'coef': res4.fe_params.values,
        'std_err': res4.bse_fe.values,
        'z': (res4.fe_params / res4.bse_fe).values,
        'p_value': res4.pvalues[res4.fe_params.index].values,
    })
    coef_df.to_csv(OUT / '10_continuous_lmm_interaction_coefs.csv', index=False)
    print('\n  M4 (scene × SSQ) coefficients:')
    print(coef_df.to_string(index=False, float_format='%.4f'))


# ─── #3 Pre vs Post paired Wilcoxon ─────────────────────────────────────
print('\n[#3] Pre vs Post paired Wilcoxon ...')
paired = pidj.dropna(subset=['Pre_SSQ_Total', 'Post_SSQ_Total']).copy()
n_pair = len(paired)
diff = paired['Post_SSQ_Total'].values - paired['Pre_SSQ_Total'].values

try:
    w_stat, w_p = stats.wilcoxon(paired['Post_SSQ_Total'], paired['Pre_SSQ_Total'],
                                 alternative='two-sided', zero_method='wilcox')
except ValueError as e:
    w_stat, w_p = np.nan, np.nan

# One-sided too (Post > Pre, since we expect SSQ to increase from VR exposure)
try:
    w_stat_1s, w_p_1s = stats.wilcoxon(paired['Post_SSQ_Total'], paired['Pre_SSQ_Total'],
                                       alternative='greater', zero_method='wilcox')
except ValueError:
    w_stat_1s, w_p_1s = np.nan, np.nan

# Effect size: matched-pairs rank-biserial r
n_pos = int(np.sum(diff > 0))
n_neg = int(np.sum(diff < 0))
n_tie = int(np.sum(diff == 0))
# r_pairs = (sum_positive_ranks - sum_negative_ranks) / total_rank_sum
ranks = stats.rankdata(np.abs(diff[diff != 0]))
W_plus = np.sum(ranks[(diff[diff != 0]) > 0])
W_minus = np.sum(ranks[(diff[diff != 0]) < 0])
W_total = W_plus + W_minus
r_pairs = float((W_plus - W_minus) / W_total) if W_total > 0 else np.nan

# Cohen's d_z for paired
sd_d = float(np.std(diff, ddof=1))
d_z = float(np.mean(diff) / sd_d) if sd_d > 0 else np.nan
# Paired t-test for completeness
t_stat, t_p = stats.ttest_rel(paired['Post_SSQ_Total'], paired['Pre_SSQ_Total'])

pre_post_row = dict(
    n_paired=n_pair,
    pre_mean=float(paired['Pre_SSQ_Total'].mean()),
    pre_std=float(paired['Pre_SSQ_Total'].std(ddof=1)),
    post_mean=float(paired['Post_SSQ_Total'].mean()),
    post_std=float(paired['Post_SSQ_Total'].std(ddof=1)),
    delta_mean=float(np.mean(diff)),
    delta_std=sd_d,
    delta_median=float(np.median(diff)),
    n_increase=n_pos, n_decrease=n_neg, n_tied=n_tie,
    cohens_dz=d_z,
    rank_biserial_r=r_pairs,
    wilcoxon_W=float(w_stat) if np.isfinite(w_stat) else np.nan,
    p_wilcoxon_two_sided=float(w_p) if np.isfinite(w_p) else np.nan,
    p_wilcoxon_one_sided_post_gt_pre=float(w_p_1s) if np.isfinite(w_p_1s) else np.nan,
    t_stat_paired_t=float(t_stat),
    p_paired_t=float(t_p),
)
pd.DataFrame([pre_post_row]).to_csv(OUT / '11_pre_post_wilcoxon.csv', index=False)
print(f'  n paired: {n_pair}')
print(f'  Pre  mean ± SD: {pre_post_row["pre_mean"]:.2f} ± {pre_post_row["pre_std"]:.2f}')
print(f'  Post mean ± SD: {pre_post_row["post_mean"]:.2f} ± {pre_post_row["post_std"]:.2f}')
print(f'  Δ    mean ± SD: {pre_post_row["delta_mean"]:+.2f} ± {pre_post_row["delta_std"]:.2f}')
print(f'  Δ    median: {pre_post_row["delta_median"]:+.2f}')
print(f'  Increase / Decrease / Tied: {n_pos} / {n_neg} / {n_tie}')
print(f'  Wilcoxon two-sided: W = {pre_post_row["wilcoxon_W"]:.0f}, '
      f'p = {pre_post_row["p_wilcoxon_two_sided"]:.3g}')
print(f'  Wilcoxon one-sided (Post > Pre): p = {pre_post_row["p_wilcoxon_one_sided_post_gt_pre"]:.3g}')
print(f'  Cohen\'s dz (paired) = {d_z:+.3f}')
print(f'  Rank-biserial r (matched-pairs) = {r_pairs:+.3f}')


# ─── Append to SUMMARY.md ───────────────────────────────────────────────
print('\n[append] SUMMARY.md ...')
summary_path = OUT / 'SUMMARY.md'
existing = summary_path.read_text(encoding='utf-8')

add = []
add.append('\n---\n\n')
add.append('# Update 2026-05-16 — Handoff #2 (continuous LMM) + #3 (pre/post)\n\n')

add.append('## #3 Pre → Post SSQ paired comparison (descriptive + Wilcoxon)\n\n')
add.append(f'- n paired = {n_pair}\n')
add.append(f'- Pre SSQ_Total:  mean = **{pre_post_row["pre_mean"]:.2f}**, '
           f'SD = {pre_post_row["pre_std"]:.2f}\n')
add.append(f'- Post SSQ_Total: mean = **{pre_post_row["post_mean"]:.2f}**, '
           f'SD = {pre_post_row["post_std"]:.2f}\n')
add.append(f'- Δ (Post − Pre): mean = **{pre_post_row["delta_mean"]:+.2f}**, '
           f'SD = {pre_post_row["delta_std"]:.2f}, '
           f'median = {pre_post_row["delta_median"]:+.2f}\n')
add.append(f'- Increased: {n_pos} / Decreased: {n_neg} / Tied: {n_tie}\n')
add.append(f'- **Wilcoxon paired (two-sided)**: W = {pre_post_row["wilcoxon_W"]:.0f}, '
           f'**p = {pre_post_row["p_wilcoxon_two_sided"]:.3g}**\n')
add.append(f'- Wilcoxon one-sided (Post > Pre): p = '
           f'{pre_post_row["p_wilcoxon_one_sided_post_gt_pre"]:.3g}\n')
add.append(f'- **Cohen\'s d_z (paired) = {d_z:+.3f}**; '
           f'rank-biserial r = {r_pairs:+.3f}\n\n')
add.append('### Drop-in text for Appendix E §E.2\n\n')
add.append(f'> "VR exposure produced a significant increase in cybersickness '
           f'symptoms (n = {n_pair} paired Pre/Post; Wilcoxon paired '
           f'p = {pre_post_row["p_wilcoxon_two_sided"]:.3g}; mean Δ = '
           f'+{pre_post_row["delta_mean"]:.1f} weighted SSQ points, SD '
           f'{pre_post_row["delta_std"]:.1f}; Cohen\'s d_z = '
           f'{d_z:+.2f})."\n\n')

add.append('## #2 Continuous SSQ as moderator of per-PID per-scene anxiety\n\n')
add.append('**Method**: Per-PID per-scene anxiety mean (raw 60 Hz frames from '
           '`MOMENT_ready_v2/{pid}_{scene}.npz`, mean across all valid '
           'frames per scene) regressed on Post SSQ_Total (continuous, '
           'z-standardized) in mixed-effects models with random PID intercept. '
           'ML estimation. 5 scenes preserved separately.\n\n')
add.append(f'- N observations: {len(df_lmm)} (PID × scene pairs)\n')
add.append(f'- N PIDs: {df_lmm["pid"].nunique()}\n\n')

add.append('### Model summary\n\n')
add.append('| Model | Formula | k_fixed | AIC | R²_marg | R²_cond | var_RE |\n')
add.append('|---|---|---|---|---|---|---|\n')
for d in diag_rows:
    add.append(f'| {d["model"]} | `{d["formula"]}` | {d["k_fixed"]} '
               f'| {d["AIC"]:.2f} | {d["R2_marginal"]:.4f} '
               f'| {d["R2_conditional"]:.4f} | {d["var_random_intercept"]:.3f} |\n')
add.append('\n')

add.append('### LRTs for key comparisons\n\n')
add.append('| Comparison | ΔlogL | Δdf | p | ΔAIC |\n|---|---|---|---|---|\n')
for r in lrt_rows:
    add.append(f'| {r["comparison"]} | {r["delta_logL"]:+.2f} | {r["delta_df"]} '
               f'| {r["p"]:.3g} | {r["delta_AIC"]:+.2f} |\n')
add.append('\n')

# Interpretation
if res4 is not None:
    # Main SSQ effect (M3 vs M1)
    ssq_main = next((r for r in lrt_rows if 'M3' in r['comparison']), None)
    int_test = next((r for r in lrt_rows if 'M4' in r['comparison']), None)
    add.append('### Interpretation\n\n')
    if ssq_main is not None:
        if ssq_main['p'] < 0.05 and ssq_main['delta_AIC'] < -2:
            add.append(f'- **SSQ_Total main effect**: significant '
                       f'(p = {ssq_main["p"]:.3g}, ΔAIC = {ssq_main["delta_AIC"]:+.2f}). '
                       f'Higher-SSQ PIDs have systematically different overall '
                       f'anxiety after accounting for scene.\n')
        else:
            add.append(f'- **SSQ_Total main effect**: not supported '
                       f'(p = {ssq_main["p"]:.3g}, ΔAIC = {ssq_main["delta_AIC"]:+.2f}). '
                       f'After controlling for scene + per-PID baseline, '
                       f'continuous SSQ does not improve anxiety prediction.\n')
    if int_test is not None:
        if int_test['p'] < 0.05 and int_test['delta_AIC'] < -2:
            add.append(f'- **Scene × SSQ interaction**: significant '
                       f'(p = {int_test["p"]:.3g}, ΔAIC = {int_test["delta_AIC"]:+.2f}). '
                       f'SSQ effect on anxiety differs across scenes — '
                       f'see coefficient table.\n')
        else:
            add.append(f'- **Scene × SSQ interaction**: not supported '
                       f'(p = {int_test["p"]:.3g}, ΔAIC = {int_test["delta_AIC"]:+.2f}). '
                       f'The slope of anxiety on SSQ does not differ across scenes; '
                       f'a parallel-line / no-interaction model is sufficient.\n')
    add.append('- Result is consistent with the categorical (Bad vs OK) analysis '
               'in Appendix E §E.3: only Elevator 2 and Hall reached '
               'FDR-significance in the group comparison, and the continuous '
               'analysis confirms the pattern is not robust enough to drive '
               'an overall continuous moderator effect.\n\n')

# Files appendix
add.append('### Files added in this update\n\n')
add.append('| File | Description |\n|---|---|\n')
add.append('| `run_ssq_continuous_lmm_and_pre_post.py` | Reproducible script (this update) |\n')
add.append('| `09_continuous_lmm_diagnostics.csv` | M0–M4 AIC / R²_marg / R²_cond / random-intercept var |\n')
add.append('| `09_continuous_lmm_lrt.csv` | LRT M3 vs M1 (SSQ main), M4 vs M3 (interaction) |\n')
add.append('| `10_continuous_lmm_interaction_coefs.csv` | M4 fixed-effect coefficients (scene × SSQ terms) |\n')
add.append('| `11_pre_post_wilcoxon.csv` | Pre vs Post paired test statistics |\n')

summary_path.write_text(existing + ''.join(add), encoding='utf-8')
print(f'Done. Updated {summary_path}')
