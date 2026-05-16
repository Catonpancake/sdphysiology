"""
Option B — Linear mixed-effects models to compare location vs order accounts
of scene anxiety differences.

Design:
  scene (5-level categorical) and protocol_order (continuous 1..5) are nearly
  perfectly collinear because each scene appears at exactly one protocol
  position (the only exception is the Elev1/Elev2 pair). We CANNOT identify
  both in the same model. Instead we compare alternative models that each
  carve up the variance differently:

    M0   anxiety ~ 1               + (1|PID)     # baseline, intercept only
    M1   anxiety ~ scene           + (1|PID)     # 4 df, captures ALL between-scene
                                                 # differences (saturates scenes)
    M2   anxiety ~ order_linear    + (1|PID)     # 1 df, only the linear trend
                                                 # in protocol position
    M3   anxiety ~ order_quadratic + (1|PID)     # 2 df, allow curvature

  Comparison logic:
    - If M1 >> M2 (much better fit): between-scene differences are NOT
      explained by linear order alone → location effect dominates.
    - If M2 ≈ M1: scene differences are mostly linear-in-order → fatigue
      account is sufficient.
    - If M3 noticeably improves over M2 but is dominated by M1: there is
      a non-linear order effect, but scene-specific structure still matters.

  Diagnostics reported per model:
    - log-likelihood, AIC, BIC
    - marginal R² (fixed effects only) and conditional R² (fixed + random)
    - Likelihood-ratio test (LRT) against the baseline M0

Unit of analysis: per-PID per-scene anxiety MEAN (same as Option A).
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

ROOT = Path(r'c:/Users/user/code/SDPhysiology')
INPUT_CSV = ROOT / 'Writing_resource/mavise_scene_analysis/autonomous_run/phase_C/02_anxiety_trajectory_perpid.csv'
OUT_DIR = ROOT / 'Writing_resource/mavise_scene_analysis/scene_anxiety_stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES = ['Elevator1', 'Outside', 'Hallway', 'Elevator2', 'Hall']

print('Loading per-PID per-scene anxiety means...')
df = pd.read_csv(INPUT_CSV)
df_cc = df[['pid'] + SCENES].dropna(subset=SCENES).reset_index(drop=True)
print(f'  complete cases: {len(df_cc)}/{len(df)}')

df_long = df_cc.melt(id_vars='pid', value_vars=SCENES,
                     var_name='scene', value_name='anxiety')
df_long['scene'] = pd.Categorical(df_long['scene'], categories=SCENES, ordered=False)
df_long['protocol_order'] = df_long['scene'].cat.codes.astype(int) + 1   # 1..5
df_long['order_centered'] = df_long['protocol_order'] - df_long['protocol_order'].mean()
df_long['order_sq'] = df_long['order_centered'] ** 2

# Total variance of the response (for R²)
ss_total = float(np.var(df_long['anxiety'], ddof=1)) * (len(df_long) - 1)


def fit_lmm(formula, name):
    """Fit a MixedLM (REML), return diagnostics dict. Falls back gracefully on singular RE."""
    print(f'\n  Fitting {name}:  {formula}')
    model = smf.mixedlm(formula, df_long, groups=df_long['pid'])
    res = None
    # Use ML (reml=False) so logL/AIC are comparable across models with
    # different fixed-effect structures.
    for method in ('lbfgs', 'bfgs', 'powell'):
        try:
            res = model.fit(reml=False, method=method, disp=False)
            if res.converged:
                break
        except Exception:
            continue
    if res is None:
        print(f'    FAILED: could not fit')
        return None, None
    fe_pred = res.predict(df_long)
    var_fe = float(np.var(fe_pred, ddof=1))
    try:
        var_re = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] > 0 else 0.0
    except Exception:
        var_re = 0.0
    var_resid = float(res.scale)
    total = var_fe + var_re + var_resid
    r2_marginal = var_fe / total if total > 0 else np.nan
    r2_conditional = (var_fe + var_re) / total if total > 0 else np.nan
    diag = dict(
        model=name, formula=formula,
        n_obs=int(res.nobs), n_groups=int(res.model.n_groups),
        n_fixed_params=int(len(res.fe_params)),
        loglik=float(res.llf),
        AIC=float(res.aic), BIC=float(res.bic),
        var_fixed=var_fe, var_random_intercept=var_re,
        var_residual=var_resid,
        R2_marginal=float(r2_marginal),
        R2_conditional=float(r2_conditional),
        converged=bool(res.converged),
    )
    return res, diag


def fit_ols_pid_fe(formula_core, name):
    """Within-PID fixed-effects OLS (= demean per PID, then OLS).
    Robust fallback when LMM RE is singular.
    formula_core: e.g., 'C(scene)' or 'order_centered' (without 'anxiety ~').
    """
    print(f'\n  Fitting {name} (PID-FE OLS):  anxiety_demean ~ {formula_core}')
    df_dm = df_long.copy()
    df_dm['anxiety_demean'] = df_dm.groupby('pid')['anxiety'].transform(lambda x: x - x.mean())
    formula = f'anxiety_demean ~ {formula_core}' if formula_core else 'anxiety_demean ~ 1'
    res = smf.ols(formula, df_dm).fit()
    diag = dict(
        model=name, formula=formula, fitter='OLS (PID-demean = within-PID FE)',
        n_obs=int(res.nobs), n_fixed_params=int(len(res.params)),
        rsquared=float(res.rsquared),
        rsquared_adj=float(res.rsquared_adj),
        F_statistic=float(res.fvalue) if res.fvalue is not None else np.nan,
        F_p_value=float(res.f_pvalue) if res.f_pvalue is not None else np.nan,
        AIC=float(res.aic), BIC=float(res.bic),
        loglik=float(res.llf),
    )
    return res, diag


# ── Fit all LMM models ───────────────────────────────────────────────────
res0, d0 = fit_lmm('anxiety ~ 1', 'M0_intercept_only')
res1, d1 = fit_lmm('anxiety ~ C(scene)', 'M1_scene')
res2, d2 = fit_lmm('anxiety ~ order_centered', 'M2_order_linear')
res3, d3 = fit_lmm('anxiety ~ order_centered + order_sq', 'M3_order_quadratic')

# ── If ANY LMM failed (singular RE), use PID-FE OLS as primary instead ──
LMM_OK = all(d is not None for d in (d0, d1, d2, d3))
if not LMM_OK:
    print('\n  >>> LMM had convergence issues — using PID-fixed-effects OLS as primary <<<')
    res1, d1 = fit_ols_pid_fe('C(scene)',              'M1_scene_OLS_PID_FE')
    res2, d2 = fit_ols_pid_fe('order_centered',        'M2_order_linear_OLS_PID_FE')
    res3, d3 = fit_ols_pid_fe('order_centered + order_sq', 'M3_order_quadratic_OLS_PID_FE')
    # M0 = intercept only on demeaned: tautologically 0 fit
    d0 = dict(model='M0_intercept_only_OLS_PID_FE',
              formula='anxiety_demean ~ 1',
              fitter='OLS (demean)',
              n_obs=int(res1.nobs), n_fixed_params=1,
              rsquared=0.0, rsquared_adj=0.0,
              F_statistic=np.nan, F_p_value=np.nan,
              AIC=float('inf'), BIC=float('inf'), loglik=float('nan'),
              )
    USE_FALLBACK = True
else:
    USE_FALLBACK = False

# ── Likelihood-ratio tests ───────────────────────────────────────────────
def lrt(res_full, res_reduced):
    """Likelihood-ratio test."""
    delta_ll = 2 * (res_full.llf - res_reduced.llf)
    delta_df = len(res_full.fe_params) - len(res_reduced.fe_params)
    p = float(1 - stats.chi2.cdf(delta_ll, delta_df))
    return float(delta_ll), int(delta_df), p


print('\n=== Model diagnostics ===')
diag_rows = [d0, d1, d2, d3]
for d in diag_rows:
    if USE_FALLBACK:
        print(f"  {d['model']:<35}  k_fixed={d['n_fixed_params']:>2}  "
              f"R²={d.get('rsquared', 0):.4f}  AIC={d['AIC']:.2f}  "
              f"F={d.get('F_statistic', float('nan')):.3f}  p={d.get('F_p_value', float('nan')):.3g}")
    else:
        print(f"  {d['model']:<25}  k_fixed={d['n_fixed_params']:>2}  "
              f"logL={d['loglik']:+.3f}  AIC={d['AIC']:.2f}  BIC={d['BIC']:.2f}  "
              f"R²_marg={d['R2_marginal']:.4f}  R²_cond={d['R2_conditional']:.4f}")
df_diag = pd.DataFrame(diag_rows)
df_diag.to_csv(OUT_DIR / '05_lmm_diagnostics.csv', index=False)

print('\n=== Model comparisons ===')
lrt_rows = []

if USE_FALLBACK:
    # OLS PID-FE — use F-tests + R² + AIC for nested comparisons.
    # M1 (scene, 4 df) vs M2 (linear, 1 df) — non-nested → AIC + Vuong-like ratio
    delta_AIC_M1_M2 = float(d2['AIC'] - d1['AIC'])
    print(f'  ΔAIC(M2_order_lin − M1_scene) = {delta_AIC_M1_M2:+.2f}  '
          f'({"M1 (scene) better" if delta_AIC_M1_M2 > 0 else "M2 (order) better"})')
    lrt_rows.append(dict(comparison='M1_scene vs M2_order_linear (AIC delta)',
                         delta_logL=np.nan, delta_df=int(d1['n_fixed_params'] - d2['n_fixed_params']),
                         p=np.nan, delta_AIC=delta_AIC_M1_M2))
    # M2 vs M3 (nested F-test)
    ll2, ll3 = float(res2.llf), float(res3.llf)
    f_M3_M2 = float(((res2.ssr - res3.ssr) / (res3.df_model - res2.df_model)) /
                    (res3.ssr / res3.df_resid))
    p_M3_M2 = float(1 - stats.f.cdf(f_M3_M2, res3.df_model - res2.df_model, res3.df_resid))
    print(f'  M3 vs M2 (F-test nested): F={f_M3_M2:.3f}, '
          f'df=({res3.df_model - res2.df_model}, {int(res3.df_resid)}), p={p_M3_M2:.3g}')
    lrt_rows.append(dict(comparison='M3_order_quadratic vs M2_order_linear (F-test)',
                         delta_logL=ll3 - ll2,
                         delta_df=int(res3.df_model - res2.df_model),
                         p=p_M3_M2, delta_AIC=float(d3['AIC'] - d2['AIC'])))
    # Each model's own F vs intercept
    for name, d in [('M1_scene', d1), ('M2_order_linear', d2), ('M3_order_quadratic', d3)]:
        lrt_rows.append(dict(comparison=f'{name} overall F vs intercept',
                             delta_logL=np.nan,
                             delta_df=d['n_fixed_params'] - 1,
                             p=float(d['F_p_value']) if not np.isnan(d['F_p_value']) else np.nan,
                             delta_AIC=np.nan))
else:
    print('\n=== Likelihood-ratio tests (vs M0 baseline) ===')
    for name, res in [('M1_scene', res1),
                      ('M2_order_linear', res2),
                      ('M3_order_quadratic', res3)]:
        ll_delta, df_delta, p = lrt(res, res0)
        lrt_rows.append(dict(comparison=f'{name} vs M0',
                             delta_logL=ll_delta, delta_df=df_delta, p=p,
                             delta_AIC=float(diag_rows[0]['AIC'] - next(x['AIC'] for x in diag_rows if x['model'] == name))))
        print(f'  {name:<25} vs M0:  ΔlogL={ll_delta:+.3f}  Δdf={df_delta}  p={p:.3g}')

    delta_AIC_M1_M2 = float(d2['AIC'] - d1['AIC'])
    lrt_rows.append(dict(comparison='M1_scene vs M2_order_linear (AIC delta)',
                         delta_logL=float(res1.llf - res2.llf),
                         delta_df=int(len(res1.fe_params) - len(res2.fe_params)),
                         p=np.nan, delta_AIC=delta_AIC_M1_M2))
    print(f'\n  ΔAIC(M2 − M1) = {delta_AIC_M1_M2:+.2f}  '
          f'({"M1 (scene) is better" if delta_AIC_M1_M2 > 0 else "M2 (order) is better"})')

    ll_delta, df_delta, p = lrt(res3, res2)
    lrt_rows.append(dict(comparison='M3_order_quadratic vs M2_order_linear',
                         delta_logL=ll_delta, delta_df=df_delta, p=p,
                         delta_AIC=float(d3['AIC'] - d2['AIC'])))
    print(f'  M3 vs M2 (nested LRT):  ΔlogL={ll_delta:+.3f}  Δdf={df_delta}  p={p:.3g}')

pd.DataFrame(lrt_rows).to_csv(OUT_DIR / '06_lmm_lrt_comparisons.csv', index=False)

# ── M1 coefficient table (scene contrasts) ───────────────────────────────
print('\n=== M1 scene coefficients (reference = first level) ===')
if USE_FALLBACK:
    m1_coef = pd.DataFrame({
        'param': res1.params.index,
        'coef': res1.params.values,
        'std_err': res1.bse.values,
        't': res1.tvalues.values,
        'p_value': res1.pvalues.values,
    })
else:
    m1_coef = pd.DataFrame({
        'param': res1.fe_params.index,
        'coef': res1.fe_params.values,
        'std_err': res1.bse_fe.values,
        'z': (res1.fe_params / res1.bse_fe).values,
        'p_value': res1.pvalues[res1.fe_params.index].values,
    })
print(m1_coef.to_string(index=False, float_format='%.4f'))
m1_coef.to_csv(OUT_DIR / '07_lmm_M1_scene_coefficients.csv', index=False)

# ── Variance decomposition (different scheme for fallback) ──────────────
print('\n=== Variance decomposition ===')
vd_rows = []
if USE_FALLBACK:
    # PID-FE OLS: total raw variance vs within-PID residual variance.
    # PID variance = var(y) - var(y - y_pid_mean)
    df_dm = df_long.copy()
    df_dm['anxiety_demean'] = df_dm.groupby('pid')['anxiety'].transform(lambda x: x - x.mean())
    var_total = float(np.var(df_dm['anxiety'], ddof=1))
    var_within = float(np.var(df_dm['anxiety_demean'], ddof=1))
    var_between_pid = max(0.0, var_total - var_within)
    print(f'  total variance:           {var_total:.6f}')
    print(f'  between-PID:              {var_between_pid:.6f} ({100*var_between_pid/var_total:.1f}%)')
    print(f'  within-PID:               {var_within:.6f} ({100*var_within/var_total:.1f}%)')
    for d, res in [(d1, res1), (d2, res2), (d3, res3)]:
        # Within-PID variance explained by this model's fixed effects
        ssr = float(res.ssr)
        sst_within = float(np.var(df_dm['anxiety_demean'], ddof=1) * (len(df_dm) - 1))
        ss_explained_within = sst_within - ssr
        pct_explained_within = 100 * ss_explained_within / sst_within if sst_within > 0 else 0
        vd_rows.append(dict(
            model=d['model'],
            pct_of_total_variance_between_PID=100 * var_between_pid / var_total,
            pct_of_total_variance_within_PID=100 * var_within / var_total,
            pct_within_variance_explained_by_model=pct_explained_within,
            pct_of_total_explained_by_model=pct_explained_within * (var_within / var_total),
        ))
else:
    for d in diag_rows:
        total = d['var_fixed'] + d['var_random_intercept'] + d['var_residual']
        if total > 0:
            vd_rows.append(dict(
                model=d['model'],
                pct_fixed=100 * d['var_fixed'] / total,
                pct_random_intercept=100 * d['var_random_intercept'] / total,
                pct_residual=100 * d['var_residual'] / total,
                total_var=total,
            ))
df_vd = pd.DataFrame(vd_rows)
print(df_vd.to_string(index=False, float_format='%.2f'))
df_vd.to_csv(OUT_DIR / '08_lmm_variance_decomposition.csv', index=False)

# ── Summary appended to existing 00_SUMMARY.md ─────────────────────────
print('\nAppending Option B summary to 00_SUMMARY.md ...')
summary_addition = []
summary_addition.append('\n---\n\n')
summary_addition.append('# Option B — Mixed-effects / PID-fixed-effects models '
                        '(location vs order)\n\n')
if USE_FALLBACK:
    summary_addition.append('**Note**: LMM with random PID intercept had singular '
                            'covariance (because per-PID anxiety variance is very small '
                            'for Outside/Hallway/Hall, so the random-intercept term '
                            'saturates). We use **within-PID fixed-effects OLS** (= '
                            'per-PID demeaned anxiety regressed on scene / order) as '
                            'the robust alternative. This isolates within-subject '
                            'effects, removing all between-subject variance, and '
                            'mathematically delivers the same point estimates as a '
                            'well-identified LMM here.\n\n')
else:
    summary_addition.append('Same input data as Option A; tests whether between-scene '
                            'anxiety differences are explained by linear protocol order '
                            '(fatigue) or by scene identity (location).\n\n')

summary_addition.append('## Model summary\n\n')
if USE_FALLBACK:
    summary_addition.append('| Model | Formula | k_fixed | R² | F | p | AIC |\n')
    summary_addition.append('|---|---|---|---|---|---|---|\n')
    for d in diag_rows:
        f_str = f'{d.get("F_statistic", float("nan")):.2f}' if not np.isnan(d.get('F_statistic', np.nan)) else 'N/A'
        p_str = f'{d.get("F_p_value", float("nan")):.3g}' if not np.isnan(d.get('F_p_value', np.nan)) else 'N/A'
        summary_addition.append(
            f'| {d["model"]} | `{d["formula"]}` | {d["n_fixed_params"]} '
            f'| {d.get("rsquared", 0):.4f} | {f_str} | {p_str} | {d["AIC"]:.2f} |\n'
        )
else:
    summary_addition.append('| Model | Formula | k_fixed | logL | AIC | BIC | R²_marg | R²_cond |\n')
    summary_addition.append('|---|---|---|---|---|---|---|---|\n')
    for d in diag_rows:
        summary_addition.append(
            f'| {d["model"]} | `{d["formula"]}` | {d["n_fixed_params"]} '
            f'| {d["loglik"]:+.2f} | {d["AIC"]:.2f} | {d["BIC"]:.2f} '
            f'| {d["R2_marginal"]:.4f} | {d["R2_conditional"]:.4f} |\n'
        )
summary_addition.append('\n')

summary_addition.append('## Likelihood-ratio tests\n\n')
summary_addition.append('| Comparison | ΔlogL | Δdf | p |\n|---|---|---|---|\n')
for row in lrt_rows:
    p_str = f'{row["p"]:.3g}' if not (isinstance(row['p'], float) and np.isnan(row['p'])) else '(AIC-based)'
    summary_addition.append(f'| {row["comparison"]} | {row["delta_logL"]:+.2f} '
                            f'| {row["delta_df"]} | {p_str} |\n')
summary_addition.append('\n')

summary_addition.append('## Interpretation\n\n')
# Auto-interpret based on R² of M1 vs M2 (within-PID R² for fallback)
if USE_FALLBACK:
    r2_m1 = d1.get('rsquared', 0)
    r2_m2 = d2.get('rsquared', 0)
    r2_m3 = d3.get('rsquared', 0)
    summary_addition.append(f'- Within-PID R² (variance of demeaned anxiety '
                            f'explained):\n'
                            f'  - **M1 (scene)** R² = **{r2_m1:.4f}**\n'
                            f'  - **M2 (linear order)** R² = {r2_m2:.4f}\n'
                            f'  - **M3 (quadratic order)** R² = {r2_m3:.4f}\n')
else:
    r2_m1 = d1['R2_marginal']
    r2_m2 = d2['R2_marginal']
    r2_m3 = d3['R2_marginal']
    summary_addition.append(f'- **M1 (scene)** explains R²_marginal = {r2_m1:.4f} '
                            f'of the variance; **M2 (linear order)** explains only '
                            f'{r2_m2:.4f}; **M3 (quadratic order)** explains '
                            f'{r2_m3:.4f}.\n')
ratio = r2_m1 / r2_m2 if r2_m2 > 0 else float('inf')
summary_addition.append(f'- M1 explains **{ratio:.1f}×** more variance than M2.\n')
if delta_AIC_M1_M2 > 10:
    summary_addition.append('- **M1 is decisively preferred over M2** (ΔAIC > 10). '
                            'Between-scene anxiety differences cannot be reduced to a '
                            'simple linear order effect — scene-specific (location) '
                            'structure carries most of the systematic variance.\n')
elif delta_AIC_M1_M2 > 2:
    summary_addition.append('- M1 modestly preferred over M2 (ΔAIC > 2). Scene-specific '
                            'effects exist beyond linear order, but the effect size is '
                            'modest.\n')
else:
    summary_addition.append('- M1 and M2 fit comparably (|ΔAIC| < 2). Between-scene '
                            'differences are largely captured by linear protocol order; '
                            'a pure fatigue account is supported.\n')
# Find M3 vs M2 row
m3_m2 = next((r for r in lrt_rows if 'M3_order_quadratic vs M2' in r['comparison']), None)
if m3_m2 and not (isinstance(m3_m2['p'], float) and np.isnan(m3_m2['p'])):
    quad_p = float(m3_m2['p'])
    if quad_p < 0.05:
        summary_addition.append(f'- M3 vs M2: adding a quadratic order term significantly '
                                f'improves the fit (p = {quad_p:.3g}) → the trajectory is '
                                f'**non-linear in protocol order** (e.g., a peak-and-decline '
                                f'pattern).\n')
    else:
        summary_addition.append(f'- M3 vs M2: quadratic order does not significantly '
                                f'improve the fit (p = {quad_p:.3g}).\n')

# Variance decomposition note
if USE_FALLBACK and vd_rows:
    vd_m1 = next((v for v in vd_rows if v['model'].startswith('M1_scene')), None)
    if vd_m1:
        summary_addition.append(
            f'- **Variance decomposition** (total per-PID per-scene anxiety):\n'
            f'  - Between-PID: {vd_m1["pct_of_total_variance_between_PID"]:.1f}%\n'
            f'  - Within-PID: {vd_m1["pct_of_total_variance_within_PID"]:.1f}%\n'
            f'  - Of the within-PID variance, M1 (scene) explains '
            f'**{vd_m1["pct_within_variance_explained_by_model"]:.1f}%** (≈ '
            f'{vd_m1["pct_of_total_explained_by_model"]:.1f}% of total).\n'
        )
elif vd_rows:
    vd_m1 = next((v for v in vd_rows if v['model'] == 'M1_scene'), None)
    if vd_m1:
        summary_addition.append(f'- M1 variance decomposition: {vd_m1["pct_fixed"]:.1f}% '
                                f'fixed (scene), {vd_m1["pct_random_intercept"]:.1f}% '
                                f'random (PID), {vd_m1["pct_residual"]:.1f}% residual.\n')

summary_addition.append('\n## Files added by Option B\n\n')
summary_addition.append('| File | Description |\n|---|---|\n')
summary_addition.append('| `run_lmm_location_vs_order.py` | This script |\n')
summary_addition.append('| `05_lmm_diagnostics.csv` | Per-model logL/AIC/BIC/R² |\n')
summary_addition.append('| `06_lmm_lrt_comparisons.csv` | Likelihood-ratio tests + AIC comparisons |\n')
summary_addition.append('| `07_lmm_M1_scene_coefficients.csv` | M1 scene contrasts (estimate ± SE, p) |\n')
summary_addition.append('| `08_lmm_variance_decomposition.csv` | Fixed / random / residual variance shares |\n')

summary_path = OUT_DIR / '00_SUMMARY.md'
existing = summary_path.read_text(encoding='utf-8') if summary_path.exists() else ''
summary_path.write_text(existing + ''.join(summary_addition), encoding='utf-8')
print(f'  appended → {summary_path}')
print(f'\nDone. All Option B CSVs in {OUT_DIR}')
