# SSQ × Decoding interaction — analysis summary

**Question**: Does cybersickness (SSQ_Total) moderate decoding/prediction performance? Higher R²/AUC for High-SSQ PIDs would suggest the model picks up sickness-related signal instead of pure anxiety.

## Sample

- N = 106 PIDs (excludes ['064', '086'])
- SSQ-Bad (Post SSQ_Total ≥ 52, Kennedy moderate cutoff): n = 15
- SSQ-OK: n = 91
- Post SSQ_Total: mean = 23.15, std = 25.72, range = [0.0, 130.9]
- Pre SSQ_Total:  mean = 11.57, std = 12.87
- Δ SSQ (Post − Pre): mean = 11.57, std = 23.31

## Significant interactions after BH-FDR (n = 1 / 96)

| task | model | condition | test | effect | p_fdr |
|---|---|---|---|---|---|
| T3-A_AR | RF_behavior_only_9ch | full_beh | Spearman_vs_Delta_SSQ | +0.210 | 0.031 |

## Tests by family (task × test type)

| task | test | n_tests | n_sig (FDR 0.05) | min p_raw |
|---|---|---|---|---|
| T1-A | MannWhitney_Bad_vs_OK | 1 | 0 | 0.835 |
| T1-A | Spearman_vs_Delta_SSQ | 1 | 0 | 0.601 |
| T1-A | Spearman_vs_Post_SSQ | 1 | 0 | 0.891 |
| T2-A | MannWhitney_Bad_vs_OK | 10 | 0 | 0.266 |
| T2-A | Spearman_vs_Delta_SSQ | 10 | 0 | 0.0869 |
| T2-A | Spearman_vs_Post_SSQ | 10 | 0 | 0.0838 |
| T3-A | MannWhitney_Bad_vs_OK | 14 | 0 | 0.0619 |
| T3-A | Spearman_vs_Delta_SSQ | 14 | 0 | 0.0283 |
| T3-A | Spearman_vs_Post_SSQ | 14 | 0 | 0.0366 |
| T3-A_AR | MannWhitney_Bad_vs_OK | 1 | 0 | 1 |
| T3-A_AR | Spearman_vs_Delta_SSQ | 1 | 1 | 0.031 |
| T3-A_AR | Spearman_vs_Post_SSQ | 1 | 0 | 0.139 |
| T3-W | MannWhitney_Bad_vs_OK | 6 | 0 | 0.0276 |
| T3-W | Spearman_vs_Delta_SSQ | 6 | 0 | 0.176 |
| T3-W | Spearman_vs_Post_SSQ | 6 | 0 | 0.199 |

## Files

| File | Description |
|---|---|
| `run_ssq_decoding_interaction.py` | This analysis script |
| `per_pid_joined.csv` | PID × (Pre/Post/Δ SSQ + all decoding metrics) for re-use |
| `test_results.csv` | Long table: every test with raw + FDR p, effect size |
| `test_results_compact.md` | Per-family markdown tables |
| `SUMMARY.md` | This file |

## Caveats

1. **SSQ_Total uses simple sum × 3.74**, matching `compute_trait_scores.py` (not the standard Kennedy subscale weighting). Subscale-level analyses (N/O/D) are still pending — see Handoff doc #4.
2. **N=106 baseline; ~100 for T2-A** (6 PIDs lost a single-class fold during LOPO and were dropped per `T2A_raw_perpid.csv`).
3. **Δ SSQ assumes Pre vs Post comparable**: same 16 items, same scoring. For 4 PIDs missing Pre, Δ is NaN and they are excluded from the Δ test.
4. **FDR is within (task × test) family**, not across the whole table; a global Bonferroni correction would be more conservative.

---

# Update 2026-05-16 — Handoff #2 (continuous LMM) + #3 (pre/post)

## #3 Pre → Post SSQ paired comparison (descriptive + Wilcoxon)

- n paired = 106
- Pre SSQ_Total:  mean = **11.57**, SD = 12.87
- Post SSQ_Total: mean = **23.15**, SD = 25.72
- Δ (Post − Pre): mean = **+11.57**, SD = 23.31, median = +3.74
- Increased: 62 / Decreased: 25 / Tied: 19
- **Wilcoxon paired (two-sided)**: W = 748, **p = 7.38e-07**
- Wilcoxon one-sided (Post > Pre): p = 3.69e-07
- **Cohen's d_z (paired) = +0.497**; rank-biserial r = +0.609

### Drop-in text for Appendix E §E.2

> "VR exposure produced a significant increase in cybersickness symptoms (n = 106 paired Pre/Post; Wilcoxon paired p = 7.38e-07; mean Δ = +11.6 weighted SSQ points, SD 23.3; Cohen's d_z = +0.50)."

## #2 Continuous SSQ as moderator of per-PID per-scene anxiety

**Method**: Per-PID per-scene anxiety mean (raw 60 Hz frames from `MOMENT_ready_v2/{pid}_{scene}.npz`, mean across all valid frames per scene) regressed on Post SSQ_Total (continuous, z-standardized) in mixed-effects models with random PID intercept. ML estimation. 5 scenes preserved separately.

- N observations: 530 (PID × scene pairs)
- N PIDs: 106

### Model summary

| Model | Formula | k_fixed | AIC | R²_marg | R²_cond | var_RE |
|---|---|---|---|---|---|---|
| M0_intercept | `anxiety_mean ~ 1` | 1 | 2070.56 | 0.0000 | 0.4204 | 1.537 |
| M1_scene | `anxiety_mean ~ C(scene)` | 5 | 1963.47 | 0.1104 | 0.5583 | 1.638 |
| M2_SSQ_only | `anxiety_mean ~ SSQ_z` | 2 | 2061.40 | 0.0537 | 0.4204 | 1.341 |
| M3_scene_plus_SSQ | `anxiety_mean ~ C(scene) + SSQ_z` | 6 | 1954.30 | 0.1641 | 0.5583 | 1.442 |
| M4_scene_times_SSQ | `anxiety_mean ~ C(scene) * SSQ_z` | 10 | 1959.53 | 0.1664 | 0.5612 | 1.444 |

### LRTs for key comparisons

| Comparison | ΔlogL | Δdf | p | ΔAIC |
|---|---|---|---|---|
| M3_scene+SSQ vs M1_scene (adds SSQ main effect) | +11.16 | 1 | 0.000835 | -9.16 |
| M4_scene×SSQ vs M3_scene+SSQ (adds interaction) | +2.78 | 4 | 0.596 | +5.22 |

### Interpretation

- **SSQ_Total main effect**: significant (p = 0.000835, ΔAIC = -9.16). Higher-SSQ PIDs have systematically different overall anxiety after accounting for scene.
- **Scene × SSQ interaction**: not supported (p = 0.596, ΔAIC = +5.22). The slope of anxiety on SSQ does not differ across scenes; a parallel-line / no-interaction model is sufficient.
- Result is consistent with the categorical (Bad vs OK) analysis in Appendix E §E.3: only Elevator 2 and Hall reached FDR-significance in the group comparison, and the continuous analysis confirms the pattern is not robust enough to drive an overall continuous moderator effect.

### Files added in this update

| File | Description |
|---|---|
| `run_ssq_continuous_lmm_and_pre_post.py` | Reproducible script (this update) |
| `09_continuous_lmm_diagnostics.csv` | M0–M4 AIC / R²_marg / R²_cond / random-intercept var |
| `09_continuous_lmm_lrt.csv` | LRT M3 vs M1 (SSQ main), M4 vs M3 (interaction) |
| `10_continuous_lmm_interaction_coefs.csv` | M4 fixed-effect coefficients (scene × SSQ terms) |
| `11_pre_post_wilcoxon.csv` | Pre vs Post paired test statistics |

---

# Update 2026-05-16 — Handoff #4 (subscale-level analysis)

Kennedy 1993 weighted subscale scoring (N / O / D) added. Both Pre and Post computed for all 106 PIDs.

## #4.1 Kennedy subscale descriptives (Post)

| Subscale | weight | mean ± SD | median | range |
|---|---|---|---|---|
| N (Nausea) | ×9.54 | 25.02 ± 33.91 | 9.54 | [0.0, 143.1] |
| O (Oculomotor) | ×7.58 | 27.60 ± 26.51 | 22.74 | [0.0, 121.3] |
| D (Disorientation) | ×13.92 | 26.26 ± 39.29 | 13.92 | [0.0, 208.8] |

Kennedy TS:  mean ± SD = 30.48 ± 34.47
Simple TS:   mean ± SD = 23.15 ± 25.72
Pearson r(Kennedy_TS, simple_TS) = 0.9968, p = 0.44 — both metrics nearly redundant; existing pipeline's simple-TS is a safe proxy for the Kennedy total.

## #4.2 Per-subscale moderator tests on decoding metrics

For each subscale (N, O, D) × Post weighted score + Δ subscale, Spearman ρ with each per-PID decoding metric. BH-FDR within (subscale × task × test) family.

**Total tests: 192, significant after FDR: 2.**

### Significant tests

| subscale | task | model | condition | test | effect | p_raw | p_FDR |
|---|---|---|---|---|---|---|---|
| D | T3-A_AR | RF_behavior_only_9ch | full_beh | Spearman_vs_Post_subscale_weighted | +0.199 | 0.0411 | 0.0411 |
| D | T3-A_AR | RF_behavior_only_9ch | full_beh | Spearman_vs_Delta_subscale_weighted | +0.277 | 0.004 | 0.004 |

### Strongest trends per subscale (top 3 lowest p_raw, may not be FDR-sig)

**N (Nausea)**

| task | model | condition | test | effect | p_raw | p_FDR | sig |
|---|---|---|---|---|---|---|---|
| T3-A | XGB | NoAR+Physio | Spearman_vs_Delta_subscale_weighted | +0.254 | 0.00848 | 0.119 |  |
| T3-A | XGB | NoAR+Physio | Spearman_vs_Post_subscale_weighted | +0.250 | 0.00983 | 0.138 |  |
| T3-A | XGB | NoAR+Beh | Spearman_vs_Delta_subscale_weighted | +0.219 | 0.0242 | 0.17 |  |

**O (Oculomotor)**

| task | model | condition | test | effect | p_raw | p_FDR | sig |
|---|---|---|---|---|---|---|---|
| T3-A | XGB | NoAR+Physio | Spearman_vs_Post_subscale_weighted | +0.218 | 0.025 | 0.209 |  |
| T2-A | Raw_RF | physio | Spearman_vs_Post_subscale_weighted | +0.215 | 0.0319 | 0.319 |  |
| T3-A | RF | NoAR+Physio | Spearman_vs_Post_subscale_weighted | +0.188 | 0.0534 | 0.209 |  |

**D (Disorientation)**

| task | model | condition | test | effect | p_raw | p_FDR | sig |
|---|---|---|---|---|---|---|---|
| T3-A_AR | RF_behavior_only_9ch | full_beh | Spearman_vs_Delta_subscale_weighted | +0.277 | 0.004 | 0.004 | ✅ |
| T3-A | XGB | AR_only | Spearman_vs_Delta_subscale_weighted | +0.260 | 0.00709 | 0.0993 |  |
| T3-A | XGB | NoAR+Beh | Spearman_vs_Delta_subscale_weighted | +0.216 | 0.0261 | 0.11 |  |

## #4.3 Per-subscale LMM on per-PID per-scene anxiety

Same per-PID per-scene anxiety mean (raw 60 Hz frames) as §7.2; subscale_z replaces the simple-TS as continuous moderator.

| Subscale | main coef (per SD) | p (main, Wald-z) | LRT add subscale: p | ΔAIC | LRT interaction: p | ΔAIC |
|---|---|---|---|---|---|---|
| N | +0.433 | 0.000828 | 0.00111 | -8.63 | 0.466 | +4.42 |
| O | +0.449 | 0.000497 | 0.000702 | -9.48 | 0.662 | +5.60 |
| D | +0.346 | 0.00863 | 0.00974 | -4.68 | 0.165 | +1.51 |

### Interpretation

- Strongest subscale main effect: **O** (β = +0.449 anxiety units per +1 SD, p = 0.000497, ΔAIC vs scene-only = -9.48)
- 3/3 subscales show a significant main effect on per-PID per-scene anxiety after controlling for scene.
- **No scene × subscale interactions significant** (consistent with §7.2: SSQ acts as a parallel-line moderator regardless of which subscale is used).

### Files added by #4

| File | Description |
|---|---|
| `run_ssq_subscale_analysis.py` | This script (#4) |
| `12_subscale_scores_per_pid.csv` | Per-PID Pre/Post N/O/D weighted + Kennedy TS + Δ |
| `13_subscale_decoding_tests.csv` | Subscale × decoding metric univariate tests |
| `14_subscale_lmm.csv` | One LMM per subscale (main effect, LRT, interaction) |
