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
