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
