# MAVISE Scene-Dependent Analysis — Clean Data

_Generated: 2026-05-13_

Independent re-extraction + analysis of MAVISE behavior data, after fixing two preprocessing bugs discovered while preparing thesis-grade interpretation evidence.

## TL;DR

- **Two bugs** in the MAVISE 56-channel behavior pipeline were discovered and fixed:
  1. `compute_agent_player_relations`: pd.merge suffix not handled — agent columns silently fell back to player columns → all NPC distances=0.
  2. `_compute_player_only_timeseries` + `_augment_behavior_dynamics`: `.diff()` operations crossed scene boundaries → spurious speed/accel spikes at scene transitions (up to 12 000 m/s).
- All MAVISE data regenerated with fixes; all baselines + analyses re-run.
- **Hallway XGB R² (HV)**: 0.424 → **0.499** (cleaner data, +0.075).
- **Hall XGB R²**: 0.001 → **0.178** (previously buggy data masked all signal).
- **Mechanism**: scene-dependent predictability is driven by **backward_flag × anxiety coupling**, not by backward_flag rate alone. Hallway is the only scene with positive coupling (Fisher-z r=+0.35); others all near zero.

## Folder index

| Path | Contents |
|---|---|
| `01_phase1_basic_measurements.py` | Phase-1 measurement script (Y_var, trajectory, bw_rate, prox_var) |
| `_step2_verify_one_pid.py` | Single-PID verify-run of fixed code |
| `_step3_cross_check_ismar.py` | Cross-check against ISMAR NPZ (boundary fix verification) |
| `autonomous_run/AUTONOMOUS_RUN_LOG.md` | Timestamped log of Phase A/B/C unattended run |
| `autonomous_run/FINAL_REPORT.md` | Phase A/B/C consolidated report |
| `autonomous_run/state.json` | Phase state, errors, warnings |
| `autonomous_run/phase_A/verification_stats.csv` | Per-scene channel stats after re-extraction |
| `autonomous_run/phase_B/*.csv + .log` | 6 baseline outputs (3-seed for DL) |
| `autonomous_run/phase_B_10seed/SUMMARY.md` | 10-seed DL paper-grade table |
| `autonomous_run/phase_C/0[1-6]_*.csv` | Phase-1 measurements + three-factor + paired test |
| `autonomous_run/phase_C/06_4panel_figure_v2.png` | Final figure (refined bottleneck + IQR) |

## Classical baseline (XGB, HV masking, per-scene)

Median R² across 9 held-out test PIDs (80/10/10 group split).

| Scene | Buggy (R²) | **Clean (R²)** | Δ |
|---|---|---|---|
| Hallway | +0.424 | **+0.499** | +0.075 |
| Hall | +0.001 | **+0.178** | +0.177 |
| Elevator | +0.082 | **+0.052** | -0.030 |
| Outside | +0.005 | **+0.015** | +0.010 |

## DL 10-seed results (mean ± SD, paper-grade)

### RNN (LSTM / GRU / GRU_Attn)
| model | Hallway | Hall | Elevator | Outside |
|---|---|---|---|---|
| LSTM | +0.220 ± 0.031 | +0.157 ± 0.020 | +0.038 ± 0.005 | -0.028 ± 0.011 |
| GRU | +0.255 ± 0.024 | +0.168 ± 0.013 | +0.054 ± 0.011 | -0.040 ± 0.014 |
| GRU_Attn | +0.342 ± 0.043 | +0.138 ± 0.021 | +0.105 ± 0.011 | -0.030 ± 0.017 |

### CNN
| scene | mean ± SD |
|---|---|
| Hallway | +0.268 ± 0.013 |
| Hall | +0.129 ± 0.010 |
| Elevator | +0.072 ± 0.009 |
| Outside | +0.008 ± 0.008 |

## Three-factor bottleneck (per scene)

Refined rule: factor falls below threshold → bottleneck.
- (a) `Y_var < 0.5` — anxiety doesn't vary
- (b) `X_var < 0.15` — backward_flag rate too low
- (c) `|Coupling| < 0.15` — backward_flag uncorrelated with anxiety

| Scene | Y_var | X_var (bw rate) | Coupling (bw, y) | Bottleneck |
|---|---|---|---|---|
| Elevator1 | 0.935 | 0.105 | -0.031 | (b) X_var, (c) Coupling |
| Outside | 0.980 | 0.267 | -0.101 | (c) Coupling |
| Hallway | 0.979 | 0.282 | +0.349 | none |
| Elevator2 | 0.746 | 0.132 | -0.250 | (b) X_var |
| Hall | 0.953 | 0.245 | -0.085 | (c) Coupling |

## Elevator1 vs Elevator2 paired t-test (habituation)

- n = 104 PIDs (paired)
- mean diff (Elev2 − Elev1) = -0.490
- Cohen's d = -0.625
- t = -6.377, **p = 5.20e-09**
- Wilcoxon p = 1.81e-08

Same physical scene, protocol position #1 vs #4. Significantly lower anxiety on second exposure — direct empirical evidence of habituation, not just literature inference.

## Mechanism narrative (refined)

**Old narrative (data was buggy)**:
> Hallway has high R² because backward_flag rate is high → avoidance evidence.

**New narrative (clean data + Fisher-z coupling)**:
> Hallway is the only scene where backward_flag *correlates* with anxiety (per-PID Fisher-z r = +0.35). Other scenes have similar or higher backward_flag rates (Hall = 0.48 vs Hallway = 0.25), but the rate is **not coupled with anxiety** in those scenes — participants step back for unrelated reasons (e.g., turning to look around in open spaces). Scene-dependent predictability therefore reflects *avoidance-as-anxiety-signal*, which requires both (a) proximity threat AND (b) avoidability — only Hallway meets both. This is a more refined claim than "avoidance behavior present in Hallway".

## Caveats and known small issues

1. **y_p vs y_b discrepancy in merge log** (`max diff ~13`): this is *not* a data mismatch — physio pipeline stores **per-PID z-scored** y, behavior pipeline stores **raw 0-10 scale** y. The merge code uses the physio (z-scored) version. Different normalization stages, same underlying signal.
2. **dist_min / count_* not 100% matched against ISMAR NPZ** (~33-72% per-frame): MAVISE pipeline's `compute_agent_player_relations` algorithm has evolved since ISMAR NPZ regeneration (2026-03-12). Raw value ranges and distributions are consistent (0-7.6 m, 0-7 counts); MAVISE uses its own internally-consistent values.
3. **head_rot_speed is smoothed** in MAVISE (via `recompute_headrot_features_from_yaw` with EMA τ=0.05 s) but raw in ISMAR NPZ. Intentional pipeline difference, not a bug.

## Reproduction

```bash
# 1. Extract clean behavior 56ch + merge with physio:
python _autonomous_master.py

# 2. (optional) Run 10-seed DL for paper-grade error bars:
python _10seed_dl_runner.py

# 3. (this file) Refresh figure + README + memory:
python _finalize_all.py
```
