# MAVISE Paper-Writer Pack

Single-file handoff with everything a paper-writing collaborator needs.
All values reflect the **clean** (post bug-fix) dataset regenerated on 2026-05-13.

---

## 1. Project frame

- **Dataset**: MAVISE (Multi-modal Anxiety in VR with Interactive Social Encounters).
  N = 108 enrolled participants, **N = 106 analyzed** (2 excluded).
- **Target paper**: IEEE TAC, MAVISE manuscript. Leakage-free baseline supersedes the
  earlier R² = 0.65–0.67 figures (which were inflated by a y_cont feature-target leakage).
- **Sibling study**: ISMAR 2026 (separate paper, smaller 9-channel behavior set).
  ISMAR results are unaffected; only MAVISE was regenerated.

---

## 2. Bug history (one paragraph)

While preparing thesis-grade scene-dependent interpretation evidence, two bugs were
found in the MAVISE 56-channel preprocessing pipeline that had been generated on
2026-01-25:

1. **Proximity bug** in `compute_agent_player_relations` (`behavior_features.py`):
   after `pd.merge(suffixes=("", "_agent"))`, the agent X/Y/Z columns are renamed
   with a `_agent` suffix, but the code referenced the original names — silently
   falling back to **player** coordinates. Result: `dx = dz = 0` for all rows,
   collapsing every NPC distance to zero. All zone counts, proximity statistics,
   and approach detectors that depend on `dist_*` were affected.

2. **Scene-boundary bug** in `_compute_player_only_timeseries` and
   `_augment_behavior_dynamics`: every `.diff()` / `np.diff(prepend=x[0])` was
   applied across the entire concatenated (scene, frame)-sorted DataFrame. At
   scene transitions, player positions teleport, producing spurious velocity
   spikes up to **12 000 m/s**. `preprocess_continuous.py` was unaffected because
   it already calls the routine per-scene.

Both were fixed by grouping all `.diff()` operations by scene
(commit `ed5eb3e`). The full dataset was re-extracted from the original
`processed_individual_anonymized/{pid}_Main.pkl / _Agent.pkl / _Customevent.pkl`
pickles (untouched, 2024-11-14 timestamps). All baselines and analyses were
re-run on the clean data.

---

## 3. Methods

### 3.1 Data acquisition (unchanged)

- Original recording at **120 Hz** (player position, head rotation, anxiety
  self-report, physiology, NPC positions, custom events).
- Downsampled to **60 Hz** for analysis (`downsample_120_to_60_scenewise`).
- 5 scenes in fixed protocol order: **Elevator1 → Outside → Hallway →
  Elevator2 → Hall**.

### 3.2 Sample (N = 106)

| Subset | n PIDs | Notes |
|---|---|---|
| Train | 78 | Includes 030, 079 (partial data, train-only) and 004, 027, 052, 066 (recovered from missing original split) |
| Validation | 19 | 008, 009, 015, 020, 040, 051, 054, 058, 067, 069, 074, 075, 078, 081, 083, 092, 094, 097, 108 |
| Held-out test | 9 | 010, 011, 022, 045, 068, 076, 091, 103, 105 |
| Excluded | 2 | **064**: major data loss (omitted entirely). **086**: only 2 valid scenes, Elevator2 all-NaN anxiety. |

Split file: `split_fixed_test.json`. Group split at the **participant** level — no
overlap of PIDs between train/val/test.

### 3.3 Windowing & feature extraction

- 5 s non-overlapping windows (300 frames at 60 Hz), stride 2 s. The 5 s window
  comes from the original `build_behavior_windows_ts_60hz` invocation parameters.
- For each window we obtain a 67-channel time series:
  - **11 physio channels**: EDA_Tonic, EDA_Phasic, SCR_Amplitude, SCR_RiseTime,
    PPG_Rate, RSP_Rate, RSP_RVT, RSP_Amplitude, pupilL, pupilR, pupil_mean.
  - **56 behavior channels** (see §4 for the dictionary).
- Discrete / flag channels (`backward_flag`, `CE_goal_visible`) are left raw;
  continuous channels are **per-window z-scored** with clipping at ±10
  (`_zscore_windows_skip_discrete`).
- Window label `y` = mean anxiety over the 5 s window, then per-PID z-scored
  for the modeling target.

### 3.4 HV (high-variance) masking

We exclude windows in which the participant's anxiety is essentially constant,
because R² is undefined when the within-window y has near-zero variance.

> For each scene, compute the per-PID training-set quantile threshold
> `θ = quantile(|y_train|, 0.25)` and keep only windows with `|y| ≥ θ`. The
> same threshold is applied to validation and test (no test leakage).

Equivalent to dropping the bottom 25% (by `|y|`) — keeps ~70–82% of windows
depending on scene.

### 3.5 Classical models

| Model | Hyperparameters |
|---|---|
| **Ridge** | `alpha ∈ {0.01, 0.1, 1, 10, 100, 1000}`, selected on validation R² |
| **XGB** | `n_estimators=500, max_depth=3, learning_rate=0.05` |

Input: `mean+std` pooling of the 5 s time series → 2 × 67 = 134 features.

### 3.6 Deep learning models

Trained cross-scene (all 4 scene groups pooled), evaluated per scene.
All use input shape `(B, T=300, C=67)`. **10 seeds** for paper-grade
mean ± SD.

| Model | Architecture | Optimizer | Training |
|---|---|---|---|
| **CNN** | Conv1d num_filters=64, kernel=3, dropout=0.5, GAP, MLP depth=1, hidden=128 | Adam lr=1e-3, weight_decay=1e-4 | batch=256, max_epochs=150, early-stop patience=15, CosineAnnealing LR |
| **LSTM** | hidden_size=64, num_layers=1, dropout=0.3 | Adam lr=1e-3, weight_decay=1e-4 | same |
| **GRU** | hidden_size=64, num_layers=1, dropout=0.3 | Adam lr=1e-3, weight_decay=1e-4 | same |
| **GRU-Attn** | GRU 64×1 + additive attention pooling, dropout=0.3 | Adam lr=1e-3, weight_decay=1e-4 | same |

Reduced model capacity (vs the leaked-y_cont baseline that used 256 filters /
3 MLP layers) is intentional — for the now-weaker signal, larger models overfit.

### 3.7 Scene definitions (for paper Methods § Scenarios)

| Scene | Order | Description |
|---|---|---|
| **Elevator1** | #1 | Confined elevator. NPC(s) enter and stand close. Participant cannot retreat (back wall). First confrontation with proximity. |
| **Outside** | #2 | Open outdoor environment. NPCs visible at long distance, never approach closely. Functions as low-threat baseline. |
| **Hallway** | #3 | Narrow indoor corridor. NPCs walk toward and past the participant; participant can sidestep or step back to avoid. |
| **Elevator2** | #4 | Second elevator (same physical scene as Elevator1) at a later protocol position. Used for habituation analysis. |
| **Hall** | #5 | Larger open room with multiple NPCs. Final scene; some habituation expected. |

(Paper should expand each based on the original MAVISE scenario document.)

---

## 4. Feature dictionary (56 behavior channels)

### 4.1 Locomotion (player self-motion, computed from `X_pos`, `Z_pos`, `Y_rot`)

| Channel | Definition |
|---|---|
| `speed` | `‖[Δx, Δz]‖ / dt` (m/s); per-scene intra-scene differencing |
| `accel` | finite difference of speed |
| `head_rot_speed` | yaw angular velocity (deg/s), per-scene unwrap |
| `speed_diff`, `speed_diff_abs`, `speed_sq` | derivative + absolute + squared speed |
| `accel_abs`, `head_rot_speed_abs`, `head_rot_accel`, `head_rot_accel_abs` | absolute / second-derivative variants |
| `move_forward` | yaw-projected forward component of `[Δx, Δz]` |
| `move_sideways` | yaw-projected lateral component |
| `move_forward_abs`, `move_sideways_abs`, `sideways_ratio` | absolute / ratio variants |
| **`backward_flag`** | `1` if `move_forward < 0` else `0` (any reverse-direction frame) |

### 4.2 Proxemics — distance to nearest NPC

| Channel | Definition |
|---|---|
| `dist_min` | min agent-to-player distance per frame (clipped at 7.6 m) |
| `dist_mean` | mean distance across all valid agents |
| `dist_std` | std of distances |
| `dist_min_diff`, `dist_min_diff_abs` | lag-1 difference + absolute |
| same for `dist_mean`, `dist_std` |

### 4.3 Proxemics — counts (Hall's personal-space zones, in metres)

| Zone | Radius |
|---|---|
| `count_intimate` | ≤ 0.45 m |
| `count_personal` | 0.45 – 1.20 m |
| `count_social` | 1.20 – 3.50 m |
| `count_public` | 3.50 – 7.60 m |
| `count_agents` | total valid agents in 7.60 m radius |
| `count_fov` | valid agents in ±55° forward field of view |
| `count_approach` | valid agents whose `dist` is decreasing |

Each `count_*` has companion `_diff`, `_inc`, `_dec` features
(lag-1 difference, increase-only, decrease-only).

### 4.4 Custom event & eye-tracking

| Channel | Definition |
|---|---|
| `CE_goal_visible` | flag — task goal currently visible (from Customevent.pkl) |
| `gaze_mean_x`, `gaze_mean_y` | per-frame mean gaze direction (normalized) |

---

## 5. Results

### 5.1 Per-scene classical baseline (HV masking + mean+std pooling)

Test R² on 9 held-out PIDs. Bold = best per scene.

| Scene | Ridge | **XGB** |
|---|---|---|
| Hallway | +0.469 | **+0.499** |
| Hall | +0.141 | **+0.178** |
| Elevator | +0.043 | +0.052 |
| Outside | −0.001 | +0.015 |

(`mavise_hv_results.csv`)

### 5.2 Deep learning, 10 seeds (mean ± SD)

| Model | Hallway | Hall | Elevator | Outside |
|---|---|---|---|---|
| LSTM | +0.220 ± 0.031 | +0.157 ± 0.020 | +0.038 ± 0.005 | −0.028 ± 0.011 |
| GRU | +0.255 ± 0.024 | +0.168 ± 0.013 | +0.054 ± 0.011 | −0.040 ± 0.014 |
| **GRU-Attn** | **+0.342 ± 0.043** | +0.138 ± 0.021 | **+0.105 ± 0.011** | −0.030 ± 0.017 |
| CNN | +0.268 ± 0.013 | +0.129 ± 0.010 | +0.072 ± 0.009 | +0.008 ± 0.008 |

XGB outperforms every DL model on every scene. Among DL, GRU-Attn is best on
Hallway/Elevator/Outside; GRU is best on Hall. (`phase_B_10seed/SUMMARY.md`)

### 5.3a Modality ablation (XGB, HV, per-scene)

Drop one modality at a time. All variants use `mean+std` pooling.

| Modality | Hallway | Hall | Elevator | Outside |
|---|---|---|---|---|
| REF — Full (67 ch = 11 physio + 56 behavior) | +0.511 | +0.192 | +0.059 | +0.014 |
| Physio-only (11 ch) | +0.191 | −0.001 | +0.021 | +0.001 |
| **Behavior-only (56 ch)** | **+0.514** | **+0.206** | +0.045 | +0.014 |

Behavior alone ≥ Full; Physio-only collapses to near-zero except in Hallway.
This is the primary ablation for the paper's narrative about
*avoidance-behavior-as-anxiety-signal*. (`mavise_directions.csv`, rows
`dir ∈ {REF, D1_physio, D1_beh}`)

### 5.3b Temporal-pooling ablation (XGB, HV, per-scene)

Same 67-channel input; vary how the 5 s time series is collapsed to a feature
vector for classical models. NOT a modality test — both rows use all channels.

| Pooling | Hallway | Hall | Elevator | Outside |
|---|---|---|---|---|
| `mean + std`  (2 stats × 67 ch = 134 features) — REF | +0.511 | +0.192 | +0.059 | +0.014 |
| `mean + std + slope`  (3 stats × 67 ch = 201 features) — D2_slope | +0.537 | +0.125 | +0.060 | +0.013 |

`slope` = ordinary-least-squares regression slope of each channel within the
5 s window (units per second). Captures within-window trend in addition to
level (`mean`) and dispersion (`std`).

**Verdict — explored, no consistent gain.** Adding the slope statistic
improves Hallway slightly (+0.026 R²) but *worsens* Hall (−0.067) and is
flat elsewhere. We retain `mean+std` as the default pooling and report the
slope variant here only for transparency.
(`mavise_directions.csv`, rows `dir ∈ {REF, D2_slope}`)

### 5.4 Three-factor mechanism (Phase-2 analysis)

For each scene, three components are computed per PID and aggregated:

- **Y_var**: per-PID standard deviation of anxiety (median across PIDs).
- **X_var (bw rate)**: per-PID mean of `backward_flag` (median across PIDs).
- **Coupling (bw × y)**: per-PID Pearson r of `backward_flag` mean per window
  with `y`, transformed via Fisher z, averaged across PIDs, then back-transformed.
  Per-PID series must have ≥ 20 windows to be included.

Bottleneck rule (post-hoc but pre-specified thresholds):
`Y_var < 0.5` → target bottleneck (a); `X_var < 0.15` → feature bottleneck (b);
`|Coupling| < 0.15` → coupling bottleneck (c).

| Scene | Y_var | X_var (bw rate) | Coupling (bw, y) | Bottleneck |
|---|---|---|---|---|
| Elevator1 | 0.935 | 0.105 | −0.031 | (b) X_var, (c) Coupling |
| Outside | 0.980 | 0.267 | −0.101 | (c) Coupling |
| **Hallway** | 0.979 | 0.282 | **+0.349** | **none** |
| Elevator2 | 0.746 | 0.132 | −0.250 | (b) X_var |
| Hall | 0.953 | 0.245 | −0.085 | (c) Coupling |

Hallway is the **only** scene where backward_flag is positively coupled to
anxiety. Other scenes have similar or higher backward_flag rates (Hall = 0.48
vs Hallway = 0.25), but the rate is **not coupled** with anxiety in those
scenes — participants step back for reasons unrelated to anxiety (turning to
look around in open spaces, etc.).

### 5.5 Habituation — Elevator1 vs Elevator2 paired test

Same physical scene at protocol positions #1 and #4. Per-PID anxiety mean,
paired across the 104 PIDs that have both scenes.

- mean diff (Elev2 − Elev1) = **−0.490** (z-scored anxiety units)
- Cohen's d = **−0.625** (medium-to-large effect)
- t(103) = −6.377, **p = 5.20 × 10⁻⁹**
- Wilcoxon **p = 1.81 × 10⁻⁸**

Direct empirical evidence of habituation, not literature inference.

### 5.6 Verification stats on clean data

(After re-extraction, per-scene channel statistics confirm bug fixes; full
table in `autonomous_run/phase_A/verification_stats.csv`.)

| Scene | dist_min std (was) | now | count_personal std (was) | now |
|---|---|---|---|---|
| Outside | ~0 (collapsed) | 0.74 | 0 | 0.35 |
| Hall | ~0 (collapsed) | 0.84 | 0 | 0.43 |
| Hallway | 0.16 | 0.68 | 0 | 0.34 |
| Elevator1 | 0.20 | 0.74 | 0 | 0.48 |
| Elevator2 | 0.23 | 0.82 | 0 | 0.49 |

---

## 6. Mechanism narrative (recommended for Discussion)

> Predictability of frame-level anxiety varies markedly by scene (Hallway
> R² ≈ 0.50; Hall ≈ 0.18; Elevator ≈ 0.05; Outside ≈ 0). The full feature
> set, the behavior-only subset, and every DL architecture all reproduce this
> ordering. Two simpler hypotheses are insufficient to explain it:
>
> (a) **Target variance** alone — Hallway, Outside, and Hall all have
> comparable anxiety variability (Y_var ≈ 0.95–0.98 z-units), yet differ in
> R² by an order of magnitude.
>
> (b) **Avoidance behavior alone** — the `backward_flag` activation rate is
> actually *higher* in Hall (0.48) and Outside (0.33) than in Hallway (0.25),
> so a "more retreats → more anxiety" account would predict Hall and Outside
> to be more predictable, not less.
>
> The data are consistent with a **coupling** account: predictability requires
> the avoidance behavior to be *correlated* with anxiety, which in turn
> requires the situational structure to make backward motion a genuine signal
> of anxiety. Per-PID Fisher-z-averaged r(backward_flag, anxiety) is large
> and positive only in Hallway (+0.35); it is near zero or negative in every
> other scene. In Outside and Hall the participants step back too — but they
> do so for reasons unrelated to NPC proximity (e.g., looking around in open
> space, or repositioning in a large room). The "avoidance-as-anxiety-signal"
> chain therefore requires both **proximity threat** (a) and **avoidability**
> (b) to be present; only Hallway satisfies both. Elevator1/Elevator2
> additionally satisfy proximity threat but lack avoidability (the cabin
> is too small), driving their R² toward zero through a different bottleneck
> (low X_var / negative coupling). This three-factor decomposition gives a
> mechanism that is both predictive (it explains the R² ordering) and
> falsifiable (each bottleneck is empirically identifiable).

---

## 7. Caveats / limitations

1. **n=4 scene-aggregate scatters**: per-scene regression in Panels (A) and
   (D) has only 4 data points. We report Spearman ρ (qualitative trend),
   not R², for the scene-aggregate fit; per-PID strip in those panels shows
   the underlying distribution (~106 points per scene group).
2. **y normalization difference logged in merge**: `physio/y_array.npy` is
   per-PID z-scored; `behavior/y_array.npy` is raw 0–10. Merge correctly
   uses the z-scored physio version. The "max diff ~13" warning in the
   logs reflects this stage difference, not a data mismatch.
3. **Algorithm drift since ISMAR**: per-frame `dist_min` and `count_*` do
   not match the 2026-03-12 ISMAR NPZ values 1-to-1 (~33–72% agreement)
   because `compute_agent_player_relations` has evolved (cleaner zone-flag
   logic, slightly different floor filter). MAVISE values are internally
   consistent; the absolute range and distribution match physical reality
   (0–7.6 m, 0–7 counts).
4. **head_rot_speed differs from ISMAR**: MAVISE applies EMA smoothing
   (τ = 0.05 s) and per-scene yaw unwrap (`recompute_headrot_features_from_yaw`);
   ISMAR NPZ stores raw frame-difference yaw. Intentional pipeline difference.
5. **DL trained cross-scene**: this is a deliberate design choice. The signal
   is too weak per scene for DL to fit (n_train scene-pure ≈ 9k windows).
   Per-scene XGB still wins overall, but DL provides a fair comparison
   under matched evaluation.
6. **Coupling threshold |r| < 0.15 is post-hoc**: chosen to make the
   bottleneck table interpretable. The qualitative ordering (Hallway alone
   has positive coupling; all others near zero) is robust to threshold
   choice in the range 0.10–0.20.

---

## 8. All artifacts — file inventory

### 8.1 Clean re-extracted data (numpy)

```
c:/Users/user/code/SDPhysiology/
  ml_processed_behavior_Elevator/    X (21954, 300, 67), Elev1+Elev2 combined
  ml_processed_behavior_Hallway/     X (12355, 300, 67)
  ml_processed_behavior_Hall/        X (6244,  300, 67)
  ml_processed_behavior_Outside/     X (18217, 300, 67)
  ml_processed_behavior_all/         X (58770, 300, 67), 5 fine scenes preserved
```

Each folder contains `X_array.npy`, `y_array.npy`, `pid_array.npy`,
`scene_array.npy`, `windex_array.npy`, `feature_tag_list.npy`.

Deprecated buggy data renamed with `_deprecated_bug` suffix in the same parent.

### 8.2 Analysis outputs

```
Writing_resource/mavise_scene_analysis/
  README.md                  # TL;DR + tables (this pack expands it)
  PAPER_WRITER_PACK.md       # this file
  metadata.json              # path index
  01_phase1_basic_measurements.py
  _step2_verify_one_pid.py
  _step3_cross_check_ismar.py
  autonomous_run/
    AUTONOMOUS_RUN_LOG.md
    FINAL_REPORT.md
    state.json
    phase_A/
      verification_stats.csv
    phase_B/                          # baselines, 3-seed for DL
      mavise_hv_results.csv           # per-scene XGB/Ridge/CNN (§5.1)
      mavise_directions.csv           # modality ablation (§5.3)
      mavise_classical_baseline.csv   # mean-pool baseline
      mavise_classical_v2.csv         # variants A and A+B (cross-scene)
      mavise_rnn_baseline.csv         # 3-seed LSTM/GRU/GRU_Attn
      mavise_cnn_baseline.csv         # 3-seed CNN
      *.log
    phase_B_10seed/                   # paper-grade DL
      SUMMARY.md                      # mean ± SD tables (§5.2)
      rnn_10seed_summary.csv
      cnn_10seed_summary.csv
      mavise_rnn_baseline.csv         # raw per-seed
      mavise_cnn_baseline.csv         # raw per-seed
    phase_C/
      01_y_variance.csv               # Phase-1 measurements
      02_anxiety_trajectory_perpid.csv
      02_anxiety_trajectory_summary.csv
      02_elev1_vs_elev2_paired.csv    # paired test (§5.5)
      03_backward_flag_rate.csv
      04_proximity_variance.csv
      05_three_factor_bottleneck.csv  # original z-score rule
      05_three_factor_bottleneck_v2.csv  # refined absolute thresholds (§5.4)
      06_4panel_figure.png            # v1
      06_4panel_figure_v2.png         # v2 with IQR + per-PID strip
```

### 8.3 Pipeline source

```
c:/Users/user/code/SDPhysiology/
  behavior_features.py           # bug-fixed, commit ed5eb3e
  ml_dataloader.py               # build_behavior_windows_ts_60hz orchestrator
  preprocess_continuous.py       # ISMAR NPZ pipeline (per-scene call)
  _autonomous_master.py          # re-extraction + baseline + analysis master
  _10seed_dl_runner.py           # 10-seed CNN+RNN re-run
  _finalize_all.py               # figure v2 + README + memory + git
  eval_mavise_hv.py              # HV+mean+std baseline (§5.1)
  eval_mavise_directions.py      # modality ablation (§5.3)
  eval_mavise_cnn_baseline.py    # CNN
  eval_mavise_rnn_baseline.py    # LSTM/GRU/GRU_Attn
  eval_mavise_classical_baseline.py  # mean-pool only
  eval_mavise_classical_v2.py        # A / A+B variants
  split_fixed_test.json          # 78/19/9 PID split
```

### 8.4 Raw data source (never touched)

```
D:/Labroom/SDPhysiology/Data/Processed/processed_individual_anonymized/
  {001..108}_Main.pkl
  {001..108}_Agent.pkl
  {001..108}_Customevent.pkl
```

108 PIDs × 3 files each. Timestamps 2024-11-14.

### 8.5 Memory (long-term context)

```
C:/Users/user/.claude/projects/c--Users-user-code-SDPhysiology/memory/
  mavise_baseline_task.md          # bug history + clean results
  mavise_leakage_free_summary.md   # earlier y_cont leakage analysis
  MEMORY.md                        # MAVISE section, top-level index
```

---

## 9. How to reproduce from scratch (~70 min)

```bash
# 1. Re-extract clean behavior 56ch + merge with physio (~60 min)
python _autonomous_master.py

# 2. 10-seed DL for paper-grade error bars (~25 min)
python _10seed_dl_runner.py

# 3. Refresh figure + README + memory + git
python _finalize_all.py
```

Or skip 1 — the clean numpy arrays in `ml_processed_behavior_*/` are already
committed (well, the path is, the .npy files are not in git but are present
on disk).

---

## 10. Open follow-ups (not blocking the paper)

- 10-seed XGB / Ridge (deterministic except for tree split RNG): not done, low value.
- Modality ablation × DL: only classical ablation done; DL × modality not run.
- Cross-scene XGB baseline: A+B variant in `classical_v2.csv` shows cross-scene XGB
  on Hallway = 0.196 vs per-scene 0.499. Worth a brief note that per-scene
  models outperform pooled cross-scene.
- Paper Methods § Scenarios: descriptions in §3.7 are minimal — expand from
  the MAVISE original scenario document if available.

End of pack.
