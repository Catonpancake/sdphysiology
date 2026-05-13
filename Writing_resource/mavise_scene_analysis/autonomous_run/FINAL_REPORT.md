# Autonomous run — final report
**Total elapsed**: 59.9 min  (1.00 h)
**Errors**: 0, **Warnings**: 5, **Anomalies**: 0
## Phase A — extraction & merge
- behavior_all/ X_shape: (59039, 300, 56), PIDs=107, skipped=1, elapsed=1351s
  - Skipped PIDs:
    - {'pid': '064', 'reason': 'no_windows', 'X_shape': (0, 0, 0), 'C_cur': 0}
- Per-scene merge results:
  - Elevator: {'X_shape': (21954, 300, 67), 'n_common': 21954, 'n_physio': 21954, 'n_behavior': 22223}
  - Hallway: {'X_shape': (12355, 300, 67), 'n_common': 12355, 'n_physio': 12355, 'n_behavior': 12355}
  - Hall: {'X_shape': (6244, 300, 67), 'n_common': 6244, 'n_physio': 6244, 'n_behavior': 6244}
  - Outside: {'X_shape': (18217, 300, 67), 'n_common': 18217, 'n_physio': 18217, 'n_behavior': 18217}
- ml_processed_behavior_all/ X: (58770, 300, 67)

### Verification stats (Phase A.6)
| scene     |   n_windows |   dist_min_mean |   dist_min_std |   dist_min_n_uniq |   count_personal_mean |   count_personal_std |   count_personal_n_uniq |   backward_flag_mean |   backward_flag_std |   backward_flag_n_uniq |   speed_mean |   speed_std |   speed_n_uniq |
|:----------|------------:|----------------:|---------------:|------------------:|----------------------:|---------------------:|------------------------:|---------------------:|--------------------:|-----------------------:|-------------:|------------:|---------------:|
| Elevator1 |       11760 |         -0.0004 |         0.7378 |            435530 |                0.0001 |               0.4800 |                    3913 |               0.1472 |              0.3544 |                      2 |      -0.0011 |      0.9871 |        1031148 |
| Outside   |       18217 |          0.0000 |         0.7394 |           1613590 |               -0.0000 |               0.3479 |                    1408 |               0.3305 |              0.4704 |                      2 |      -0.0000 |      0.9997 |        3260318 |
| Hallway   |       12355 |          0.0000 |         0.6846 |           1016797 |               -0.0000 |               0.3359 |                    1288 |               0.2451 |              0.4301 |                      2 |      -0.0001 |      0.8594 |        1795170 |
| Elevator2 |       10194 |         -0.0004 |         0.8168 |            434016 |                0.0002 |               0.4887 |                    3515 |               0.1350 |              0.3417 |                      2 |      -0.0017 |      0.9792 |         772952 |
| Hall      |        6244 |          0.0001 |         0.8366 |            805471 |               -0.0000 |               0.4333 |                     937 |               0.4756 |              0.4994 |                      2 |      -0.0001 |      0.9984 |        1252547 |

## Phase B — baselines
- **classical_baseline**: ok (23s)
- **classical_v2**: ok (35s)
- **hv**: ok (236s)
- **directions**: ok (1259s)
- **cnn**: ok (235s)
- **rnn**: ok (380s)

### MAVISE HV (XGB) per-scene R²
| variant      | scene    | model   |      test_r2 |      val_r2 |   pct_kept |
|:-------------|:---------|:--------|-------------:|------------:|-----------:|
| HV_classical | Hallway  | Ridge   |  0.469214    |   0.414827  |    70.9602 |
| HV_classical | Hallway  | XGB     |  0.498914    |   0.497711  |    70.9602 |
| HV_classical | Hall     | Ridge   |  0.141283    |   0.191599  |    68.7371 |
| HV_classical | Hall     | XGB     |  0.17821     |   0.149475  |    68.7371 |
| HV_classical | Elevator | Ridge   |  0.0430885   |   0.0488847 |    74.3711 |
| HV_classical | Elevator | XGB     |  0.0522842   |   0.0471131 |    74.3711 |
| HV_classical | Outside  | Ridge   | -0.000765324 |   0.0028336 |    81.9016 |
| HV_classical | Outside  | XGB     |  0.0147573   |   0.0193381 |    81.9016 |
| HV_CNN       | Hallway  | CNN     |  0.337062    | nan         |    70.9602 |
| HV_CNN       | Hall     | CNN     |  0.160101    | nan         |    68.7371 |
| HV_CNN       | Elevator | CNN     |  0.103098    | nan         |    74.3711 |
| HV_CNN       | Outside  | CNN     |  0.0154717   | nan         |    81.9016 |

## Phase C — analyses

### Three-factor decomposition (Phase 2)
| scene     |   n_pids_kept |   Y_var |   X_var_bw |   coupling_bw_fisher |   coupling_dist_fisher |   n_bw_corrs |   n_dist_corrs |   Y_var_z |   X_var_bw_z |   coupling_bw_fisher_z | Bottleneck                |
|:----------|--------------:|--------:|-----------:|---------------------:|-----------------------:|-------------:|---------------:|----------:|-------------:|-----------------------:|:--------------------------|
| Elevator1 |           105 |  0.9350 |     0.1048 |              -0.0307 |                -0.0819 |          105 |            105 |    0.1663 |      -1.2395 |                -0.0311 | X_var (feature) (z=-1.24) |
| Outside   |           107 |  0.9796 |     0.2666 |              -0.1013 |                -0.0048 |          107 |            100 |    0.6193 |       0.7416 |                -0.3475 | none (all sufficient)     |
| Hallway   |           107 |  0.9795 |     0.2817 |               0.3485 |                -0.0641 |          106 |            104 |    0.6176 |       0.9264 |                 1.6664 | none (all sufficient)     |
| Elevator2 |           105 |  0.7459 |     0.1318 |              -0.2499 |                -0.0614 |          105 |            105 |   -1.7558 |      -0.9084 |                -1.0127 | Y_var (target) (z=-1.76)  |
| Hall      |           106 |  0.9534 |     0.2452 |              -0.0852 |                 0.0723 |          106 |            105 |    0.3527 |       0.4799 |                -0.2752 | none (all sufficient)     |

4-panel figure: `c:\Users\user\code\SDPhysiology\Writing_resource\mavise_scene_analysis\autonomous_run\phase_C\06_4panel_figure.png`

## Anomalies & warnings (sorted)
- WARNING [2026-05-13 02:11:05] [A.3] Physio shape (21954, 11, 300): appears (N,C,T). Will transpose.
- WARNING [2026-05-13 02:11:06]       [A.4] y_p vs y_b differ max=9.8262, using physio y
- WARNING [2026-05-13 02:11:11]       [A.4] y_p vs y_b differ max=12.8061, using physio y
- WARNING [2026-05-13 02:11:16]       [A.4] y_p vs y_b differ max=9.3782, using physio y
- WARNING [2026-05-13 02:11:20]       [A.4] y_p vs y_b differ max=13.0036, using physio y
