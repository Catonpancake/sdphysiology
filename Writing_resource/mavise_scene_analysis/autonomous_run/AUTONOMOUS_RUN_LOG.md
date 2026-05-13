[2026-05-13 01:48:24] [INFO] === Autonomous run START ===
[2026-05-13 01:48:24] [INFO] CWD: C:\Users\user\code\SDPhysiology
[2026-05-13 01:48:24] [INFO] OUT_BASE: c:\Users\user\code\SDPhysiology\Writing_resource\mavise_scene_analysis\autonomous_run
[2026-05-13 01:48:24] [INFO] 
==============================================================================
[2026-05-13 01:48:24] [INFO] PHASE A — Re-extract behavior 56ch & merge with physio → 67ch
[2026-05-13 01:48:24] [INFO] ==============================================================================
[2026-05-13 01:48:24] [INFO] [A.1] Running build_behavior_windows_ts_60hz for all 5 scenes ...
[2026-05-13 02:10:56] [INFO] [A.1] behavior_all/ saved. X=(59039, 300, 56), PIDs=107, skipped=1, elapsed=1351s
[2026-05-13 02:10:56] [INFO] [A.2] Splitting behavior_all → per-scene behavior_{scene}/ ...
[2026-05-13 02:11:00] [INFO]    [A.2] behavior_Elevator: N=22223, scenes=['Elevator1', 'Elevator2']
[2026-05-13 02:11:01] [INFO]    [A.2] behavior_Hallway: N=12355, scenes=['Hallway']
[2026-05-13 02:11:02] [INFO]    [A.2] behavior_Hall: N=6244, scenes=['Hall']
[2026-05-13 02:11:05] [INFO]    [A.2] behavior_Outside: N=18217, scenes=['Outside']
[2026-05-13 02:11:05] [INFO] [A.3] Verifying physio data (ml_processed_{scene}/) ...
[2026-05-13 02:11:05] [INFO]    [A.3] ml_processed_Elevator: X=(21954, 11, 300)
[2026-05-13 02:11:05] [INFO]    [A.3] ml_processed_Hallway: X=(12355, 11, 300)
[2026-05-13 02:11:05] [INFO]    [A.3] ml_processed_Hall: X=(6244, 11, 300)
[2026-05-13 02:11:05] [INFO]    [A.3] ml_processed_Outside: X=(18217, 11, 300)
[2026-05-13 02:11:05] [WARNING] [A.3] Physio shape (21954, 11, 300): appears (N,C,T). Will transpose.
[2026-05-13 02:11:05] [INFO] [A.4] Merging physio + behavior → 67ch per scene ...
[2026-05-13 02:11:05] [INFO]    [A.4] Merging Elevator ...
[2026-05-13 02:11:06] [WARNING]       [A.4] y_p vs y_b differ max=9.8262, using physio y
[2026-05-13 02:11:10] [INFO]       [A.4] Elevator: X_merged=(21954, 300, 67), common=21954/(p=21954/b=22223)
[2026-05-13 02:11:10] [INFO]    [A.4] Merging Hallway ...
[2026-05-13 02:11:11] [WARNING]       [A.4] y_p vs y_b differ max=12.8061, using physio y
[2026-05-13 02:11:16] [INFO]       [A.4] Hallway: X_merged=(12355, 300, 67), common=12355/(p=12355/b=12355)
[2026-05-13 02:11:16] [INFO]    [A.4] Merging Hall ...
[2026-05-13 02:11:16] [WARNING]       [A.4] y_p vs y_b differ max=9.3782, using physio y
[2026-05-13 02:11:18] [INFO]       [A.4] Hall: X_merged=(6244, 300, 67), common=6244/(p=6244/b=6244)
[2026-05-13 02:11:18] [INFO]    [A.4] Merging Outside ...
[2026-05-13 02:11:20] [WARNING]       [A.4] y_p vs y_b differ max=13.0036, using physio y
[2026-05-13 02:11:26] [INFO]       [A.4] Outside: X_merged=(18217, 300, 67), common=18217/(p=18217/b=18217)
[2026-05-13 02:11:26] [INFO] [A.5] Combining per-scene merged → ml_processed_behavior_all/ ...
[2026-05-13 02:11:56] [INFO]    [A.5] ml_processed_behavior_all/ saved: X=(58770, 300, 67)
[2026-05-13 02:11:56] [INFO] [A.6] Verification stats on regenerated data ...
[2026-05-13 02:11:59] [INFO]    [A.6] verification_stats.csv written
[2026-05-13 02:12:00] [INFO] 
    scene  n_windows  dist_min_mean  dist_min_std  dist_min_n_uniq  count_personal_mean  count_personal_std  count_personal_n_uniq  backward_flag_mean  backward_flag_std  backward_flag_n_uniq  speed_mean  speed_std  speed_n_uniq
Elevator1      11760        -0.0004        0.7378           435530               0.0001              0.4800                   3913              0.1472             0.3544                     2     -0.0011     0.9871       1031148
  Outside      18217         0.0000        0.7394          1613590              -0.0000              0.3479                   1408              0.3305             0.4704                     2     -0.0000     0.9997       3260318
  Hallway      12355         0.0000        0.6846          1016797              -0.0000              0.3359                   1288              0.2451             0.4301                     2     -0.0001     0.8594       1795170
Elevator2      10194        -0.0004        0.8168           434016               0.0002              0.4887                   3515              0.1350             0.3417                     2     -0.0017     0.9792        772952
     Hall       6244         0.0001        0.8366           805471              -0.0000              0.4333                    937              0.4756             0.4994                     2     -0.0001     0.9984       1252547
[2026-05-13 02:12:01] [INFO] 
==============================================================================
[2026-05-13 02:12:01] [INFO] PHASE B — Re-run baselines on clean data
[2026-05-13 02:12:01] [INFO] ==============================================================================
[2026-05-13 02:12:01] [INFO] [B] Running classical_baseline (eval_mavise_classical_baseline.py) ...
[2026-05-13 02:12:24] [INFO] [B] classical_baseline → ok (23s)
[2026-05-13 02:12:24] [INFO] [B] Running classical_v2 (eval_mavise_classical_v2.py) ...
[2026-05-13 02:12:59] [INFO] [B] classical_v2 → ok (35s)
[2026-05-13 02:12:59] [INFO] [B] Running hv (eval_mavise_hv.py) ...
[2026-05-13 02:16:55] [INFO] [B] hv → ok (236s)
[2026-05-13 02:16:55] [INFO] [B] Running directions (eval_mavise_directions.py) ...
[2026-05-13 02:37:54] [INFO] [B] directions → ok (1259s)
[2026-05-13 02:37:54] [INFO] [B] Running cnn (eval_mavise_cnn_baseline.py) ...
[2026-05-13 02:41:49] [INFO] [B] cnn → ok (235s)
[2026-05-13 02:41:49] [INFO] [B] Running rnn (eval_mavise_rnn_baseline.py) ...
[2026-05-13 02:48:08] [INFO] [B] rnn → ok (380s)
[2026-05-13 02:48:08] [INFO] [B] Snapshotting result CSVs ...
[2026-05-13 02:48:08] [INFO]    copied mavise_classical_baseline.csv
[2026-05-13 02:48:08] [INFO]    copied mavise_classical_v2.csv
[2026-05-13 02:48:08] [INFO]    copied mavise_hv_results.csv
[2026-05-13 02:48:09] [INFO]    copied mavise_directions.csv
[2026-05-13 02:48:09] [INFO]    copied mavise_cnn_baseline.csv
[2026-05-13 02:48:09] [INFO]    copied mavise_rnn_baseline.csv
[2026-05-13 02:48:09] [INFO] 
==============================================================================
[2026-05-13 02:48:09] [INFO] PHASE C — analyses on clean data (Phase 1 + Phase 2 + Phase 3)
[2026-05-13 02:48:09] [INFO] ==============================================================================
[2026-05-13 02:48:09] [INFO] [C.1] Phase-1 measurements ...
[2026-05-13 02:48:13] [INFO]    [C.1] done. 4 CSVs written.
[2026-05-13 02:48:13] [INFO] [C.2] Phase-2 three-factor ...
[2026-05-13 02:48:15] [INFO]    [C.2] done. CSV written.
[2026-05-13 02:48:15] [INFO] 
    scene  n_pids_kept  Y_var  X_var_bw  coupling_bw_fisher  coupling_dist_fisher  n_bw_corrs  n_dist_corrs  Y_var_z  X_var_bw_z  coupling_bw_fisher_z                Bottleneck
Elevator1          105 0.9350    0.1048             -0.0307               -0.0819         105           105   0.1663     -1.2395               -0.0311 X_var (feature) (z=-1.24)
  Outside          107 0.9796    0.2666             -0.1013               -0.0048         107           100   0.6193      0.7416               -0.3475     none (all sufficient)
  Hallway          107 0.9795    0.2817              0.3485               -0.0641         106           104   0.6176      0.9264                1.6664     none (all sufficient)
Elevator2          105 0.7459    0.1318             -0.2499               -0.0614         105           105  -1.7558     -0.9084               -1.0127  Y_var (target) (z=-1.76)
     Hall          106 0.9534    0.2452             -0.0852                0.0723         106           105   0.3527      0.4799               -0.2752     none (all sufficient)
[2026-05-13 02:48:15] [INFO] [C.3] Phase-3 4-panel figure ...
[2026-05-13 02:48:16] [INFO]    [C.3] done. c:\Users\user\code\SDPhysiology\Writing_resource\mavise_scene_analysis\autonomous_run\phase_C\06_4panel_figure.png
[2026-05-13 02:48:16] [INFO] === Autonomous run END (59.9 min) ===
[2026-05-13 02:48:16] [INFO] Final report written: c:\Users\user\code\SDPhysiology\Writing_resource\mavise_scene_analysis\autonomous_run\FINAL_REPORT.md
