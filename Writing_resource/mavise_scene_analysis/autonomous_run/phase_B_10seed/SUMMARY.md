# 10-seed DL re-run — summary
Total time: 21.0 min

## Run durations
- rnn: 14.7 min
- cnn: 6.3 min

## RNN (LSTM / GRU / GRU_Attn) — mean ± SD across 10 seeds

| model | Hallway | Hall | Elevator | Outside |
|---|---|---|---|---|
| GRU | +0.255 ± 0.024 | +0.168 ± 0.013 | +0.054 ± 0.011 | -0.040 ± 0.014 |
| GRU_Attn | +0.342 ± 0.043 | +0.138 ± 0.021 | +0.105 ± 0.011 | -0.030 ± 0.017 |
| LSTM | +0.220 ± 0.031 | +0.157 ± 0.020 | +0.038 ± 0.005 | -0.028 ± 0.011 |

## CNN — mean ± SD across 10 seeds

| scene | mean R² ± SD | n_seeds |
|---|---|---|
| Hallway | +0.268 ± 0.013 | 10 |
| Hall | +0.129 ± 0.010 | 10 |
| Elevator | +0.072 ± 0.009 | 10 |
| Outside | +0.008 ± 0.008 | 10 |

## Comparison vs 3-seed (from autonomous_run/phase_B/)

See `phase_B/mavise_rnn_baseline.csv` (3-seed) vs `phase_B_10seed/mavise_rnn_baseline.csv` (10-seed).
