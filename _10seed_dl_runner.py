"""
_10seed_dl_runner.py — re-run CNN + RNN baselines with NUM_SEEDS=10.

Sequential execution (CNN then RNN) to avoid GPU contention.
Snapshots results into autonomous_run/phase_B_10seed/ and writes a
consolidated summary at the end.
"""
from __future__ import annotations
import os, sys, time, json, shutil, subprocess, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = Path(r'c:/Users/user/code/SDPhysiology')
OUT_DIR = ROOT / 'Writing_resource/mavise_scene_analysis/autonomous_run/phase_B_10seed'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / 'RUN_10SEED_LOG.md'
SUMMARY = OUT_DIR / 'SUMMARY.md'

PYTHON = r'C:/Users/user/anaconda3/envs/ml_env/python.exe'
WR = ROOT / 'Writing_resource'


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', errors='replace').decode('ascii'), flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_baseline(name: str, script_name: str, timeout_s: int = 90 * 60):
    log(f'=== {name} START ({script_name}) ===')
    t0 = time.time()
    logfile = OUT_DIR / f'{name}.log'
    try:
        with open(logfile, 'w', encoding='utf-8') as lf:
            proc = subprocess.run(
                [PYTHON, str(ROOT / script_name)],
                stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT),
                timeout=timeout_s, encoding='utf-8', errors='replace',
            )
        elapsed = time.time() - t0
        status = 'ok' if proc.returncode == 0 else f'rc={proc.returncode}'
        log(f'=== {name} {status} ({elapsed:.0f}s = {elapsed/60:.1f}min) ===')
        return {'status': status, 'elapsed': elapsed, 'log': str(logfile)}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log(f'=== {name} TIMEOUT at {elapsed:.0f}s ===')
        return {'status': 'timeout', 'elapsed': elapsed}
    except Exception as e:
        log(f'=== {name} EXCEPTION: {e} ===')
        return {'status': 'exception', 'error': str(e)}


def snapshot_csv(src_name: str, dst_name: str | None = None):
    src = WR / src_name
    dst = OUT_DIR / (dst_name or src_name)
    if src.exists():
        shutil.copy2(src, dst)
        log(f'  snapshotted: {src_name} -> {dst}')
        return dst
    log(f'  MISSING: {src_name}')
    return None


def build_summary(rnn_csv: Path | None, cnn_csv: Path | None, durations: dict):
    lines = []
    lines.append('# 10-seed DL re-run — summary\n')
    lines.append(f'Total time: {sum(durations.values())/60:.1f} min\n\n')

    lines.append('## Run durations\n')
    for k, v in durations.items():
        lines.append(f'- {k}: {v/60:.1f} min\n')
    lines.append('\n')

    # RNN table
    if rnn_csv and rnn_csv.exists():
        rdf = pd.read_csv(rnn_csv)
        seed_cols = [c for c in rdf.columns if c.startswith('s') and c[1:].isdigit()]
        log(f'RNN seed cols: {seed_cols}')

        # per (model, scene): mean ± std across seeds
        out_rows = []
        for _, r in rdf.iterrows():
            seed_vals = [r[c] for c in seed_cols if pd.notna(r[c])]
            out_rows.append(dict(
                model=r['model'], scene=r['scene'],
                mean_r2=float(np.mean(seed_vals)),
                std_r2=float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0,
                n_seeds=len(seed_vals),
            ))
        rdf_sum = pd.DataFrame(out_rows)
        rdf_sum.to_csv(OUT_DIR / 'rnn_10seed_summary.csv', index=False)
        lines.append('## RNN (LSTM / GRU / GRU_Attn) — mean ± SD across 10 seeds\n\n')
        # pivot per model
        pivot_mean = rdf_sum.pivot(index='model', columns='scene', values='mean_r2')
        pivot_std = rdf_sum.pivot(index='model', columns='scene', values='std_r2')
        scenes = ['Hallway', 'Hall', 'Elevator', 'Outside']
        present_scenes = [s for s in scenes if s in pivot_mean.columns]
        header = '| model | ' + ' | '.join(present_scenes) + ' |'
        sep = '|' + '|'.join(['---'] * (len(present_scenes) + 1)) + '|'
        lines.append(header + '\n')
        lines.append(sep + '\n')
        for m in pivot_mean.index:
            cells = [f'{pivot_mean.loc[m, s]:+.3f} ± {pivot_std.loc[m, s]:.3f}'
                     for s in present_scenes]
            lines.append(f'| {m} | ' + ' | '.join(cells) + ' |\n')
        lines.append('\n')

    # CNN table
    if cnn_csv and cnn_csv.exists():
        cdf = pd.read_csv(cnn_csv)
        seed_cols = [c for c in cdf.columns if 'seed' in c and 'r2' in c]
        log(f'CNN seed cols: {seed_cols}')
        out_rows = []
        for _, r in cdf.iterrows():
            seed_vals = [r[c] for c in seed_cols if pd.notna(r[c])]
            out_rows.append(dict(
                scene=r['scene'],
                mean_r2=float(np.mean(seed_vals)),
                std_r2=float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0,
                n_seeds=len(seed_vals),
            ))
        cdf_sum = pd.DataFrame(out_rows)
        cdf_sum.to_csv(OUT_DIR / 'cnn_10seed_summary.csv', index=False)
        lines.append('## CNN — mean ± SD across 10 seeds\n\n')
        lines.append('| scene | mean R² ± SD | n_seeds |\n|---|---|---|\n')
        for _, r in cdf_sum.iterrows():
            lines.append(f'| {r["scene"]} | {r["mean_r2"]:+.3f} ± {r["std_r2"]:.3f} '
                         f'| {int(r["n_seeds"])} |\n')
        lines.append('\n')

    # Comparison vs 3-seed
    lines.append('## Comparison vs 3-seed (from autonomous_run/phase_B/)\n\n')
    lines.append('See `phase_B/mavise_rnn_baseline.csv` (3-seed) vs '
                 '`phase_B_10seed/mavise_rnn_baseline.csv` (10-seed).\n')

    SUMMARY.write_text(''.join(lines), encoding='utf-8')
    log(f'Summary written: {SUMMARY}')


def main():
    t_start = time.time()
    log('=== 10-seed DL runner START ===')
    durations = {}

    # 1) RNN
    rnn_res = run_baseline('mavise_rnn_baseline_10seed', 'eval_mavise_rnn_baseline.py',
                           timeout_s=90 * 60)
    durations['rnn'] = rnn_res.get('elapsed', 0)
    rnn_csv = snapshot_csv('mavise_rnn_baseline.csv', 'mavise_rnn_baseline.csv')

    # 2) CNN
    cnn_res = run_baseline('mavise_cnn_baseline_10seed', 'eval_mavise_cnn_baseline.py',
                           timeout_s=90 * 60)
    durations['cnn'] = cnn_res.get('elapsed', 0)
    cnn_csv = snapshot_csv('mavise_cnn_baseline.csv', 'mavise_cnn_baseline.csv')

    # 3) Build summary
    try:
        build_summary(rnn_csv, cnn_csv, durations)
    except Exception as e:
        log(f'Summary build failed: {e}\n{traceback.format_exc()}')

    total = time.time() - t_start
    log(f'=== ALL DONE ({total/60:.1f} min) ===')


if __name__ == '__main__':
    main()
