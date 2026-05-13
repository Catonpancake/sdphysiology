"""
eval_mavise_rnn_baseline.py
============================
MAVISE — LSTM / GRU / GRUAttention baseline.

설정:
  - per-PID X normalization + HV masking (이전 best 조합)
  - Cross-scene combined training, per-scene evaluation
  - 3 seeds avg
  - hidden_size=64, num_layers=1 (weak-signal에 맞게 작은 모델)

Reference (이전 결과):
  CNN (cross-scene, HV):  Hallway +0.309  Hall +0.036  Elevator +0.061  Outside -0.013
  XGB (per-scene, HV):    Hallway +0.424  Hall +0.001  Elevator +0.082  Outside +0.005

Usage:
  C:/Users/user/anaconda3/envs/ml_env/python.exe eval_mavise_rnn_baseline.py
"""

import json, time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from models import LSTMRegressor, GRURegressor, GRUAttentionRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("c:/Users/user/code/SDPhysiology")
SPLIT_FILE  = DATA_ROOT / "split_fixed_test.json"
OUT_FILE    = DATA_ROOT / "Writing_resource" / "mavise_rnn_baseline.csv"

SCENES      = ["Hallway", "Hall", "Elevator", "Outside"]
HV_QUANTILE = 0.25
NUM_SEEDS   = 10

MODELS = {
    "LSTM": lambda in_ch: LSTMRegressor(
        input_size=in_ch, hidden_size=64, num_layers=1, dropout=0.3),
    "GRU": lambda in_ch: GRURegressor(
        input_size=in_ch, hidden_size=64, num_layers=1, dropout=0.3),
    "GRU_Attn": lambda in_ch: GRUAttentionRegressor(
        input_size=in_ch, hidden_size=64, num_layers=1, dropout=0.3),
}

TRAIN_P = dict(lr=1e-3, batch_size=256, max_epochs=150,
               patience=15, min_delta=1e-4, weight_decay=1e-4)

# ── Helpers ───────────────────────────────────────────────────────────────────
def r2(yt, yp):
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 1e-12 else np.nan

def perpid_normalize_X(X, pids):
    X_out = X.copy().astype(np.float32)
    for pid in np.unique(pids):
        m = pids == pid
        Xp = X_out[m]
        mu = Xp.mean(axis=(0,1), keepdims=True)
        sg = Xp.std(axis=(0,1), keepdims=True)
        sg[sg < 1e-8] = 1.0
        X_out[m] = (Xp - mu) / sg
    return np.nan_to_num(X_out, nan=0.0)

def load_scene(scene):
    d = DATA_ROOT / f"ml_processed_behavior_{scene}"
    return (np.load(d / "X_array.npy"),
            np.load(d / "y_array.npy"),
            np.load(d / "pid_array.npy", allow_pickle=True))

def make_loader(X, y, batch_size, shuffle=True, rnn=True):
    # RNN: (B, T, C),  CNN: (B, C, T)
    if rnn:
        Xt = torch.from_numpy(X)                        # (N, T, C) already
    else:
        Xt = torch.from_numpy(X.transpose(0, 2, 1))    # (N, C, T)
    yt = torch.from_numpy(y.astype(np.float32))
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)

def train_model(model, X_tr, y_tr, X_va, y_va, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN_P["lr"],
                              weight_decay=TRAIN_P["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_P["max_epochs"])
    tr_ld = make_loader(X_tr, y_tr, TRAIN_P["batch_size"], shuffle=True)
    va_ld = make_loader(X_va, y_va, TRAIN_P["batch_size"], shuffle=False)

    best_loss, best_state, no_imp = float("inf"), None, 0
    for ep in range(1, TRAIN_P["max_epochs"] + 1):
        model.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            F.mse_loss(model(Xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        vl = []
        with torch.no_grad():
            for Xb, yb in va_ld:
                vl.append(F.mse_loss(model(Xb.to(device)), yb.to(device)).item())
        vl = float(np.mean(vl))
        if vl < best_loss - TRAIN_P["min_delta"]:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= TRAIN_P["patience"]:
                break
    model.load_state_dict(best_state)
    print(f"      seed={seed}  val_loss={best_loss:.4f}  ep={ep}")
    return model

def predict(model, X, device, batch_size=512):
    ld = make_loader(X, np.zeros(len(X)), batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in ld:
            preds.append(model(Xb.to(device)).cpu().numpy())
    return np.concatenate(preds)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)
    train_pids = split["train_pids"]
    val_pids   = split["val_pids"]
    test_pids  = split["test_pids"]
    print(f"Split: train={len(train_pids)} val={len(val_pids)} test={len(test_pids)} PIDs")

    # Load & per-PID normalize
    print("\nLoading & normalizing...")
    scene_raw = {s: load_scene(s) for s in SCENES}
    all_X   = np.concatenate([scene_raw[s][0] for s in SCENES])
    all_pid = np.concatenate([scene_raw[s][2] for s in SCENES])
    all_X_n = perpid_normalize_X(all_X, all_pid)
    offsets = np.concatenate([[0], np.cumsum([len(scene_raw[s][1]) for s in SCENES])])

    scene_norm = {}
    for i, scene in enumerate(SCENES):
        X_n  = all_X_n[offsets[i]:offsets[i+1]]
        y    = scene_raw[scene][1]
        pid  = scene_raw[scene][2]
        tr   = np.isin(pid, train_pids)
        va   = np.isin(pid, val_pids)
        te   = np.isin(pid, test_pids)
        scene_norm[scene] = dict(X_n=X_n, y=y,
            X_tr=X_n[tr], y_tr=y[tr],
            X_va=X_n[va], y_va=y[va],
            X_te=X_n[te], y_te=y[te])
        print(f"  {scene}: train={tr.sum()} val={va.sum()} test={te.sum()}")

    # HV masking per scene, then combine
    scene_thr = {s: float(np.quantile(np.abs(scene_norm[s]["y_tr"]), HV_QUANTILE))
                 for s in SCENES}

    X_tr_all = np.concatenate([
        scene_norm[s]["X_tr"][np.abs(scene_norm[s]["y_tr"]) >= scene_thr[s]]
        for s in SCENES])
    y_tr_all = np.concatenate([
        scene_norm[s]["y_tr"][np.abs(scene_norm[s]["y_tr"]) >= scene_thr[s]]
        for s in SCENES])
    X_va_all = np.concatenate([
        scene_norm[s]["X_va"][np.abs(scene_norm[s]["y_va"]) >= scene_thr[s]]
        for s in SCENES])
    y_va_all = np.concatenate([
        scene_norm[s]["y_va"][np.abs(scene_norm[s]["y_va"]) >= scene_thr[s]]
        for s in SCENES])

    print(f"\nCombined HV-masked: train={len(y_tr_all)} val={len(y_va_all)}")
    in_ch = X_tr_all.shape[2]  # 67

    rows = []

    for model_name, model_fn in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  {model_name}")
        print(f"{'='*55}")

        seed_preds = {s: [] for s in SCENES}

        for si in range(NUM_SEEDS):
            print(f"  -- Seed {si+1}/{NUM_SEEDS} --")
            model = train_model(model_fn(in_ch), X_tr_all, y_tr_all,
                                X_va_all, y_va_all, device, seed=si*7)

            for scene in SCENES:
                d = scene_norm[scene]
                thr = scene_thr[scene]
                mask_te = np.abs(d["y_te"]) >= thr
                X_te_hv = d["X_te"][mask_te]
                y_te_hv = d["y_te"][mask_te]
                if len(y_te_hv) < 2:
                    continue
                pred = predict(model, X_te_hv, device)
                seed_preds[scene].append((y_te_hv, pred))
                print(f"      {scene:<12} R²={r2(y_te_hv, pred):+.4f}")

        # Average across seeds
        print(f"\n  [{model_name}] avg across seeds:")
        for scene in SCENES:
            sp = seed_preds[scene]
            if not sp:
                continue
            y_te = sp[0][0]
            pred_avg = np.mean([p for _, p in sp], axis=0)
            r2_avg   = r2(y_te, pred_avg)
            r2_each  = [r2(y, p) for y, p in sp]
            print(f"    {scene:<12} R²={r2_avg:+.4f}  "
                  f"(seeds: {[f'{v:+.3f}' for v in r2_each]})")
            rows.append(dict(model=model_name, scene=scene,
                             test_r2=r2_avg,
                             **{f"s{i}": r2_each[i] for i in range(len(r2_each))}))

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    def get(model, scene):
        sub = df[(df.model == model) & (df.scene == scene)]["test_r2"].values
        return f"{sub[0]:+.4f}" if len(sub) else "   N/A"

    model_names = list(MODELS.keys())
    header = f"{'Scene':<12}" + "".join(f"{m:>12}" for m in model_names)
    print(f"\n{'='*(len(header)+4)}")
    print(header)
    print(f"{'-'*(len(header)+4)}")
    for scene in SCENES:
        print(f"{scene:<12}" + "".join(f"{get(m,scene):>12}" for m in model_names))

    # 이전 결과 비교
    prev = {
        "CNN":  {"Hallway":+0.309, "Hall":+0.036, "Elevator":+0.061, "Outside":-0.013},
        "XGB":  {"Hallway":+0.424, "Hall":+0.001, "Elevator":+0.082, "Outside":+0.005},
    }
    print(f"\n  Reference:")
    for ref_name, ref_vals in prev.items():
        row = f"  {ref_name:<10}" + "".join(f"{ref_vals[s]:>12.4f}" for s in SCENES)
        print(f"  {ref_name:<10}  " + "  ".join(
            f"{sc:<12} {ref_vals[sc]:+.4f}" for sc in SCENES))

    print(f"\nSaved: {OUT_FILE}")
    print(f"Total: {int(time.time()-t0)}s")

if __name__ == "__main__":
    main()
