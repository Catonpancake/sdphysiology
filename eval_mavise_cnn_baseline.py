"""
eval_mavise_cnn_baseline.py
============================
MAVISE CNN baseline — leakage-free, cross-scene training.

Architecture: CNNRegressor (reduced capacity for weak signal)
  Original (with y_cont leakage): num_filters=256, mlp_depth=3  -> R²=0.65-0.67
  This (leakage-free):            num_filters=64,  mlp_depth=1  -> ?

Pipeline:
  1. Load X (N, 300, 67), y (N,) from all 4 scenes
  2. Per-PID X normalization (all scenes per PID)
  3. Cross-scene combined training (80/10/10 group split)
  4. Early stopping on combined val
  5. Evaluate per scene

Usage:
  C:/Users/user/anaconda3/envs/ml_env/python.exe eval_mavise_cnn_baseline.py
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
from models import CNNRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("c:/Users/user/code/SDPhysiology")
SPLIT_FILE = DATA_ROOT / "split_fixed_test.json"
OUT_FILE   = DATA_ROOT / "Writing_resource" / "mavise_cnn_baseline.csv"

SCENES = ["Hallway", "Hall", "Elevator", "Outside"]

# Reduced capacity (no leakage → weaker signal → smaller model)
CNN_PARAMS = dict(
    input_channels=67,
    num_filters=64,
    kernel_size=3,
    dropout=0.5,
    pool_bins=50,
    mlp_depth=1,
    mlp_hidden=128,
)

TRAIN_PARAMS = dict(
    lr=1e-3,
    batch_size=256,
    max_epochs=150,
    patience=15,
    min_delta=1e-4,
    weight_decay=1e-4,
)

NUM_SEEDS = 10   # 여러 seed → 평균 (variance 줄이기). 10 for paper-grade error bars.

# ── Helpers ───────────────────────────────────────────────────────────────────
def r2(yt, yp):
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 1e-12 else np.nan


def perpid_normalize_X(X, pids):
    X_out = X.copy().astype(np.float32)
    for pid in np.unique(pids):
        m = pids == pid
        Xp = X_out[m]
        mu = Xp.mean(axis=(0, 1), keepdims=True)
        sg = Xp.std(axis=(0, 1), keepdims=True)
        sg[sg < 1e-8] = 1.0
        X_out[m] = (Xp - mu) / sg
    X_out = np.nan_to_num(X_out, nan=0.0)
    return X_out


def load_scene(scene):
    d = DATA_ROOT / f"ml_processed_behavior_{scene}"
    X   = np.load(d / "X_array.npy")
    y   = np.load(d / "y_array.npy")
    pid = np.load(d / "pid_array.npy", allow_pickle=True)
    return X, y, pid


def make_loader(X, y, batch_size, shuffle=True):
    # CNN expects (B, C, T) — transpose from (N, T, C)
    Xt = torch.from_numpy(X.transpose(0, 2, 1))   # (N, 67, 300)
    yt = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Training ─────────────────────────────────────────────────────────────────
def train_one_seed(X_tr, y_tr, X_va, y_va, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CNNRegressor(**CNN_PARAMS).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=TRAIN_PARAMS["lr"],
                             weight_decay=TRAIN_PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TRAIN_PARAMS["max_epochs"])

    tr_loader = make_loader(X_tr, y_tr, TRAIN_PARAMS["batch_size"], shuffle=True)
    va_loader = make_loader(X_va, y_va, TRAIN_PARAMS["batch_size"], shuffle=False)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, TRAIN_PARAMS["max_epochs"] + 1):
        # Train
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(Xb), yb)
            loss.backward()
            opt.step()
        scheduler.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in va_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_losses.append(F.mse_loss(model(Xb), yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss - TRAIN_PARAMS["min_delta"]:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= TRAIN_PARAMS["patience"]:
                print(f"    seed={seed} early stop @ epoch {epoch}, best_val_loss={best_val_loss:.4f}")
                break
    else:
        print(f"    seed={seed} max epochs, best_val_loss={best_val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


def predict(model, X, device, batch_size=512):
    loader = make_loader(X, np.zeros(len(X)), batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
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
    print(f"Split: train={len(train_pids)} val={len(val_pids)} test={len(test_pids)} PIDs\n")

    # Load & normalize (all PIDs across all scenes combined)
    print("Loading & per-PID normalizing...")
    scene_raw = {}
    for scene in SCENES:
        X, y, pid = load_scene(scene)
        scene_raw[scene] = (X, y, pid)

    all_X   = np.concatenate([scene_raw[s][0] for s in SCENES], axis=0)
    all_pid = np.concatenate([scene_raw[s][2] for s in SCENES], axis=0)
    all_X_n = perpid_normalize_X(all_X, all_pid)

    n_each = [len(scene_raw[s][1]) for s in SCENES]
    offsets = np.concatenate([[0], np.cumsum(n_each)])

    scene_norm = {}
    for i, scene in enumerate(SCENES):
        X_n = all_X_n[offsets[i]:offsets[i+1]]
        y   = scene_raw[scene][1]
        pid = scene_raw[scene][2]
        tr  = np.isin(pid, train_pids)
        va  = np.isin(pid, val_pids)
        te  = np.isin(pid, test_pids)
        scene_norm[scene] = dict(
            X_n=X_n, y=y,
            X_tr=X_n[tr], y_tr=y[tr],
            X_va=X_n[va], y_va=y[va],
            X_te=X_n[te], y_te=y[te],
        )
        print(f"  {scene}: train={tr.sum()} val={va.sum()} test={te.sum()}")

    # Cross-scene combined train/val
    X_tr_all = np.concatenate([scene_norm[s]["X_tr"] for s in SCENES], axis=0)
    y_tr_all = np.concatenate([scene_norm[s]["y_tr"] for s in SCENES], axis=0)
    X_va_all = np.concatenate([scene_norm[s]["X_va"] for s in SCENES], axis=0)
    y_va_all = np.concatenate([scene_norm[s]["y_va"] for s in SCENES], axis=0)
    print(f"\nCombined: train={len(y_tr_all)} val={len(y_va_all)} windows")
    print(f"CNN params: {CNN_PARAMS}")
    print(f"Train params: {TRAIN_PARAMS}")

    rows = []
    seed_preds = {scene: [] for scene in SCENES}  # collect per-seed predictions

    for seed in range(NUM_SEEDS):
        print(f"\n── Seed {seed+1}/{NUM_SEEDS} ──")
        model = train_one_seed(X_tr_all, y_tr_all, X_va_all, y_va_all, device, seed=seed*7)

        for scene in SCENES:
            X_te = scene_norm[scene]["X_te"]
            y_te = scene_norm[scene]["y_te"]
            if len(y_te) < 2:
                continue
            pred = predict(model, X_te, device)
            seed_preds[scene].append((y_te, pred))
            r2_te = r2(y_te, pred)
            print(f"    {scene:<12} R²={r2_te:+.4f}")

    # Average predictions across seeds
    print(f"\n{'='*55}")
    print(f"{'Scene':<12} {'CNN R² (avg seeds)':>20}")
    print(f"{'-'*55}")
    for scene in SCENES:
        sp = seed_preds[scene]
        if not sp:
            continue
        y_te = sp[0][0]
        pred_avg = np.mean([p for _, p in sp], axis=0)
        r2_avg = r2(y_te, pred_avg)
        # also per-seed
        r2_each = [r2(y, p) for y, p in sp]
        print(f"  {scene:<12} R²={r2_avg:+.4f}  (per-seed: {[f'{v:+.3f}' for v in r2_each]})")
        rows.append(dict(scene=scene, model="CNN",
                         test_r2_avg=r2_avg,
                         **{f"seed{i}_r2": r2_each[i] for i in range(len(r2_each))}))

    df = pd.DataFrame(rows)
    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")
    print(f"Total: {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
