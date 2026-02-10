import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models import CNNRegressor, GRURegressor, GRUAttentionRegressor, LSTMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from itertools import product
import random
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import ml_dataloader
# --------------------- 기본 유틸 ---------------------
def set_seed(seed=42):
    seed = int(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_nct_for_cnn(X: np.ndarray, input_channels: int):
    """
    X를 (N,C,T)로 보장. 
    - 이미 (N,C,T)이고 C==input_channels 이면 그대로 반환
    - (N,T,C)이고 C==input_channels 이면 (N,C,T)로 전치
    - 아니면 명확한 에러
    """
    assert isinstance(X, np.ndarray) and X.ndim == 3, f"X must be 3D np.ndarray, got {type(X)} with ndim={getattr(X,'ndim',None)}"
    N, A, B = X.shape

    # 이미 (N,C,T)인 경우
    if A == input_channels:
        return X  # (N,C,T)

    # (N,T,C)인 경우
    if B == input_channels:
        return np.transpose(X, (0, 2, 1))  # (N,C,T)

    raise ValueError(
        f"ensure_nct_for_cnn: shape mismatch. X.shape={X.shape}, expected one axis==input_channels({input_channels})"
    )


def maybe_permute(X, model_type):
    return torch.tensor(X, dtype=torch.float32).permute(0, 2, 1) if model_type == "CNN" else torch.tensor(X, dtype=torch.float32)

def to_tensor_dataset(X, y, model_type):
    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"❌ Empty dataset passed to to_tensor_dataset. X shape: {X.shape}, y shape: {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"❌ Mismatch between X and y: {X.shape[0]} != {y.shape[0]}")
    
    X_tensor = maybe_permute(X, model_type)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)

def to_loader(X, y, model_type, batch_size=32, shuffle=False, input_channels=None):
    if model_type.upper() == "CNN":
        if input_channels is None:
            input_channels = X.shape[-1]  # (N,T,C) 가정
        X_nct = ensure_nct_for_cnn(X, input_channels)   # ← 여기서만 변환
        X_tensor = torch.tensor(X_nct, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

        # 🔎 디버그(임시): 첫 배치 shape 확인
        xb, yb = next(iter(loader))
        print(f"[LOADER-CNN] batch shape={tuple(xb.shape)}  (expect: (B,{input_channels},T≥{max(3,7)}))")

        return loader
    # (비-CNN) 기존 경로 유지
    dataset = to_tensor_dataset(X, y, model_type)
    if len(dataset) == 0:
        raise ValueError("❌ DataLoader received an empty dataset.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_grid_results(results, filename="grid_results.csv"):
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"💾 Saved grid search results to {filename}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, device, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

def save_predictions(y_true, y_pred, filename="predictions.npz"):
    np.savez_compressed(filename, y_true=y_true, y_pred=y_pred)
    print(f"Saved predictions to {filename}")

def plot_train_val_loss(train_losses, val_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_ablation_results(df, title="Feature Ablation (Validation R²)"):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, y="feature_removed", x="val_r2", palette="viridis")
    plt.xlabel("Validation R²")
    plt.ylabel("Feature Removed")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def summarize_split_scores(name, scores):
    if scores is None or len(scores) == 0:
        return {f"{name}_r2_mean": None, f"{name}_r2_std": None,
                f"{name}_rmse_mean": None, f"{name}_rmse_std": None,
                f"{name}_mae_mean": None, f"{name}_mae_std": None}
    arr = np.array(scores, dtype=np.float64)
    out = {
        f"{name}_r2_mean": float(np.mean(arr[:,0])),
        f"{name}_r2_std":  float(np.std(arr[:,0], ddof=1)) if len(arr)>1 else 0.0,
        f"{name}_rmse_mean": float(np.mean(arr[:,1])),
        f"{name}_rmse_std":  float(np.std(arr[:,1], ddof=1)) if len(arr)>1 else 0.0,
        f"{name}_mae_mean": float(np.mean(arr[:,2])),
        f"{name}_mae_std":  float(np.std(arr[:,2], ddof=1)) if len(arr)>1 else 0.0,
    }
    print(f"➡ {name.upper()}  R² {out[f'{name}_r2_mean']:.4f}±{out[f'{name}_r2_std']:.4f} | "
          f"RMSE {out[f'{name}_rmse_mean']:.4f}±{out[f'{name}_rmse_std']:.4f} | "
          f"MAE {out[f'{name}_mae_mean']:.4f}±{out[f'{name}_mae_std']:.4f}")
    return out
import csv

def write_seed_metrics_csv(path_csv, all_train_scores, all_val_scores, test_scores):
    S = max(len(all_train_scores or []), len(all_val_scores or []), len(test_scores or []))
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["seed","split","r2","rmse","mae"])
        for i in range(S):
            if all_train_scores and i < len(all_train_scores):
                r2, rmse, mae = all_train_scores[i]; w.writerow([i,"train",f"{r2:.6f}",f"{rmse:.6f}",f"{mae:.6f}"])
            if all_val_scores and i < len(all_val_scores):
                r2, rmse, mae = all_val_scores[i];  w.writerow([i,"val",  f"{r2:.6f}",f"{rmse:.6f}",f"{mae:.6f}"])
            if test_scores and i < len(test_scores):
                r2, rmse, mae = test_scores[i];     w.writerow([i,"test", f"{r2:.6f}",f"{rmse:.6f}",f"{mae:.6f}"])
                
def count_lstm_params(input_size: int, hidden_size: int, num_layers: int) -> int:
    """
    Pytorch LSTM with bias: per layer params = 4*H*(I_or_H + H + 2)
    (because i2h: 4H*(in), h2h: 4H*H, bias_ih:4H, bias_hh:4H)
    First layer in_dim=I, subsequent layers in_dim=H.
    """
    total = 0
    for layer in range(num_layers):
        in_dim = input_size if layer == 0 else hidden_size
        total += 4 * hidden_size * in_dim   # weight_ih
        total += 4 * hidden_size * hidden_size  # weight_hh
        total += 8 * hidden_size  # two biases
    return total
def make_masks_from_fixed_test(pid_arr, test_pid_list, val_ratio, seed):
    all_pids = np.unique(pid_arr).tolist()
    test_set = set(test_pid_list)
    trainval_pids = [p for p in all_pids if p not in test_set]
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(trainval_pids) * val_ratio)))
    val_pids = set(rng.choice(trainval_pids, size=n_val, replace=False).tolist())
    train_pids = set([p for p in trainval_pids if p not in val_pids])
    pid_arr = np.asarray(pid_arr)
    test_mask  = np.isin(pid_arr, list(test_set))
    val_mask   = np.isin(pid_arr, list(val_pids))
    train_mask = np.isin(pid_arr, list(train_pids))
    return train_mask, val_mask, test_mask

def recipe_for(H:int, L:int, BASE_BATCH_SIZE=16):
    if H <= 128:
        lr, epochs = 1e-3, 400
    elif H <= 256:
        lr, epochs = 3e-4, 600
    else:  # H >= 512
        lr, epochs = 1e-4, 800
    if L >= 2:
        epochs += 200
    return {"learning_rate": lr, "epochs": epochs, "batch_size": BASE_BATCH_SIZE}

def center_from_train_split(y_tr, pid_tr, scene_tr):
    """
    Across-participant 완화 버전:
    - TRAIN 전체 global mean만 빼고,
    - pid/scene 별 평균은 건드리지 않는다.
    """
    y_tr = y_tr.astype(np.float32)
    global_mean = float(y_tr.mean()) if len(y_tr) else 0.0

    def center_fn(y, pid, scene):
        # pid, scene 인자는 인터페이스 유지용
        y = y.astype(np.float32)
        return y - global_mean

    return center_fn, {"global_mean": global_mean}

# def center_from_train_split(y_tr, pid_tr, scene_tr):
#     y_tr = y_tr.astype(np.float32)
#     ps_mean, pid_mean = {}, {}
#     ps_keys = np.stack([pid_tr, scene_tr], axis=1)
#     for (k_pid, k_sc) in np.unique(ps_keys, axis=0):
#         m = (pid_tr == k_pid) & (scene_tr == k_sc)
#         ps_mean[(k_pid, k_sc)] = float(y_tr[m].mean()) if np.any(m) else np.nan
#     for k_pid in np.unique(pid_tr):
#         m = (pid_tr == k_pid)
#         pid_mean[k_pid] = float(y_tr[m].mean()) if np.any(m) else np.nan
#     global_mean = float(y_tr.mean()) if len(y_tr) else 0.0

#     def _mu(pid_i, sc_i):
#         m = ps_mean.get((pid_i, sc_i))
#         if m is None or np.isnan(m):
#             m = pid_mean.get(pid_i, global_mean)
#             if m is None or np.isnan(m):
#                 m = global_mean
#         return m

#     def center_fn(y, pid, scene):
#         y = y.astype(np.float32)
#         mu = np.array([_mu(p, s) for p, s in zip(pid, scene)], dtype=np.float32)
#         return y - mu

#     return center_fn, {"global_mean": global_mean}

def hv_mask_from_train_x(X_all, train_mask, q=0.30):
    """
    Train-derived HV mask (input-based, safer):
    - Per-sample std over all time*channels
    - Threshold: train q-quantile; apply unchanged to all splits
    """
    X_all = np.asarray(X_all, dtype=np.float32)
    X_flat = X_all.reshape(X_all.shape[0], -1)
    std_all = X_flat.std(axis=1)
    thr = float(np.quantile(std_all[train_mask], q)) if np.any(train_mask) else 0.0
    keep_all = std_all >= thr
    return keep_all

import numpy as np

def make_y_transform_train_only(
    y_train: np.ndarray,
    scene_train: np.ndarray,
    mode: str = "scene",          # {"global", "scene"}
    standardize: bool = True,     # True면 z-score, False면 mean-center만
    eps: float = 1e-6,
):
    """
    Across-safe y normalization:
    - NEVER uses pid.
    - Fit stats on TRAIN ONLY, apply unchanged to val/test.

    Returns:
      transform(y, scene) -> y_norm
      inv_transform(y_norm, scene) -> y_orig (optional)
      stats dict
    """
    y_train = y_train.astype(np.float32)
    scene_train = scene_train.astype(object)

    # global stats (train only)
    mu_g = float(np.mean(y_train)) if len(y_train) else 0.0
    sd_g = float(np.std(y_train))  if len(y_train) else 1.0
    if sd_g < eps: sd_g = 1.0

    stats = {"mode": mode, "standardize": standardize, "mu_global": mu_g, "sd_global": sd_g}

    if mode == "global":
        def _mu(scene_i): return mu_g
        def _sd(scene_i): return sd_g

    elif mode == "scene":
        mu_s = {}
        sd_s = {}
        for sc in np.unique(scene_train):
            m = (scene_train == sc)
            ys = y_train[m]
            mu = float(np.mean(ys)) if len(ys) else mu_g
            sd = float(np.std(ys))  if len(ys) else sd_g
            if sd < eps: sd = sd_g
            mu_s[sc] = mu
            sd_s[sc] = sd

        stats["mu_scene"] = {str(k): float(v) for k, v in mu_s.items()}
        stats["sd_scene"] = {str(k): float(v) for k, v in sd_s.items()}

        def _mu(scene_i): return mu_s.get(scene_i, mu_g)
        def _sd(scene_i): return sd_s.get(scene_i, sd_g)

    else:
        raise ValueError(f"mode must be 'global' or 'scene', got {mode}")

    def transform(y: np.ndarray, scene: np.ndarray):
        y = y.astype(np.float32)
        scene = scene.astype(object)
        mu = np.array([_mu(sc) for sc in scene], dtype=np.float32)
        if standardize:
            sd = np.array([_sd(sc) for sc in scene], dtype=np.float32)
            return (y - mu) / (sd + eps)
        else:
            return (y - mu)

    def inv_transform(y_norm: np.ndarray, scene: np.ndarray):
        y_norm = y_norm.astype(np.float32)
        scene = scene.astype(object)
        mu = np.array([_mu(sc) for sc in scene], dtype=np.float32)
        if standardize:
            sd = np.array([_sd(sc) for sc in scene], dtype=np.float32)
            return y_norm * (sd + eps) + mu
        else:
            return y_norm + mu

    return transform, inv_transform, stats


def hv_mask_from_train_y(y_all, pid_all, scene_all, train_mask, q=0.30):
    """
    Train-derived HV mask (target-based; leakage-safe since computed on TRAIN ONLY):
    - abs deviation from (pid,scene) means on TRAIN; train q-quantile → threshold
    - apply unchanged to all splits
    """
    y_all = y_all.astype(np.float32); p_all = pid_all; s_all = scene_all
    y_tr = y_all[train_mask]; p_tr = p_all[train_mask]; s_tr = s_all[train_mask]
    # (pid,scene) means from TRAIN
    ps_mean = {}
    ps_keys = np.stack([p_tr, s_tr], axis=1)
    uniq_ps = np.unique(ps_keys, axis=0)
    for k_pid, k_sc in uniq_ps:
        m = (p_tr == k_pid) & (s_tr == k_sc)
        ps_mean[(k_pid, k_sc)] = float(y_tr[m].mean()) if np.any(m) else np.nan
    # pid means & global
    pid_mean = {}
    for k_pid in np.unique(p_tr):
        m = (p_tr == k_pid)
        pid_mean[k_pid] = float(y_tr[m].mean()) if np.any(m) else np.nan
    global_mean = float(y_tr.mean()) if len(y_tr) else 0.0

    def _mu(pid_i, sc_i):
        m = ps_mean.get((pid_i, sc_i))
        if m is None or np.isnan(m):
            m = pid_mean.get(pid_i, global_mean)
            if m is None or np.isnan(m):
                m = global_mean
        return m

    abs_dev_tr = np.abs(y_tr - np.array([_mu(p, s) for p, s in zip(p_tr, s_tr)], dtype=np.float32))
    thr = float(np.quantile(abs_dev_tr, q)) if len(abs_dev_tr) else 0.0

    abs_dev_all = np.abs(y_all - np.array([_mu(p, s) for p, s in zip(p_all, s_all)], dtype=np.float32))
    keep_all = abs_dev_all >= thr
    return keep_all

def apply_per_split_mask(X, y, pid, scene, widx, train_m, val_m, test_m, keep_all):
    """Intersect split masks with keep_all and return split tuples."""
    tr_idx = np.where(train_m & keep_all)[0]
    va_idx = np.where(val_m & keep_all)[0]
    te_idx = np.where(test_m & keep_all)[0]
    return (X[tr_idx], y[tr_idx], pid[tr_idx], scene[tr_idx], widx[tr_idx]), \
           (X[va_idx], y[va_idx], pid[va_idx], scene[va_idx], widx[va_idx]), \
           (X[te_idx], y[te_idx], pid[te_idx], scene[te_idx], widx[te_idx])

    
# === ml_utils.py ===
from sklearn.model_selection import GroupKFold
import numpy as np
# ============================================================
# Fast FFT-based modality×scene lag estimation + application
# - Step-A: 라그 추정(비모델식, FFT cross-corr, 서브샘플/조기종료/캐시)
# - Step-B: 모달리티별 라그를 각 채널에 적용(씬별 블록 내부에서 shift)
# Author: you
# ============================================================
import os, gc, json, math
import numpy as np
import pandas as pd
from collections import defaultdict

# -----------------------------
# 0) User config
# -----------------------------
STRIDE_SECONDS   = 2.0                     # 윈도우 hop(초) ← 현재 파이프라인 기준
LAG_GRID_SECONDS = np.arange(-6, 7, 2)     # 후보 라그(초): ±6s, 2s step (필요시 조정)
MAX_PIDS_PER_SCENE    = 20                 # scene별 PID 서브샘플 상한
MAX_WINDOWS_PER_PID   = 200                # PID 내 윈도우 서브샘플 상한
EPSILON_FLAT     = 1e-6                    # span < EPSILON_FLAT → flat → 0s
GAMMA_TOL        = 0.01                    # (top1 - score@0s) < GAMMA_TOL → 0s
DELTA_TOL        = 0.005                   # (top1 - top2) < DELTA_TOL → 0s
CACHE_PATH       = "lags_consensus_modality_scene.csv"  # 캐시 저장 경로

# 모달리티별 대표 채널(라그 추정용 1채널)과 적용 대상 채널(라그 적용 시 함께 이동할 채널들)
valid_cols = {
    "EDA":   ["EDA_Tonic", "EDA_Phasic", "SCR_Amplitude", "SCR_RiseTime"],
    "PPG":   ["PPG_Rate"],
    "RSP":   ["RSP_Rate", "RSP_RVT", "RSP_Amplitude"],
    "Pupil": ["pupilL", "pupilR", "pupil_mean"],
}
rep_col_per_modality = {        # 대표 1채널(FFT 상호상관에 사용)
    "EDA":   "EDA_Tonic",
    "PPG":   "PPG_Rate",
    "RSP":   "RSP_Rate",
    "Pupil": "pupil_mean",
}

# -----------------------------
# 1) Utilities
# -----------------------------
# ============================================================
# Drop-in replacement: robust modality×scene lag applier
#   - Fixes IndexError by using scene/pid-local slices → global indices
#   - Shifts per (scene, pid) block; applies to all channels of a modality
#   - Trims to common valid range across modalities (if drop_trimmed=True)
# Requires in scope: STRIDE_SECONDS
# df_lags columns: ["scene","modality","best_lag_s"]
# valid_cols: {"EDA":[...], "PPG":[...], "RSP":[...], "Pupil":[...]}
# ============================================================
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def _build_tag_to_mod_map(feature_tag_list: List[str],
                          valid_cols: Dict[str, List[str]]) -> Dict[str, str]:
    tag2mod = {}
    for mod, tags in valid_cols.items():
        for t in tags:
            tag2mod[t] = mod
    # feature_tag_list에 실제 존재하는 태그만 유지
    return {t: tag2mod[t] for t in feature_tag_list if t in tag2mod}

def _channels_of_mod(feature_tag_list: List[str],
                     valid_cols: Dict[str, List[str]],
                     modality: str) -> List[int]:
    want = set(valid_cols.get(modality, []))
    return [i for i, t in enumerate(feature_tag_list) if t in want]

def _lag_steps(best_lag_s: float, stride_seconds: float) -> int:
    if not np.isfinite(best_lag_s):
        return 0
    return int(np.round(best_lag_s / float(stride_seconds)))

def apply_modality_scene_lags(
    X: np.ndarray, y: np.ndarray, pid: np.ndarray,
    scene: np.ndarray, widx: np.ndarray,
    feature_tag_list: List[str],
    df_lags: pd.DataFrame,
    *,
    valid_cols: Dict[str, List[str]],
    drop_trimmed: bool = True,
    stride_seconds: float = None   # ← 없으면 STRIDE_SECONDS를 외부에서 partial로 바인딩해도 됨
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns shifted (X,y,pid,scene,widx). If drop_trimmed=True, only keeps indices
    that are valid for ALL modalities within each (scene,pid) block.
    """
    assert X.ndim == 3, "X must be (N,T,C)"
    N, T, C = X.shape

    if stride_seconds is None:
        # 전역 상수를 쓰고 있다면 다음 한 줄을 STRIDE_SECONDS로 바꿔도 됩니다.
        raise ValueError("stride_seconds must be provided")

    # (1) 라그 테이블 정리: scene→modality→best_lag_s
    lag_map = {}
    for _, row in df_lags.iterrows():
        sc = row["scene"]
        md = row["modality"]
        lag_map.setdefault(sc, {})[md] = float(row["best_lag_s"])

    # (2) 태그→모달리티 매핑
    tag2mod = _build_tag_to_mod_map(feature_tag_list, valid_cols)
    modalities = sorted(set(tag2mod.values()))

    # (3) 출력용 배열 (in-place 수정 → 트림까지 고려하면 인덱스 모음 후 슬라이싱)
    #     시프트는 덮어쓰기를 유발할 수 있으니, 항상 temp copy로 오른쪽에 대입
    X_out = X.copy()
    y_out = y.copy()  # y도 shift? ← 라그는 피처 정렬용이라 보통 y는 고정 유지

    keep_global_indices: List[int] = []  # 최종 보관할 전역 인덱스 집합

    # === 씬 단위 루프 ===
    for sc in np.unique(scene):
        idx_sc = np.where(scene == sc)[0]            # 전역 인덱스
        if idx_sc.size == 0:
            continue

        # 씬 내부 PID 목록
        pids_sc = np.unique(pid[idx_sc])

        # 모달리티별 best lag(초)→스텝
        mod_to_k = {}
        for md in modalities:
            k = _lag_steps(lag_map.get(sc, {}).get(md, 0.0), stride_seconds)
            mod_to_k[md] = k

        # === PID 블록 단위로 시프트 수행 ===
        # 공통 유효 마스크(씬 전체 길이 기준이 아니라, 블록 별로 구해서 모아서 붙임)
        for p in pids_sc:
            idx_block = idx_sc[pid[idx_sc] == p]     # 전역 인덱스 (길이 L)
            L = idx_block.size
            if L == 0:
                continue

            # 블록 로컬에서의 공통 valid 마스크 (모든 모달리티 교집합)
            # 처음엔 전부 True → 모달리티별로 &=
            valid_mask_local = np.ones(L, dtype=bool)

            # ---- 모달리티별 시프트 ----
            for md in modalities:
                k = mod_to_k.get(md, 0)
                ch_idx = _channels_of_mod(feature_tag_list, valid_cols, md)
                if len(ch_idx) == 0 or k == 0:
                    # 라그 0이면 전체가 유효 범위
                    continue

                if abs(k) >= L:
                    # 라그가 블록 길이 이상이면 전부 날아감 → 결국 이 PID 블록은 버려짐
                    valid_mask_local[:] = False
                    continue

                if k > 0:
                    # 과거 → 미래 정렬: dst는 [k:L), src는 [0:L-k)
                    src_local = np.arange(0, L - k, dtype=int)
                    dst_local = np.arange(k, L, dtype=int)
                else:
                    # k<0: 미래 → 과거 정렬: dst는 [0:L+k), src는 [(-k):L)
                    kk = -k
                    src_local = np.arange(kk, L, dtype=int)
                    dst_local = np.arange(0, L - kk, dtype=int)

                # 전역 인덱스로 변환
                src = idx_block[src_local]
                dst = idx_block[dst_local]

                # 안전: 길이 동일 보장
                if src.size != dst.size or src.size == 0:
                    valid_mask_local[:] = False
                    continue

                # 덮어쓰기 방지: temp copy 후 대입
                # (N,T,C)에서 N축을 src/dst로 선택, 채널 축은 ch_idx
                temp = X_out[src, :, :][:, :, ch_idx].copy()
                X_out[dst, :, :][:, :, ch_idx] = temp

                # 유효 마스크 갱신(이 모달리티 관점에서 유효한 위치만 True)
                mask_this = np.zeros(L, dtype=bool)
                mask_this[dst_local] = True
                valid_mask_local &= mask_this

            # ---- 블록 내 최종 keep 인덱스 확정 ----
            if drop_trimmed:
                keep_block = idx_block[valid_mask_local]
            else:
                # 트림하지 않으면 가장 넓은 범위를 keep (여기선 전부 유지)
                keep_block = idx_block

            if keep_block.size > 0:
                keep_global_indices.append(keep_block)

    # === 최종 트림/정렬 ===
    if drop_trimmed:
        if len(keep_global_indices) == 0:
            # 전부 잘려버린 경우(극단적인 라그 설정) — 빈 배열 반환
            return (X_out[:0], y_out[:0], pid[:0], scene[:0], widx[:0])
        keep_idx = np.concatenate(keep_global_indices, axis=0)
        # 정렬(안 하면 씬/피드 순서가 섞일 수 있음)
        keep_idx.sort(kind="mergesort")
        X_out = X_out[keep_idx]
        y_out = y_out[keep_idx]
        pid_out = pid[keep_idx]
        scene_out = scene[keep_idx]
        widx_out = widx[keep_idx]
    else:
        pid_out, scene_out, widx_out = pid.copy(), scene.copy(), widx.copy()

    # (선택) 안정성 위해 최종 정렬: (pid, widx)
    order = np.lexsort((widx_out, pid_out))
    return (X_out[order], y_out[order], pid_out[order], scene_out[order], widx_out[order])

def _ensure_ntc(X, feature_tag_list):
    """(N,T,C) 강제. (N,C,T)인 경우 변환."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must be 3D")
    # Heuristic: T dimension is the 'middle' for (N,T,C)
    if X.shape[1] < X.shape[2]:  # already (N,T,C)
        return X
    else:                        # (N,C,T) → (N,T,C)
        return np.transpose(X, (0, 2, 1))

def _to_float32(*arrs):
    return [np.asarray(a, dtype=np.float32) for a in arrs]

def _idx_map(feature_tag_list):
    """feature_tag_list → {col_name: channel_index}"""
    return {name: i for i, name in enumerate(feature_tag_list)}

def _uniform_subsample_indices(idx, k, rng):
    """idx 배열에서 균등 서브샘플 k개."""
    idx = np.asarray(idx)
    if len(idx) <= k:
        return idx
    # 균등 간격 선택
    lin = np.linspace(0, len(idx)-1, num=k, dtype=int)
    return idx[lin]

def _group_sort_indices(pid, scene, widx):
    """(scene, pid) 블록별로 widx 오름차순 인덱스 리스트를 반환."""
    df = pd.DataFrame({"pid":pid, "scene":scene, "widx":widx, "i":np.arange(len(pid))})
    out = {}
    for (sc, p), g in df.groupby(["scene","pid"]):
        out[(sc, p)] = g.sort_values("widx")["i"].to_numpy()
    return out

def _xcorr_fft(a, b, max_lag_steps):
    """
    정상화 상호상관(FFT). 입력은 1D float32, 같은 길이.
    반환: lags_steps[-max..+max], corr[len=2*max+1]
    """
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    n = len(a)
    # next pow2 for speed
    nfft = 1 << (n*2-1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    cc = np.fft.irfft(fa * np.conj(fb), nfft)
    cc = np.concatenate([cc[-(n-1):], cc[:n]])  # shift zero-lag center
    # normalize by length at each lag ~ use n for approximate NCC
    cc = cc / (n + 1e-8)
    # center slice around zero-lag
    mid = len(cc)//2
    lags = np.arange(-max_lag_steps, max_lag_steps+1)
    corr = np.take(cc, mid + lags)
    return lags, corr.astype(np.float32)

def _pick_best_lag(curve_lags_s, curve_vals):
    """
    곡선에서 최고점과 안정화 규칙으로 최종 라그 선택.
    반환: best_lag_s, reason, delta_vs0, span
    """
    vals = np.asarray(curve_vals, dtype=np.float32)
    lags = np.asarray(curve_lags_s, dtype=np.float32)
    if not np.isfinite(vals).any():
        return 0.0, "invalid", np.nan, np.nan

    vmax_idx = int(np.nanargmax(vals))
    top1 = float(vals[vmax_idx])
    lag_top1 = float(lags[vmax_idx])
    # top2
    tmp = vals.copy()
    tmp[vmax_idx] = -np.inf
    top2 = float(tmp[np.nanargmax(tmp)]) if np.isfinite(tmp).any() else top1
    # span
    span = float(np.nanmax(vals) - np.nanmin(vals))
    # score@0s
    if 0.0 in lags:
        score0 = float(vals[lags.tolist().index(0.0)])
    else:
        score0 = np.nan
    delta_vs0 = (top1 - score0) if np.isfinite(score0) else np.nan

    # rules
    if (not np.isfinite(top1)) or (np.isfinite(span) and span < EPSILON_FLAT):
        return 0.0, "flat/invalid", delta_vs0, span
    if np.isfinite(top2) and (top1 - top2) < DELTA_TOL:
        return 0.0, f"Δtop1-top2<{DELTA_TOL}", delta_vs0, span
    if np.isfinite(delta_vs0) and (delta_vs0 < GAMMA_TOL):
        return 0.0, f"Δvs0<{GAMMA_TOL}", delta_vs0, span
    return lag_top1, "peak", delta_vs0, span

# -----------------------------
# 2) Step-A: estimate lags (FFT, subsample, cache)
# -----------------------------
def estimate_modality_scene_lags_fft(
    X, y, pid, scene, widx, feature_tag_list,
    *,
    stride_seconds=STRIDE_SECONDS,
    lag_grid_seconds=LAG_GRID_SECONDS,
    max_pids_per_scene=MAX_PIDS_PER_SCENE,
    max_windows_per_pid=MAX_WINDOWS_PER_PID,
    cache_path=CACHE_PATH,
    random_state=42
):
    """
    출력: DataFrame [scene, modality, best_lag_s, delta_vs0, span, n_pid_used, n_win_used]
    """
    # 캐시 있으면 로드
    if cache_path and os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path)
            # 캐시 스키마 간단 검증
            required = {"scene","modality","best_lag_s"}
            if required.issubset(df_cache.columns):
                return df_cache
        except Exception:
            pass  # 캐시 무시하고 재계산

    rng = np.random.default_rng(random_state)
    X = _ensure_ntc(X, feature_tag_list)
    X, y = _to_float32(X, y)
    pid = np.asarray(pid)
    scene = np.asarray(scene)
    widx = np.asarray(widx)

    # 채널 인덱스 매핑
    name_to_ci = _idx_map(feature_tag_list)

    # scene/pid 그룹 인덱스 정렬표
    grp = _group_sort_indices(pid, scene, widx)

    # 라그→윈도우 스텝 변환
    lag_steps = np.array(np.round(lag_grid_seconds / stride_seconds), dtype=int)
    lag_steps = np.unique(lag_steps)
    lag_lut_s = (lag_steps * stride_seconds).astype(np.float32)

    rows = []
    for sc in sorted(np.unique(scene)):
        # scene 내 PID 수집
        pids_sc = sorted({p for (sc2, p) in grp.keys() if sc2 == sc})
        if len(pids_sc) == 0:
            continue
        # PID 서브샘플
        if len(pids_sc) > max_pids_per_scene:
            pids_sc = list(rng.choice(pids_sc, size=max_pids_per_scene, replace=False))

        for modality, rep_col in rep_col_per_modality.items():
            # 대표 채널 존재 체크
            if rep_col not in name_to_ci:
                # 모달리티 전체 skip
                continue
            ci = name_to_ci[rep_col]

            # 라그 곡선 누적(가중 평균: PID별 유효 윈도우 수)
            acc_num = np.zeros(len(lag_steps), dtype=np.float64)
            acc_den = np.zeros(len(lag_steps), dtype=np.float64)
            total_used_windows = 0
            used_pids = 0

            for p in pids_sc:
                idx = grp.get((sc, p), None)
                if idx is None or len(idx) < 5:
                    continue
                # PID 내 균등 서브샘플
                idx_use = _uniform_subsample_indices(idx, max_windows_per_pid, rng)
                # y_seq (per-window)
                y_seq = y[idx_use].astype(np.float32)
                # x_seq: 윈도우 평균(대표 채널)
                x_seq = X[idx_use, :, ci].mean(axis=1).astype(np.float32)

                n = len(y_seq)
                if n < 10 or np.isnan(y_seq).any() or np.isnan(x_seq).any():
                    continue

                # FFT 상호상관(정규화)
                # 여기서는 전체 라그 곡선에서 필요한 lag만 샘플링 (간단화를 위해 직접 시프트 계산)
                # cc: 우리 lag_steps에 맞춰 직접 계산(길이 짧아 메모리 안정적)
                pid_curve = np.full(len(lag_steps), np.nan, dtype=np.float32)
                for j, k in enumerate(lag_steps):
                    if k >= 0:
                        y_s = y_seq[: n-k]
                        x_s = x_seq[k : n]
                    else:
                        k2 = -k
                        y_s = y_seq[k2 : n]
                        x_s = x_seq[: n-k2]
                    if len(y_s) < 5:
                        continue
                    ys = (y_s - y_s.mean()) / (y_s.std() + 1e-8)
                    xs = (x_s - x_s.mean()) / (x_s.std() + 1e-8)
                    pid_curve[j] = float(np.dot(ys, xs) / (len(ys) + 1e-8))

                # 누적(가중치 = 유효 샘플 길이)
                w = float(len(y_s)) if 'y_s' in locals() else 0.0
                if np.isfinite(pid_curve).any() and w > 0:
                    mask = np.isfinite(pid_curve)
                    acc_num[mask] += (pid_curve[mask] * w)
                    acc_den[mask] += w
                    total_used_windows += int(len(idx_use))
                    used_pids += 1

            if used_pids == 0 or (acc_den <= 0).all():
                best_lag_s, reason, delta_vs0, span = 0.0, "no_data", np.nan, np.nan
            else:
                mean_curve = np.divide(acc_num, np.maximum(acc_den, 1e-8), out=np.full_like(acc_num, np.nan), where=(acc_den>0))
                best_lag_s, reason, delta_vs0, span = _pick_best_lag(lag_lut_s, mean_curve)

            rows.append({
                "scene": sc,
                "modality": modality,
                "rep_col": rep_col,
                "best_lag_s": float(best_lag_s),
                "delta_vs0": (float(delta_vs0) if np.isfinite(delta_vs0) else np.nan),
                "span": (float(span) if np.isfinite(span) else np.nan),
                "n_pid_used": int(used_pids),
                "n_win_used": int(total_used_windows),
            })
            gc.collect()

    df = pd.DataFrame(rows, columns=["scene","modality","rep_col","best_lag_s","delta_vs0","span","n_pid_used","n_win_used"])
    # 캐시 저장
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


# -----------------------------
# 4) Example usage (drop-in)
# -----------------------------
# (전제) 메모리에 다음이 준비되어 있어야 합니다:
#   X: (N,T,C) or (N,C,T)
#   y: (N,)
#   pid, scene, widx: (N,)
#   feature_tag_list: list[str] (len=C)
#
# 1) 라그 추정 (캐시 재사용)
# df_lags = estimate_modality_scene_lags_fft(
#     X, y, pid, scene, widx, feature_tag_list,
#     stride_seconds=STRIDE_SECONDS,
#     lag_grid_seconds=LAG_GRID_SECONDS,
#     max_pids_per_scene=MAX_PIDS_PER_SCENE,
#     max_windows_per_pid=MAX_WINDOWS_PER_PID,
#     cache_path=CACHE_PATH,
#     random_state=42
# )
# print(df_lags.head())
#
# 2) 라그 적용 → 학습 파이프라인에 연결
# X2, y2, pid2, scene2, widx2 = apply_modality_scene_lags(
#     X, y, pid, scene, widx, feature_tag_list, df_lags,
#     valid_cols=valid_cols, drop_trimmed=True
# )
# 이후에는 기존 split_across_with_gap(...), HV masking, y-centering, z-score, ablation/grid/train 등을 그대로 사용하세요.

# === ml_utils.py ===
import numpy as np
from sklearn.model_selection import GroupKFold
from typing import List, Tuple, Dict

def make_group_kfold_indices(groups: np.ndarray, n_splits: int = 5):
    """
    groups: 예) pid_train (Train 마스크 적용 후의 PID)
    반환: [(train_idx, val_idx), ...]  # 둘 다 Train 서브셋 내 인덱스
    """
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(groups))
    return list(gkf.split(idx, groups=groups, y=None))

def lag_sweep_kfold(
    X_tr: np.ndarray, y_tr: np.ndarray, pid_tr: np.ndarray, widx_tr: np.ndarray,
    *,
    stride_seconds: float,
    lag_candidates_sec: List[float],
    n_splits: int = 5,
    scorer_fn=None,
    seed: int = 42
):
    """
    Train 전용으로 라그 후보를 KFold(GroupKFold by PID)로 평가해 best lag 선택
    - scorer_fn: (y_true, y_pred) -> R2 같은 스코어 반환
    - 반환: {"best_lag_sec": float, "cv_r2_mean": float}
    """
    assert scorer_fn is not None, "scorer_fn을 전달하세요."
    folds = make_group_kfold_indices(pid_tr, n_splits=n_splits)
    rng = np.random.default_rng(seed)
    best_lag, best_mean = None, -1e9

    for lag_s in lag_candidates_sec:
        r2s = []
        for tr_idx, va_idx in folds:
            # 🅐 lag 적용(윈도우 격자 기반이면 lag_seconds를 frame 단위 정수 스텝으로 변환해서 shift)
            X_tr_fold, y_tr_fold = X_tr[tr_idx], y_tr[tr_idx]
            X_va_fold, y_va_fold = X_tr[va_idx], y_tr[va_idx]
            # NOTE: 아래 apply_lag_timeseries_like는 사용중인 함수 형태에 맞게 바꾸세요
            X_tr_l, y_tr_l, _, _, _ = apply_lag_timeseries(X_tr_fold, y_tr_fold,
                                                            pid_tr[tr_idx], np.zeros_like(pid_tr[tr_idx]),
                                                            widx_tr[tr_idx],
                                                            stride_seconds=stride_seconds,
                                                            lag_seconds=lag_s)
            X_va_l, y_va_l, _, _, _ = apply_lag_timeseries(X_va_fold, y_va_fold,
                                                            pid_tr[va_idx], np.zeros_like(pid_tr[va_idx]),
                                                            widx_tr[va_idx],
                                                            stride_seconds=stride_seconds,
                                                            lag_seconds=lag_s)
            # 🅑 간단 모델로 빠른 스코어 산출(또는 현재 사용 모델의 “미니” 설정)
            # 여기서는 예시로 mean-predictor를 쓰지 말고, 실제 파이프라인의 빠른 trial을 호출하세요.
            # y_hat = quick_fit_and_predict(X_tr_l, y_tr_l, X_va_l)   # 사용중인 빠른 래퍼
            # 아래는 자리표시자
            y_hat = np.full_like(y_va_l, fill_value=y_tr_l.mean())
            r2 = scorer_fn(y_va_l, y_hat)
            r2s.append(float(r2))
        mean_r2 = float(np.mean(r2s))
        if mean_r2 > best_mean:
            best_mean, best_lag = mean_r2, float(lag_s)

    return {"best_lag_sec": best_lag, "cv_r2_mean": best_mean}

# --------------------- 모델 생성 ---------------------
MODEL_REGISTRY = {
    "CNN": CNNRegressor,
    "GRU": GRURegressor,
    "GRU_Attn": GRUAttentionRegressor,
    "LSTM": LSTMRegressor  # ✅ 추가됨!
}

def get_model(model_type: str, input_size: int, params: dict):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    ModelClass = MODEL_REGISTRY[model_type]
    
    if model_type == "CNN":
        return ModelClass(
            input_channels=params["input_channels"],  # ✅ 올바르게 수정!
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"]
        )
    else:
        return ModelClass(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Returns (r2, rmse, mae)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape, f"shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # RMSE / MAE
    diff = y_true - y_pred
    mse  = np.mean(diff ** 2)
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))

    # R^2
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return r2, rmse, mae

# --------------------- 데이터 분할 ---------------------
def create_dataloaders(
    X_array,
    y_array,
    pid_array,
    *,
    batch_size: int = 32,
    seed: int = 42,
    mode: str = "train_val",           # ✅ 기본값: 내부 test 떼지 않음
    val_ratio: float = 0.20,           # ✅ 가변화
    model_type: str = "CNN",
    input_channels=None,
    return_pid_splits: bool = False,   # ✅ 선택적으로 PID 분할 정보 반환
):
    """
    PID 그룹을 보장하는 내부 분할용 로더 생성기.

    Parameters
    ----------
    mode : {"train_val", "train_only", "train_val_test"}
        - "train_val"   : Train/Val만 생성(권장; 데이터 낭비 없음)
        - "train_only"  : Val도 떼지 않고 전부 Train으로 사용(최종 고정 에폭 학습에 권장)
        - "train_val_test" : (하위호환/비권장) 내부 Test까지 생성
    val_ratio : float
        mode="train_val"일 때만 사용되는 내부 검증 비율.

    Returns
    -------
    (train_loader, val_loader, test_loader)  # test_loader는 mode에 따라 None
    + (선택) pid_splits 딕셔너리(return_pid_splits=True일 때)
    """
    import numpy as np

    pid_array = np.asarray(pid_array)
    unique_pids = np.unique(pid_array)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)

    if mode not in {"train_val", "train_only", "train_val_test"}:
        raise ValueError("mode must be one of {'train_val','train_only','train_val_test'}")

    if mode == "train_only":
        pids_train = unique_pids
        pids_val   = np.array([], dtype=unique_pids.dtype)
        pids_test  = np.array([], dtype=unique_pids.dtype)

    elif mode == "train_val":
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be in (0,1) for mode='train_val'")
        n_total = len(unique_pids)
        n_val   = max(1, int(round(n_total * val_ratio)))
        n_train = max(1, n_total - n_val)
        pids_train = unique_pids[:n_train]
        pids_val   = unique_pids[n_train:n_train + n_val]
        pids_test  = np.array([], dtype=unique_pids.dtype)

    else:  # "train_val_test"  (하위호환; 비권장)
        n_total = len(unique_pids)
        n_train = int(n_total * 0.8)
        n_val   = int(n_total * 0.1)
        pids_train = unique_pids[:n_train]
        pids_val   = unique_pids[n_train:n_train + n_val]
        pids_test  = unique_pids[n_train + n_val:]

    def _select_by_pids(pids):
        mask = np.isin(pid_array, pids)
        return X_array[mask], y_array[mask]

    X_train, y_train = _select_by_pids(pids_train)
    X_val,   y_val   = (None, None) if pids_val.size  == 0 else _select_by_pids(pids_val)
    X_test,  y_test  = (None, None) if pids_test.size == 0 else _select_by_pids(pids_test)

    print(f"🎯 create_dataloaders mode={mode} | "
          f"train={len(pids_train)} PIDs, val={len(pids_val)} PIDs, test={len(pids_test)} PIDs")
    print(f"   X_train={None if X_train is None else X_train.shape}, "
          f"X_val={None if X_val is None else X_val.shape}, "
          f"X_test={None if X_test is None else X_test.shape}")

    # CNN 채널 수 명시(없으면 데이터에서 추론)
    if model_type.upper() == "CNN":
        if input_channels is None and X_train is not None:
            input_channels = X_train.shape[-1]  # (N,T,C) 가정

    train_loader = to_loader(
        X_train, y_train, model_type, batch_size, shuffle=True,
        input_channels=input_channels if model_type.upper() == "CNN" else None
    )
    val_loader = None if X_val is None else to_loader(
        X_val, y_val, model_type, batch_size, shuffle=False,
        input_channels=input_channels if model_type.upper() == "CNN" else None
    )
    test_loader = None if X_test is None else to_loader(
        X_test, y_test, model_type, batch_size, shuffle=False,
        input_channels=input_channels if model_type.upper() == "CNN" else None
    )

    if return_pid_splits:
        pid_splits = {
            "train_pids": pids_train.tolist(),
            "val_pids":   pids_val.tolist(),
            "test_pids":  pids_test.tolist(),
        }
        return train_loader, val_loader, test_loader, pid_splits
    return train_loader, val_loader, test_loader


# --------------------- 평가 함수 ---------------------
# ← 반드시 열 0에 위치 (클래스 밖, 전역)
import torch

# @torch.no_grad()
# def evaluate(model, loader, device, model_type="CNN", return_arrays=True):
#     """
#     Safe evaluation:
#       - Always normalizes preds / targets to 1-D (B,) before accumulation
#       - Avoids 0-D scalar issues when batch size == 1
#       - Returns (r2, rmse, mae, avg_loss, y_true, y_pred)
#     """
#     model.eval()
#     criterion = torch.nn.MSELoss(reduction="mean")

#     total_loss, total_n = 0.0, 0
#     all_y_true, all_y_pred = [], []

#     for X_batch, y_batch in loader:
#         X_batch = X_batch.to(device, non_blocking=True)
#         y_batch = y_batch.to(device, non_blocking=True)

#         # --- forward ---
#         preds = model(X_batch)

#         # --- shape normalization: ALWAYS 1-D (B,) ---
#         if preds.dim() == 2 and preds.size(-1) == 1:
#             preds = preds.squeeze(-1)   # (B,1) -> (B,)
#         preds = preds.reshape(-1)        # safeguard: (B,) no matter what
#         y_batch = y_batch.reshape(-1)    # targets also (B,)

#         # --- loss ---
#         loss = criterion(preds, y_batch)
#         bs = y_batch.size(0)
#         total_loss += loss.item() * bs
#         total_n += bs

#         # --- collect ---
#         if return_arrays:
#             all_y_true.append(y_batch.detach().view(-1).cpu())
#             all_y_pred.append(preds.detach().view(-1).cpu())

#     avg_loss = total_loss / max(1, total_n)

#     if return_arrays:
#         y_true = torch.cat(all_y_true, dim=0).cpu().numpy() if all_y_true else None
#         y_pred = torch.cat(all_y_pred, dim=0).cpu().numpy() if all_y_pred else None
#         r2, rmse, mae = compute_metrics(y_true, y_pred) if (y_true is not None and y_pred is not None) else (float("nan"), float("nan"), float("nan"))
#     else:
#         y_true = y_pred = None
#         r2 = rmse = mae = float("nan")

#     return r2, rmse, mae, avg_loss, y_true, y_pred
@torch.no_grad()
def evaluate(model, loader, device, model_type="CNN", return_arrays=True):
    """
    Safe evaluation:
      - Always normalizes preds / targets to 1-D (B,) before accumulation
      - Avoids 0-D scalar issues when batch size == 1
      - Returns (r2, rmse, mae, avg_loss, y_true, y_pred)

    ✅ FIX:
      - return_arrays=False여도 metrics(r2/rmse/mae)는 계산한다.
      - 단지 y_true/y_pred 배열만 None으로 반환한다.
    """
    import numpy as np
    import torch

    model.eval()
    criterion = torch.nn.MSELoss(reduction="mean")

    total_loss, total_n = 0.0, 0
    all_y_true, all_y_pred = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds = model(X_batch)

        # --- shape normalization: ALWAYS 1-D (B,) ---
        if preds.dim() == 2 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)   # (B,1) -> (B,)
        preds = preds.reshape(-1)
        y_batch = y_batch.reshape(-1)

        # --- loss ---
        loss = criterion(preds, y_batch)
        bs = y_batch.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        # --- collect ALWAYS (for metrics) ---
        all_y_true.append(y_batch.detach().view(-1).cpu())
        all_y_pred.append(preds.detach().view(-1).cpu())

    avg_loss = total_loss / max(1, total_n)

    if all_y_true:
        y_true = torch.cat(all_y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(all_y_pred, dim=0).cpu().numpy()
        r2, rmse, mae = compute_metrics(y_true, y_pred)
    else:
        y_true = y_pred = None
        r2 = rmse = mae = float("nan")

    # return_arrays=False면 배열만 숨김
    if not return_arrays:
        y_true = None
        y_pred = None

    return r2, rmse, mae, avg_loss, y_true, y_pred


def windowwise_zscore(X, eps=1e-6):
    """
    X: (N,T,C)
    각 (n,c)별로 time축(T) 기준 z-score
    leakage 없음(샘플 내부만 씀)
    """
    mu = X.mean(axis=1, keepdims=True)         # (N,1,C)
    sd = X.std(axis=1, keepdims=True)          # (N,1,C)
    return (X - mu) / (sd + eps)

# def evaluate_and_save(
#     model,
#     test_data,                      # DataLoader or (X, y) or [(X, y)]
#     device,
#     filename: str = "test_predictions.npz",
#     model_type: str = "CNN",
#     batch_size: int = 64,
#     input_channels: int = None,     # CNN: 명시 시 우선, 없으면 모델/데이터에서 추론
# ):
#     """
#     Robust test evaluator:
#     - Accepts DataLoader or raw (X,y) (tuple/list).
#     - For CNN, guarantees (N,C,T) via ensure_nct_for_cnn, then builds a DataLoader.
#     - Calls evaluate() and saves predictions.
#     """
#     from ml_utils import evaluate, save_predictions, to_loader, ensure_nct_for_cnn

#     # 1) 데이터 → DataLoader 통일
#     if isinstance(test_data, torch.utils.data.DataLoader):
#         test_loader = test_data

#     else:
#         # tuple or list -> (X, y) 추출
#         if isinstance(test_data, tuple):
#             X, y = test_data
#         elif isinstance(test_data, list):
#             # [(X, y)] 형태 허용
#             if len(test_data) == 0:
#                 raise ValueError("evaluate_and_save: empty test_data list.")
#             if isinstance(test_data[0], tuple):
#                 X, y = test_data[0]
#             else:
#                 raise TypeError(f"evaluate_and_save: unsupported list element type: {type(test_data[0])}")
#         else:
#             raise TypeError(f"evaluate_and_save: unsupported test_data type: {type(test_data)}")

#         # 기본 검증
#         if not isinstance(X, np.ndarray) or X.ndim != 3:
#             raise ValueError(f"evaluate_and_save expects X as 3D np.ndarray, got {type(X)} with ndim={getattr(X,'ndim',None)}")
#         if not isinstance(y, np.ndarray):
#             y = np.asarray(y)

#         # CNN: (N,C,T) 강제 + input_channels 지정
#         if model_type.upper() == "CNN":
#             # input_channels 추론 우선순위: 인자 > 모델.conv1.in_channels > X.shape[-1](=C)
#             if input_channels is None:
#                 input_channels = getattr(getattr(model, "conv1", None), "in_channels", None)
#             if input_channels is None:
#                 input_channels = X.shape[-1]  # (N,T,C)일 가능성 고려

#             # (N,C,T) 강제 정렬
#             X_nct = ensure_nct_for_cnn(X, input_channels=input_channels)  # (N,C,T)
#             # DataLoader 구성
#             X_tensor = torch.tensor(X_nct, dtype=torch.float32)
#             y_tensor = torch.tensor(y, dtype=torch.float32)
#             test_loader = torch.utils.data.DataLoader(
#                 torch.utils.data.TensorDataset(X_tensor, y_tensor),
#                 batch_size=batch_size, shuffle=False
#             )
#         else:
#             # 비-CNN: 표준 로더로 생성 (내부가 (N,T,C) 가정)
#             test_loader = to_loader(X, y, model_type=model_type, batch_size=batch_size, shuffle=False)

#     # 2) 평가
#     r2, rmse, mae, loss, y_true, y_pred = evaluate(model, test_loader, device, model_type=model_type)

#     # 3) 저장
#     save_predictions(y_true, y_pred, filename)
#     print(f"📊 Test R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}  → saved to {filename}")

#     return r2, rmse, mae, loss
# # 
def evaluate_and_save(
    model,
    test_data,                      # DataLoader or (X, y) or [(X, y)]
    device,
    filename: str = "test_predictions.npz",
    model_type: str = "CNN",
    batch_size: int = 64,
    input_channels: int = None,     # CNN: 명시 시 우선, 없으면 모델/데이터에서 추론
):
    """
    Robust test evaluator:
    - Accepts DataLoader or raw (X,y) (tuple/list).
    - For CNN, guarantees (N,C,T) via ensure_nct_for_cnn, then builds a DataLoader.
    - Calls evaluate() and saves predictions.
    - 추가: Pearson r, baseline RMSE(=std(y_true) in centered setup), ΔRMSE 저장/출력
    """
    import os, json, re, glob
    import numpy as np
    import torch
    from ml_utils import evaluate, save_predictions, to_loader, ensure_nct_for_cnn

    def _pearsonr_safe(a, b, eps=1e-12):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if a.size < 2 or b.size < 2:
            return float("nan")
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + eps
        return float((a*b).sum() / denom)

    # 1) 데이터 → DataLoader 통일
    if isinstance(test_data, torch.utils.data.DataLoader):
        test_loader = test_data
    else:
        # tuple or list -> (X, y) 추출
        if isinstance(test_data, tuple):
            X, y = test_data
        elif isinstance(test_data, list):
            if len(test_data) == 0:
                raise ValueError("evaluate_and_save: empty test_data list.")
            if isinstance(test_data[0], tuple):
                X, y = test_data[0]
            else:
                raise TypeError(f"evaluate_and_save: unsupported list element type: {type(test_data[0])}")
        else:
            raise TypeError(f"evaluate_and_save: unsupported test_data type: {type(test_data)}")

        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(f"evaluate_and_save expects X as 3D np.ndarray, got {type(X)} with ndim={getattr(X,'ndim',None)}")
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        # CNN: (N,C,T) 강제 + input_channels 지정
        if model_type.upper() == "CNN":
            if input_channels is None:
                input_channels = getattr(getattr(model, "conv1", None), "in_channels", None)
            if input_channels is None:
                input_channels = X.shape[-1]  # (N,T,C)일 가능성 고려

            X_nct = ensure_nct_for_cnn(X, input_channels=input_channels)  # (N,C,T)
            X_tensor = torch.tensor(X_nct, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size, shuffle=False
            )
        else:
            # 비-CNN: 표준 로더로 생성 (내부가 (N,T,C) 가정)
            test_loader = to_loader(X, y, model_type=model_type, batch_size=batch_size, shuffle=False)

    # 2) 평가
    r2, rmse, mae, loss, y_true, y_pred = evaluate(model, test_loader, device, model_type=model_type)

    # 3) 추가 메트릭 계산 (centered 셋업 가정: baseline RMSE = std(y_true))
    baseline_rmse = float(np.std(y_true.astype(np.float64)))
    pearson_r     = _pearsonr_safe(y_true, y_pred)
    delta_rmse    = baseline_rmse - float(rmse)

    # 4) 저장 (npz: 예측), (json: 메트릭)
    save_predictions(y_true, y_pred, filename)

    metrics_path = os.path.splitext(filename)[0] + "_metrics.json"
    metrics_obj = {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "loss": float(loss) if hasattr(loss, "__float__") else loss,
        "pearson_r": float(pearson_r),
        "baseline_rmse": float(baseline_rmse),
        "delta_rmse": float(delta_rmse),
        "n": int(len(y_true)),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, ensure_ascii=False, indent=2)

    # 5) 콘솔 출력
    print(
        f"📊 Test R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | "
        f"r: {pearson_r:.4f} | ΔRMSE: {delta_rmse:+.4f} (baseline={baseline_rmse:.4f})"
        f"  → saved to {filename} (+ {os.path.basename(metrics_path)})"
    )

    # 리턴 시그니처는 그대로 유지(호환성)
    return r2, rmse, mae, loss
def train_model(
    X, y, params, model_type="GRU",
    num_epochs=20, seed=42,
    return_curve=False, pid_array=None,
    use_internal_split=True, external_val_data=None,
    patience=5, min_delta=1e-3,
    delta_is_relative=False,
    optimizer_name="adamw", weight_decay=1e-4,
    scheduler_name="plateau",
    scheduler_patience=2, scheduler_factor=0.5,
    max_lr=None,
    grad_clip_norm=0.5,
    amp=False,
    deterministic=True,
    criterion = torch.nn.MSELoss(reduction="mean"),
    # Internal split params
    internal_split_mode="train_val",   # {"train_val","train_only","train_val_test","two_stage"}
    internal_val_ratio=0.20,
):
    """
    Drop-in replacement.

    ✅ 변경 요약:
      - internal_split_mode="two_stage" 추가
        Stage-A: 작은 내부 val로 early-stopping → best_epoch 추정
        Stage-B: train_only(전체 trainval)로 best_epoch 재학습(early-stop 없음)
      - 리턴 시 train/val 분할 인덱스(가능시) 포함
      - return_curve=True:
            (model, train_losses, val_losses, val_r2, val_rmse, val_mae, train_indices, val_indices, train_r2, train_rmse, train_mae)
        False:
            (model, val_r2, val_rmse, val_mae, train_indices, val_indices, train_r2, train_rmse, train_mae)
    """

    def _get_indices_from_loader(loader):
        """
        가능한 경우, 원본 X 기준의 절대 인덱스를 복원.
        - SubsetRandomSampler: loader.sampler.indices
        - torch.utils.data.Subset: loader.dataset.indices
        - 위가 없으면 None (외부에서 평가 시 None 처리)
        """
        if loader is None:
            return None
        idx = None
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "indices"):
            idx = loader.sampler.indices
        elif hasattr(loader, "dataset") and hasattr(loader.dataset, "indices"):
            idx = loader.dataset.indices
        if idx is not None:
            try:
                idx = np.asarray(idx, dtype=int)
            except Exception:
                idx = None
        return idx

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(X) == 0:
        raise ValueError("🚨 X is empty at start of train_model")

    # ---------- params ----------
    p = dict(params)

    # ---------- 입력 크기 (데이터 변형은 로더에서) ----------
    if model_type.upper() == "CNN":
        p["input_channels"] = p.get("input_channels", X.shape[-1] if X.ndim == 3 else None)
        input_size = p["input_channels"]
    else:
        p["input_size"] = p.get("input_size", X.shape[2] if X.ndim == 3 else None)
        input_size = p["input_size"]

    if input_size is None:
        raise ValueError("❌ Unable to infer input_size/input_channels from X; please provide in params.")

    # ---------- Build loaders ----------
    if use_internal_split:
        if pid_array is None:
            raise ValueError("❌ pid_array must be provided when use_internal_split=True")

        if internal_split_mode == "two_stage":
            # Stage-A: 작은 내부 검증셋(권장 0.02~0.10)
            stageA_val_ratio = max(0.02, float(internal_val_ratio))
            train_loader_A, val_loader_A, _ = create_dataloaders(
                X, y, pid_array=pid_array,
                batch_size=p["batch_size"], seed=seed,
                mode="train_val",
                val_ratio=stageA_val_ratio,
                model_type=model_type,
                input_channels=p.get("input_channels", None)
            )
            # Stage-B: full train_only (검증 없음)
            train_loader_B, _, _ = create_dataloaders(
                X, y, pid_array=pid_array,
                batch_size=p["batch_size"], seed=seed,
                mode="train_only",
                model_type=model_type,
                input_channels=p.get("input_channels", None)
            )
            # 인덱스 복원용(가능시)
            val_loader = val_loader_A
            train_loader = train_loader_A
            two_stage = True

        else:
            train_loader, val_loader, _ = create_dataloaders(
                X, y, pid_array=pid_array,
                batch_size=p["batch_size"], seed=seed,
                mode=internal_split_mode,
                val_ratio=internal_val_ratio,
                model_type=model_type,
                input_channels=p.get("input_channels", None)
            )
            two_stage = False

    else:
        if external_val_data is None:
            raise ValueError("❌ external_val_data must be provided when use_internal_split=False")
        X_val, y_val = external_val_data
        if len(X) == 0 or len(y) == 0:
            raise ValueError("❌ Empty training data passed.")
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("❌ Empty validation data passed.")

        train_loader = to_loader(
            X, y, model_type, batch_size=p["batch_size"], shuffle=True,
            input_channels=p.get("input_channels", None)
        )
        val_loader = to_loader(
            X_val, y_val, model_type, batch_size=p["batch_size"], shuffle=False,
            input_channels=p.get("input_channels", None)
        )
        two_stage = False

    # ---------- 분할 인덱스(가능 시) ----------
    train_indices = _get_indices_from_loader(train_loader)
    val_indices   = _get_indices_from_loader(val_loader)

    # ---------- 공통 빌더 ----------
    def _build_fresh_model_and_optim():
        _m = get_model(model_type, input_size=input_size, params=p).to(device)
        if optimizer_name.lower() == "adamw":
            _opt = torch.optim.AdamW(_m.parameters(), lr=p["learning_rate"], weight_decay=weight_decay)
        else:
            _opt = torch.optim.Adam(_m.parameters(), lr=p["learning_rate"])
        if scheduler_name == "plateau":
            _sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                _opt, mode="min", factor=scheduler_factor, patience=scheduler_patience, verbose=False
            )
        elif scheduler_name == "cosine":
            T_max = max(1, num_epochs)
            base_lr = _opt.param_groups[0]["lr"]
            if max_lr is not None and max_lr > base_lr:
                for g in _opt.param_groups:
                    g["lr"] = max_lr
            _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=T_max)
        else:
            _sch = None
        return _m, _opt, _sch

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    # ========== Two-Stage ==========
    if two_stage:
        # ----- Stage-A: early-stop -----
        model, optimizer, scheduler = _build_fresh_model_and_optim()
        train_losses_A, val_losses_A = [], []
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None
        best_epoch = -1

        for epoch in range(num_epochs):
            model.train()
            epoch_loss, total_n = 0.0, 0

            for X_batch, y_batch in train_loader_A:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                    preds = model(X_batch)
                    if preds.dim() == 2 and preds.size(-1) == 1:
                        preds = preds.squeeze(-1)
                    loss = criterion(preds.reshape(-1), y_batch.reshape(-1))
                if not torch.isfinite(loss):
                    print(f"[WARN] Non-finite loss detected (epoch={epoch}). "
                        f"Skipping this batch. loss={loss.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                if grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                bs = y_batch.size(0)
                epoch_loss += loss.item() * bs
                total_n += bs

            train_losses_A.append(epoch_loss / max(1, total_n))

            val_r2_A, val_rmse_A, val_mae_A, val_loss, _, _ = evaluate(
                model, val_loader_A, device, model_type=model_type
            )
            val_losses_A.append(val_loss)

            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            elif scheduler_name == "cosine":
                scheduler.step()

            # early-stopping
            if delta_is_relative:
                improved = (best_val_loss == float("inf")) or (
                    (best_val_loss - val_loss) / max(1e-12, best_val_loss) > min_delta
                )
            else:
                improved = (val_loss + min_delta) < best_val_loss

            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
                best_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"⏹️ [Stage-A] Early stopping at epoch {epoch + 1} (best @ {best_epoch}, best_val_loss={best_val_loss:.6f})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # (보고용) Stage-A의 train/val 지표
        train_r2_A, train_rmse_A, train_mae_A, _, _, _ = evaluate(
            model, train_loader_A, device, model_type=model_type
        )

        # ----- Stage-B: full train_only로 best_epoch 재학습 (early-stop 없음) -----
        est_epochs = int(best_epoch if best_epoch > 0 else max(1, num_epochs // 3))
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, optimizer, scheduler = _build_fresh_model_and_optim()
        train_losses_B = []

        for epoch in range(est_epochs):
            model.train()
            epoch_loss, total_n = 0.0, 0

            for X_batch, y_batch in train_loader_B:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                    preds = model(X_batch)
                    if preds.dim() == 2 and preds.size(-1) == 1:
                        preds = preds.squeeze(-1)
                    loss = criterion(preds.reshape(-1), y_batch.reshape(-1))

                scaler.scale(loss).backward()
                if grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                bs = y_batch.size(0)
                epoch_loss += loss.item() * bs
                total_n += bs

            train_losses_B.append(epoch_loss / max(1, total_n))
            if scheduler_name == "cosine":
                scheduler.step()

        # 최종 리턴: 모델은 Stage-B 최종, val curve는 Stage-A 기준
        train_indices = None
        val_indices   = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if return_curve:
            return (
                model,
                (train_losses_A + train_losses_B),
                val_losses_A,
                float(val_r2_A), float(val_rmse_A), float(val_mae_A),
                train_indices, val_indices,
                float(train_r2_A), float(train_rmse_A), float(train_mae_A)
            )
        else:
            return (
                model,
                float(val_r2_A), float(val_rmse_A), float(val_mae_A),
                train_indices, val_indices,
                float(train_r2_A), float(train_rmse_A), float(train_mae_A)
            )

    # ========== 단일 스테이지 (train_val / train_only / train_val_test) ==========
    else:
        model = get_model(model_type, input_size=input_size, params=p).to(device)
        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, verbose=False
            )
        elif scheduler_name == "cosine":
            T_max = max(1, num_epochs)
            base_lr = optimizer.param_groups[0]["lr"]
            if max_lr is not None and max_lr > base_lr:
                for g in optimizer.param_groups:
                    g["lr"] = max_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        else:
            scheduler = None

        scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None
        best_epoch = -1

        for epoch in range(num_epochs):
            model.train()
            epoch_loss, total_n = 0.0, 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                    preds = model(X_batch)
                    if preds.dim() == 2 and preds.size(-1) == 1:
                        preds = preds.squeeze(-1)
                    loss = criterion(preds.reshape(-1), y_batch.reshape(-1))

                scaler.scale(loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                bs = y_batch.size(0)
                epoch_loss += loss.item() * bs
                total_n += bs

            train_losses.append(epoch_loss / max(1, total_n))

            # Validation
            if val_loader is not None:
                val_r2, val_rmse, val_mae, val_loss, _, _ = evaluate(
                    model, val_loader, device, model_type=model_type, return_arrays=False
                )
                val_losses.append(val_loss)

                if scheduler_name == "plateau":
                    scheduler.step(val_loss)
                elif scheduler_name == "cosine":
                    scheduler.step()

                # Early stopping
                if delta_is_relative:
                    improved = (best_val_loss == float("inf")) or (
                        (best_val_loss - val_loss) / max(1e-12, best_val_loss) > min_delta
                    )
                else:
                    improved = (val_loss + min_delta) < best_val_loss

                if improved:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_epoch = epoch + 1
                    best_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                  for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch + 1} (best @ {best_epoch}, best_val_loss={best_val_loss:.6f})")
                    break
            else:
                # val이 없는 경우(=train_only) 스케줄러만 진행
                if scheduler_name == "cosine":
                    scheduler.step()
        # --- [DEBUG] last-epoch(=현재 model) 성능 먼저 계산 ---
        train_r2_last, train_rmse_last, train_mae_last, _, _, _ = evaluate(
            model, train_loader, device, model_type=model_type
        )
        if val_loader is not None:
            val_r2_last, val_rmse_last, val_mae_last, val_loss_last, _, _ = evaluate(
                model, val_loader, device, model_type=model_type, return_arrays=False
            )
        else:
            val_r2_last = val_rmse_last = val_mae_last = val_loss_last = float("nan")

        last_epoch = len(train_losses)  # 실제 수행 epoch 수
        print(f"[DEBUG] LAST  epoch={last_epoch} | TrainR2={train_r2_last:.8f} | ValR2={val_r2_last:.4f} | ValLoss={val_loss_last:.6f} | best@{best_epoch}")

        # --- best_state 적용 (기존 로직) ---
        if best_state is not None:
            model.load_state_dict(best_state)

        # --- [DEBUG] best_state 성능(기존처럼) ---
        train_r2, train_rmse, train_mae, _, _, _ = evaluate(
            model, train_loader, device, model_type=model_type
        )
        if val_loader is not None:
            val_r2, val_rmse, val_mae, val_loss_best, _, _ = evaluate(
                model, val_loader, device, model_type=model_type, return_arrays=False
            )
        else:
            val_r2 = val_rmse = val_mae = float("nan")
            val_loss_best = float("nan")

        print(f"[DEBUG] BEST  epoch={best_epoch} | TrainR2={train_r2:.4f} | ValR2={val_r2:.4f} | ValLoss={val_loss_best:.6f}")

        # 보고용 train/val 지표
        train_r2, train_rmse, train_mae, _, _, _ = evaluate(
            model, train_loader, device, model_type=model_type
        )
        if val_loader is not None:
            val_r2, val_rmse, val_mae, _, _, _ = evaluate(
                model, val_loader, device, model_type=model_type
            )
        else:
            val_r2 = val_rmse = val_mae = float("nan")
        print(f"Train R2={train_r2:.4f}, Val R2={val_r2:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if return_curve:
            return (
                model,
                train_losses,
                val_losses,
                float(val_r2), float(val_rmse), float(val_mae),
                train_indices, val_indices,
                float(train_r2), float(train_rmse), float(train_mae)
            )
        else:
            return (
                model,
                float(val_r2), float(val_rmse), float(val_mae),
                train_indices, val_indices,
                float(train_r2), float(train_rmse), float(train_mae)
            )

# def train_model(
#     X, y, params, model_type="GRU",
#     num_epochs=20, seed=42,
#     return_curve=False, pid_array=None,
#     use_internal_split=True, external_val_data=None,
#     patience=5, min_delta=1e-3,
#     delta_is_relative=False,
#     optimizer_name="adamw", weight_decay=1e-4,
#     scheduler_name="plateau",
#     scheduler_patience=2, scheduler_factor=0.5,
#     max_lr=None,
#     grad_clip_norm=1.0,
#     amp=True,
#     deterministic=True,
#     criterion = torch.nn.MSELoss(reduction="mean"),
#     # Internal split params
#     internal_split_mode="train_val",   #{"train_val","train_only","train_val_test"}
#     internal_val_ratio=0.20,
# ):
#     """
#     Drop-in replacement.

#     ✅ 변경 요약(로직 최소 변경):
#       - 학습 내부 동작은 그대로 유지
#       - 리턴 시 train/val 분할 인덱스를 함께 반환
#         * DataLoader의 sampler/dataset에 indices가 있을 때만 절대 인덱스 복구 가능
#         * 없으면 None 반환
#       - return_curve=True:
#             (model, train_losses, val_losses, val_r2, val_rmse, val_mae, train_indices, val_indices)
#         False:
#             (model, val_r2, val_rmse, val_mae, train_indices, val_indices)
#     """
#     import gc
#     import torch
#     import numpy as np

#     def _get_indices_from_loader(loader):
#         """
#         가능한 경우, 원본 X 기준의 절대 인덱스를 복원.
#         - SubsetRandomSampler: loader.sampler.indices
#         - torch.utils.data.Subset: loader.dataset.indices
#         - 위가 없으면 None (외부에서 평가 시 None 처리)
#         """
#         idx = None
#         # (1) sampler.indices
#         if hasattr(loader, "sampler") and hasattr(loader.sampler, "indices"):
#             idx = loader.sampler.indices
#         # (2) dataset.indices (Subset)
#         elif hasattr(loader, "dataset") and hasattr(loader.dataset, "indices"):
#             idx = loader.dataset.indices
#         # numpy/torch 텐서로 정규화
#         if idx is not None:
#             try:
#                 import numpy as np
#                 idx = np.asarray(idx, dtype=int)
#             except Exception:
#                 idx = None
#         return idx

#     if deterministic:
#         try:
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#             torch.use_deterministic_algorithms(True)
#         except Exception:
#             pass

#     set_seed(seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if len(X) == 0:
#         raise ValueError("🚨 X is empty at start of train_model")

#     # ✅ params 우선 준비
#     p = dict(params)

#     # ✅ 입력 크기 설정(데이터는 변형하지 않음 — 변형은 로더에서)
#     if model_type.upper() == "CNN":
#         p["input_channels"] = p.get("input_channels", X.shape[-1] if X.ndim == 3 else None)
#         input_size = p["input_channels"]
#     else:
#         p["input_size"] = p.get("input_size", X.shape[2] if X.ndim == 3 else None)
#         input_size = p["input_size"]

#     if input_size is None:
#         raise ValueError("❌ Unable to infer input_size/input_channels from X; please provide in params.")

#     # --------- Build loaders ---------
#     if use_internal_split:
#         if pid_array is None:
#             raise ValueError("❌ pid_array must be provided when use_internal_split=True")
#         train_loader, val_loader, _ = create_dataloaders(
#             X, y, pid_array=pid_array,
#             batch_size=p["batch_size"],
#             seed=seed,
#             mode=internal_split_mode,          # ✅ 내부 모드 선택
#             val_ratio=internal_val_ratio,      # ✅ 내부 val 비율
#             model_type=model_type,
#             input_channels=p.get("input_channels", None)  # ✅ CNN 채널 전달
#         )
#     else:
#         if external_val_data is None:
#             raise ValueError("❌ external_val_data must be provided when use_internal_split=False")
#         X_val, y_val = external_val_data

#         if len(X) == 0 or len(y) == 0:
#             raise ValueError("❌ Empty training data passed.")
#         if len(X_val) == 0 or len(y_val) == 0:
#             raise ValueError("❌ Empty validation data passed.")

#         # ❌ 여기에서 X/X_val transpose 금지 — 로더가 처리
#         train_loader = to_loader(X, y, model_type, batch_size=p["batch_size"], shuffle=True,
#                                  input_channels=p.get("input_channels", None))
#         val_loader   = to_loader(X_val, y_val, model_type, batch_size=p["batch_size"], shuffle=False,
#                                  input_channels=p.get("input_channels", None))

#     # 🔎 분할 인덱스(가능할 경우만) 확보
#     train_indices = _get_indices_from_loader(train_loader)
#     val_indices   = _get_indices_from_loader(val_loader)

#     # --------- Model / Optim / Scheduler ---------
#     model = get_model(model_type, input_size=input_size, params=p).to(device)

#     if optimizer_name.lower() == "adamw":
#         optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=weight_decay)
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

#     criterion = criterion

#     if scheduler_name == "plateau":
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, verbose=False
#         )
#     elif scheduler_name == "cosine":
#         T_max = max(1, num_epochs)
#         base_lr = optimizer.param_groups[0]["lr"]
#         if max_lr is not None and max_lr > base_lr:
#             for g in optimizer.param_groups:
#                 g["lr"] = max_lr
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
#     else:
#         scheduler = None

#     scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

#     train_losses, val_losses = [], []
#     best_val_loss = float("inf")
#     epochs_no_improve = 0
#     best_state = None
#     best_epoch = -1

#     # --------- Train loop ---------
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0
#         total_n = 0

#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.to(device, non_blocking=True)
#             y_batch = y_batch.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
#                 preds = model(X_batch)
#                 if preds.dim() == 2 and preds.size(-1) == 1:
#                     preds = preds.squeeze(-1)
#                 preds   = preds.reshape(-1)
#                 y_batch = y_batch.reshape(-1)
#                 loss = criterion(preds, y_batch)

#             scaler.scale(loss).backward()

#             if grad_clip_norm is not None and grad_clip_norm > 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

#             scaler.step(optimizer)
#             scaler.update()

#             bs = y_batch.size(0)
#             epoch_loss += loss.item() * bs
#             total_n += bs

#         train_loss = epoch_loss / max(1, total_n)
#         train_losses.append(train_loss)

#         # ---- Validation
#         val_r2, val_rmse, val_mae, val_loss, _, _ = evaluate(
#             model, val_loader, device, model_type=model_type, return_arrays=False
#         )
#         val_losses.append(val_loss)
#         # ---- Scheduler step
#         if scheduler_name == "plateau":
#             scheduler.step(val_loss)
#         elif scheduler_name == "cosine":
#             scheduler.step()

#         # ---- Early stopping
#         if delta_is_relative:
#             improved = (best_val_loss == float("inf")) or ((best_val_loss - val_loss) / max(1e-12, best_val_loss) > min_delta)
#         else:
#             improved = (val_loss + min_delta) < best_val_loss

#         if improved:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#             best_epoch = epoch + 1
#             best_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print(f"⏹️ Early stopping at epoch {epoch + 1} (best @ {best_epoch}, best_val_loss={best_val_loss:.6f})")
#             break

#     if best_state is not None:
#         model.load_state_dict(best_state)

#     # ✅ 여기서 Train/Val 평가를 둘 다 수행
#     #    - train_loader는 shuffle=True여도 지표 계산에는 영향 없음 (평균/오차 기반)
#     train_r2, train_rmse, train_mae, _, _, _ = evaluate(
#         model, train_loader, device, model_type=model_type
#     )
#     val_r2, val_rmse, val_mae, _, _, _ = evaluate(
#         model, val_loader, device, model_type=model_type
#     )

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     gc.collect()

#     # ✅ 리턴만 확장 (indices는 앞서 잡은 값; None일 수 있음)
#     if return_curve:
#         return (
#             model,                # 0
#             train_losses,         # 1
#             val_losses,           # 2
#             val_r2, val_rmse, val_mae,   # 3,4,5 (기존)
#             train_indices, val_indices,   # 6,7 (있으면 np.ndarray, 없으면 None)
#             train_r2, train_rmse, train_mae  # 8,9,10 🔥 추가
#         )
#     else:
#         return (
#             model,
#             val_r2, val_rmse, val_mae,
#             train_indices, val_indices,
#             train_r2, train_rmse, train_mae
#         )
# --------------------- 기타 ---------------------
def mask(X, y, pids, sel):
    m = np.isin(pids, sel)
    return X[m], y[m], pids[m]


def to_loader_simple(X, y, batch_size=32, permute=True, shuffle=True, model_type=None, input_channels=None):
    # CNN이면 먼저 (N,C,T) 강제
    if (model_type is not None) and (model_type.upper() == "CNN"):
        if input_channels is None:
            input_channels = X.shape[-1]  # (N,T,C) 가정
        X = ensure_nct_for_cnn(X, input_channels)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # 기존 permute 옵션은 호환성 유지용. CNN이면 이미 (N,C,T) 상태이므로 permute 필요 없음.
    if permute and (model_type is None or model_type.upper() != "CNN"):
        X_tensor = X_tensor.permute(0, 2, 1)  # (B,T,C)→(B,C,T) 용 구경

    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def grid_search_model(
    X, y, pid_array,
    model_type,
    search_space,
    num_epochs=20,
    seed=42,
    seed_list=(42, 43, 44),   # ✅ multi-seed
    use_internal_split=False, # ✅ 외부 val 고정 권장
    external_val_data=None,   # (X_val, y_val)
    patience=10,
    min_delta=1e-6,
    criterion=torch.nn.MSELoss(reduction="mean"),
):
    """
    Multi-seed grid search for CNN/GRU etc.
    Returns average & std performance for each param combo.
    """
    from itertools import product
    import numpy as np
    import pandas as pd
    if seed_list is None:
        seed_list = (seed,)      # ✅ 기존 seed → tuple 변환

    results = []
    keys, values = list(search_space.keys()), list(search_space.values())

    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        print(f"🔍 Trying {param_dict}")
        r2_list, rmse_list, mae_list = [], [], []

        for s in seed_list:
            try:
                _model, val_r2, val_rmse, val_mae, *_ = train_model(
                    X, y, {**param_dict, "input_size": X.shape[-1]},  # 🔒 input_size 보장
                    model_type=model_type,
                    num_epochs=num_epochs,
                    seed=s,
                    pid_array=pid_array,
                    use_internal_split=use_internal_split,
                    external_val_data=external_val_data,
                    patience=patience,
                    min_delta=min_delta,
                    criterion=criterion
                )
                r2_list.append(val_r2)
                rmse_list.append(val_rmse)
                mae_list.append(val_mae)
            except Exception as e:
                print(f"❌ Error with params {param_dict}, seed {s}: {e}")
                import traceback
                traceback.print_exc()
                r2_list.append(-9999)
                rmse_list.append(9999)
                mae_list.append(9999)

        result = {
            **param_dict,
            "val_r2_mean": np.mean(r2_list),
            "val_r2_std": np.std(r2_list),
            "val_rmse_mean": np.mean(rmse_list),
            "val_mae_mean": np.mean(mae_list)
        }
        results.append(result)

    best_result = max(results, key=lambda x: x["val_r2_mean"])

    print("\n🏆 Best Hyperparameters:")
    for k, v in best_result.items():
        if not k.startswith("val_"):
            print(f"{k:<12}: {v}")

    return best_result, results

def run_feature_ablation(
    X_train, y_train, pid_train,
    X_val, y_val, pid_val,
    feature_tags, model_type,
    fixed_params, num_epochs=10,
    seed=42,
    seed_list=(42, 43, 44),       # ✅ multi-seed
    batch_size=32,
    patience=10,
    min_delta=1e-6,
    criterion = torch.nn.MSELoss(reduction="mean")
):
    """
    Feature ablation:
      - 채널 완전 제거 대신 마스킹(0으로 대체) → input_size 유지
      - 외부 val 고정 (use_internal_split=False)
      - ΔR² = ablation - baseline 계산
      - multi-seed 평균 성능 산정
    """
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score
    
    if seed_list is None:
        seed_list = (seed,)      # ✅ 기존 seed → tuple 변환

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    def train_eval_with_mask(X_tr, y_tr, X_va, y_va, mask_idx, desc):
        """mask_idx: None이면 baseline, int이면 해당 채널 마스킹"""
        # 마스킹 적용
        X_tr_masked = X_tr.copy()
        X_va_masked = X_va.copy()
        if mask_idx is not None:
            X_tr_masked[:, :, mask_idx] = 0.0
            X_va_masked[:, :, mask_idx] = 0.0

        r2_list = []
        for s in seed_list:
            params = fixed_params.copy()
            params["input_size"] = X_tr_masked.shape[-1]  # ✅ 동기화
            model, *_ = train_model(
                X_tr_masked, y_tr, params,
                model_type=model_type,
                num_epochs=num_epochs,
                seed=s,
                pid_array=pid_train,
                use_internal_split=False,  # ✅ 외부 val 고정
                external_val_data=(X_va_masked, y_va),
                patience=patience,
                min_delta=min_delta,
                criterion=criterion
            )
            val_loader = to_loader(X_va_masked, y_va, model_type, batch_size=batch_size, shuffle=False)
            _, _, _, _, _, y_pred = evaluate(model, val_loader, device, model_type=model_type)
            r2_list.append(r2_score(y_va, y_pred))

        return np.mean(r2_list), np.std(r2_list)

    # Baseline
    base_r2_mean, base_r2_std = train_eval_with_mask(X_train, y_train, X_val, y_val, None, "baseline")
    results.append({
        "feature_removed": "None (baseline)",
        "val_r2_mean": base_r2_mean,
        "val_r2_std": base_r2_std,
        "delta_r2": 0.0
    })

    # Ablation loop (masking)
    for i, feat in enumerate(feature_tags):
        print(f"🔍 Masking {feat} ({i+1}/{len(feature_tags)})")
        ab_r2_mean, ab_r2_std = train_eval_with_mask(X_train, y_train, X_val, y_val, i, feat)
        delta = ab_r2_mean - base_r2_mean
        results.append({
            "feature_removed": feat,
            "val_r2_mean": ab_r2_mean,
            "val_r2_std": ab_r2_std,
            "delta_r2": delta
        })

    df_result = pd.DataFrame(results)
    if "val_r2" not in df_result.columns and "val_r2_mean" in df_result.columns:
        df_result["val_r2"] = df_result["val_r2_mean"]
    return df_result


# ==============================
# Paired multi-seed ablation (stable)
# ==============================
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch

@dataclass
class AblationResult:
    df: pd.DataFrame
    baseline_by_seed: List[Dict[str, float]]
    masked_by_seed: Dict[str, List[Dict[str, float]]]

def _to_loader_std(X, y, model_type: str, batch_size: int, shuffle: bool):
    from ml_utils import to_loader
    return to_loader(X, y, model_type=model_type, batch_size=batch_size, shuffle=shuffle)

def _train_eval_once(
    X_tr, y_tr, X_val, y_val,
    pid_tr, pid_val,
    model_type: str,
    params: Dict[str, Any],
    device,
    seed: int,
    num_epochs: int,
    patience: int,
    min_delta: float,
    criterion,
):
    """
    1회 학습 후 VAL 성능 리턴: (r2, rmse, mae, epochs_run)
    """
    from ml_utils import train_model, evaluate
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 내부 split 없이 external val로만 early stopping
    model, train_losses, val_losses, val_r2, val_rmse, val_mae, *__rest = train_model(
        X_tr, y_tr, params, model_type=model_type,
        num_epochs=num_epochs, seed=seed,
        return_curve=True, pid_array=pid_tr,
        use_internal_split=False, external_val_data=(X_val, y_val),
        patience=patience, min_delta=min_delta,
        delta_is_relative=False,
        optimizer_name="adamw", weight_decay=1e-4,
        scheduler_name="plateau",
        scheduler_patience=2, scheduler_factor=0.5,
        grad_clip_norm=0.5, amp=False, deterministic=True,###############################^^###############
        criterion=criterion
    )
    epochs_run = len(val_losses) if isinstance(val_losses, list) else np.nan
    # 평가 (val)
    val_loader = _to_loader_std(X_val, y_val, model_type=model_type, batch_size=params.get("batch_size", 32), shuffle=False)
    r2, rmse, mae, _, _, _ = evaluate(model, val_loader, device, model_type=model_type)
    return float(r2), float(rmse), float(mae), int(epochs_run) if epochs_run==epochs_run else -1  # -1 if nan

def run_ablation_paired(
    X_train: np.ndarray, y_train: np.ndarray, pid_train: np.ndarray,
    X_val: np.ndarray,   y_val:   np.ndarray,   pid_val:   np.ndarray,
    feature_tag_list: List[str],
    *,
    model_type: str,
    fixed_params: Dict[str, Any],
    device=None,
    # stability knobs
    seeds: List[int] = (42, 43, 44),
    num_epochs: int = 8,
    patience: int = 3,
    min_delta: float = 5e-4,
    criterion=None,
    # I/O
    save_path: str = None,
    # sub-sample (optional; default OFF for stability)
    subsample_ratio: float = 1.0,
):
    """
    Paired multi-seed ablation:
    - baseline를 seeds로 여러 번 학습 → seed별 baseline 기록
    - 각 feature를 제거한 masked 실험도 동일 seeds로 학습/평가
    - seed-paired delta_r2 = r2(masked) - r2(baseline) → mean/std 집계
    - 모든 입력은 '이미 전처리/라그/정규화 적용 완료된' 배열을 가정(transform-only 일관성 유지)
    """
    assert X_train.ndim == 3 and X_val.ndim == 3
    C = X_train.shape[-1]
    assert C == len(feature_tag_list), f"C={C} vs len(feature_tag_list)={len(feature_tag_list)}"

    # (1) optional subsample (train only) — PID 균형을 해치지 않도록 단순 무작위만 사용 (원하면 교체)
    if 0 < subsample_ratio < 1.0:
        n = len(X_train)
        k = max(32, int(n * subsample_ratio))
        idx = np.random.RandomState(123).choice(n, size=k, replace=False)
        X_tr0, y_tr0, pid_tr0 = X_train[idx], y_train[idx], pid_train[idx]
    else:
        X_tr0, y_tr0, pid_tr0 = X_train, y_train, pid_train

    # (2) baseline across seeds
    batch_size = fixed_params.get("batch_size", 32)
    base_params = dict(fixed_params)
    # CNN/GRU 입력 차원 주입은 외부에서 해왔을 것이므로, 고대로 사용
    # ✅ baseline과 모든 실험이 동일한 입력 차원을 바라보도록 1회 주입
    if model_type.upper() == "CNN":
        base_params["input_channels"] = C
    else:
        base_params["input_size"] = C
        
        
    baseline_recs = []
    for s in seeds:
        r2_b, rmse_b, mae_b, ep_b = _train_eval_once(
            X_tr0, y_tr0, X_val, y_val,
            pid_tr0, pid_val,
            model_type, base_params, device, s,
            num_epochs, patience, min_delta, criterion
        )
        baseline_recs.append({"seed": s, "val_r2": r2_b, "val_rmse": rmse_b, "val_mae": mae_b, "epochs": ep_b})

    # seed -> baseline r2 map
    bmap = {rec["seed"]: rec["val_r2"] for rec in baseline_recs}

    rows = []
    masked_by_seed: Dict[str, List[Dict[str, float]]] = {}

    # (3) loop over features: drop one (mask) and train with SAME seeds
    for ci, fname in enumerate(feature_tag_list):
        # channel drop
        keep_idx = [j for j in range(C) if j != ci]
        X_tr = X_tr0[:, :, keep_idx]
        X_vl = X_val[:,   :, keep_idx]

        # ❗️여기서 파라미터에 현재 채널 수 주입 (가장 중요)
        params_m = dict(base_params)
        cur_C = X_tr.shape[-1]
        if model_type.upper() == "CNN":
            params_m["input_channels"] = cur_C
        else:
            params_m["input_size"] = cur_C

        # (선택) 안전 체크: 입력-파라미터 일치
        # assert params_m.get("input_size", cur_C) == cur_C, f"Mismatch: {params_m.get('input_size')} vs {cur_C}"

        seed_records = []
        r2_list = []
        for s in seeds:
            r2_m, rmse_m, mae_m, ep_m = _train_eval_once(
                X_tr,
                (y_train if subsample_ratio >= 1.0 else y_tr0),  # y는 drop 없음
                X_vl, y_val,
                pid_tr0, pid_val,
                model_type, params_m, device, s,      # ← 여기서 params_m 사용
                num_epochs, patience, min_delta, criterion
            )
            r2_base = bmap[s]
            delta_r2 = r2_m - r2_base
            seed_records.append({"seed": s, "val_r2": r2_m, "delta_r_2": delta_r2, "epochs": ep_m})
            r2_list.append(r2_m)


        masked_by_seed[fname] = seed_records
        r2_arr   = np.array([r["val_r2"]   for r in seed_records], dtype=float)
        d_arr    = np.array([r["delta_r_2"] for r in seed_records], dtype=float)
        row = {
            "feature_removed": fname,
            "val_r2_mean": float(np.mean(r2_arr)),
            "val_r2_std":  float(np.std(r2_arr, ddof=1)) if len(r2_arr) > 1 else 0.0,
            "delta_r2_mean": float(np.mean(d_arr)),
            "delta_r2_std":  float(np.std(d_arr, ddof=1)) if len(d_arr) > 1 else 0.0,
            "val_r2": float(np.mean(r2_arr)),  # for continuity
        }
        rows.append(row)

    # (4) build DataFrame (+ baseline first row)
    df = pd.DataFrame(rows, columns=["feature_removed","val_r2_mean","val_r2_std","delta_r2_mean","delta_r2_std","val_r2"])
    base_row = {
        "feature_removed": "None (baseline)",
        "val_r2_mean": float(np.mean([r["val_r2"] for r in baseline_recs])),
        "val_r2_std":  float(np.std([r["val_r2"] for r in baseline_recs], ddof=1)) if len(baseline_recs)>1 else 0.0,
        "delta_r2_mean": 0.0,
        "delta_r2_std":  0.0,
        "val_r2": float(np.mean([r["val_r2"] for r in baseline_recs])),
    }
    df = pd.concat([pd.DataFrame([base_row]), df], ignore_index=True)

    # (5) save
    if save_path:
        df.to_csv(save_path, index=False, encoding="utf-8")

    return AblationResult(df=df, baseline_by_seed=baseline_recs, masked_by_seed=masked_by_seed)


# ==============================
# Selection helper (delta_r2 mean-std rule + top-k fallback)
# ==============================
def select_features_by_delta_r2_stable(
    df, feature_tag_list,
    delta_col_mean="delta_r2_mean",
    delta_col_std="delta_r2_std",
    *,
    thr_mean: float = 1e-3,     # 제거 시 R2가 이만큼이라도 떨어져야 "유익"으로 간주
    k_std: float = 1.0,         # 불확실성 패널티 가중
    min_top_k: int = 30,        # ✅ 최소 유지 개수 (예: 25~40 권장)
    max_top_k: int | None = None
):
    """
    delta_r2 = (masked_r2 - baseline_r2)
      · delta_r2 < 0  → 그 특성을 '빼면' 성능이 떨어짐 → "유익한" 특성
      · delta_r2 > 0  → 그 특성을 '빼면' 성능이 좋아짐 → '유해' 특성(제거 후보)

    선택 규칙:
      1) 1차: delta_mean < -thr_mean 인 것들 중에서 보수적 점수(-mean - k*std)가 큰 순으로 정렬
      2) 2차: 1차 후보가 너무 적으면 최소 top-k를 fallback으로 강제
    """
    import numpy as np
    import pandas as pd

    need_cols = [delta_col_mean, delta_col_std, "feature_removed"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not in ablation DF columns {list(df.columns)}")

    # baseline 행 (feature_removed == "None (baseline)") 제거
    d = df.loc[df["feature_removed"] != "None (baseline)"].copy()

    # 보수적 점수: 제거 시 손해(-delta)와 표준편차 패널티를 함께 고려
    d["score"] = -d[delta_col_mean] - k_std * d[delta_col_std].fillna(0.0)

    # 1차 기준: 평균 기준으로 확실히 '유익'하다고 말할 수 있는 것들
    mask_confident = (d[delta_col_mean] < -abs(thr_mean))
    d_conf = d.loc[mask_confident].copy()

    # 정렬(점수 내림차순)
    d_sorted = d_conf.sort_values("score", ascending=False)

    # 선택 개수 계산
    if len(d_sorted) >= min_top_k:
        keep_names = d_sorted["feature_removed"].tolist()
    else:
        # ✅ Fallback: 전체에서 점수 기준으로 최소 top-k 확보
        d_all_sorted = d.sort_values("score", ascending=False)
        top_k = min_top_k if max_top_k is None else min(min_top_k, max_top_k)
        keep_names = d_all_sorted["feature_removed"].head(top_k).tolist()

    # 최종 인덱스 변환
    name2idx = {name: i for i, name in enumerate(feature_tag_list)}
    keep_idx = []
    for nm in keep_names:
        if nm in name2idx:
            keep_idx.append(name2idx[nm])

    # 안전장치: 최소 1개 이상
    if len(keep_idx) == 0:
        keep_idx = list(range(min(len(feature_tag_list), min_top_k)))

    # drop도 리턴(디버깅/로그용)
    keep_set = set(keep_idx)
    drop_idx = [i for i in range(len(feature_tag_list)) if i not in keep_set]

    return keep_idx, drop_idx


# ml_utils_time.py
# -*- coding: utf-8 -*-
"""
Timeseries 학습 '직전 단계' 유틸 모듈
- 입력: 이미 저장된 npy들 (X:[N,C,T], y:[N], pid:[N], scene:[N], windex:[N])
- 기능: 고분산 필터 → 타깃 중심화 → 라그 적용(인덱스 재매칭) → 스플릿+갭 → 베이스라인
- 출력: 기존 파이프라인이 기대하는 모양 그대로 반환 가능
"""

from typing import Dict, Tuple, Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score


# ---------- 0) 작은 헬퍼 ----------
def apply_mask_arrays(X, y, pid, scene, windex, mask):
    """Boolean mask를 X/y/pid/scene/windex에 일괄 적용."""
    return X[mask], y[mask], pid[mask], scene[mask], windex[mask]


# ---------- 1) 타깃 중심화 / 복원 ----------
def center_target(y: np.ndarray, pid: np.ndarray, scene: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    (pid×scene) 단위로 y 평균을 빼서 중심화. 학습 안정화 및 개인/씬 오프셋 제거.
    Returns
    - y_centered: 중심화된 y
    - y_means: {(pid, scene) -> float 평균}
    """
    yc = y.astype(np.float32).copy()
    y_means: Dict[Tuple[str, str], float] = {}
    meta = pd.DataFrame({"pid": pid, "scene": scene})
    for (p, s), idx in meta.groupby(["pid", "scene"]).groups.items():
        mu = float(y[list(idx)].mean())
        yc[list(idx)] = y[list(idx)] - mu
        y_means[(p, s)] = mu
    return yc, y_means


def restore_target(y_centered: np.ndarray, pid: np.ndarray, scene: np.ndarray, y_means: Dict) -> np.ndarray:
    """
    중심화된 y를 원 스케일로 복원. 리포트용(R²/RMSE/MAE) 계산 때 사용.
    """
    out = y_centered.astype(np.float32).copy()
    for i, (p, s) in enumerate(zip(pid, scene)):
        out[i] = out[i] + y_means[(p, s)]
    return out


# ---------- 2) 고분산 윈도 필터 ----------
def high_variance_mask(y: np.ndarray, pid: np.ndarray, scene: np.ndarray, quantile: float = 0.5) -> np.ndarray:
    """
    같은 (pid×scene) 그룹 내에서 |y - 그룹평균| 상위 quantile 윈도만 선택하는 마스크.
    SNR↑ 목적. 예: quantile=0.5 → 상위 50%만 사용.
    """
    keep = np.zeros(len(y), dtype=bool)
    df = pd.DataFrame({"y": y, "pid": pid, "scene": scene})
    for (_, _), sub in df.groupby(["pid", "scene"]):
        dev = np.abs(sub["y"] - sub["y"].mean())
        thr = dev.quantile(quantile)
        keep[sub.index[dev >= thr]] = True
    return keep



# ---------- 4) 라그 스윕(CV로 최적 라그 선택) : frame/seconds 겸용 + 피처 강화 ----------
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from typing import Dict, Tuple, Optional, Any, Callable

# ------------------------------------------------------------
# 0) 유틸: 입력 정규화 / 피처 구성 / 간단 라그 적용
# ------------------------------------------------------------
def _ensure_NTC(X: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    X를 (N,T,C)로 통일. True=원래가 NTC였음, False=원래가 NCT였음
    """
    if X.ndim != 3:
        raise ValueError("X must be 3D")
    if X.shape[1] >= X.shape[2]:  # (N,T,C) 추정
        return X, True
    else:  # (N,C,T) -> (N,T,C)
        return np.transpose(X, (0, 2, 1)), False

def _maybe_back_to_NCT(X_ntc: np.ndarray, was_ntc: bool) -> np.ndarray:
    return X_ntc if was_ntc else np.transpose(X_ntc, (0, 2, 1))

def _build_simple_features(X_: np.ndarray) -> np.ndarray:
    """
    (N,T,C) or (N,C,T) 입력 지원. 채널별 통계 3C개.
    """
    X_ntc, was_ntc = _ensure_NTC(X_)
    mean_t = X_ntc.mean(axis=1)                # (N,C)
    std_t  = X_ntc.std(axis=1)                 # (N,C)
    slope  = np.diff(X_ntc, axis=1).mean(axis=1)  # (N,C)
    feats  = np.concatenate([mean_t, std_t, slope], axis=1)
    return feats.astype(np.float32)

def _apply_lag_timeseries_simple(
    X, y, pid, scene, windex,
    *, stride_seconds: int, lag_seconds: float
):
    """
    윈도우 단위 '정렬' 버전: y를 lag_seconds만큼 시프트하여
    공통 구간만 유지. (양끝단 손실을 잘라냄)
    +lag: 라벨이 지연(=y가 뒤), -lag: 생리가 지연(=y가 앞)
    """
    X_ntc, was_ntc = _ensure_NTC(np.asarray(X))
    y = np.asarray(y).reshape(-1)
    pid = np.asarray(pid)
    scene = np.asarray(scene)
    windex = np.asarray(windex)

    if len(windex) < 2:
        raise ValueError("windex length must be >=2 to infer hop.")
    hop_frames = int(np.median(np.diff(windex)))  # 윈도우 간 프레임 hop
    sr_hz = hop_frames / float(stride_seconds)    # '초당 윈도우'에 해당하는 비율
    lag_frames = int(np.round(lag_seconds * sr_hz))

    # y를 이동: 양수 lag -> y 뒤로 이동(앞쪽 잘림), 음수 lag -> y 앞으로 이동(뒤쪽 잘림)
    if lag_frames > 0:
        y2 = y[lag_frames:]
        X2 = X_ntc[:len(y2)]
        pid2 = pid[:len(y2)]
        scene2 = scene[:len(y2)]
        w2 = windex[:len(y2)]
    elif lag_frames < 0:
        shift = abs(lag_frames)
        y2 = y[:-shift]
        X2 = X_ntc[shift:shift+len(y2)]
        pid2 = pid[shift:shift+len(y2)]
        scene2 = scene[shift:shift+len(y2)]
        w2 = windex[shift:shift+len(y2)]
    else:
        X2, y2, pid2, scene2, w2 = X_ntc, y, pid, scene, windex

    X2 = _maybe_back_to_NCT(X2, was_ntc)
    return X2, y2, pid2, scene2, w2

# ------------------------------------------------------------
# 1) CV 스코어러 (GroupKFold, ElasticNet 파이프라인)
# ------------------------------------------------------------
def _cv_score_enet(
    feats: np.ndarray, y: np.ndarray, groups: np.ndarray,
    cv_folds: Optional[int] = None, random_state: int = 42
) -> Tuple[float, float]:
    g = np.asarray(groups)
    n_groups = len(np.unique(g))
    if cv_folds is None:
        n_splits = max(2, min(5, n_groups))
    else:
        n_splits = max(2, min(cv_folds, n_groups))
    if n_splits < 2:
        return np.nan, np.nan

    gkf = GroupKFold(n_splits=n_splits)
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=6000, random_state=random_state)
    )
    scores = []
    for tr, va in gkf.split(feats, y, groups=g):
        pipe.fit(feats[tr], y[tr])
        p = pipe.predict(feats[va])
        scores.append(r2_score(y[va], p))
    return float(np.mean(scores)), float(np.std(scores))

# ------------------------------------------------------------
# 2) 외부 스플릿(없으면 생성)
# ------------------------------------------------------------
def make_outer_split_indices(
    pid: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    participant 단위 그룹 셔플 스플릿으로 train/val/test 인덱스 반환.
    """
    pid = np.asarray(pid)
    uniq = np.unique(pid)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    trainval_idx, test_idx = next(gss1.split(np.zeros_like(pid), groups=pid))
    # 이제 trainval에서 val 분리
    pid_tv = pid[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio / (1.0 - test_ratio), random_state=random_state)
    train_idx_rel, val_idx_rel = next(gss2.split(np.zeros_like(pid_tv), groups=pid_tv))
    return dict(
        train_idx=trainval_idx[train_idx_rel],
        val_idx=trainval_idx[val_idx_rel],
        test_idx=test_idx
    )

# ------------------------------------------------------------
# 3) 씬별 lag 스윕(훈련에서만) → best_lag[scene] 선택
# ------------------------------------------------------------
def select_scene_lags_on_train(
    X, y, pid, scene, windex,
    *, stride_seconds: int,
    lag_grid: Tuple[float, ...] = tuple(np.arange(-8, 9, 1)),
    groups: str = "pid",
    cv_folds: Optional[int] = None,
    random_state: int = 42,
    epsilon_flat: float = 1e-6,
    apply_fn: Optional[Callable[..., Tuple[np.ndarray, ...]]] = None,
    train_mask: Optional[np.ndarray] = None
) -> Tuple[Dict[Any, float], pd.DataFrame]:
    """
    train_mask가 지정되면 그 구간으로만 lag 스윕을 수행.
    반환: (scene_to_best_lag, summary_df)
    """
    if apply_fn is None:
        apply_fn = _apply_lag_timeseries_simple

    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    pid = np.asarray(pid); scene = np.asarray(scene); windex = np.asarray(windex)
    if train_mask is None:
        train_mask = np.ones_like(y, dtype=bool)

    scenes = np.unique(scene[train_mask])
    rows = []
    best = {}

    for sc in scenes:
        m = (scene == sc) & train_mask
        if m.sum() < 10:
            best[sc] = 0.0
            rows.append(dict(scene=sc, lag_s=np.nan, cv_r2_mean=np.nan, cv_r2_std=np.nan,
                             n_samples=int(m.sum()), flat_curve=True, chosen=False))
            continue

        curve = []
        for lag_s in lag_grid:
            X2, y2, pid2, scene2, w2 = apply_fn(
                X[m], y[m], pid[m], scene[m], windex[m],
                stride_seconds=stride_seconds, lag_seconds=lag_s
            )
            if len(y2) < 10:
                curve.append((lag_s, np.nan, np.nan, len(y2)))
                continue
            feats = _build_simple_features(X2)
            g_arr = pid2 if groups == "pid" else scene2
            mean_r2, std_r2 = _cv_score_enet(feats, y2, g_arr, cv_folds=cv_folds, random_state=random_state)
            curve.append((lag_s, mean_r2, std_r2, len(y2)))

        dfc = pd.DataFrame(curve, columns=["lag_s", "cv_r2_mean", "cv_r2_std", "n_samples"])
        span = dfc["cv_r2_mean"].max() - dfc["cv_r2_mean"].min()
        flat = bool(span < epsilon_flat)
        # best 선택(평평하면 0 고정)
        if flat:
            lag_star = 0.0
        else:
            lag_star = float(dfc.sort_values("cv_r2_mean", ascending=False).iloc[0]["lag_s"])

        best[sc] = lag_star
        dfc["scene"] = sc
        dfc["chosen"] = (dfc["lag_s"] == lag_star)
        dfc["flat_curve"] = flat
        rows.append(dfc)

    summary = pd.concat(rows, ignore_index=True) if len(rows) and isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
    return best, summary

# ------------------------------------------------------------
# 4) 씬별 lag를 고정 적용하여 전체 세트 정렬
# ------------------------------------------------------------
def apply_scene_lags_fixed(
    X, y, pid, scene, windex,
    *, stride_seconds: int,
    scene_to_lag: Dict[Any, float],
    apply_fn: Optional[Callable[..., Tuple[np.ndarray, ...]]] = None,
    drop_unknown_scene: bool = True
):
    """
    샘플을 씬별로 분할하여 해당 씬의 lag를 적용 → 다시 concat.
    """
    if apply_fn is None:
        apply_fn = _apply_lag_timeseries_simple

    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    pid = np.asarray(pid); scene = np.asarray(scene); windex = np.asarray(windex)

    X_parts, y_parts, pid_parts, scene_parts, w_parts = [], [], [], [], []
    for sc, lag_s in scene_to_lag.items():
        m = (scene == sc)
        if m.sum() == 0:
            continue
        X2, y2, pid2, scene2, w2 = apply_fn(
            X[m], y[m], pid[m], scene[m], windex[m],
            stride_seconds=stride_seconds, lag_seconds=lag_s
        )
        X_parts.append(X2); y_parts.append(y2)
        pid_parts.append(pid2); scene_parts.append(scene2); w_parts.append(w2)

    if not X_parts:
        if drop_unknown_scene:
            return None, None, None, None, None
        else:
            return X, y, pid, scene, windex

    X_cat = np.concatenate(X_parts, axis=0)
    y_cat = np.concatenate(y_parts, axis=0)
    pid_cat = np.concatenate(pid_parts, axis=0)
    scene_cat = np.concatenate(scene_parts, axis=0)
    w_cat = np.concatenate(w_parts, axis=0)

    # 원래 순서로 정렬하고 싶다면 windex 기준 정렬 가능(선택)
    order = np.argsort(w_cat)
    return X_cat[order], y_cat[order], pid_cat[order], scene_cat[order], w_cat[order]

# ------------------------------------------------------------
# 5) 풀 파이프라인: (a) outer split → (b) train에서 씬별 lag 선정
#                     → (c) 선정값 고정 적용 → (d) 정렬된 세트 반환
# ------------------------------------------------------------
def nested_scene_lag_pipeline(
    X, y, pid, scene, windex,
    *, stride_seconds: int,
    lag_grid: Tuple[float, ...] = tuple(np.arange(-8, 9, 1)),
    groups: str = "pid",
    cv_folds: Optional[int] = None,
    random_state: int = 42,
    outer_split: Optional[Dict[str, np.ndarray]] = None,
    epsilon_flat: float = 1e-6,
    apply_fn: Optional[Callable[..., Tuple[np.ndarray, ...]]] = None,
    gap_steps: int = 0
) -> Dict[str, Any]:
    """
    반환:
      {
        'scene_to_lag': {scene: lag_s, ...},
        'lag_curve_df': DataFrame(씬별 스윕 곡선),
        'X_train','y_train','pid_train','scene_train','w_train',
        'X_val', ...,
        'X_test', ...
      }
    """
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    pid = np.asarray(pid); scene = np.asarray(scene); windex = np.asarray(windex)

    # (a) outer split
    if outer_split is None:
        split = make_outer_split_indices(pid, val_ratio=0.1, test_ratio=0.1, random_state=random_state)
    else:
        split = outer_split
    train_idx = np.asarray(split["train_idx"])
    val_idx   = np.asarray(split["val_idx"])
    test_idx  = np.asarray(split["test_idx"])

    # (b) 훈련에서만 씬별 lag 스윕
    train_mask = np.zeros_like(y, dtype=bool)
    train_mask[train_idx] = True
    scene_to_lag, curve_df = select_scene_lags_on_train(
        X, y, pid, scene, windex,
        stride_seconds=stride_seconds,
        lag_grid=lag_grid,
        groups=groups,
        cv_folds=cv_folds,
        random_state=random_state,
        epsilon_flat=epsilon_flat,
        apply_fn=apply_fn,
        train_mask=train_mask
    )

    # (c) 선정값 고정 적용 (train/val/test 각각)
    def _pick(idx):
        Xs = X[idx]; ys = y[idx]; pids = pid[idx]; scs = scene[idx]; ws = windex[idx]
        out = apply_scene_lags_fixed(
            Xs, ys, pids, scs, ws,
            stride_seconds=stride_seconds,
            scene_to_lag=scene_to_lag,
            apply_fn=apply_fn
        )
        return out

    X_tr, y_tr, pid_tr, sc_tr, w_tr = _pick(train_idx)
    X_va, y_va, pid_va, sc_va, w_va = _pick(val_idx)
    X_te, y_te, pid_te, sc_te, w_te = _pick(test_idx)

    # (d) 경계/누수 완충: gap_steps가 양수면, 각 세트 양끝에서 일정 샘플 제거
    def _apply_gap(X_, y_, pid_, sc_, w_, gap_steps=0):
        if X_ is None:  # 씬이 모두 빠졌을 때
            return None, None, None, None, None
        if gap_steps <= 0 or len(y_) <= 2 * gap_steps:
            return X_, y_, pid_, sc_, w_
        return X_[gap_steps:-gap_steps], y_[gap_steps:-gap_steps], pid_[gap_steps:-gap_steps], sc_[gap_steps:-gap_steps], w_[gap_steps:-gap_steps]

    X_tr, y_tr, pid_tr, sc_tr, w_tr = _apply_gap(X_tr, y_tr, pid_tr, sc_tr, w_tr, gap_steps)
    X_va, y_va, pid_va, sc_va, w_va = _apply_gap(X_va, y_va, pid_va, sc_va, w_va, gap_steps)
    X_te, y_te, pid_te, sc_te, w_te = _apply_gap(X_te, y_te, pid_te, sc_te, w_te, gap_steps)

    return dict(
        scene_to_lag=scene_to_lag,
        lag_curve_df=curve_df,
        X_train=X_tr, y_train=y_tr, pid_train=pid_tr, scene_train=sc_tr, w_train=w_tr,
        X_val=X_va,   y_val=y_va,   pid_val=pid_va,   scene_val=sc_va,   w_val=w_va,
        X_test=X_te,  y_test=y_te,  pid_test=pid_te,  scene_test=sc_te,  w_test=w_te
    )

# ------------------------------------------------------------
# 6) 사용 예시 (주석만; 실행 X)
# ------------------------------------------------------------
# outer = make_outer_split_indices(pid_array, val_ratio=0.1, test_ratio=0.1, random_state=42)
# out = nested_scene_lag_pipeline(
#     X_array, y_array, pid_array, scene_array, windex_array,
#     stride_seconds=2,
#     lag_grid=tuple(range(-8, 9, 1)),
#     groups="pid",
#     cv_folds=5,
#     random_state=42,
#     outer_split=outer,
#     epsilon_flat=1e-6,
#     apply_fn=None,        # 여러분이 이미 가진 apply_lag_timeseries를 쓰려면: apply_fn=apply_lag_timeseries
#     gap_steps=2
# )
# print(out['scene_to_lag'])        # {'Elevator1': +7.0, 'Elevator2': +4.0, 'Hall': +3.0, 'Outside': -7.0, ...}
# model.fit(out['X_train'], out['y_train']); ...

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
def lag_sweep_cv_timeseries(
    X, y, pid, scene, windex,
    stride_seconds: int,
    lag_grid=tuple(range(-8, 9, 1)),      # seconds 또는 frames(아래 lag_unit으로 지정)
    groups: str = "pid",
    cv_folds=None,                         # ★ fold 개수
    random_state: int = 42,
    *,
    lag_unit: str = "seconds",             # {"seconds","frames"}
    sampling_rate_hz: float = None,        # lag_unit="frames"일 때 필요(없으면 windex/stride로 추정)
    epsilon_flat: float = 1e-6,            # 평탄 플래그 임계값
    r2_min_valid: float = -5.0,            # ★ R² sanity 범위
    r2_max_valid: float = 1.001
):
    """
    반환 DF 컬럼:
      - lag_s, lag_frames
      - cv_r2_mean, cv_r2_std (NaN-safe, sanity 필터 후)
      - n_samples, n_pid, n_groups (라그별 유효성 진단용)
      - flat_curve (span<epsilon_flat)
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import r2_score
    import warnings

    # ---------- 내부 유틸 ----------
    def _ensure_ntc(X_):
        X_ = np.asarray(X_)
        if X_.ndim != 3:
            raise ValueError("X must be 3D")
        if X_.shape[1] >= X_.shape[2]:  # (N,T,C)
            return X_, True
        return np.transpose(X_, (0, 2, 1)), False  # (N,C,T)->(N,T,C)

    def _build_features(X_):
        """채널별 mean/std/1차차분 평균 → (N, 3C)"""
        X_ntc, was_ntc = _ensure_ntc(X_)
        mean_t = X_ntc.mean(axis=1)                 # (N,C)
        std_t  = X_ntc.std(axis=1)                  # (N,C)
        slope  = np.diff(X_ntc, axis=1).mean(axis=1)  # (N,C)
        feats  = np.concatenate([mean_t, std_t, slope], axis=1).astype(np.float32)
        return feats

    def _sanitize_scores(vals):
        """비유한/비현실 범위의 R²는 NaN으로."""
        v = pd.to_numeric(pd.Series(vals), errors="coerce").astype(float)
        bad = (~np.isfinite(v)) | (v < r2_min_valid) | (v > r2_max_valid)
        v[bad] = np.nan
        return v.values

    def cv_score(feats_, y_, groups_arr, cv_folds=cv_folds, random_state=42):
        g = np.asarray(groups_arr)
        uniq = np.unique(g)
        n_groups = len(uniq)
        if cv_folds is None:
            n_splits = max(2, min(5, n_groups))
        else:
            n_splits = max(2, min(int(cv_folds), n_groups))
        if n_splits < 2:
            return np.nan, np.nan, n_groups

        gkf = GroupKFold(n_splits=n_splits)
        pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=6000, random_state=random_state)
        )

        scores = []
        for tr, va in gkf.split(feats_, y_, groups=g):
            # 폴드 유효성 검사
            if len(np.unique(g[tr])) < 1 or len(np.unique(g[va])) < 1:
                continue
            # y 분산 0 방지 (R² 정의 문제)
            if np.nanstd(y_[va]) < 1e-12:
                # 분산 0이면 예측이 상수여도 R² 불안정 → 스킵
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    pipe.fit(feats_[tr], y_[tr])
                    p = pipe.predict(feats_[va])
                    scores.append(r2_score(y_[va], p))
                except Exception:
                    continue

        if len(scores) == 0:
            return np.nan, np.nan, n_groups

        scores = _sanitize_scores(scores)
        if np.all(np.isnan(scores)):
            return np.nan, np.nan, n_groups
        return float(np.nanmean(scores)), float(np.nanstd(scores)), n_groups

    # ---------- 라그 격자/프레임 해상도 ----------
    if lag_unit == "frames":
        if sampling_rate_hz is None:
            if len(windex) >= 2:
                step_frames = int(np.median(np.diff(windex)))
                sampling_rate_hz = step_frames / float(stride_seconds)
            else:
                raise ValueError("sampling_rate_hz가 필요합니다 (windex로 추정 불가).")
        lag_seconds_grid = [int(l) / float(sampling_rate_hz) for l in lag_grid]
        lag_frames_grid  = list(lag_grid)
    else:
        # seconds 모드
        lag_seconds_grid = list(lag_grid)
        # 0초 포함 보장
        if 0.0 not in set(float(s) for s in lag_seconds_grid):
            lag_seconds_grid = sorted(list(set(lag_seconds_grid + [0.0])))
        # seconds→frames(보고용)
        if len(windex) >= 2:
            step_frames = int(np.median(np.diff(windex)))
            sr_hz = step_frames / float(stride_seconds)  # '초당 윈도우'
            lag_frames_grid = [int(np.round(s * sr_hz)) for s in lag_seconds_grid]
        else:
            lag_frames_grid = [None for _ in lag_seconds_grid]

    # ---------- 스윕 ----------
    rows = []
    for lag_s, lag_f in zip(lag_seconds_grid, lag_frames_grid):
        X2, y2, pid2, scene2, widx2 = apply_lag_timeseries(
            X, y, pid, scene, windex,
            stride_seconds=stride_seconds,
            lag_seconds=lag_s
        )
        n_samples = int(len(y2))
        if n_samples < 10:
            rows.append({
                "lag_s": lag_s, "lag_frames": lag_f,
                "cv_r2_mean": np.nan, "cv_r2_std": np.nan,
                "n_samples": n_samples,
                "n_pid": int(len(np.unique(pid2))) if n_samples>0 else 0,
                "n_groups": 0
            })
            continue

        feats2 = _build_features(X2)
        g_arr = pid2 if groups == "pid" else scene2
        m, s, n_groups = cv_score(feats2, y2, g_arr, cv_folds=cv_folds, random_state=random_state)

        rows.append({
            "lag_s": lag_s, "lag_frames": lag_f,
            "cv_r2_mean": m, "cv_r2_std": s,
            "n_samples": n_samples,
            "n_pid": int(len(np.unique(pid2))),
            "n_groups": int(n_groups)
        })

    df = pd.DataFrame(rows)

    # R² sanity 필터(최종 한 번 더)
    if "cv_r2_mean" in df.columns:
        df["cv_r2_mean"] = _sanitize_scores(df["cv_r2_mean"])
    if "cv_r2_std" in df.columns:
        df["cv_r2_std"] = pd.to_numeric(df["cv_r2_std"], errors="coerce")

    # 평탄성 플래그 (유효값 범위에서만)
    if len(df):
        valid_vals = df["cv_r2_mean"].dropna()
        span = (valid_vals.max() - valid_vals.min()) if len(valid_vals) else np.nan
        df["flat_curve"] = bool(np.isfinite(span) and (span < epsilon_flat))

    # 정렬(NA는 뒤로)
    df = df.sort_values("cv_r2_mean", ascending=False, na_position="last").reset_index(drop=True)
    return df

# def lag_sweep_cv_timeseries(
#     X, y, pid, scene, windex,
#     stride_seconds: int,
#     lag_grid=tuple(range(-8, 9, 1)),      # seconds 또는 frames(아래 lag_unit으로 지정)
#     groups: str = "pid",
#     cv_folds=None,               # ★ 추가: fold 개수
#     random_state: int = 42,
#     *,
#     lag_unit: str = "seconds",            # "seconds" | "frames"
#     sampling_rate_hz: float = None,       # lag_unit="frames"일 때 필요(없으면 windex/stride로 추정)
#     epsilon_flat: float = 1e-6            # 평탄 곡선 판정 임계값(최대-최소 < epsilon이면 flat)
# ):
#     """
#     반환 DF는 기존 컬럼에 더해:
#       - 'lag_frames': 프레임 단위 라그
#       - 'flat_curve': 해당 라그 스윕이 (max-min)<epsilon 인지 여부(한 번만 기록; 라그별로 동일)
#     """
#     # 샘플링레이트 추정 (필요 시)
#     if lag_unit == "frames":
#         if sampling_rate_hz is None:
#             if len(windex) >= 2:
#                 step_frames = int(np.median(np.diff(windex)))
#                 sampling_rate_hz = step_frames / float(stride_seconds)
#             else:
#                 raise ValueError("sampling_rate_hz가 필요합니다 (windex로 추정 불가).")
#         lag_seconds_grid = [int(l)/float(sampling_rate_hz) for l in lag_grid]
#         lag_frames_grid  = list(lag_grid)
#     else:
#         lag_seconds_grid = list(lag_grid)
#         # seconds→frames(보고용)
#         if len(windex) >= 2:
#             step_frames = int(np.median(np.diff(windex)))
#             sr_hz = step_frames / float(stride_seconds)
#             lag_frames_grid = [int(np.round(s * sr_hz)) for s in lag_seconds_grid]
#         else:
#             lag_frames_grid = [None for _ in lag_seconds_grid]

#     def _build_features(X_):
#         """
#         간단 평균 대신 채널별 통계로 피처 강화:
#         - mean_t: 시간축 평균 (N, C)
#         - std_t : 시간축 표준편차 (N, C)
#         - slope : 1차차분 평균(시간 변화율) (N, C)
#         합쳐서 (N, 3C)
#         """
#         if X_.ndim != 3:
#             raise ValueError("X must be 3D")
#         # (N, T, C) 또는 (N, C, T) 모두 지원
#         if X_.shape[1] >= X_.shape[2]:  # (N, T, C)
#             Taxis, Caxis = 1, 2
#         else:                           # (N, C, T)
#             Taxis, Caxis = 2, 1
#         mean_t = X_.mean(axis=Taxis)
#         std_t  = X_.std(axis=Taxis)
#         diff   = np.diff(X_, axis=Taxis)
#         slope  = diff.mean(axis=Taxis)
#         feats  = np.concatenate([mean_t, std_t, slope], axis=1)
#         return feats.astype(np.float32)

#     def cv_score(feats_, y_, groups_arr, cv_folds=cv_folds, random_state=42):
#         g = np.asarray(groups_arr)
#         n_groups = len(np.unique(g))
#         if cv_folds is None:
#             n_splits = max(2, min(5, n_groups))   # 기존 자동 규칙
#         else:
#             n_splits = max(2, min(cv_folds, n_groups))

#         if n_splits < 2:
#             return np.nan, np.nan

#         gkf = GroupKFold(n_splits=n_splits)

#         pipe = make_pipeline(
#             StandardScaler(with_mean=True, with_std=True),
#             ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=6000, random_state=random_state)
#         )

#         scores = []
#         for tr, va in gkf.split(feats_, y_, groups=g):
#             pipe.fit(feats_[tr], y_[tr])
#             p = pipe.predict(feats_[va])
#             scores.append(r2_score(y_[va], p))

#         return float(np.mean(scores)), float(np.std(scores))


#     rows = []
#     for lag_s, lag_f in zip(lag_seconds_grid, lag_frames_grid):
#         X2, y2, pid2, scene2, widx2 = apply_lag_timeseries(
#             X, y, pid, scene, windex,
#             stride_seconds=stride_seconds,
#             lag_seconds=lag_s
#         )
#         if len(y2) < 10:
#             rows.append({"lag_s": lag_s, "lag_frames": lag_f, "cv_r2_mean": np.nan, "cv_r2_std": np.nan, "n_samples": int(len(y2))})
#             continue
#         feats2 = _build_features(X2)
#         g_arr = pid2 if groups == "pid" else scene2
#         m, s = cv_score(feats2, y2, g_arr)
#         rows.append({"lag_s": lag_s, "lag_frames": lag_f, "cv_r2_mean": m, "cv_r2_std": s, "n_samples": int(len(y2))})

#     df = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False, na_position="last").reset_index(drop=True)
#     # 평탄성 플래그(보고/후처리용)
#     if len(df):
#         span = (df["cv_r2_mean"].max() - df["cv_r2_mean"].min())
#         df["flat_curve"] = bool(span < epsilon_flat)
#     return df

# ---------- 5) 분할 + 갭(누수 방지) ----------

def split_across_with_gap(
    pid: np.ndarray, scene: np.ndarray, windex: np.ndarray,
    val_ratio: float = 0.2, gap_steps: int = 2, seed: int = 42
):
    """
    Across-participant split + 테스트 주변 ±gap_steps를 train/val에서 제거.
    Returns: train_mask, val_mask, test_mask, split_info(dict)
    """
    rng = np.random.RandomState(seed)
    uniq = np.unique(pid)
    if len(uniq) == 0:
        raise ValueError("No participants found")

    test_pid = rng.choice(uniq)
    rest = [p for p in uniq if p != test_pid]
    n_val = max(1, int(len(rest) * val_ratio)) if len(rest) else 0
    val_pids = set(rng.choice(rest, size=n_val, replace=False)) if n_val > 0 else set()

    train_mask = np.array([(p not in val_pids) and (p != test_pid) for p in pid], dtype=bool)
    val_mask   = np.array([(p in  val_pids) and (p != test_pid) for p in pid], dtype=bool)
    test_mask  = (pid == test_pid)

    # 테스트 주변 gap 제거 (같은 pid 내 scene별로 독립 적용)
    df = pd.DataFrame({"pid": pid, "scene": scene, "windex": windex})
    for (_, sc), sub in df[df["pid"] == test_pid].groupby(["pid", "scene"]):
        sub = sub.sort_values("windex")
        idxs = sub.index.to_numpy()
        lo = np.maximum(np.arange(len(sub)) - gap_steps, 0)
        hi = np.minimum(np.arange(len(sub)) + gap_steps, len(sub) - 1)
        banned = set()
        for a, b in zip(lo, hi):
            banned.update(idxs[a:b + 1].tolist())
        banned = list(banned)
        train_mask[banned] = False
        val_mask[banned]   = False

    info = dict(test_pid=str(test_pid), val_pids=[str(p) for p in val_pids], gap_steps=int(gap_steps))
    return train_mask, val_mask, test_mask, info

# ---------- apply_lag_timeseries : 정수배면 윈도우 시프트, 아니면 프레임 시프트 ----------

import numpy as np

def apply_lag_timeseries(
    X, y, pid, scene, widx,
    stride_seconds: float,
    lag_seconds: float,
    *,
    drop_edge: bool = True,
    integer_multiple_tol: float = 0.05  # lag/stride가 정수배인지 판정 허용오차(비율)
):
    N = len(y)
    assert len(pid) == N and len(scene) == N and len(widx) == N
    if X.ndim != 3:
        raise ValueError("X must be 3D")

    # 레이아웃
    if X.shape[1] >= X.shape[2]:  # (N, T, C)
        is_rnn = True
        N_, T, C = X.shape
    else:                         # (N, C, T)
        is_rnn = False
        N_, C, T = X.shape
    assert N_ == N

    # 샘플링레이트 추정
    if N >= 2:
        step_frames = int(np.median(np.diff(widx)))  # ≈ stride_seconds * sr_hz
        step_frames = max(step_frames, 1)
    else:
        step_frames = 1
    sr_hz = step_frames / float(stride_seconds)

    # 정수배 판정
    k_real = lag_seconds / float(stride_seconds)
    k_round = int(np.round(k_real))
    is_integer_multiple = np.isclose(k_real, k_round, atol=integer_multiple_tol)

    if is_integer_multiple:
        # 윈도우 단위 시프트(기존 동작)
        shift = k_round
        if shift == 0:
            return X, y, pid, scene, widx
        if shift > 0:
            x_sl, y_sl = slice(0, N - shift), slice(shift, N)
        else:
            x_sl, y_sl = slice(-shift, N), slice(0, N + shift)
        X2 = X[x_sl]
        y2 = y[y_sl]
        pid2 = pid[x_sl]
        scene2 = scene[x_sl]
        widx2 = widx[x_sl]
        m = min(len(X2), len(y2))
        X2 = X2[:m]; y2 = y2[:m]; pid2 = pid2[:m]; scene2 = scene2[:m]; widx2 = widx2[:m]
        return X2, y2, pid2, scene2, widx2

    # 프레임 단위 시프트(창 내부 시간축 이동)
    frame_shift = int(np.round(lag_seconds * sr_hz))
    if frame_shift == 0:
        return X, y, pid, scene, widx

    if frame_shift > 0:
        # X를 앞으로 당김: 앞쪽 frame_shift 프레임 버림
        if is_rnn:  X_cut = X[:, frame_shift:, :]
        else:       X_cut = X[:, :, frame_shift:]
    else:
        fs = -frame_shift
        if is_rnn:  X_cut = X[:, :T - fs, :]
        else:       X_cut = X[:, :, :T - fs]

    # 패딩 없이 트리밍 (drop_edge=True)
    X2 = X_cut
    y2, pid2, scene2, widx2 = y.copy(), pid.copy(), scene.copy(), widx.copy()

    if not drop_edge:
        # 필요 시 0패딩으로 T 복원 (대개 권장하지 않음)
        pad_len = T - (X2.shape[1] if is_rnn else X2.shape[2])
        if pad_len > 0:
            if is_rnn:
                pad = np.zeros((N, pad_len, C), dtype=X.dtype)
                if frame_shift > 0: X2 = np.concatenate([X2, pad], axis=1)
                else:               X2 = np.concatenate([pad, X2], axis=1)
            else:
                pad = np.zeros((N, C, pad_len), dtype=X.dtype)
                if frame_shift > 0: X2 = np.concatenate([X2, pad], axis=2)
                else:               X2 = np.concatenate([pad, X2], axis=2)

    return X2, y2, pid2, scene2, widx2


# ---------- split_within_finetune_gap : test 주변 ±gap만 제외하도록 수정 ----------

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

def split_within_finetune_gap(
    pid: np.ndarray, scene: np.ndarray, windex: np.ndarray,
    target_pid: Optional[str] = None, finetune_ratio: float = 0.7, gap_steps: int = 2, seed: int = 42
):
    """
    대상 pid의 각 scene에서:
      - windex 오름차순으로 앞쪽 finetune, 뒤쪽 test
      - test 인접 구간 ±gap_steps 만 finetune에서 제외 (기존처럼 전구간 ban하지 않음)
    """
    rng = np.random.RandomState(seed)
    up = np.unique(pid)
    if target_pid is None:
        target_pid = rng.choice(up)

    pretrain_mask = (pid != target_pid)
    finetune_mask = np.zeros_like(pretrain_mask, dtype=bool)
    test_mask     = np.zeros_like(pretrain_mask, dtype=bool)

    df = pd.DataFrame({"pid": pid, "scene": scene, "windex": windex})
    for (_, sc), sub in df[df["pid"] == target_pid].groupby(["pid", "scene"]):
        sub = sub.sort_values("windex")
        n = len(sub)
        if n == 0:
            continue
        cut = max(1, int(np.floor(n * finetune_ratio)))
        fin_idx = sub.index[:cut].to_numpy()
        tst_idx = sub.index[cut:].to_numpy()

        finetune_mask[fin_idx] = True
        test_mask[tst_idx]     = True

        # 테스트 인덱스 주변만 ban
        if len(tst_idx) > 0 and gap_steps > 0:
            # sub 내 위치 인덱스 맵
            pos = {idx: i for i, idx in enumerate(sub.index.to_numpy())}
            banned = set()
            for t in tst_idx:
                i = pos[t]
                a = max(0, i - gap_steps)
                b = min(n - 1, i + gap_steps)
                banned.update(sub.index[a:b + 1].tolist())
            # finetune에서만 제거 (pretrain_mask는 타 참가자이므로 영향 없음)
            finetune_mask[list(banned)] = False

    info = dict(target_pid=str(target_pid), finetune_ratio=float(finetune_ratio), gap_steps=int(gap_steps))
    return pretrain_mask, finetune_mask, test_mask, info

# ---------- 6) 간단 베이스라인 ----------
def baseline_participant_mean(y: np.ndarray, pid: np.ndarray):
    """참가자별 y 평균을 예측치로 사용."""
    df = pd.DataFrame({"y": y, "pid": pid})
    mu = df.groupby("pid")["y"].transform("mean").to_numpy()
    return mu.astype(np.float32)


def baseline_persistence(y: np.ndarray, pid: np.ndarray, windex: np.ndarray) -> np.ndarray:
    """
    같은 참가자 내에서 windex 순으로 이전 윈도의 y를 그대로 예측.
    첫 윈도는 다음 값으로 대체(간단 백필).
    """
    pred = np.zeros_like(y, dtype=np.float32)
    df = pd.DataFrame({"y": y, "pid": pid, "windex": windex})
    for p, sub in df.groupby("pid"):
        sub = sub.sort_values("windex")
        p_ = sub["y"].shift(1).fillna(method="bfill").to_numpy()
        pred[sub.index] = p_.astype(np.float32)
    return pred


# ---------- 7) 오케스트레이터(원클릭 준비 단계) ----------
def prepare_timeseries_for_training(
    X: np.ndarray, y: np.ndarray, pid: np.ndarray, scene: np.ndarray, windex: np.ndarray,
    *,
    use_high_variance: bool = True, high_var_q: float = 0.5,
    use_center_target: bool = True,
    use_lag: bool = True, stride_seconds: int = 5, lag_seconds: int = 0,
    split_mode: str = "across",  # 'across' | 'within'
    val_ratio: float = 0.2, gap_steps: int = 2, seed: int = 42,
    within_target_pid: Optional[str] = None, finetune_ratio: float = 0.7
):
    """
    고분산→중심화→라그→분할(+갭)까지 한번에 수행.
    Returns:
      dict(
        X_train, y_train, pid_train,
        X_val,   y_val,   pid_val,
        X_test,  y_test,  pid_test,
        y_val_raw, y_test_raw,  # 중심화 복원본(리포트용)
        meta={ 'y_means':..., 'split_info':... }
      )
    """
    # 1) 고분산 필터
    if use_high_variance:
        m = high_variance_mask(y, pid, scene, quantile=high_var_q)
        X, y, pid, scene, windex = apply_mask_arrays(X, y, pid, scene, windex, m)

    # 2) 타깃 중심화
    y_c, y_means = (y, None)
    if use_center_target:
        y_c, y_means = center_target(y, pid, scene)

    # 3) 라그
    if use_lag and lag_seconds != 0:
        X, y_c, pid, scene, windex = apply_lag_timeseries(
            X, y_c, pid, scene, windex, stride_seconds=stride_seconds, lag_seconds=lag_seconds
        )

    # 4) 분할 + 갭
    if split_mode == "across":
        tr_m, va_m, te_m, info = split_across_with_gap(pid, scene, windex, val_ratio=val_ratio, gap_steps=gap_steps, seed=seed)
        pretrain_mask = None
    elif split_mode == "within":
        pre_m, fin_m, te_m, info = split_within_finetune_gap(pid, scene, windex, target_pid=within_target_pid,
                                                             finetune_ratio=finetune_ratio, gap_steps=gap_steps, seed=seed)
        # 간단화: finetune 마스크를 train, pretrain은 원하면 따로 사용
        tr_m, va_m = fin_m, np.zeros_like(fin_m, dtype=bool)
        pretrain_mask = pre_m
    else:
        raise ValueError("split_mode must be 'across' or 'within'")

    # 5) 마스크 적용
    X_train, y_train, pid_train, _, _ = apply_mask_arrays(X, y_c, pid, scene, windex, tr_m)
    X_val,   y_val,   pid_val,   _, _ = apply_mask_arrays(X, y_c, pid, scene, windex, va_m)
    X_test,  y_test,  pid_test,  _, _ = apply_mask_arrays(X, y_c, pid, scene, windex, te_m)

    # 6) 리포트용 원 스케일 복원
    if y_means is not None:
        y_val_raw  = restore_target(y_val,  pid_val,  pid_val*0 + "", y_means)  # scene 필요로 인해 아래에서 다시 계산
        y_test_raw = restore_target(y_test, pid_test, pid_test*0 + "", y_means)
        # 위 한 줄은 scene 정보가 없으므로 올바르지 않음. 아래에서 scene을 포함하여 다시 계산.
    # 정확 복원을 위해 scene도 넘겨주자
    # (위 apply_mask_arrays에서 scene을 버렸으므로 y 복원 시 scene 필요하면 별도로 반환하는 게 안전)
    # 간단히 원 스케일 복원은 호출부에서 restore_target(y_*, pid_*, scene_*, y_means)로 처리 권장.
    y_val_raw = None
    y_test_raw = None

    meta = dict(y_means=y_means, split_info=info, pretrain_mask=pretrain_mask if split_mode == "within" else None)
    return dict(
        X_train=X_train, y_train=y_train, pid_train=pid_train,
        X_val=X_val,     y_val=y_val,     pid_val=pid_val,
        X_test=X_test,   y_test=y_test,   pid_test=pid_test,
        y_val_raw=y_val_raw, y_test_raw=y_test_raw,
        meta=meta
    )


###########PLOT#################

# build_paper_assets.py
# -*- coding: utf-8 -*-
"""
논문용 결과 패키지(표/텍스트) + 그림 세트 생성 스크립트
 - 메인 표(HV=none, seed=20), 민감도 표(y_train/x_variance)
 - 테스트 코호트(10명) 명시 텍스트 템플릿
 - 네거티브 컨트롤(Log 파일에서 자동 파싱)
 - 그림:
     (1) HV 모드별 Test R² (mean ± SE) 막대그래프
     (2) 네거티브 컨트롤 막대그래프 (옵션)
     (3) Ablation Top-K (기본 15개) 막대그래프 — HV별 1장씩
"""

import os, json, math, re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------- 설정 ---------------
IN_DIR = Path(".")         # 입력 파일 위치 (필요시 수정)
OUT_DIR = Path("./paper_assets")  # 출력 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = IN_DIR / "split_fixed_test.json"
SUMMARY_FILES = {
    "none": IN_DIR / "summary_none.json",
    "x_variance": IN_DIR / "summary_x_variance.json",
    "y_train": IN_DIR / "summary_y_train.json",
}
ABLATION_FILES = {
    "none": IN_DIR / "ablation_none.csv",
    "x_variance": IN_DIR / "ablation_x_variance.csv",
    "y_train": IN_DIR / "ablation_y_train.csv",
}
NEG_LOG_FILE = IN_DIR / "negative_controls.log"  # 옵션(없어도 됨)

TOPK_ABLATION = 15  # Ablation 상위 k개
FIG_DPI = 300

# --------------- 유틸 ---------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def se_from_std(std, n):
    if n is None or n <= 1:
        return None
    return std / math.sqrt(n)

def parse_negative_controls(log_path: Path):
    """
    로그 파일에서 네거티브 컨트롤 수치 추출.
    기대 패턴:
      [QC] Label-shift R² ≈ -0.0123
      [QC] Time-order destroyed R² ≈ 0.0345
    """
    if not log_path.exists():
        return None

    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    label_shift = None
    time_destroy = None

    m1 = re.search(r"Label-shift R²\s*≈\s*([\-+]?\d*\.?\d+)", txt)
    if m1:
        label_shift = float(m1.group(1))
    m2 = re.search(r"Time-order destroyed R²\s*≈\s*([\-+]?\d*\.?\d+)", txt)
    if m2:
        time_destroy = float(m2.group(1))

    return {"label_shift_r2": label_shift, "time_destroy_r2": time_destroy}

def to_md_table(df: pd.DataFrame) -> str:
    """DataFrame → Markdown table 문자열"""
    return df.to_markdown(index=False)

def to_latex_table(df: pd.DataFrame) -> str:
    """DataFrame → LaTeX tabular 문자열"""
    return df.to_latex(index=False, float_format="%.4f", escape=False)

# --------------- 1) 결과 패키지 (표/텍스트) ---------------
def build_results_package():
    # a) 테스트 코호트 목록
    split = load_json(SPLIT_FILE)
    test_pids = split.get("test_pids", [])
    train_pids = split.get("train_pids", [])
    val_pids = split.get("val_pids", [])
    seed_used = split.get("seed", None)

    # b) 3개 요약 불러오기
    rows = []
    for mode, fpath in SUMMARY_FILES.items():
        js = load_json(fpath)
        mean_r2 = js.get("seed_mean_r2", None)
        std_r2  = js.get("seed_std_r2", None)
        n_seeds = js.get("n_seeds", None)
        se_r2   = se_from_std(std_r2, n_seeds)
        feat_count = js.get("feat_count", None)
        best_params = js.get("best_params", {})
        num_filters = best_params.get("num_filters", None)
        kernel_size = best_params.get("kernel_size", None)
        dropout = best_params.get("dropout", None)
        lr = best_params.get("learning_rate", None)
        rows.append({
            "HV_MODE": mode,
            "Test R² (mean)": mean_r2,
            "Test R² (SE)": se_r2,
            "Seeds": n_seeds,
            "Features Used": feat_count,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "learning_rate": lr
        })

    df_summary = pd.DataFrame(rows).sort_values(by="HV_MODE")
    df_summary_path_csv = OUT_DIR / "summary_hv_modes.csv"
    df_summary.to_csv(df_summary_path_csv, index=False, encoding="utf-8-sig")

    # c) 메인 표(논문): HV=none만 추려서 깔끔 표
    df_main = df_summary[df_summary["HV_MODE"] == "none"].copy()
    df_main.rename(columns={
        "Test R² (mean)": "Test R² (mean)",
        "Test R² (SE)": "SE",
        "Features Used": "Feat",
        "num_filters": "Filters",
        "kernel_size": "Kernel",
        "learning_rate": "LR"
    }, inplace=True)
    df_main = df_main[["HV_MODE", "Test R² (mean)", "SE", "Seeds", "Feat", "Filters", "Kernel", "dropout", "LR"]]
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    (OUT_DIR / "texts").mkdir(exist_ok=True)

    # 저장: CSV/Markdown/LaTeX
    df_main.to_csv(OUT_DIR / "tables" / "main_table_hv_none.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "tables" / "main_table_hv_none.md").write_text(to_md_table(df_main), encoding="utf-8")
    (OUT_DIR / "tables" / "main_table_hv_none.tex").write_text(to_latex_table(df_main), encoding="utf-8")

    # d) 민감도 표(3모드)
    (OUT_DIR / "tables" / "sensitivity_hv_all.csv").write_text(df_summary.to_csv(index=False), encoding="utf-8")
    (OUT_DIR / "tables" / "sensitivity_hv_all.md").write_text(to_md_table(df_summary), encoding="utf-8")
    (OUT_DIR / "tables" / "sensitivity_hv_all.tex").write_text(to_latex_table(df_summary), encoding="utf-8")

    # e) 네거티브 컨트롤(있으면)
    neg = parse_negative_controls(NEG_LOG_FILE)
    if neg:
        df_neg = pd.DataFrame([
            {"Control": "Label-shift", "Test R²": neg.get("label_shift_r2")},
            {"Control": "Time-order destroyed", "Test R²": neg.get("time_destroy_r2")},
        ])
        df_neg.dropna().to_csv(OUT_DIR / "tables" / "negative_controls.csv", index=False, encoding="utf-8-sig")
        (OUT_DIR / "tables" / "negative_controls.md").write_text(to_md_table(df_neg.dropna()), encoding="utf-8")
        (OUT_DIR / "tables" / "negative_controls.tex").write_text(to_latex_table(df_neg.dropna()), encoding="utf-8")

    # f) 텍스트 템플릿(Methods/Results 문구 뼈대)
    methods_txt = f"""\
[Methods – Evaluation Setup (Template)]
• Participant-disjoint split (fixed cohort): test={len(test_pids)} PIDs (seed={seed_used}), val={len(val_pids)} PIDs, train={len(train_pids)} PIDs.
• Hyperparameter tuning and feature selection used only the validation set (external validation; no internal split).
• Final reporting used a seed ensemble of {int(df_summary['Seeds'].iloc[0])} with deterministic settings; TF32 disabled explicitly.
• Lag = OFF; window format = (N,T,C); GAP between windows = 10; target centering via train-only hierarchical means (pid, scene -> pid -> global).

[Test cohort PIDs]
{", ".join(map(str, test_pids))}
"""
    (OUT_DIR / "texts" / "methods_template.txt").write_text(methods_txt, encoding="utf-8")

    # 결과 텍스트(모드별 성능 요약)
    lines = ["[Results – HV sensitivity (seed mean ± SE)]"]
    for _, r in df_summary.iterrows():
        m = r["Test R² (mean)"]
        se = r["Test R² (SE)"]
        s = r["Seeds"]
        lines.append(f"- HV={r['HV_MODE']}: R² = {m:.4f} ± {se:.4f} (seeds={int(s)})")
    if neg:
        ls = neg.get("label_shift_r2")
        td = neg.get("time_destroy_r2")
        lines.append(f"- Negative controls: Label-shift R² ≈ {ls}, Time-order destroyed R² ≈ {td}")
    results_txt = "\n".join(lines)
    (OUT_DIR / "texts" / "results_summary.txt").write_text(results_txt, encoding="utf-8")

    print("[OK] Results package saved under:", OUT_DIR)

# --------------- 2) 그림 세트 ---------------
def plot_hv_bar(df_summary: pd.DataFrame):
    # 막대: mean, 오차막대: SE
    modes = df_summary["HV_MODE"].tolist()
    means = df_summary["Test R² (mean)"].tolist()
    ses   = df_summary["Test R² (SE)"].tolist()

    plt.figure()
    x = np.arange(len(modes))
    plt.bar(x, means, yerr=ses, capsize=4)
    plt.xticks(x, modes)
    plt.ylabel("Test R² (mean ± SE)")
    plt.title("HV Mode Sensitivity (Test)")
    plt.tight_layout()
    (OUT_DIR / "figs").mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "figs" / "hv_modes_test_r2.png", dpi=FIG_DPI)
    plt.close()

def plot_negative_controls(neg_dict):
    if not neg_dict:
        return
    labels = []
    values = []
    if neg_dict.get("label_shift_r2") is not None:
        labels.append("Label-shift")
        values.append(neg_dict["label_shift_r2"])
    if neg_dict.get("time_destroy_r2") is not None:
        labels.append("Time-order destroyed")
        values.append(neg_dict["time_destroy_r2"])
    if not labels:
        return

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Test R²")
    plt.title("Negative Controls")
    plt.tight_layout()
    (OUT_DIR / "figs").mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / "figs" / "negative_controls.png", dpi=FIG_DPI)
    plt.close()

def plot_ablation_topk(ablation_path: Path, mode_name: str, topk: int = 15):
    df = safe_read_csv(ablation_path)
    if df is None:
        return
    # baseline 탐색
    # 기대 컬럼: ["feature_removed", "val_r2"], baseline row: "None (baseline)"
    assert "feature_removed" in df.columns and "val_r2" in df.columns, "Ablation CSV에 필요한 컬럼이 없습니다."
    base_row = df[df["feature_removed"].str.contains("None", case=False, na=False)]
    assert len(base_row) == 1, "baseline 행을 찾지 못했습니다."
    r2_base = float(base_row["val_r2"].iloc[0])

    df2 = df.copy()
    df2 = df2[~df2["feature_removed"].str.contains("None", case=False, na=False)]
    df2["drop_in_r2"] = r2_base - df2["val_r2"]
    df2 = df2.sort_values("drop_in_r2", ascending=False).head(topk)

    plt.figure()
    y_labels = df2["feature_removed"].tolist()[::-1]
    x_vals   = df2["drop_in_r2"].tolist()[::-1]
    y = np.arange(len(y_labels))
    plt.barh(y, x_vals)
    plt.yticks(y, y_labels)
    plt.xlabel("Δ R² on VAL (baseline - removed)")
    plt.title(f"Ablation Top-{topk} (HV={mode_name})")
    plt.tight_layout()
    (OUT_DIR / "figs").mkdir(exist_ok=True)
    out = OUT_DIR / "figs" / f"ablation_top{topk}_{mode_name}.png"
    plt.savefig(out, dpi=FIG_DPI)
    plt.close()

def build_figures():
    # 요약 불러와서 막대+오차막대
    rows = []
    for mode, fpath in SUMMARY_FILES.items():
        js = load_json(fpath)
        rows.append({
            "HV_MODE": mode,
            "Test R² (mean)": js.get("seed_mean_r2", None),
            "Test R² (SE)": se_from_std(js.get("seed_std_r2", None), js.get("n_seeds", None)),
        })
    df_summary = pd.DataFrame(rows).sort_values(by="HV_MODE")
    plot_hv_bar(df_summary)

    # 네거티브 컨트롤
    neg = parse_negative_controls(NEG_LOG_FILE)
    plot_negative_controls(neg)

    # Ablation Top-K (모드별)
    for mode, path in ABLATION_FILES.items():
        if Path(path).exists():
            plot_ablation_topk(Path(path), mode, topk=TOPK_ABLATION)

    print("[OK] Figures saved under:", OUT_DIR / "figs")

# --------------- main ---------------
if __name__ == "__main__":
    build_results_package()
    build_figures()


# -*- coding: utf-8 -*-
# planA_utils.py
import os, json, datetime
import numpy as np
import torch

# ============ Determinism / TF32 ============
def set_reproducible():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "42")
    torch.set_float32_matmul_precision("highest")  # effectively disables TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def dbg(msg: str):
    print(f"[DBG] {msg}")

# ============ Shape / Split helpers ============
def to_NTC_strict(X: np.ndarray, feature_tags: list):
    assert isinstance(X, np.ndarray) and X.ndim == 3
    L = len(feature_tags)
    N, A, B = X.shape
    if B == L and A != L:     # (N,T,C)
        return X
    if A == L and B != L:     # (N,C,T)
        return np.transpose(X, (0, 2, 1))
    if A == L and B == L:
        raise ValueError(f"Ambiguous: both middle and last dims equal |features|={L}.")
    raise ValueError(f"Mismatch: X.shape={X.shape}, |features|={L}.")

def sample_or_load_fixed_test_pids(pid_array, manual_list=None, k=10, save_path=None, seed=42):
    uniq = np.unique(pid_array).tolist()
    if manual_list is not None:
        test_pids = [p for p in manual_list if p in uniq]
    else:
        rng = np.random.default_rng(seed)
        test_pids = rng.choice(uniq, size=min(k, len(uniq)), replace=False).tolist()
    meta = {
        "date": datetime.datetime.now().isoformat(),
        "seed": seed,
        "test_pids": test_pids,
        "note": "Fixed test cohort for Plan A"
    }
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return test_pids

def make_participant_disjoint_masks(pid, test_pid_list, val_ratio=0.20, seed=42):
    all_pids = np.unique(pid).tolist()
    test_set = set(test_pid_list)
    trainval_pids = [p for p in all_pids if p not in test_set]
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(trainval_pids) * val_ratio)))
    val_pids = set(rng.choice(trainval_pids, size=n_val, replace=False).tolist())
    train_pids = set([p for p in trainval_pids if p not in val_pids])

    pid = np.asarray(pid)
    test_mask  = np.isin(pid, list(test_set))
    val_mask   = np.isin(pid, list(val_pids))
    train_mask = np.isin(pid, list(train_pids))
    return train_mask, val_mask, test_mask, sorted(list(train_pids)), sorted(list(val_pids))

def assert_no_pid_overlap(pid_a, pid_b):
    dup = set(pid_a.tolist()).intersection(set(pid_b.tolist()))
    assert len(dup) == 0, f"PID overlap detected: {sorted(list(dup))[:10]}"

# ============ HV / Centering ============
def hv_mask_from_train_x(X_all, train_mask, q=0.30):
    X_all = np.asarray(X_all, dtype=np.float32)
    X_flat = X_all.reshape(X_all.shape[0], -1)
    std_all = X_flat.std(axis=1)
    thr = float(np.quantile(std_all[train_mask], q)) if np.any(train_mask) else 0.0
    keep_all = std_all >= thr
    return keep_all

def center_from_train_split(y_tr, pid_tr, scene_tr):
    y_tr = y_tr.astype(np.float32)
    ps_mean, pid_mean = {}, {}
    ps_keys = np.stack([pid_tr, scene_tr], axis=1)
    for (k_pid, k_sc) in np.unique(ps_keys, axis=0):
        m = (pid_tr == k_pid) & (scene_tr == k_sc)
        ps_mean[(k_pid, k_sc)] = float(y_tr[m].mean()) if np.any(m) else np.nan
    for k_pid in np.unique(pid_tr):
        m = (pid_tr == k_pid)
        pid_mean[k_pid] = float(y_tr[m].mean()) if np.any(m) else np.nan
    global_mean = float(y_tr.mean()) if len(y_tr) else 0.0

    def _mu(pid_i, sc_i):
        m = ps_mean.get((pid_i, sc_i))
        if m is None or np.isnan(m):
            m = pid_mean.get(pid_i, global_mean)
            if m is None or np.isnan(m):
                m = global_mean
        return m

    def center_fn(y, pid, scene):
        y = y.astype(np.float32)
        mu = np.array([_mu(p, s) for p, s in zip(pid, scene)], dtype=np.float32)
        return y - mu

    return center_fn, {"global_mean": global_mean}

# ============ Negative controls ============
def negative_controls_once(X_trainval, y_trainval, pid_trainval, X_test, y_test, best_params, model_type, device):
    from ml_pipeline import train_and_evaluate_seeds  # local import to avoid circularity

    def destroy_time_order(X, rng=None):
        rng = np.random.default_rng(2025) if rng is None else rng
        Nn, Tt, Cc = X.shape
        out = np.empty_like(X)
        for i in range(Nn):
            perm = rng.permutation(Tt)
            out[i] = X[i, perm, :]
        return out

    def shift_labels_per_pid(y_, pid_, k=5):
        y_out = y_.copy()
        for p in np.unique(pid_):
            idx = np.where(pid_ == p)[0]
            if len(idx) > 0:
                y_out[idx] = np.roll(y_[idx], k % len(idx))
        return y_out

    print("\n[QC] Label-shift negative control...")
    y_shift = shift_labels_per_pid(y_trainval, pid_trainval, k=max(1, X_trainval.shape[1]//4))
    _, _, scores_neg, _, _ = train_and_evaluate_seeds(
        X_trainval, y_shift, pid_trainval,
        X_test, y_test,
        model_type=model_type,
        best_params=best_params,
        device=device,
        num_seeds=2, num_epochs=5, patience=3, min_delta=1e-3
    )
    r2_neg = float(np.mean([s[0] for s in scores_neg]))
    print(f"[QC] Label-shift R² ≈ {r2_neg:.4f} (≈0 근처가 정상)")

    print("[QC] Time-order destroyed control...")
    X_trv_perm = destroy_time_order(X_trainval)
    X_te_perm  = destroy_time_order(X_test)
    _, _, scores_perm, _, _ = train_and_evaluate_seeds(
        X_trv_perm, y_trainval, pid_trainval,
        X_te_perm,  y_test,
        model_type=model_type,
        best_params=best_params,
        device=device,
        num_seeds=2, num_epochs=5, patience=3, min_delta=1e-3
    )
    r2_perm = float(np.mean([s[0] for s in scores_perm]))
    print(f"[QC] Time-order destroyed R² ≈ {r2_perm:.4f} (낮아야 정상)")
# ===========================
# X scene-wise z-score normalization (fit on TRAIN only → apply to VAL/TEST)
# Place this block RIGHT AFTER y centering and BEFORE ablation/grid/train
# ===========================
import numpy as np
from collections import defaultdict

def _safe_mean_std(arr):
    """Flatten over time axis, ignore NaN/Inf, return (mean, std) with std>=eps."""
    eps = 1e-6
    # arr shape: (n_win, T) or (n_samples*T,)
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0  # fallback to unit scale
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(s) or s < eps:
        s = 1.0  # unit scale fallback
    return m, s
def _fit_scene_stats(X_train, scene_train, eps=1e-6):
    """
    TRAIN에서 scene별 (채널별) 평균/표준편차를 구해 반환합니다.
    - scene 라벨 타입(str/int 등)에 의존하지 않도록 key를 그대로 사용합니다.
    - 표준편차가 0에 가까우면 분모 안정화를 위해 1.0으로 교체합니다.
    Returns:
        stats: dict[scene_key] -> (mu_scene[C], sigma_scene[C])
        global_stats: (mu_global[C], sigma_global[C])
    """
    import numpy as np

    scenes = np.asarray(scene_train)
    # 전역 통계 (time, batch 차원 평균/표준편차)
    mu_global = np.nanmean(X_train, axis=(0, 1))  # shape (C,)
    sg_global = np.nanstd(X_train,  axis=(0, 1))  # shape (C,)
    sg_global = np.where(sg_global < eps, 1.0, sg_global)

    stats = {}
    for s in np.unique(scenes):
        m = (scenes == s)
        if not np.any(m):
            continue
        Xs = X_train[m]  # shape (N_s, T, C)
        mu_s = np.nanmean(Xs, axis=(0, 1))  # (C,)
        sg_s = np.nanstd( Xs, axis=(0, 1))  # (C,)
        sg_s = np.where(sg_s < eps, 1.0, sg_s)
        # ✅ 문자열/정수 어떤 라벨이든 key 그대로 저장
        stats[s] = (mu_s, sg_s)

    return stats, (mu_global, sg_global)

def _transform_scenewise(X, scene_vec, stats, global_stats, eps=1e-6):
    """
    TRAIN에서 구한 scene별 z-score 통계를 이용해 (N,T,C) 배열을 정규화합니다.
    - scene 라벨이 stats에 없으면 전역 통계를 사용합니다.
    """
    import numpy as np

    scenes = np.asarray(scene_vec)
    X_out  = np.array(X, copy=True)
    mu_g, sg_g = global_stats
    sg_g = np.where(sg_g < eps, 1.0, sg_g)

    uniq = np.unique(scenes)
    for s in uniq:
        idx = (scenes == s)
        mu_s, sg_s = stats.get(s, (mu_g, sg_g))
        denom = np.where(sg_s < eps, 1.0, sg_s)
        X_out[idx] = (X_out[idx] - mu_s) / denom

    return X_out

def _quick_check(X_arr, scene_vec, tag="SPLIT", k=3):
    """
    scene별 평균/표준편차를 대략 확인하는 간이 진단 함수.
    """
    import numpy as np
    k=len(np.unique(scene_vec))
    scenes = np.asarray(scene_vec)
    uniq = np.unique(scenes)
    print(f"[QC:{tag}] scenes={list(uniq)}")
    for i, s in enumerate(uniq[:k]):
        m = (scenes == s)
        xs = X_arr[m]
        mu = np.nanmean(xs, axis=(0,1))
        sg = np.nanstd( xs, axis=(0,1))
        print(f"  - {s}: mu[0..2]={mu[:3]!r}, std[0..2]={sg[:3]!r}, N={xs.shape[0]}")

# ============ Core runner (1 HV mode) ============
def run_planA_one_mode(
    HV_MODE: str,
    # raw splits
    X_train_raw, y_train_raw, pid_train, scene_train,
    X_val_raw,   y_val_raw,   pid_val,   scene_val,
    X_test_raw,  y_test_raw,  pid_test,  scene_test,
    # config
    feature_tag_list, OUT_DIR, seed_master, device, model_type,
    RUN_ABLATION=True, ABLATION_EPOCHS=10, RUN_GRID=True, GRID_EPOCHS=20,
    NUM_SEEDS_FINAL=20, EPOCHS_FINAL=50,
    patience_ablation=10, min_delta_ablation=1e-6,
    patience_grid=7, min_delta_grid=1e-5,
    patience_train=7, min_delta_train=1e-3,
    HV_QUANTILE=0.25, selection_threshold=0.0005,
    # model param templates
    fixed_params_base=None, search_space=None,
    use_internal_split=True,                  # ✅ 외부 고정 val 쓰게
    external_val_data=None,           # ✅ 바깥에서 만든 고정 검증셋
    deterministic=True                        # ✅ (원하면 True + CUBLAS 환경변수)
):
    from ml_pipeline import (
        run_ablation, select_features_by_ablation, run_grid_search,
        train_and_evaluate_seeds, summarize_test_results
    )
    def _ensure_dir(p: str):
        p = os.path.abspath(p)
        os.makedirs(p, exist_ok=True)
        return p

    # 0) OUT_DIR 절대경로 + 즉시 보장
    OUT_DIR = _ensure_dir(OUT_DIR)
    dbg(f"[{HV_MODE}] OUT_DIR: {OUT_DIR}")

    # ----- HV keepers -----
    if HV_MODE == "none":
        keep_train = np.ones(len(y_train_raw), dtype=bool)
        keep_val   = np.ones(len(y_val_raw), dtype=bool)
        keep_test  = np.ones(len(y_test_raw), dtype=bool)
    elif HV_MODE == "x_variance":
        keep_all = hv_mask_from_train_x(
            np.concatenate([X_train_raw, X_val_raw, X_test_raw], axis=0),
            np.concatenate([np.ones(len(y_train_raw), bool),
                            np.zeros(len(y_val_raw) + len(y_test_raw), bool)], axis=0),
            q=HV_QUANTILE
        )
        keep_train = keep_all[:len(y_train_raw)]
        keep_val   = keep_all[len(y_train_raw):len(y_train_raw)+len(y_val_raw)]
        keep_test  = keep_all[len(y_train_raw)+len(y_val_raw):]
    elif HV_MODE == "y_train":
        center_fn_tmp, _ = center_from_train_split(y_train_raw, pid_train, scene_train)
        y_dev_tr = np.abs(center_fn_tmp(y_train_raw, pid_train, scene_train))
        thr = float(np.quantile(y_dev_tr, HV_QUANTILE)) if len(y_dev_tr) else 0.0
        keep_train = (y_dev_tr >= thr)
        keep_val   = (np.abs(center_fn_tmp(y_val_raw, pid_val, scene_val))   >= thr)
        keep_test  = (np.abs(center_fn_tmp(y_test_raw, pid_test, scene_test)) >= thr)
    else:
        raise ValueError("Unknown HV_MODE")

    # slice
    X_tr, y_tr, pid_tr, scene_tr = X_train_raw[keep_train], y_train_raw[keep_train], pid_train[keep_train], scene_train[keep_train]
    X_va, y_va, pid_va, scene_va = X_val_raw[keep_val],   y_val_raw[keep_val],   pid_val[keep_val],   scene_val[keep_val]
    X_te, y_te, pid_te, scene_te = X_test_raw[keep_test], y_test_raw[keep_test], pid_test[keep_test], scene_test[keep_test]
    dbg(f"[{HV_MODE}] kept → train:{len(y_tr)} | val:{len(y_va)} | test:{len(y_te)}")

    # ----- Center from TRAIN only -----
    center_fn, _stat = center_from_train_split(y_tr, pid_tr, scene_tr)
    y_tr_c = center_fn(y_tr, pid_tr, scene_tr)
    y_va_c = center_fn(y_va, pid_va, scene_va)
    y_te_c = center_fn(y_te, pid_te, scene_te)

    # ----- Ablation -----
    fixed_params = dict(fixed_params_base or {})
    fixed_params["input_size"] = X_tr.shape[-1]  # RNN 계열은 input_size=C

    if RUN_ABLATION:
        ablation_path = os.path.join(OUT_DIR, f"ablation_{HV_MODE}.csv")
        os.makedirs(os.path.dirname(ablation_path), exist_ok=True)
        df_ablation = run_ablation(
            X_tr, y_tr_c, pid_tr,
            X_va, y_va_c, pid_va,
            feature_tag_list,
            model_type=model_type,
            fixed_params=fixed_params,
            seed=seed_master,
            num_epochs=ABLATION_EPOCHS,
            save_path=ablation_path,
            patience=patience_ablation,
            min_delta=min_delta_ablation
        )
        keep_features, keep_indices = select_features_by_ablation(
            df_ablation, feature_tag_list, threshold=selection_threshold
        )
    else:
        keep_indices = np.arange(X_tr.shape[-1]).tolist()
        keep_features = feature_tag_list

    # slice channels
    X_tr = X_tr[:, :, keep_indices]
    X_va = X_va[:, :, keep_indices]
    X_te = X_te[:, :, keep_indices]
    feat_tags_sel = (np.array(feature_tag_list)[keep_indices]).tolist()
    new_C = X_tr.shape[-1]
    dbg(f"[{HV_MODE}] features selected: {new_C}")

    # ----- Grid Search (external val) -----
    if RUN_GRID:
        best_params, _ = run_grid_search(
            X_tr, y_tr_c, pid_tr,
            model_type=model_type,
            search_space=search_space or {},
            seed=seed_master,
            num_epochs=GRID_EPOCHS,
            patience=patience_grid,
            min_delta=min_delta_grid,
            use_internal_split=False,
            external_val_data=(X_va, y_va_c)
        )
        best_params = dict(best_params)
    else:
        best_params = dict(fixed_params_base or {})
    best_params["input_size"] = new_C

    dbg(f"[{HV_MODE}] best_params={best_params}")

    # ----- Final train on train+val → test -----
    X_trv = np.concatenate([X_tr, X_va], axis=0)
    y_trv = np.concatenate([y_tr_c, y_va_c], axis=0)
    pid_trv = np.concatenate([pid_tr, pid_va], axis=0)

    assert X_trv.shape[-1] == X_te.shape[-1] == best_params["input_size"]

    train_losses, val_losses, test_scores, _, _ = train_and_evaluate_seeds(
        X_trv, y_trv, pid_trv,
        X_te, y_te_c,
        model_type=model_type,
        best_params=best_params,
        device=device,
        num_seeds=NUM_SEEDS_FINAL,
        num_epochs=EPOCHS_FINAL,
        patience=patience_train,
        min_delta=min_delta_train,
        use_internal_split=use_internal_split,                  # ✅ 외부 고정 val 쓰게
        external_val_data=external_val_data,           # ✅ 바깥에서 만든 고정 검증셋
        deterministic=deterministic                        # ✅ (원하면 True + CUBLAS 환경변수)
    )
    summarize_test_results(test_scores)

    r2s = [s[0] for s in test_scores]
    out = {
        "HV_MODE": HV_MODE,
        "test_pids": sorted(np.unique(pid_te).tolist()),
        "feat_count": new_C,
        "best_params": best_params,
        "seed_mean_r2": float(np.mean(r2s)),
        "seed_std_r2": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
        "n_seeds": len(r2s)
    }
    with open(os.path.join(OUT_DIR, f"summary_{HV_MODE}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return {
        "best_params": best_params, "feat_tags": feat_tags_sel,
        "X_trv": X_trv, "y_trv": y_trv, "pid_trv": pid_trv,
        "X_te": X_te, "y_te_c": y_te_c
    }
#########################
##Binary##
########################

# binary_eval_utils.py

import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def make_binary_thresholds_from_train(y_train_cont, q_low=0.4, q_high=0.6):
    """
    연속 y_train(centering 이후 값)에 대해 40%, 60% quantile을 기준으로
    high/low cut을 정의.
    """
    ql, qh = np.quantile(y_train_cont, [q_low, q_high])
    return float(ql), float(qh)


def binarize_with_thresholds(y_cont, thr_low, thr_high):
    """
    연속값 y_cont를 threshold 기반으로 0/1로 binarize.
      - y <= thr_low → 0
      - y >= thr_high → 1
      - 그 사이 값은 -1 (중립/버리는 구간)
    """
    y = np.asarray(y_cont, dtype=np.float32)
    labels = np.full_like(y, fill_value=-1, dtype=np.int64)
    labels[y <= thr_low] = 0
    labels[y >= thr_high] = 1
    mask = labels >= 0
    return labels, mask


def evaluate_binary_from_continuous(y_true_cont, y_pred_cont, thr_low, thr_high):
    """
    연속형 y_true, y_pred를 받아서:
      1) 둘 다 동일 threshold로 binarize
      2) 공통으로 유효한(mask) 샘플만 골라
      3) ACC, F1, AUROC 계산

    반환: dict(metric_name -> value, n_samples 등)
    """
    y_true_cont = np.asarray(y_true_cont, dtype=np.float32)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)

    # 1) true / pred 각각 binarize
    y_true_bin, mask_true = binarize_with_thresholds(y_true_cont, thr_low, thr_high)
    y_pred_bin, mask_pred = binarize_with_thresholds(y_pred_cont, thr_low, thr_high)

    # 2) 공통 마스크
    mask = mask_true & mask_pred
    if mask.sum() == 0:
        raise ValueError("No samples left after masking for binary thresholds.")

    yt = y_true_bin[mask]
    yp = y_pred_bin[mask]

    # 3) metrics
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp)

    # AUROC는 score가 연속이면 되므로, pred의 연속값(y_pred_cont)을 그대로 사용
    scores_for_auc = y_pred_cont[mask]
    # y_true는 0/1 bin label 사용
    auc = roc_auc_score(yt, scores_for_auc)

    return {
        "n": int(mask.sum()),
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "thr_low": float(thr_low),
        "thr_high": float(thr_high),
    }


def evaluate_binary_for_all_seeds_from_npz(
    model_type,
    out_dir,
    y_train_cont_for_threshold,
    q_low=0.4,
    q_high=0.6,
    npz_pattern=None,
):
    """
    회귀 학습 후, 저장된 *_test_predictions_seed*.npz들을 이용해
    각 seed별 binary metric을 계산.

    - model_type: "GRU", "CNN" 등 (파일 패턴에 사용)
    - out_dir: npz가 모여 있는 디렉토리 (OUT_DIR)
    - y_train_cont_for_threshold: train y (continuous, centered) 전체
    - q_low, q_high: quantile cut (ex. 0.4, 0.6)
    - npz_pattern: 커스텀 패턴이 필요한 경우 지정, 없으면
       f"{model_type.lower()}_test_predictions_seed*.npz" 사용

    반환: list of dict (seed별 metric)
    """
    if npz_pattern is None:
        npz_pattern = f"{model_type.lower()}_test_predictions_seed*.npz"

    glob_pat = os.path.join(out_dir, npz_pattern)
    files = sorted(glob.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No prediction npz found with pattern: {glob_pat}")

    thr_low, thr_high = make_binary_thresholds_from_train(
        y_train_cont_for_threshold, q_low=q_low, q_high=q_high
    )

    results = []
    for path in files:
        basename = os.path.basename(path)
        # seed 번호 추출 시도
        seed = None
        import re
        m = re.search(r"seed(\d+)", basename)
        if m:
            seed = int(m.group(1))

        data = np.load(path)
        # NOTE: 여기 키 이름은 현재 npz 구조에 맞춰 조정 필요할 수 있음.
        # 일반적으로 'y_true', 'y_pred' 형태라고 가정.
        if "y_true" in data:
            y_true = data["y_true"]
        elif "y_test" in data:
            y_true = data["y_test"]
        else:
            raise KeyError(f"{basename}에 y_true/y_test 키가 없음. 실제 키 이름을 확인해 조정 필요.")

        if "y_pred" in data:
            y_pred = data["y_pred"]
        elif "y_hat" in data:
            y_pred = data["y_hat"]
        else:
            raise KeyError(f"{basename}에 y_pred/y_hat 키가 없음. 실제 키 이름을 확인해 조정 필요.")

        m_bin = evaluate_binary_from_continuous(y_true, y_pred, thr_low, thr_high)
        m_bin["seed"] = seed
        m_bin["file"] = basename
        results.append(m_bin)

    return results

def make_delta_labels(y, pid, scene, widx):
    """
    y: (N,) 원래 anxiety level (연속값)
    pid, scene, widx: 같은 길이의 배열
    return:
        y_delta: (N,) delta label
        valid_mask: (N,) bool, delta가 정의된 샘플(True)만 표시
    """
    y = np.asarray(y, dtype=np.float32)
    pid = np.asarray(pid)
    scene = np.asarray(scene)
    widx = np.asarray(widx)

    N = len(y)
    assert len(pid) == N and len(scene) == N and len(widx) == N

    # (pid, scene, widx) 순으로 정렬해서 이웃 window를 찾기 쉽게 만듦
    order = np.lexsort((widx, scene, pid))
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(N)

    pid_s   = pid[order]
    scene_s = scene[order]
    widx_s  = widx[order]
    y_s     = y[order]

    y_delta_s  = np.zeros_like(y_s, dtype=np.float32)
    valid_s    = np.zeros_like(y_s, dtype=bool)

    for i in range(1, N):
        same_pid   = (pid_s[i]   == pid_s[i-1])
        same_scene = (scene_s[i] == scene_s[i-1])
        # 같은 (pid,scene) 안에서 widx가 바로 이전(=k-1)인 경우에만 delta 정의
        if same_pid and same_scene and (widx_s[i] == widx_s[i-1] + 1):
            y_delta_s[i] = y_s[i] - y_s[i-1]
            valid_s[i]   = True
        # 그 외(첫 window, 혹은 widx가 이어지지 않는 경우)는 valid=False 그대로 둠

    # 원래 인덱스로 되돌리기
    y_delta = y_delta_s[inv_order]
    valid   = valid_s[inv_order]
    return y_delta, valid
