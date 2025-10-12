# normwear_utill.py
# --------------------------------------------------------------------------------------
# 재사용 가능한 유틸 함수 모음 (파일럿 → 전량 실행까지 호환)
# - 다운스트림 포맷 생성(리샘플/윈도우/정규화/라벨/품질필터)
# - leakage-free split 생성/검증
# - 배치 로딩([N,C,T]), 임베딩 추출(NormWear), 풀링(mean/concat)
# - 임베딩 캐시 저장/로드, 간이 회귀평가(베이스라인 포함)
#
# 사용 예시는 파일 하단의 __doc__ 문자열 참고.
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os
import sys
import json
import pickle
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Iterable, Any

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas가 필요합니다. `pip install pandas`") from e

# neurokit2는 선택(품질 필터 사용 시 권장)
try:
    import neurokit2 as nk  # type: ignore
except Exception:
    nk = None  # 품질 필터 비활성

# sklearn은 평가 시 사용(선택)
try:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
except Exception:
    # 필요한 순간에만 import 실패 안내
    Ridge = ElasticNet = None  # type: ignore


# =========================================
# 설정/스펙
# =========================================

@dataclass
class DownstreamSpec:
    data_dir: str                           # 원천 pkl 폴더 (예: D:\...\processed_individual_anonymized)
    output_dir: str                         # 다운스트림 저장 폴더
    scenes: List[str] = None                # 사용할 scene 목록 (None이면 전체)
    fs_raw: int = 120                       # 원 샘플링
    fs_target: int = 50                     # 목표 샘플링
    window_seconds: int = 20                # 윈도우 길이(초)
    stride_seconds: int = 2                 # 스트라이드(초)
    channel_order: List[str] = None         # 예: ["PPG_Clean","EDA_Clean","RSP_Clean"]
    waveform_norm: str = "raw"              # "raw" | "zscore10s"
    label_mode: str = "raw"                 # "raw" | "zscore10s"
    baseline_seconds: int = 10              # z-score 기준(초)
    enable_ppg_quality_filter: bool = False # 품질 필터 on/off
    ppg_quality_threshold: float = 0.5      # 품질 임계값
    max_windows_per_pid: Optional[int] = None  # 참가자당 최대 윈도우 수 제한(파일럿용)
    verbose: bool = True
    label_lag_sec: int = 0  #생리 지연 보정(초 단위, 음수/양수 허용)

    def __post_init__(self):
        if self.channel_order is None:
            self.channel_order = ["PPG_Clean", "EDA_Clean"]
        if self.scenes is None:
            self.scenes = ["Hallway"]

@dataclass
class SplitPlan:
    # 참가자 단위 분할 (유출 방지)
    train_pids: List[str]                   # 예: ["001","002",...]
    test_pids: List[str]                    # 예: ["101","102",...]
    # 필요 시 valid_pids 등 확장 가능


# =========================================
# 로깅/유틸
# =========================================

def _header(msg: str):
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)

def _check_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"경로/파일이 없습니다: {path}")

def set_seed(seed: int = 42):
    np.random.seed(seed)

def pid_str(i: int) -> str:
    return f"{i:03d}"

def pid_from_fname(fname: str):
    # "001_win000000.pkl" → "001"
    return fname.split("_")[0]


# =========================================
# 리샘플/윈도우/정규화/라벨
# =========================================

def _resample_1d(x: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    n_new = int(round(len(x) * float(fs_to) / float(fs_from)))
    if n_new <= 1:
        return x.astype(np.float32, copy=False)
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    t_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False, dtype=np.float64)
    return np.interp(t_new, t_old, x.astype(np.float32)).astype(np.float32)

def _resample_df(df: pd.DataFrame, cols: List[str], fs_from: int, fs_to: int) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"컬럼 없음: {c}")
        out[c] = _resample_1d(df[c].to_numpy(dtype=np.float32), fs_from, fs_to)
    n = min(len(v) for v in out.values())
    for c in out:
        out[c] = out[c][:n]
    return pd.DataFrame(out)

def _iter_windows(n: int, win: int, stride: int) -> Iterable[Tuple[int, int]]:
    for s in range(0, n - win + 1, stride):
        yield s, s + win

def _compute_baseline_stats(arr_dict: Dict[str, np.ndarray], fs: int, baseline_seconds: int) -> Dict[str, Tuple[float, float]]:
    n_base = int(fs * baseline_seconds)
    stats: Dict[str, Tuple[float, float]] = {}
    for c, x in arr_dict.items():
        x0 = x[:n_base] if len(x) >= n_base else x
        m = float(np.mean(x0)) if len(x0) else 0.0
        s = float(np.std(x0)) if len(x0) else 1.0
        stats[c] = (m, s if s > 1e-6 else 1.0)
    return stats

def _apply_zscore(arr_dict: Dict[str, np.ndarray], stats: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c, x in arr_dict.items():
        m, s = stats.get(c, (0.0, 1.0))
        out[c] = (x - m) / s
    return out

def _compute_label(anx: np.ndarray, fs: int, mode: str, baseline_seconds: int) -> float:
    if mode == "raw":
        return float(np.mean(anx))
    elif mode == "zscore10s":
        n_base = int(fs * baseline_seconds)
        base = anx[:n_base] if len(anx) >= n_base else anx
        m = float(np.mean(base)) if len(base) else 0.0
        s = float(np.std(base)) if len(base) else 1.0
        z = (anx - m) / (s if s > 0.5 else 1.0)  # 라벨 std 하한 0.5
        return float(np.mean(z))
    else:
        raise ValueError(f"Unknown label_mode={mode}")

def _ppg_quality_ok(x_ppg: np.ndarray, fs: int, thr: float) -> bool:
    if nk is None:
        return True
    try:
        q = nk.ppg_quality(x_ppg.astype(np.float32), sampling_rate=fs)
        return float(np.nanmean(q)) >= thr
    except Exception:
        return True  # 품질 계산 실패 시 필터 생략(보수적)


# =========================================
# 다운스트림 생성 / split
# =========================================

def list_participants(data_dir: str, pattern_suffix: str = "_Main.pkl") -> List[str]:
    pids = sorted([p.split("_")[0] for p in os.listdir(data_dir) if p.endswith(pattern_suffix)])
    return pids

def save_downstream_for_participants(
    spec: DownstreamSpec,
    participants: List[str],
    make_split: Optional[SplitPlan] = None,
) -> Dict[str, Any]:
    """
    참가자 목록에 대해 다운스트림 포맷(.pkl) 생성.
    - scene별로 독립적으로 윈도우링하여 저장(서로 다른 scene 사이를跨해서 윈도우 만들지 않음).
    - 반환: meta 사전(저장 파일 수, 설정 등)
    """
    os.makedirs(spec.output_dir, exist_ok=True)
    meta = {
        "spec": asdict(spec),
        "participants": participants,
        "saved_counts": {},
        "files_by_pid": {},
    }

    total_saved = 0
    for pid in participants:
        path = os.path.join(spec.data_dir, f"{pid}_Main.pkl")
        if not os.path.exists(path):
            print(f"[{pid}] 파일 없음 → 스킵")
            continue
        try:
            df_all = pd.read_pickle(path)
        except Exception as e:
            print(f"[{pid}] 읽기 실패: {e} → 스킵")
            continue

        saved_files: List[str] = []
        # scene별로 분리 윈도우링
        for scene in spec.scenes:
            if "scene" in df_all.columns:
                df = df_all.loc[df_all["scene"] == scene].reset_index(drop=True)
            else:
                df = df_all.copy() 
            if df.empty:
                continue

            # 필요한 컬럼만
            needed = set(spec.channel_order + ["anxiety"])
            keep = [c for c in df.columns if c in needed]
            df = df[keep].copy()
            if df.empty or any(c not in df.columns for c in spec.channel_order):
                print(f"[{pid}|{scene}] 필요한 채널 누락 → 스킵")
                continue

            # 리샘플
            try:
                df_rs = _resample_df(df, spec.channel_order + ["anxiety"], spec.fs_raw, spec.fs_target)
            except KeyError as e:
                print(f"[{pid}|{scene}] 컬럼 누락: {e} → 스킵")
                continue

            arr_all = {c: df_rs[c].to_numpy(dtype=np.float32) for c in spec.channel_order}
            anx_all = df_rs["anxiety"].to_numpy(dtype=np.float32)
            n_base = int(spec.fs_target * spec.baseline_seconds)
            anx_base = anx_all[:n_base] if len(anx_all) >= n_base else anx_all
            anx_m = float(np.mean(anx_base)) if len(anx_base) else 0.0
            anx_s = float(np.std(anx_base)) if len(anx_base) else 1.0
            if anx_s < 0.5:
                anx_s = 1.0  # 라벨 분산 하한(너무 작으면 z-score가 폭주)

            # 파형 정규화
            if spec.waveform_norm == "zscore10s":
                stats = _compute_baseline_stats(arr_all, spec.fs_target, spec.baseline_seconds)
                arr_all = _apply_zscore(arr_all, stats)

            T_total = len(anx_all)
            win = spec.fs_target * spec.window_seconds
            stride = spec.fs_target * spec.stride_seconds
            lag_samples = int(spec.fs_target * spec.label_lag_sec)
            if T_total < win:
                continue

            count = 0
            for s, e in _iter_windows(T_total, win, stride):
                if spec.max_windows_per_pid is not None and count >= spec.max_windows_per_pid:
                    break

                seg = {c: arr_all[c][s:e] for c in spec.channel_order}
                if any(len(v) < win for v in seg.values()):
                    continue

                # (선택) PPG 품질 필터
                if spec.enable_ppg_quality_filter and ("PPG_Clean" in spec.channel_order):
                    if not _ppg_quality_ok(seg["PPG_Clean"], spec.fs_target, spec.ppg_quality_threshold):
                        continue
                # 라벨 래깅 적용: anx[s+lag : e+lag] 사용(범위 벗어나면 스킵)
                s_lag, e_lag = s + lag_samples, e + lag_samples
                if s_lag < 0 or e_lag > T_total:
                    continue
                anx_seg = anx_all[s_lag:e_lag]

                if spec.label_mode == "zscore10s":
                    y_val = float(np.mean((anx_seg - anx_m) / anx_s))
                else:
                    y_val = float(np.mean(anx_seg))

                data_ct = np.stack([seg[c] for c in spec.channel_order], axis=0).astype(np.float16)
                fname = f"{pid}_{scene}_win{s:06d}.pkl"  # scene 포함(혼합 저장 대비)
                fpath = os.path.join(spec.output_dir, fname)
                payload = {
                    "uid": pid,
                    "scene": scene,
                    "data": data_ct,                      # [C,T] float16
                    "sampling_rate": int(spec.fs_target),
                    "label": [{"reg": float(y_val)}],
                    "channels": list(spec.channel_order), # 추적용
                    "label_lag_sec": int(spec.label_lag_sec)
                }
                with open(fpath, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

                saved_files.append(fname)
                count += 1
                total_saved += 1

        meta["saved_counts"][pid] = len(saved_files)
        meta["files_by_pid"][pid] = saved_files
        if spec.verbose:
            print(f"[{pid}] 저장 {len(saved_files)}개")

    # split 생성
    if make_split is not None:
        split_path = os.path.join(spec.output_dir, "train_test_split.json")
        split = build_split_from_files(meta["files_by_pid"], make_split)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        if spec.verbose:
            print(f"[split] train={len(split['train'])}, test={len(split['test'])}")

    # 메타 저장
    with open(os.path.join(spec.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if spec.verbose:
        print(f"✅ 다운스트림 생성 완료 | 총 {total_saved}개 | 저장: {spec.output_dir}")
    return meta


def build_split_from_files(files_by_pid: Dict[str, List[str]], plan: SplitPlan) -> Dict[str, Any]:
    train, test = [], []
    for pid, files in files_by_pid.items():
        if pid in plan.train_pids:
            train.extend(files)
        elif pid in plan.test_pids:
            test.extend(files)
    return {"train": train, "test": test, "train_pids": plan.train_pids, "test_pids": plan.test_pids}


# utils/load_split.py 혹은 normwear_utill.py 안
import os, json
from typing import List, Dict, Optional, Tuple, Union

def load_split(*args, **kwargs):
    """
    유연한 시그니처:
      1) load_split(split_path='.../train_test_split.json', base_dir='.../run_dir')
      2) load_split(base_dir='.../run_dir')  # split_path 생략 시 run_dir/train_test_split.json 사용
      3) load_split('.../train_test_split.json', '.../run_dir')  # 구버전 위치 인자
      4) load_split('.../run_dir')  # 구버전: 위치 인자 하나 → base_dir로 간주

    반환:
      {"train": [abs_path,...], "test": [abs_path,...]}
    """
    split_path: Optional[str] = kwargs.get("split_path")
    base_dir:   Optional[str] = kwargs.get("base_dir")

    # 위치 인자 호환
    if len(args) == 2 and split_path is None and base_dir is None:
        split_path, base_dir = args
    elif len(args) == 1 and split_path is None and base_dir is None:
        # 하나만 주면 base_dir로 간주
        base_dir = args[0]

    if base_dir is None and split_path is None:
        raise TypeError("load_split requires base_dir or split_path")

    if split_path is None and base_dir is not None:
        split_path = os.path.join(base_dir, "train_test_split.json")

    if base_dir is None and split_path is not None:
        base_dir = os.path.dirname(os.path.abspath(split_path))

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    for k in ["train", "test"]:
        if k not in split:
            raise KeyError(f"missing key '{k}' in {split_path}")

    def to_abs(lst: List[str]) -> List[str]:
        return [p if os.path.isabs(p) else os.path.join(base_dir, p) for p in lst]

    return {"train": to_abs(split["train"]), "test": to_abs(split["test"])}



def leakage_free_check(train_files: List[str], test_files: List[str], verbose: bool = True):
    train_pids = sorted(list({pid_from_fname(os.path.basename(p)) for p in train_files}))
    test_pids  = sorted(list({pid_from_fname(os.path.basename(p)) for p in test_files}))
    inter = sorted(list(set(train_pids) & set(test_pids)))
    if verbose:
        print(f" - train PIDs: {train_pids}")
        print(f" - test  PIDs: {test_pids}")
    if inter:
        raise RuntimeError(f"Leakage 감지: train/test 교집합 PID={inter}")
    if verbose:
        print(" ✅ Leakage-free 통과")


# =========================================
# 배치 로딩 / 검증
# =========================================

def load_one_downstream(path: str) -> Tuple[str, np.ndarray, int, float, List[str]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    for key in ["uid", "data", "sampling_rate", "label"]:
        if key not in payload:
            raise KeyError(f"[{os.path.basename(path)}] 키 누락: {key}")
    uid = str(payload["uid"])
    data = payload["data"]
    sr   = int(payload["sampling_rate"])
    labels = payload["label"]
    chans = payload.get("channels", None)
    if not isinstance(labels, (list, tuple)) or len(labels) == 0:
        raise ValueError(f"[{os.path.basename(path)}] label 형식 오류")
    y = None
    for d in labels:
        if isinstance(d, dict) and "reg" in d:
            y = float(d["reg"]); break
    if y is None:
        raise ValueError(f"[{os.path.basename(path)}] reg 라벨 없음")
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError(f"[{os.path.basename(path)}] data 형태가 [C,T]가 아님")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError(f"[{os.path.basename(path)}] data에 NaN/Inf 존재")
    return uid, data, sr, y, (chans if chans is not None else [])

def stack_batch(paths: List[str], expect_C: Optional[int] = None, expect_T: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    Xs, ys, uids = [], [], []
    SR_REF = None
    C_ref, T_ref = expect_C, expect_T
    for p in paths:
        uid, data, sr, y, _ = load_one_downstream(p)
        if SR_REF is None: SR_REF = sr
        if sr != SR_REF: raise ValueError(f"[{os.path.basename(p)}] sampling_rate 불일치: {sr} vs {SR_REF}")
        C, T = data.shape
        if C_ref is None: C_ref = C
        if T_ref is None: T_ref = T
        if C != C_ref or T != T_ref:
            raise ValueError(f"[{os.path.basename(p)}] shape 불일치: got [{C},{T}], expect [{C_ref},{T_ref}]")
        Xs.append(data.astype(np.float32, copy=False))
        ys.append(float(y))
        uids.append(uid)
    X = np.stack(Xs, axis=0)  # [N,C,T]
    y = np.asarray(ys, dtype=np.float32)
    u = np.asarray(uids, dtype=object)
    return X, y, u, C_ref, T_ref, SR_REF


# =========================================
# NormWear 임베딩 / 풀링
# =========================================

def ensure_normwear_on_syspath(normwear_repo: str):
    parent = os.path.dirname(normwear_repo)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    print(f" - sys.path 추가: {parent}")

def select_device(force_cpu: bool = False):
    import torch
    if force_cpu:
        print(" - 장치: CPU (FORCE_CPU=True)")
        return torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" - 장치: {device}")
    return device

def init_normwear_model(normwear_repo: str, weight_path: str, device=None, optimized_cwt: bool = True):
    ensure_normwear_on_syspath(normwear_repo)
    from NormWear.main_model import NormWearModel  # type: ignore
    import torch
    if device is None:
        device = select_device(force_cpu=False)
    model = NormWearModel(weight_path=weight_path, optimized_cwt=optimized_cwt).to(device)
    model.eval()
    print(" - NormWearModel 로드 완료")
    return model, device

# --- embed_with_pooling 교체 (NumPy 입력 유지) ---
def embed_with_pooling(
    model,
    X: np.ndarray,                     # [N,C,T] float32
    sampling_rate: int,
    device,
    batch_size: int = 8,
    pooling: str = "concat",           # "mean" | "concat" | "meanstd_concat"
) -> np.ndarray:
    """
    반환:
      - "mean"           → [N, 768]
      - "concat"         → [N, C*768]              (채널별 patch-mean concat)
      - "meanstd_concat" → [N, C*2*768]            (채널별 patch-mean & patch-std concat)
    """
    import torch
    assert X.ndim == 3, f"X ndim={X.ndim}, [N,C,T] 필요"
    N, C, T = X.shape
    outs: List[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb_np = X[i:i+batch_size]  # NumPy로 유지 (get_embedding이 내부에서 np 기대)
            out = model.get_embedding(xb_np, sampling_rate=sampling_rate, device=device)  # [B,C,P,768]
            if not torch.is_tensor(out):
                out = torch.as_tensor(out, device=device)
            if out.ndim != 4 or out.shape[-1] != 768:
                raise RuntimeError(f"임베딩 출력 형태 이상: {tuple(out.shape)} (기대: [B,C,P,768])")

            pm = out.mean(dim=2)                                # [B,C,768]
            if pooling == "mean":
                pooled = pm.mean(dim=1)                         # [B,768]
            elif pooling == "concat":
                pooled = pm.reshape(pm.shape[0], -1)            # [B,C*768]
            elif pooling == "meanstd_concat":
                ps = out.std(dim=2, unbiased=False)             # [B,C,768]
                pooled = torch.cat([pm, ps], dim=2).reshape(pm.shape[0], -1)  # [B,C*2*768]
            else:
                raise ValueError(f"Unknown pooling={pooling}")

            outs.append(pooled.detach().cpu().numpy().astype(np.float32))

    Z = np.concatenate(outs, axis=0)
    t1 = time.time()
    print(f" - 임베딩 완료: N={N}, C={C}, T={T}, Z.shape={Z.shape}, 소요={t1-t0:.2f}s, pooling={pooling}, batch={batch_size}")
    if np.isnan(Z).any() or np.isinf(Z).any():
        raise ValueError("임베딩에 NaN/Inf 존재")
    return Z

# --- PCA 유틸 추가 (train 기준 fit → test transform) ---
def pca_fit_transform(Z_train: np.ndarray, Z_test: np.ndarray, n_components: int = 256, whiten: bool = False, random_state: int = 42):
    """
    임베딩 차원 축소(PCA). train으로 fit 후 test에 동일 변환 적용.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    Ztr_p = pca.fit_transform(Z_train)
    Zte_p = pca.transform(Z_test)
    print(f" - PCA: {Z_train.shape[1]} → {n_components} | explained_var={pca.explained_variance_ratio_.sum():.3f}")
    return Ztr_p.astype(np.float32), Zte_p.astype(np.float32)


# =========================================
# 임베딩 캐시 / 평가
# =========================================

def save_embedding_cache(save_dir: str, Xtr: np.ndarray, ytr: np.ndarray, utr: np.ndarray,
                         Xte: np.ndarray, yte: np.ndarray, ute: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_train.npy"), Xtr)
    np.save(os.path.join(save_dir, "y_train.npy"), ytr)
    np.save(os.path.join(save_dir, "uid_train.npy"), utr)
    np.save(os.path.join(save_dir, "X_test.npy"),  Xte)
    np.save(os.path.join(save_dir, "y_test.npy"),  yte)
    np.save(os.path.join(save_dir, "uid_test.npy"), ute)
    print(f" - 임베딩 캐시 저장 완료: {save_dir}")

def evaluate_regressors_with_baseline(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    use_ridge: bool = True, use_enet: bool = True,
    ridge_alpha: float = 1.0, enet_alpha: float = 0.01, enet_l1_ratio: float = 0.2,
):
    """
    baseline(평균예측)과 Ridge/ElasticNet 비교.
    반환 dict 예:
      {
        "baseline": {"R2": ..., "MAE": ..., "RMSE": ...},
        "Ridge":    {"R2": ..., "MAE": ..., "RMSE": ...},
        "ElasticNet": {...}
      }
    """
    results: Dict[str, Dict[str, float]] = {}

    # Baseline: yhat = mean(y_train)
    yhat_base = np.full_like(yte, fill_value=float(np.mean(ytr)))
    results["baseline"] = _regression_metrics(yte, yhat_base, name="baseline")

    if use_ridge:
        _ensure_sklearn()
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))  # type: ignore
        results["Ridge"] = _fit_eval_model(model, Xtr, ytr, Xte, yte, name="Ridge")

    if use_enet:
        _ensure_sklearn()
        model = make_pipeline(StandardScaler(), ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1_ratio, max_iter=10000))  # type: ignore
        results["ElasticNet"] = _fit_eval_model(model, Xtr, ytr, Xte, yte, name="ElasticNet")

    _print_results(results)
    return results

def _ensure_sklearn():
    if Ridge is None or ElasticNet is None:
        raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`")

def _fit_eval_model(model, Xtr, ytr, Xte, yte, name="model"):
    t0 = time.time()
    model.fit(Xtr, ytr)
    t1 = time.time()
    yhat = model.predict(Xte)
    t2 = time.time()
    r2  = float(r2_score(yte, yhat))
    mae = float(mean_absolute_error(yte, yhat))
    rmse= float(np.sqrt(mean_squared_error(yte, yhat)))
    print(f" - [{name}] train_time={t1-t0:.3f}s, pred_time={t2-t1:.3f}s | R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return {"R2": r2, "MAE": mae, "RMSE": rmse}

def _regression_metrics(y_true, y_pred, name="baseline"):
    r2  = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse= float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f" - [{name}] R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return {"R2": r2, "MAE": mae, "RMSE": rmse}

def _print_results(res: Dict[str, Dict[str, float]]):
    _header("평가 요약")
    for k, v in res.items():
        print(f"{k:>10s} → R2={v['R2']:.4f}, MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}")

# ================================
# (NEW) GroupKFold 기반 튜닝 & 평가
# ================================
from typing import Callable
import warnings

def _uid_groups(uids: np.ndarray) -> np.ndarray:
    """UID 문자열 배열 → 그룹 인덱스(0..G-1). GroupKFold 등에 사용."""
    uniq = {u: i for i, u in enumerate(np.unique(uids))}
    return np.asarray([uniq[u] for u in uids], dtype=np.int32)

def _build_pipeline(model_name: str, **params):
    """모델명에 따른 파이프라인 생성. 스케일러 포함/미포함을 내부에서 처리."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import SVR

    m = model_name.lower()
    if m == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if m == "enet":
        return make_pipeline(StandardScaler(), ElasticNet(max_iter=10000, **params))
    if m == "svr":
        # 비선형 RBF 커널 + 스케일러
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", **params))
    if m == "lgbm":
        try:
            import lightgbm as lgb
        except Exception:
            raise ImportError("LightGBM이 설치되어 있지 않습니다. `pip install lightgbm` 후 다시 시도하세요.")
        # LGBMRegressor는 자체적으로 스케일 불필요
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"Unknown model_name={model_name}")

def _default_param_grid(model_name: str) -> Dict[str, List]:
    """모델별 기본 그리드(안전/소형). 필요시 YAML에서 덮어쓸 수 있음."""
    m = model_name.lower()
    if m == "ridge":
        return {"ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
    if m == "enet":
        return {
            "elasticnet__alpha":    [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    if m == "svr":
        return {
            "svr__C":      [0.1, 1, 10, 100],
            "svr__gamma":  ["scale", 1e-3, 1e-2, 1e-1],
            "svr__epsilon":[0.01, 0.1, 0.5],
        }
    if m == "lgbm":
        return {
            "num_leaves":   [31, 63],
            "learning_rate":[0.05, 0.1],
            "n_estimators": [300, 600],
            "subsample":    [0.8, 1.0],
            "colsample_bytree":[0.8, 1.0],
            "random_state": [42],
        }
    raise ValueError(f"Unknown model_name={model_name}")

def _extract_best_estimator(gs) -> Any:
    """GridSearchCV/RandomizedSearchCV에서 best_estimator_ 얻기."""
    if hasattr(gs, "best_estimator_"):
        return gs.best_estimator_
    raise AttributeError("best_estimator_를 찾을 수 없습니다.")

# --- tune_and_eval_groupcv 시그니처와 내부 로직 확장 ---
def tune_and_eval_groupcv(
    model_name: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    utr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    cv_splits: int = 5,
    refit_scoring: str = "neg_mean_absolute_error",
    verbose: int = 1,
    scale_y: bool = False,   # ⬅️ NEW
):
    from sklearn.model_selection import GroupKFold, GridSearchCV
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor

    groups = _uid_groups(utr)
    if param_grid is None:
        param_grid = _default_param_grid(model_name)

    base_est = _build_pipeline(model_name)  # e.g., StandardScaler()+SVR or Ridge/ENet

    estimator = base_est
    grid = param_grid

    # y 스케일링을 사용할 경우: TTR로 래핑하고 grid key에 'regressor__' 접두사 부여
    if scale_y and model_name.lower() in ("ridge", "enet", "svr"):
        estimator = TransformedTargetRegressor(
            regressor=base_est,
            transformer=StandardScaler(with_mean=True, with_std=True)
        )
        grid = {f"regressor__{k}": v for k, v in param_grid.items()}

    scoring = refit_scoring
    cv = GroupKFold(n_splits=cv_splits)

    print(f"[CV] model={model_name} | scoring={scoring} | splits={cv_splits} | scale_y={scale_y}")
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(Xtr, ytr, groups=groups)

    best_est = _extract_best_estimator(gs)
    best_params = getattr(gs, "best_params_", {})
    cv_best_score = float(getattr(gs, "best_score_", np.nan))
    print(f" - best_params: {best_params}")
    print(f" - cv best {scoring} = {cv_best_score:.6f}")

    yhat = best_est.predict(Xte)  # TTR이면 자동 역변환
    r2  = float(r2_score(yte, yhat))
    mae = float(mean_absolute_error(yte, yhat))
    rmse= float(np.sqrt(mean_squared_error(yte, yhat)))
    print(f" - [TEST] R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    return {"best_params": best_params, "cv_best_score": cv_best_score, "test": {"R2": r2, "MAE": mae, "RMSE": rmse}}

# --- run_models_with_groupcv에서 scale_y 전달 ---
def run_models_with_groupcv(
    Ztr: np.ndarray, ytr: np.ndarray, utr: np.ndarray,
    Zte: np.ndarray, yte: np.ndarray,
    eval_cfg: Dict[str, Any],
):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    results: Dict[str, Any] = {}

    # baseline
    yhat_base = np.full_like(yte, fill_value=float(np.mean(ytr)))
    r2b  = float(r2_score(yte, yhat_base))
    maeb = float(mean_absolute_error(yte, yhat_base))
    rmseb= float(np.sqrt(mean_squared_error(yte, yhat_base)))
    print(f" - [baseline] R2={r2b:.4f}, MAE={maeb:.4f}, RMSE={rmseb:.4f}")
    results["baseline"] = {"R2": r2b, "MAE": maeb, "RMSE": rmseb}

    models    = [m.lower() for m in eval_cfg.get("models", ["ridge","enet"])]
    cv_splits = int(eval_cfg.get("cv_splits", 5))
    tune      = bool(eval_cfg.get("tune", True))
    grids     = eval_cfg.get("grids", {})
    scale_y   = bool(eval_cfg.get("scale_y", False))   # ⬅️ NEW

    for m in models:
        print("="*80)
        print(f"[MODEL] {m}")
        if not tune:
            pipe = _build_pipeline(m)
            pipe.fit(Ztr, ytr)
            yhat = pipe.predict(Zte)
            r2  = float(r2_score(yte, yhat))
            mae = float(mean_absolute_error(yte, yhat))
            rmse= float(np.sqrt(mean_squared_error(yte, yhat)))
            print(f" - [TEST] (no-tune) R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
            results[m] = {"no_tune_test": {"R2": r2, "MAE": mae, "RMSE": rmse}}
            continue

        grid = grids.get(m, None)
        try:
            res = tune_and_eval_groupcv(
                model_name=m,
                Xtr=Ztr, ytr=ytr, utr=utr,
                Xte=Zte, yte=yte,
                param_grid=grid, cv_splits=cv_splits,
                refit_scoring="neg_mean_absolute_error",
                scale_y=scale_y,     # ⬅️ pass
            )
            results[m] = res
        except ImportError as e:
            import warnings; warnings.warn(str(e))
            print(f" ! {m} 스킵: {e}")
        except Exception as e:
            print(f" ! {m} 실패: {e}")

    print("="*80); print("완료: 모델별 결과 dict 반환")
    return results

# =========================================================
# One-click Runner (A→B→C) + Sweep with Caching
# =========================================================
import os, json, time, hashlib, copy
from typing import Dict, Any, List, Optional, Tuple

# --- 안전한 딥머지 ---
def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

# --- 해시/지문 ---
def _short_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

# --- spec 빌더: 어디에 두든 label_lag_sec 인식 ---
def build_spec_from_cfg(cfg: Dict[str, Any]) -> DownstreamSpec:
    label_lag = cfg.get("label_lag_sec",
                        cfg.get("normalization", {}).get("label_lag_sec",
                                                         cfg.get("windowing", {}).get("label_lag_sec", 0)))
    return DownstreamSpec(
        data_dir       = cfg["data"]["data_dir"],
        output_dir     = cfg["data"]["output_dir"],
        scenes         = cfg["data"]["scenes"],
        fs_raw         = cfg["windowing"]["fs_raw"],
        fs_target      = cfg["windowing"]["fs_target"],
        window_seconds = cfg["windowing"]["window_seconds"],
        stride_seconds = cfg["windowing"]["stride_seconds"],
        channel_order  = cfg["channels"]["channel_order"],
        waveform_norm  = cfg["normalization"]["waveform_norm"],
        label_mode     = cfg["normalization"]["label_mode"],
        baseline_seconds = cfg["normalization"]["baseline_seconds"],
        enable_ppg_quality_filter = cfg["quality"]["enable_ppg_quality_filter"],
        ppg_quality_threshold     = cfg["quality"]["ppg_quality_threshold"],
        max_windows_per_pid = cfg["scale"]["max_windows_per_pid"],
        verbose = True,
        label_lag_sec = int(label_lag),
    )

# --- 런 이름 규칙(폴더 자동 네이밍) ---
def make_run_name(cfg: Dict[str, Any]) -> str:
    s   = cfg["data"]["scenes"][0] if cfg["data"]["scenes"] else "Scene"
    lag = cfg.get("label_lag_sec",
                  cfg.get("normalization",{}).get("label_lag_sec",
                                                  cfg.get("windowing",{}).get("label_lag_sec", 0)))
    strd= cfg["windowing"]["stride_seconds"]
    pool= cfg["embed"]["pooling"]
    pca = cfg["embed"].get("pca_dim", None)
    bits = [f"{s}", f"lag{lag}", f"str{strd}", f"pool-{pool}"]
    if pca: bits.append(f"pca{pca}")
    return "__".join(bits)

# --- A(윈도우/X 형성)에 영향을 주는 구성요소만 추려 해시 ---
def _fingerprint_A(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data_dir": cfg["data"]["data_dir"],
        "scenes": tuple(cfg["data"]["scenes"]),
        "fs_raw": cfg["windowing"]["fs_raw"],
        "fs_target": cfg["windowing"]["fs_target"],
        "window_seconds": cfg["windowing"]["window_seconds"],
        "stride_seconds": cfg["windowing"]["stride_seconds"],
        "channels": tuple(cfg["channels"]["channel_order"]),
        "waveform_norm": cfg["normalization"]["waveform_norm"],
        "baseline_seconds": cfg["normalization"]["baseline_seconds"],
        "ppg_quality": {
            "enable": cfg["quality"]["enable_ppg_quality_filter"],
            "thr": cfg["quality"]["ppg_quality_threshold"],
        },
        # 주의: label_lag_sec 은 X에는 영향 X → A 해시에서 제외
    }

# --- C(임베딩 Z)에 영향을 주는 구성요소만 추려 해시 ---
def _fingerprint_Z(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "A": _fingerprint_A(cfg),
        "pooling": cfg["embed"]["pooling"],
        "pca_dim": cfg["embed"].get("pca_dim", None),
        "pca_whiten": cfg["embed"].get("pca_whiten", False),
        "weight_path": cfg["embed"]["weight_path"],
        "sr": cfg["windowing"]["fs_target"],     # ← 추가!
    }

# --- 글로벌 임베딩 캐시 경로(해시 기반) ---
def _resolve_global_emb_cache(base_out_dir: str, cfg: Dict[str, Any]) -> str:
    z_hash = _short_hash(_fingerprint_Z(cfg))
    return os.path.join(base_out_dir, "_emb_cache_global", z_hash)

def _emb_cache_exists(cache_dir: str) -> bool:
    need = ["Z_train.npy", "Z_test.npy", "uids_train.npy", "uids_test.npy", "y_train.npy", "y_test.npy"]
    return all(os.path.exists(os.path.join(cache_dir, n)) for n in need)

def _load_emb_cache(cache_dir: str):
    Ztr = np.load(os.path.join(cache_dir, "Z_train.npy"))
    Zte = np.load(os.path.join(cache_dir, "Z_test.npy"))
    utr = np.load(os.path.join(cache_dir, "uids_train.npy"), allow_pickle=True)
    ute = np.load(os.path.join(cache_dir, "uids_test.npy"), allow_pickle=True)
    ytr = np.load(os.path.join(cache_dir, "y_train.npy"))
    yte = np.load(os.path.join(cache_dir, "y_test.npy"))
    return Ztr, Zte, ytr, yte, utr, ute

def _save_emb_cache(cache_dir: str, Ztr, Zte, ytr, yte, utr, ute):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "Z_train.npy"), Ztr)
    np.save(os.path.join(cache_dir, "Z_test.npy"), Zte)
    np.save(os.path.join(cache_dir, "y_train.npy"), ytr)
    np.save(os.path.join(cache_dir, "y_test.npy"), yte)
    np.save(os.path.join(cache_dir, "uids_train.npy"), utr)
    np.save(os.path.join(cache_dir, "uids_test.npy"), ute)

# --- 핵심: 단일 실험 실행(A→B→C) ---
def run_experiment(
    base_cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    force_regen_A: bool = False,   # A 강제 재생성
    force_reembed: bool = False,   # C 임베딩 강제 재계산
) -> Dict[str, Any]:
    """
    base_cfg + overrides로 구성 후:
      A) save_downstream_for_participants (split 생성 또는 재사용)
      B) stack_batch 로딩/검증
      C) 임베딩(+PCA) → GroupCV 평가
    임베딩은 글로벌 캐시(_emb_cache_global/<hash>)로 재사용.
    """
    cfg = _deep_update(base_cfg, overrides or {})
    # label_lag_sec 은 어느 섹션에 있든 최상위 키로도 복제해둠(로그 편의)
    if "label_lag_sec" not in cfg:
        cfg["label_lag_sec"] = cfg.get("normalization", {}).get("label_lag_sec",
                                 cfg.get("windowing", {}).get("label_lag_sec", 0))

    run_name = make_run_name(cfg)
    base_out = cfg["data"]["output_dir"]
    run_out  = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)

    print("="*100)
    print(f"[RUN] {run_name}")
    print(f" - output_dir: {run_out}")
    print(f" - label_lag_sec={cfg['label_lag_sec']}, stride={cfg['windowing']['stride_seconds']}, pooling={cfg['embed']['pooling']}, pca_dim={cfg['embed'].get('pca_dim', None)}")

    # A) 생성 (split 재사용 규칙: run_out에 split이 있으면 재사용)
    spec = build_spec_from_cfg(_deep_update(cfg, {"data": {"output_dir": run_out}}))
    split_path = os.path.join(run_out, "train_test_split.json")
    need_A = force_regen_A or (not os.path.exists(split_path))
    if need_A:
        train_pids = cfg["split"].get("train_pids") or []
        test_pids  = cfg["split"].get("test_pids") or []
        if not (train_pids and test_pids):
            # 자동 샘플링
            all_pids = list_participants(spec.data_dir)
            rng = np.random.default_rng(cfg["split"].get("seed", 42))
            all_pids = rng.permutation(all_pids).tolist()
            ntr, nte = int(cfg["split"]["train_count"]), int(cfg["split"]["test_count"])
            train_pids, test_pids = all_pids[:ntr], all_pids[ntr:ntr+nte]
        print(f"[A] Creating downstream… (lag={spec.label_lag_sec})")
        save_downstream_for_participants(
            spec,
            participants=train_pids + test_pids,
            make_split=SplitPlan(train_pids=train_pids, test_pids=test_pids),
        )
    else:
        print("[A] Reusing existing split/files (found train_test_split.json)")

    # B) 로딩/검증 — 유연 버전 load_split 사용
    split = load_split(base_dir=run_out)  # {'train': [...], 'test': [...]}
    tr_files, te_files = split["train"], split["test"]

    print(f"[B] stack_batch: train={len(tr_files)} | test={len(te_files)}")

    # --- train ---
    ret_tr = stack_batch(tr_files)  # 가변 반환 대응
    if not isinstance(ret_tr, tuple):
        raise TypeError(f"stack_batch(train) returned non-tuple: {type(ret_tr)}")

    if len(ret_tr) >= 3:
        Xtr, ytr, utr = ret_tr[:3]
    else:
        raise ValueError(f"stack_batch(train) returned {len(ret_tr)} values (<3)")

    # --- test ---
    ret_te = stack_batch(te_files)
    if not isinstance(ret_te, tuple):
        raise TypeError(f"stack_batch(test) returned non-tuple: {type(ret_te)}")

    if len(ret_te) >= 3:
        Xte, yte, ute = ret_te[:3]
    else:
        raise ValueError(f"stack_batch(test) returned {len(ret_te)} values (<3)")

    # 샘플링레이트: 설정값을 항상 사용
    SR = int(cfg["windowing"]["fs_target"])

    # (선택) 일치성 검증: 윈도우 길이로 역산한 SR과 다르면 경고만 출력
    SR_from_T = int(Xtr.shape[-1] / cfg["windowing"]["window_seconds"])
    if SR_from_T != SR:
        print(f"⚠️ SR mismatch: cfg={SR}, derived={SR_from_T}  → cfg 값을 사용합니다.")


    print(f" - [INPUT] Xtr{Xtr.shape}, Xte{Xte.shape}, SR={SR}, C={Xtr.shape[1]}, T={Xtr.shape[2]}")
    print(f" - [LABEL] train N={len(ytr)} mean={np.mean(ytr):.4f} std={np.std(ytr):.4f}")
    print(f" - [LABEL] test  N={len(yte)} mean={np.mean(yte):.4f} std={np.std(yte):.4f}")

    # C) 임베딩(+PCA) with 글로벌 캐시

    global_cache = _resolve_global_emb_cache(base_out, cfg)
    print(f"[C] Embedding cache key: {_short_hash(_fingerprint_Z(cfg))}")
    if (not force_reembed) and _emb_cache_exists(global_cache):
        print(f" - Load embeddings from cache: {global_cache}")
        Ztr, Zte, ytr_c, yte_c, utr_c, ute_c = _load_emb_cache(global_cache)
        # 같은 split인지(UID/길이) 체크; 다르면 재계산
        if Ztr.shape[0] != len(ytr) or Zte.shape[0] != len(yte) or not np.array_equal(utr_c, utr) or not np.array_equal(ute_c, ute):
            print(" ! Cache UID/length mismatch → re-embed")
            use_cache = False
        else:
            use_cache = True
    else:
        use_cache = False

    if not use_cache:
        model, device = init_normwear_model(cfg["embed"]["normwear_repo"],
                                            cfg["embed"]["weight_path"],
                                            cfg["embed"].get("device", None))
        Ztr = embed_with_pooling(model, Xtr, sampling_rate=SR, device=device,
                                 batch_size=int(cfg["embed"]["batch_size"]),
                                 pooling=cfg["embed"]["pooling"])
        Zte = embed_with_pooling(model, Xte, sampling_rate=SR, device=device,
                                 batch_size=int(cfg["embed"]["batch_size"]),
                                 pooling=cfg["embed"]["pooling"])
        # PCA
        if cfg["embed"].get("pca_dim", None):
            Ztr, Zte = pca_fit_transform(Ztr, Zte,
                                         n_components=int(cfg["embed"]["pca_dim"]),
                                         whiten=bool(cfg["embed"].get("pca_whiten", False)),
                                         random_state=42)
        # 글로벌 캐시 저장
        _save_emb_cache(global_cache, Ztr, Zte, ytr, yte, utr, ute)
        print(f" - Saved embeddings to cache: {global_cache}")

    # (선택) 러닝로그용 로컬 캐시도 기록
    try:
        save_embedding_cache(os.path.join(run_out, "_emb_cache"), Ztr, Zte, ytr, yte, utr, ute)
        print(f" - Local cache linked: {os.path.join(run_out, '_emb_cache')}")
    except Exception:
        pass

    # 평가 (baseline 포함)
    results = run_models_with_groupcv(Ztr, ytr, utr, Zte, yte, eval_cfg=cfg["eval"])

    # 결과 저장
    res_path = os.path.join(run_out, "results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # 요약 CSV(append)
    csv_row = {
        "run_name": run_name,
        "label_lag_sec": cfg["label_lag_sec"],
        "stride": cfg["windowing"]["stride_seconds"],
        "pooling": cfg["embed"]["pooling"],
        "pca_dim": cfg["embed"].get("pca_dim", None),
        "baseline_MAE": results["baseline"]["MAE"],
    }
    # 승자 모델 하나 고르기(예: svr 우선)
    for m in ["svr","ridge","enet","lgbm"]:
        if m in results:
            for k,v in results[m]["test"].items():
                csv_row[f"{m}_{k}"] = v
    csv_file = os.path.join(base_out, "sweep_summary.csv")
    header_needed = not os.path.exists(csv_file)
    import csv as _csv
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if header_needed: w.writeheader()
        w.writerow(csv_row)

    print(f"[DONE] saved: {res_path}")
    return {"run_name": run_name, "results": results, "out_dir": run_out}

# --- 여러 조합 스윕 ---
def run_sweep(base_cfg: Dict[str, Any], sweep_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    sweep_list: overrides 딕셔너리들의 리스트
    """
    outs = []
    for i, ov in enumerate(sweep_list, 1):
        print("\n" + "#"*100)
        print(f"[SWEEP] {i}/{len(sweep_list)} overrides = {json.dumps(ov, ensure_ascii=False)}")
        outs.append(run_experiment(base_cfg, ov))
    print("\n== SWEEP DONE ==")
    return outs

# =========================================
# 모듈 docstring에 간단 사용 예시
# =========================================

__doc__ = """
# normwear_utill.py — 간단 사용 예

from normwear_utill import (
    DownstreamSpec, SplitPlan, list_participants, save_downstream_for_participants,
    load_split, leakage_free_check, stack_batch,
    init_normwear_model, embed_with_pooling, save_embedding_cache,
    evaluate_regressors_with_baseline,
)

# 1) 다운스트림 생성 (파일럿 설정)
spec = DownstreamSpec(
    data_dir       = r"D:\\Labroom\\SDPhysiology\\Data\\processed_individual_anonymized",
    output_dir     = r"D:\\Labroom\\SDPhysiology\\sample_for_downstream_v2_zscore",
    scenes         = ["Hallway","Outside"],
    fs_raw         = 120,
    fs_target      = 50,
    window_seconds = 20,
    stride_seconds = 2,
    channel_order  = ["PPG_Clean","EDA_Clean","RSP_Clean"],
    waveform_norm  = "zscore10s",
    label_mode     = "zscore10s",
    baseline_seconds = 10,
    enable_ppg_quality_filter = True,
    ppg_quality_threshold     = 0.5,
    max_windows_per_pid = 24,
)

pids_all = list_participants(spec.data_dir)      # ["001","002",...]
train_pids = pids_all[:10]
test_pids  = pids_all[10:14]

meta = save_downstream_for_participants(
    spec=spec,
    participants=train_pids + test_pids,
    make_split=SplitPlan(train_pids=train_pids, test_pids=test_pids)
)

# 2) split 로드 + 배치 적재
split_path = os.path.join(spec.output_dir, "train_test_split.json")
train_files, test_files = load_split(split_path, base_dir=spec.output_dir)
leakage_free_check(train_files, test_files)

Xtr, ytr, utr, C_ref, T_ref, SR = stack_batch(train_files, expect_C=None, expect_T=None)
Xte, yte, ute, _, _, _          = stack_batch(test_files,  expect_C=C_ref, expect_T=T_ref)

# 3) 임베딩 추출 (concat 풀링 권장)
model, device = init_normwear_model(
    normwear_repo = r"C:\\Users\\user\\code\\NormWear",
    weight_path   = r"C:\\Users\\user\\code\\SDPhysiology\\weights\\normwear\\normwear_last_checkpoint-15470-correct.pth",
)

Ztr = embed_with_pooling(model, Xtr, sampling_rate=SR, device=device, batch_size=8, pooling="concat")
Zte = embed_with_pooling(model, Xte, sampling_rate=SR, device=device, batch_size=8, pooling="concat")

# 4) 임베딩 캐시 저장(선택)
save_embedding_cache(os.path.join(spec.output_dir, "_emb_cache"), Ztr, ytr, utr, Zte, yte, ute)

# 5) 간이 회귀 + baseline 비교
results = evaluate_regressors_with_baseline(Ztr, ytr, Zte, yte, use_ridge=True, use_enet=True)
"""
