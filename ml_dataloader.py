import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# === [ADD] Robust baseline utilities =========================================
def _robust_center_scale(arr, *, use_median=True, mad_c=1.4826, eps=1e-6):
    """
    1D 배열의 중심/스케일 계산.
    - use_median=True  → (median, mad_c * MAD)
    - use_median=False → (mean, std)
    """
    a = np.asarray(arr, dtype=np.float32)
    if use_median:
        med = np.median(a)
        mad = np.median(np.abs(a - med))
        sc = float(mad_c * mad)
        if not np.isfinite(sc) or sc < eps:
            sc = 1.0
        return float(med), sc
    else:
        mu = float(np.nanmean(a))
        sd = float(np.nanstd(a))
        if not np.isfinite(sd) or sd < eps:
            sd = 1.0
        return mu, sd

def _make_baseline_fn(mode: str, *, first_seconds: int, fs: int, mad_c=1.4826):
    """
    mode: "first10s_meanstd" | "first10s_medmad" | "scene_medmad"
    반환: baseline_center_scale(series: np.ndarray) -> (center, scale)
    """
    def _fn(series: np.ndarray):
        x = np.asarray(series, dtype=np.float32)
        if mode == "scene_medmad":
            return _robust_center_scale(x, use_median=True, mad_c=mad_c)
        elif mode == "first10s_medmad":
            L = max(1, int(first_seconds * fs))
            base = x[:L]
            return _robust_center_scale(base, use_median=True, mad_c=mad_c)
        else:  # "first10s_meanstd"
            L = max(1, int(first_seconds * fs))
            base = x[:L]
            return _robust_center_scale(base, use_median=False, mad_c=mad_c)
    return _fn
# ============================================================================

def _ema_causal(x: np.ndarray, alpha: float):
    """
    Causal EMA: y[t] = alpha*x[t] + (1-alpha)*y[t-1]
    alpha = 1 - exp(-Δt/τ).  Δt=1/fs, τ: seconds
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    one_minus = 1.0 - alpha
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + one_minus * y[t-1]
    return y


def _rolling_slope(x: np.ndarray, k: int):
    """
    Causal rolling slope over last k samples (linear regression on indices [0..k-1]).
    첫 (k-1) 구간은 첫 유효 기울기로 채움.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if k <= 1 or n == 0:
        return np.zeros_like(x, dtype=np.float32)

    # 미리 고정된 시간축 통계(0..k-1) 준비
    t = np.arange(k, dtype=np.float32)
    t_mean = t.mean()
    denom = float(np.sum((t - t_mean) ** 2)) + 1e-12  # var(t)*k

    out = np.empty_like(x, dtype=np.float32)
    first_slope = None
    for i in range(n):
        j0 = max(0, i - k + 1)
        seg = x[j0:i+1]
        if len(seg) < k:
            # 길이 k 되기 전엔 slope 계산을 뒤로 미룸
            out[i] = 0.0
            continue
        # 길이 정확히 k인 구간만 사용
        y = seg.astype(np.float32)
        y_mean = y.mean()
        # cov(y,t) / var(t)
        num = float(np.sum((y - y_mean) * (t - t_mean)))
        slope = num / denom
        if first_slope is None:
            first_slope = slope
        out[i] = slope

    # 앞 구간 채우기
    if first_slope is None:
        first_slope = 0.0
    for i in range(min(k-1, n)):
        out[i] = first_slope
    return out


def _welch_bandpowers_fft(x: np.ndarray, fs: float, bands: list[tuple[float,float,str]]):
    """
    간단 FFT-PSD 기반 밴드파워(창 전체)를 구하고 dict로 반환.
    bands: [(f_lo, f_hi, tag), ...]
    또한 total(0..Nyq)과 ratio(Low/All, Low/Mid) 계산을 위해 total과 mid도 함께 반환.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return {}, 0.0

    x = x - np.mean(x)
    X = np.fft.rfft(x)                       # N/2+1
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    psd = (np.abs(X) ** 2) / (n * fs)        # 간단 PSD 근사

    res = {}
    total = float(np.sum(psd))               # 0..Nyquist 총파워
    for (flo, fhi, tag) in bands:
        m = (freqs >= flo) & (freqs < fhi)
        res[tag] = float(np.sum(psd[m]))
    return res, total

# def process_physiology_data(
#     data_path,
#     output_path="./ml_processed",
#     window_seconds=20,
#     stride_seconds=2,
#     sampling_rate=120,
#     scenename="Hallway"
    
# ):
#     os.makedirs(output_path, exist_ok=True)

#     window_size = sampling_rate * window_seconds
#     stride_size = sampling_rate * stride_seconds

#     valid_cols = {
#         "EDA": ["EDA_Tonic", "EDA_Phasic", "SCR_Amplitude", "SCR_RiseTime"],
#         "PPG": ["PPG_Rate"],
#         "RSP": ["RSP_Rate", "RSP_RVT", "RSP_Amplitude"],
#         "Pupil": ["pupilL", "pupilR", "pupil_mean"]
#     }

#     clip_dict = {
#         "EDA_Tonic": 30, "EDA_Phasic": 10, "SCR_Amplitude": 10, "SCR_RiseTime": 10,
#         "PPG_Rate": 5, "RSP_Rate": 5, "RSP_RVT": 7, "RSP_Amplitude": 10,
#         "pupilL": 10, "pupilR": 10, "pupil_mean": 10
#     }

#     participants = sorted([f.split("_")[0] for f in os.listdir(data_path) if f.endswith("_Main.pkl")])

#     baseline_dict = {}
#     anxiety_baseline_dict = {}
#     all_features = []
#     X_array = []
#     y_array = []
#     pid_array = []
#     feature_tag_list = []

#     for pid in tqdm(participants, desc="Processing"):
#         try:
#             df = pd.read_pickle(os.path.join(data_path, f"{pid}_Main.pkl"))
#             df = df[df["scene"] == scenename].dropna().reset_index(drop=True)

#             if "pupilL" in df.columns and "pupilR" in df.columns:
#                 df["pupil_mean"] = df[["pupilL", "pupilR"]].mean(axis=1)

#             base = df.iloc[:sampling_rate * 10]
#             baseline_dict[pid] = {
#                 col: (base[col].mean(), base[col].std() if base[col].std() > 1e-6 else 1.0)
#                 for mod, cols in valid_cols.items() for col in cols if col in base.columns
#             }

#             if "anxiety" in base.columns:
#                 mean = base["anxiety"].mean()
#                 std = base["anxiety"].std()
#                 anxiety_baseline_dict[pid] = (mean, std if std > 0.5 else 1.0)

#             for start in range(0, len(df) - window_size + 1, stride_size):
#                 window = df.iloc[start:start + window_size].copy()
#                 if len(window) < window_size:
#                     continue

#                 norm_window = window.copy()
#                 for mod, cols in valid_cols.items():
#                     for col in cols:
#                         if col in norm_window.columns and col in baseline_dict[pid]:
#                             mean, std = baseline_dict[pid][col]
#                             norm_window[col] = (norm_window[col] - mean) / std

#                 if "PPG_Clean" in norm_window.columns:
#                     quality = nk.ppg_quality(norm_window["PPG_Clean"].values, sampling_rate=sampling_rate)
#                     if np.nanmean(quality) < 0.5:
#                         continue

#                 try:
#                     peaks = np.where(window["PPG_Peaks"].values == 1)[0]
#                     if len(peaks) >= 4:
#                         ibi = np.diff(peaks) / sampling_rate
#                         cv = np.std(ibi) / np.mean(ibi)
#                         if cv > 0.5:
#                             raise ValueError(f"High HRV CV: {cv:.2f}")
#                         hrv = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False, method="time")
#                         hrv_features = hrv[["HRV_RMSSD", "HRV_SDNN", "HRV_pNN50"]].iloc[0].to_dict()
#                     else:
#                         hrv_features = {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_pNN50": np.nan}
#                 except Exception:
#                     hrv_features = {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_pNN50": np.nan}

#                 row = {"participant": pid, "start_idx": start}
#                 if "anxiety" in window.columns and pid in anxiety_baseline_dict:
#                     mean, std = anxiety_baseline_dict[pid]
#                     z_scored = (window["anxiety"] - mean) / std
#                     row["anxiety"] = z_scored.mean()
#                     y_array.append(z_scored.mean())

#                 feature_sequence = []
#                 feature_tags = []

#                 for mod, cols in valid_cols.items():
#                     for col in cols:
#                         if col in norm_window.columns:
#                             clipped = norm_window[col].clip(-clip_dict[col], clip_dict[col])
#                             row[f"{col}_mean"] = clipped.mean()
#                             row[f"{col}_std"] = clipped.std()
#                             row[f"{col}_max"] = clipped.max()
#                             row[f"{col}_slope"] = np.polyfit(np.arange(len(clipped)), clipped, 1)[0]
#                             feature_sequence.append(clipped.values)
#                             feature_tags.append(f"{col}")

#                 if feature_sequence:
#                     X_array.append(np.stack(feature_sequence, axis=1))  # [T, C]
#                     pid_array.append(pid)
#                     feature_tag_list.append(feature_tags)

#                 row.update(hrv_features)
#                 all_features.append(row)

#         except Exception as e:
#             print(f"[{pid}] 전체 처리 오류: {e}")
#             continue

#     df_feat = pd.DataFrame(all_features)
#     X_array = np.array(X_array)
#     y_array = np.array(y_array)
#     pid_array = np.array(pid_array)
#     feature_tag_list = feature_tag_list[0] if feature_tag_list else []

#     np.save(os.path.join(output_path, "X_array.npy"), X_array)
#     np.save(os.path.join(output_path, "y_array.npy"), y_array)
#     np.save(os.path.join(output_path, "pid_array.npy"), pid_array)
#     np.save(os.path.join(output_path, "feature_tag_list.npy"), feature_tag_list)
#     df_feat.to_csv(os.path.join(output_path, "df_feat.csv"), index=False)

#     print("✅ 저장 완료:", output_path)
#     print(f"📊 X shape: {X_array.shape} | y shape: {y_array.shape} | feature dim: {len(feature_tag_list)}")
def process_physiology_data(
    data_path,
    output_path="./ml_processed",
    window_seconds=20,
    stride_seconds=2,
    sampling_rate=120,
    scenename="Hallway",
    *,
    # === NEW: baseline options ===
    baseline_mode="first10s_medmad",   # "first10s_meanstd" | "first10s_medmad" | "scene_medmad"
    baseline_first_seconds=10,
    mad_c=1.4826,
    eps=1e-6
):
    """
    전처리 파이프라인 (scene별):
      - pupil_mean 생성
      - baseline_mode에 따라 scene 내 baseline(z-score) 적용
      - 윈도우링 + 클리핑 + feature 요약(평균/표준편차/최대/기울기)
      - HRV(time) 일부 추출
      - anxiety는 baseline 기준으로 z-score 후, 윈도 평균을 타깃으로 사용

    baseline_mode:
      - "first10s_meanstd" : scene 시작 10초의 mean/std
      - "first10s_medmad"  : scene 시작 10초의 median/MAD(×1.4826)
      - "scene_medmad"     : scene 전체 median/MAD(×1.4826)
    """
    import os
    import numpy as np
    import pandas as pd
    import neurokit2 as nk
    from tqdm import tqdm

    os.makedirs(output_path, exist_ok=True)

    window_size = int(sampling_rate * window_seconds)
    stride_size = int(sampling_rate * stride_seconds)

    valid_cols = {
        "EDA": ["EDA_Tonic", "EDA_Phasic", "SCR_Amplitude", "SCR_RiseTime"],
        "PPG": ["PPG_Rate"],
        "RSP": ["RSP_Rate", "RSP_RVT", "RSP_Amplitude"],
        "Pupil": ["pupilL", "pupilR", "pupil_mean"]
    }

    clip_dict = {
        "EDA_Tonic": 30, "EDA_Phasic": 10, "SCR_Amplitude": 10, "SCR_RiseTime": 10,
        "PPG_Rate": 5, "RSP_Rate": 5, "RSP_RVT": 7, "RSP_Amplitude": 10,
        "pupilL": 10, "pupilR": 10, "pupil_mean": 10
    }

    # 참가자 목록
    participants = sorted([f.split("_")[0] for f in os.listdir(data_path) if f.endswith("_Main.pkl")])

    baseline_dict = {}
    anxiety_baseline_dict = {}
    all_features = []
    X_array, y_array, pid_array = [], [], []
    feature_tag_list = []

    # === baseline 함수 구성 (이미 상단에 추가한 helper 사용 가정) ===
    baseline_fn = _make_baseline_fn(
        baseline_mode, first_seconds=baseline_first_seconds, fs=sampling_rate, mad_c=mad_c
    )

    for pid in tqdm(participants, desc="Processing"):
        try:
            df = pd.read_pickle(os.path.join(data_path, f"{pid}_Main.pkl"))
            df = df[df.get("scene") == scenename].dropna().reset_index(drop=True)
            if df.empty:
                continue

            # pupil_mean 생성
            if "pupilL" in df.columns and "pupilR" in df.columns:
                df["pupil_mean"] = df[["pupilL", "pupilR"]].mean(axis=1)

            # ===== 1) scene-level baseline 계산 (모든 사용 컬럼) =====
            baseline_dict[pid] = {}
            for mod, cols in valid_cols.items():
                for col in cols:
                    if col in df.columns:
                        c, s = baseline_fn(df[col].to_numpy())
                        if not np.isfinite(s) or s < eps:
                            s = 1.0
                        baseline_dict[pid][col] = (c, s)

            # anxiety baseline (scene 기준)
            if "anxiety" in df.columns:
                c, s = baseline_fn(df["anxiety"].to_numpy())
                if not np.isfinite(s) or s < eps:
                    s = 1.0
                anxiety_baseline_dict[pid] = (c, s)

            # ===== 2) 윈도우 루프 =====
            for start in range(0, len(df) - window_size + 1, stride_size):
                window = df.iloc[start:start + window_size].copy()
                if len(window) < window_size:
                    continue

                # (옵션) 품질 체크 (PPG_Clean 있을 때만)
                if "PPG_Clean" in window.columns:
                    try:
                        quality = nk.ppg_quality(window["PPG_Clean"].to_numpy(), sampling_rate=sampling_rate)
                        if float(np.nanmean(quality)) < 0.5:
                            continue
                    except Exception:
                        pass  # 품질 계산 실패 시 통과(보수적으로 유지)

                # ===== 2-1) scene-baseline z-score 적용 =====
                norm_window = window.copy()
                for mod, cols in valid_cols.items():
                    for col in cols:
                        if col in norm_window.columns and col in baseline_dict.get(pid, {}):
                            mean_c, std_c = baseline_dict[pid][col]
                            std_c = std_c if std_c > eps else 1.0
                            norm_window[col] = (norm_window[col] - mean_c) / std_c

                # ===== 2-2) HRV(time) 일부 추출 (PPG_Peaks 사용) =====
                try:
                    hrv_features = {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_pNN50": np.nan}
                    if "PPG_Peaks" in window.columns:
                        peaks = np.where(window["PPG_Peaks"].to_numpy() == 1)[0]
                        if len(peaks) >= 4:
                            ibi = np.diff(peaks) / float(sampling_rate)
                            if (np.mean(ibi) > eps) and (np.std(ibi) / (np.mean(ibi) + eps) <= 0.5):
                                # neurokit2 hrv_time 입력은 R-peak 인덱스를 기대하지만,
                                # 여기서는 PPG peak index로 근사 사용
                                hrv_df = nk.hrv_time(
                                    peaks, sampling_rate=sampling_rate, show=False, method="time"
                                )
                                if not hrv_df.empty:
                                    hrv_features = {
                                        "HRV_RMSSD": float(hrv_df["HRV_RMSSD"].iloc[0]) if "HRV_RMSSD" in hrv_df else np.nan,
                                        "HRV_SDNN":  float(hrv_df["HRV_SDNN"].iloc[0])  if "HRV_SDNN"  in hrv_df else np.nan,
                                        "HRV_pNN50": float(hrv_df["HRV_pNN50"].iloc[0]) if "HRV_pNN50" in hrv_df else np.nan
                                    }
                except Exception:
                    hrv_features = {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_pNN50": np.nan}

                # ===== 2-3) 타깃(anxiety) 윈도 평균 =====
                row = {"participant": pid, "start_idx": start}
                if "anxiety" in window.columns and pid in anxiety_baseline_dict:
                    mean_a, std_a = anxiety_baseline_dict[pid]
                    std_a = std_a if std_a > eps else 1.0
                    z_anx = (window["anxiety"] - mean_a) / std_a
                    row["anxiety"] = float(np.nanmean(z_anx))
                    y_array.append(row["anxiety"])

                # ===== 2-4) 피처 요약 + 시퀀스 쌓기 =====
                feature_sequence = []
                feature_tags = []

                t_idx = np.arange(len(norm_window), dtype=np.float32)
                for mod, cols in valid_cols.items():
                    for col in cols:
                        if col in norm_window.columns:
                            # 클리핑 후 통계/기울기
                            clipped = norm_window[col].clip(-clip_dict[col], clip_dict[col]).to_numpy(dtype=np.float32)
                            row[f"{col}_mean"]  = float(np.nanmean(clipped))
                            row[f"{col}_std"]   = float(np.nanstd(clipped))
                            row[f"{col}_max"]   = float(np.nanmax(clipped))
                            # slope (1차 선형회귀 계수)
                            try:
                                # polyfit은 NaN이 있으면 실패 → NaN 처리
                                if np.isnan(clipped).any():
                                    slope = np.nan
                                else:
                                    slope = np.polyfit(t_idx, clipped, 1)[0]
                            except Exception:
                                slope = np.nan
                            row[f"{col}_slope"] = float(slope) if np.isfinite(slope) else np.nan

                            # 원시 시퀀스 [T] → 나중 스택 시 [T,C]
                            feature_sequence.append(clipped)
                            feature_tags.append(f"{col}")

                if feature_sequence:
                    X_array.append(np.stack(feature_sequence, axis=1))  # [T, C]
                    pid_array.append(pid)
                    feature_tag_list = feature_tags  # 창마다 동일 구성 가정 → 마지막 값 사용

                # HRV 병합
                row.update(hrv_features)
                all_features.append(row)

        except Exception as e:
            print(f"[{pid}] 전체 처리 오류: {e}")
            continue

    # ===== 3) 아웃풋 저장 =====
    df_feat = pd.DataFrame(all_features)
    X_array = np.asarray(X_array, dtype=np.float32) if len(X_array) else np.empty((0, window_size, 0), dtype=np.float32)
    y_array = np.asarray(y_array, dtype=np.float32) if len(y_array) else np.empty((0,), dtype=np.float32)
    pid_array = np.asarray(pid_array)

    np.save(os.path.join(output_path, "X_array.npy"), X_array)
    np.save(os.path.join(output_path, "y_array.npy"), y_array)
    np.save(os.path.join(output_path, "pid_array.npy"), pid_array)
    np.save(os.path.join(output_path, "feature_tag_list.npy"), np.array(feature_tag_list, dtype=object))
    df_feat.to_csv(os.path.join(output_path, "df_feat.csv"), index=False)

    print("✅ 저장 완료:", output_path)
    print(f"📊 X shape: {X_array.shape} | y shape: {y_array.shape} | feature dim: {len(feature_tag_list)}")


# === [PATCH 1/3] Low-frequency utils (hop 기반, causal) ======================
import numpy as np

def _lf_alpha_from_tau(hop_seconds: float, tau_seconds: float) -> float:
    """
    EMA alpha = 1 - exp(-hop/tau), hop/tau 모두 '초' 단위.
    """
    hop = float(max(hop_seconds, 1e-9))
    tau = float(max(tau_seconds, 1e-9))
    return 1.0 - np.exp(-hop / tau)

def _ema_causal_hop(x: np.ndarray, hop_seconds: float, tau_seconds: float) -> np.ndarray:
    """
    Causal EMA on window sequence x (shape: (T,)).
    EMA[t] = alpha*x[t] + (1-alpha)*EMA[t-1], alpha = 1 - exp(-hop/tau).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    alpha = _lf_alpha_from_tau(hop_seconds, tau_seconds)
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    one_minus = 1.0 - alpha
    for t in range(1, x.shape[0]):
        y[t] = alpha * x[t] + one_minus * y[t-1]
    return y

def _rolling_slope_causal_hop(x: np.ndarray, hop_seconds: float, window_seconds: float) -> np.ndarray:
    """
    Causal rolling slope over last k samples, with k = round(window_seconds / hop_seconds).
    리턴 단위: '초당 변화량' (per second).
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    if T == 0:
        return x
    hop = float(max(hop_seconds, 1e-9))
    k = int(max(round(float(window_seconds) / hop), 2))  # 최소 2
    out = np.zeros(T, dtype=np.float32)

    # 고정 시간축 통계(0..k-1)
    t = np.arange(k, dtype=np.float32)
    t_mean = t.mean()
    denom = float(np.sum((t - t_mean) ** 2)) + 1e-12  # var(t)*k

    first_slope = None
    for i in range(T):
        j0 = i - k + 1
        if j0 < 0:
            continue  # 앞 구간은 나중에 채움
        y = x[j0:i+1].astype(np.float32)  # 길이 k
        y_mean = y.mean()
        num = float(np.sum((y - y_mean) * (t - t_mean)))
        slope_per_step = num / denom
        slope_per_sec = slope_per_step / hop
        if first_slope is None:
            first_slope = slope_per_sec
        out[i] = slope_per_sec

    if first_slope is None:
        first_slope = 0.0
    # 초기 미충족 구간 채우기
    for i in range(min(k-1, T)):
        out[i] = first_slope
    return out
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ---------------------------
# Helper: interpolation-based downsampling (column-wise)
# ---------------------------
def interpolate_downsample(df: pd.DataFrame, target_hz: int, original_hz: int = 120, time_col: str = None):
    """
    선형보간 기반 다운샘플링. 숫자형 컬럼만 처리.
    time_col이 주어지면 해당 컬럼(밀리초) 기준으로 리샘플, 없으면 가상 시간축 사용.
    target_hz >= original_hz 이면 그대로 반환.
    """
    if target_hz >= original_hz:
        return df.copy()

    num_rows = len(df)
    if num_rows == 0:
        return df.copy()

    # 원래/타겟 시간축
    if time_col is None:
        t_orig = np.arange(num_rows) / original_hz
    else:
        t_orig = (df[time_col].to_numpy() / 1000.0)

    total_time = t_orig[-1] - t_orig[0]
    n_target = int(np.floor(total_time * target_hz)) + 1
    if n_target < 2:
        n_target = max(2, int(num_rows * target_hz / original_hz))

    t_new = np.linspace(t_orig[0], t_orig[-1], n_target)

    # 숫자형 컬럼만 보간
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = {}
    for col in numeric_cols:
        x = df[col].to_numpy()
        # NaN 임시 채우기(앞/뒤 확장)
        if np.isnan(x).any():
            s = pd.Series(x).ffill().bfill().to_numpy()
        else:
            s = x
        out[col] = np.interp(t_new, t_orig, s)

    # 숫자 아닌 컬럼은 최근접 인덱스로 서브샘플
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    if non_numeric_cols:
        idx_new = np.searchsorted(t_orig, t_new, side="left")
        idx_new = np.clip(idx_new, 0, num_rows - 1)
        for col in non_numeric_cols:
            out[col] = df[col].iloc[idx_new].to_numpy()

    return pd.DataFrame(out)
# ---------------------------
# Main: extract raw windows (scene 고정 컬럼명 사용)
#  - 추가: feature expansion(옵션), target smoothing(옵션)
# ---------------------------
import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ---- 간단 롤링/스펙트럼 유틸 ----
def _rolling_mean(x, k):
    if k <= 1: return x.copy()
    # padding='reflect'로 가장자리 왜곡 최소화
    pad = k // 2
    xpad = np.pad(x, (pad, k - 1 - pad), mode='reflect')
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(xpad, ker, mode='valid').astype(np.float32)

def _rolling_std(x, k):
    if k <= 1: return np.zeros_like(x, dtype=np.float32)
    m = _rolling_mean(x, k)
    # (x-m)^2의 평균의 루트
    pad = k // 2
    xpad = np.pad(x, (pad, k - 1 - pad), mode='reflect')
    ker = np.ones(k, dtype=np.float32) / k
    v = np.convolve((xpad - np.mean(xpad))**2, ker, mode='valid')
    # 근사: 국소분산 대신 전역평균 보정 피하기 위해 m 이용
    return np.sqrt(np.maximum(1e-12, _rolling_mean((x - m)**2, k))).astype(np.float32)

def _diff(x, order=1):
    if order == 1:
        d = np.diff(x, n=1, prepend=x[0])
    elif order == 2:
        d = np.diff(x, n=2, prepend=[x[0], x[1] if len(x) > 1 else x[0]])
    else:
        raise ValueError("order must be 1 or 2")
    return d.astype(np.float32)

def _slope_whole_window(x):
    # 창 전체에 대해 선형회귀 기울기 (상수 채널로 반환)
    n = len(x)
    t = np.arange(n, dtype=np.float32)
    t -= t.mean()
    denom = np.sum(t*t) + 1e-12
    slope = np.sum((x - x.mean()) * t) / denom
    return slope

def _iqr_whole_window(x):
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def _band_energy_fft(x, fs, f_lo, f_hi):
    # 창 내 FFT 기반 대역 에너지 (상대적 합)
    x = x.astype(np.float32)
    n = len(x)
    x = x - np.mean(x)
    X = np.fft.rfft(x)  # N/2+1
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    psd = (np.abs(X) ** 2) / (n * fs)  # 간단 PSD 근사
    m = (freqs >= f_lo) & (freqs < f_hi)
    return float(np.sum(psd[m]))
def extract_raw_physio_windows(
    data_path: str,
    output_path: str = "./ml_processed_raw",
    window_seconds: int = 5,     # 권장: 5초
    stride_seconds: int = 5,     # 권장: 5초 (겹침 없음)
    sampling_rate: int = 120,
    scenes="Outside",            # None=전체, str 또는 list[str]
    original_hz: int = 120,      # 원본 저장 주파수(기본 120Hz로 가정)
    save_meta: bool = True,
    # ---- 타깃 스무딩 옵션 ----
    enable_target_smoothing: bool = False,
    target_smoothing_method: str = "ema",  # "ema" | "median"
    target_smoothing_steps: int = 3,
    smooth_before_zscore: bool = True,
    # ---- 파생 피처 옵션 ----
    enable_feature_expansion: bool = False,
    fe_diff_orders=(1, 2),
    fe_ma_seconds=(2,),
    fe_std_seconds=(5,),
    fe_enable_slope=True,
    fe_enable_iqr=True,
    fe_enable_band_energy=True,
    # ---- 저역 번들 옵션 ----
    fe_enable_lowfreq_bundle: bool = False,
    fe_lowfreq_hop_seconds: float = None,
    fe_lowfreq_targets: tuple = ("EDA_Tonic","EDA_Phasic","PPG_Rate","RSP_Rate","pupilL"),
    fe_lowfreq_spec: dict = None,
    lf_ema_seconds: tuple = (10,),
    lf_slope_seconds: tuple = (10,),
    lf_bands: tuple = ((0.00, 0.02, "LF"), (0.02, 0.08, "MF"), (0.08, 0.20, "HF")),
    # ==== ✅ 신규: baseline 모드 옵션 ====
    baseline_mode_signals: str = None,     # None | "first10s_meanstd" | "first10s_medmad" | "scene_medmad"
    baseline_mode_target: str  = None,     # None | 동일 선택지
    baseline_first_seconds: int = 10,
    mad_c: float = 1.4826,
    eps: float = 1e-6
):
    """
    출력: X_array [N, C, T], y_array [N], pid_array [N], scene_array [N], windex_array [N]
    feature_tag_list.npy, (선택) meta.json

    변경점(중요):
      • baseline_mode_signals / baseline_mode_target 로 scene-level baseline 1차 정규화 지원
      • 기존 창 내부 z-score는 유지(2단 정규화)
    """
    import os, json
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from tqdm import tqdm

    # ==== 내부 유틸(이미 파일 상단에 있다면 생략 가능) =========================
    def _robust_center_scale(arr, *, use_median=True, mad_c=1.4826, eps=1e-6):
        a = np.asarray(arr, dtype=np.float32)
        if use_median:
            med = np.median(a)
            mad = np.median(np.abs(a - med))
            sc = float(mad_c * mad)
            if not np.isfinite(sc) or sc < eps:
                sc = 1.0
            return float(med), sc
        else:
            mu = float(np.nanmean(a))
            sd = float(np.nanstd(a))
            if not np.isfinite(sd) or sd < eps:
                sd = 1.0
            return mu, sd

    def _make_baseline_fn(mode: str, *, first_seconds: int, fs: int, mad_c=1.4826, eps=1e-6):
        def _fn(x: np.ndarray):
            x = np.asarray(x, dtype=np.float32)
            if mode == "scene_medmad":
                return _robust_center_scale(x, use_median=True, mad_c=mad_c, eps=eps)
            elif mode == "first10s_medmad":
                L = max(1, int(first_seconds * fs))
                base = x[:L]
                return _robust_center_scale(base, use_median=True, mad_c=mad_c, eps=eps)
            elif mode == "first10s_meanstd":
                L = max(1, int(first_seconds * fs))
                base = x[:L]
                return _robust_center_scale(base, use_median=False, mad_c=mad_c, eps=eps)
            else:
                # None or unknown → no-op: (0,1)
                return 0.0, 1.0
        return _fn
    # ========================================================================

    os.makedirs(output_path, exist_ok=True)

    window_size = int(window_seconds * sampling_rate)
    stride_size = int(stride_seconds * sampling_rate)

    # 사용 신호 컬럼
    signal_dict = {
        "EDA":   ["EDA_Tonic", "EDA_Phasic", "SCR_Amplitude", "SCR_RiseTime"],
        "PPG":   ["PPG_Rate"],
        "RSP":   ["RSP_Rate", "RSP_RVT", "RSP_Amplitude"],
        "Pupil": ["pupilL", "pupilR", "pupil_mean"],
    }
    base_cols = sum(signal_dict.values(), [])
    # ✅ 추가 physiology 채널 (있으면 쓰고, 없어도 스킵)
    extra_physio_cols = [
        "EDA_Clean",
        "PPG_Clean",
        "RSP_Clean",
        "RSP_Phase",
    ]
    band_map = {
        "EDA_Tonic": [(0.0, 0.4, "EDA_0_0.4")],
        "RSP_Rate":  [(0.2, 0.5, "RSP_0.2_0.5")],
        "RSP_RVT":   [(0.2, 0.5, "RSP_0.2_0.5")],
        "RSP_Amplitude": [(0.2, 0.5, "RSP_0.2_0.5")],
        "PPG_Rate":  [(0.04, 0.15, "HRV_LF_approx"),
                      (0.15, 0.40, "HRV_HF_approx")],
    }
    # ✅ physiology cross-modality 조합 (창 내부에서 사용)
    combo_pairs = [
        # 심박 × EDA (전반적 arousal proxy)
        ("EDA_Tonic",  "PPG_Rate", "EDA_Tonic_x_PPG_Rate"),
        ("EDA_Phasic", "PPG_Rate", "EDA_Phasic_x_PPG_Rate"),
        # cardio-respiratory coupling 비슷한 proxy
        ("RSP_Rate",   "PPG_Rate", "RSP_Rate_x_PPG_Rate"),
    ]
    ratio_pairs = [
        # 심박 수준을 나눈 EDA / RSP
        ("EDA_Tonic", "PPG_Rate", "EDA_Tonic_over_PPG_Rate"),
        ("RSP_Rate",  "PPG_Rate", "RSP_Rate_over_PPG_Rate"),
    ]
    eps_div = 1e-3

    # 참가자
    participants = sorted([f.split("_")[0] for f in os.listdir(data_path) if f.endswith("_Main.pkl")])

    # 결과 컨테이너
    X_list, y_list, pid_list = [], [], []
    scene_list, windex_list = [], []

    # scenes 인자 정규화
    if scenes is None:
        scenes_set = None
    elif isinstance(scenes, str):
        scenes_set = {scenes}
    else:
        scenes_set = set(scenes)

    # 롤링 커널 크기
    ma_ks = [max(1, int(round(s * sampling_rate))) for s in fe_ma_seconds]
    std_ks = [max(1, int(round(s * sampling_rate))) for s in fe_std_seconds]

    # baseline 함수 준비(신호/타깃 각각)
    bl_sig_fn = _make_baseline_fn(
        baseline_mode_signals, first_seconds=baseline_first_seconds, fs=sampling_rate, mad_c=mad_c, eps=eps
    )
    bl_tgt_fn = _make_baseline_fn(
        baseline_mode_target, first_seconds=baseline_first_seconds, fs=sampling_rate, mad_c=mad_c, eps=eps
    )

    for pid in tqdm(participants, desc="Extracting Raw Signals"):
        try:
            df = pd.read_pickle(os.path.join(data_path, f"{pid}_Main.pkl"))
            if 'scene' not in df.columns:
                df['scene'] = 'unknown'

            # scene 필터링
            if scenes_set is None:
                df_scene_all = df.copy()
            else:
                df_scene_all = df[df['scene'].isin(scenes_set)].copy()
            if df_scene_all.empty or "anxiety" not in df_scene_all.columns:
                continue

            # pupil_mean 생성
            if "pupil_mean" not in df_scene_all.columns and {"pupilL", "pupilR"}.issubset(df_scene_all.columns):
                df_scene_all["pupil_mean"] = df_scene_all[["pupilL", "pupilR"]].mean(axis=1)

            # 필요한 컬럼만 유지 + 결측 제거
            keep_cols = ["scene", "anxiety"] + [c for c in base_cols if c in df_scene_all.columns]
            df_scene_all = df_scene_all[keep_cols].dropna().reset_index(drop=True)
            if len(df_scene_all) < window_size:
                continue

            # 다운샘플(필요 시)
            if sampling_rate < original_hz:
                df_scene_all = interpolate_downsample(
                    df_scene_all, target_hz=sampling_rate, original_hz=original_hz
                )

            # ==== 타깃(y) 처리 ====
            anxiety_raw = df_scene_all["anxiety"].to_numpy(dtype=np.float32)

            # (옵션) 스무딩
            if enable_target_smoothing:
                k = max(1, int(target_smoothing_steps))
                if target_smoothing_method.lower() == "median":
                    if k % 2 == 0: k += 1
                    pad = k // 2
                    xp = np.pad(anxiety_raw, (pad, pad), mode="reflect")
                    sm = np.array([np.median(xp[i:i+k]) for i in range(len(xp)-k+1)], dtype=np.float32)
                else:
                    alpha = 2.0 / (k + 1.0)
                    sm = np.empty_like(anxiety_raw)
                    acc = anxiety_raw[0]
                    for i, v in enumerate(anxiety_raw):
                        acc = alpha * v + (1 - alpha) * acc
                        sm[i] = acc
                anxiety_for_norm = sm if smooth_before_zscore else anxiety_raw
            else:
                anxiety_for_norm = anxiety_raw

            # (선택) scene-level baseline 1차 정규화
            c_t, s_t = bl_tgt_fn(anxiety_for_norm)
            if s_t < eps: s_t = 1.0
            anxiety_bl = (anxiety_for_norm - c_t) / s_t

            # (기존) scene 전체 z-score (원하면 생략 가능)
            a_mean, a_std = float(np.nanmean(anxiety_bl)), float(np.nanstd(anxiety_bl))
            if not np.isfinite(a_std) or a_std < eps: a_std = 1.0
            anxiety_z = (anxiety_bl - a_mean) / a_std

            # 모든 "필수" physiology 컬럼 존재 확인
            present_main = [c for c in base_cols if c in df_scene_all.columns]
            if len(present_main) != len(base_cols):
                # 필수 채널 중 하나라도 없으면 이 참가자/scene은 스킵
                continue

            # 추가 physiology 채널은 있으면 같이 사용
            extra_present = [c for c in extra_physio_cols if c in df_scene_all.columns]

            # 최종 사용할 채널 목록 = 필수 + 추가
            present_cols = present_main + extra_present

            # 원본 시계열 캐시
            series_map = {c: df_scene_all[c].to_numpy(dtype=np.float32) for c in present_cols}


            # (선택) 신호 scene-level baseline 1차 정규화
            if baseline_mode_signals is not None:
                for col in present_cols:
                    c_s, s_s = bl_sig_fn(series_map[col])
                    if s_s < eps: s_s = 1.0
                    series_map[col] = (series_map[col] - c_s) / s_s

            # 윈도 루프
            n = len(df_scene_all)
            scene_series = df_scene_all['scene'].to_numpy()
            widx_counter = defaultdict(int)

            for start in range(0, n - window_size + 1, stride_size):
                end = start + window_size

                # scene 경계 안전
                window_scenes = scene_series[start:end]
                if np.any(window_scenes != window_scenes[0]):
                    continue
                sc_name = str(window_scenes[0])

                channel_data, channel_tags = [], []
                t_idx = np.arange(window_size, dtype=np.float32)

                # 채널 확장 + 창 내부 z-score
                for col in present_cols:
                    seg = series_map[col][start:end]  # scene-baseline 반영된 원본 창

                    candidates = [(seg, col)]
                    if enable_feature_expansion:
                        # 1) 차분
                        for od in fe_diff_orders:
                            d = _diff(seg, order=od)
                            candidates.append((d, f"{col}_diff{od}"))
                        # 2) 이동통계
                        for klen in ma_ks:
                            ma = _rolling_mean(seg, klen)
                            candidates.append((ma, f"{col}_ma{klen}"))
                        for klen in std_ks:
                            rs = _rolling_std(seg, klen)
                            candidates.append((rs, f"{col}_std{klen}"))
                        # 3) slope/IQR/대역에너지(상수채널)
                        if fe_enable_slope:
                            s = _slope_whole_window(seg)
                            candidates.append((np.full_like(seg, s), f"{col}_slope"))
                        if fe_enable_iqr:
                            q = _iqr_whole_window(seg)
                            candidates.append((np.full_like(seg, q), f"{col}_iqr"))
                        if fe_enable_band_energy and col in band_map:
                            for (flo, fhi, tag) in band_map[col]:
                                be = _band_energy_fft(seg, sampling_rate, flo, fhi)
                                candidates.append((np.full_like(seg, be), f"{col}_{tag}"))
                        # 4) 저역 번들(옵션)
                        if fe_enable_lowfreq_bundle and (col in fe_lowfreq_targets):
                            _hop_sec = float(fe_lowfreq_hop_seconds) if fe_lowfreq_hop_seconds is not None else float(stride_seconds)
                            _spec = dict(ema_taus=list(lf_ema_seconds), slope_secs=list(lf_slope_seconds), use_bandpower=False)
                            if isinstance(fe_lowfreq_spec, dict): _spec.update(fe_lowfreq_spec)
                            for tau in _spec.get("ema_taus", []) or []:
                                ema = _ema_causal_hop(seg, hop_seconds=_hop_sec, tau_seconds=float(tau))
                                candidates.append((ema.astype(np.float32), f"{col}__LF_EMA_{int(round(tau))}s"))
                            for wsec in _spec.get("slope_secs", []) or []:
                                rs = _rolling_slope_causal_hop(seg, hop_seconds=_hop_sec, window_seconds=float(wsec))
                                candidates.append((rs.astype(np.float32), f"{col}__LF_RSLOPE_{int(round(wsec))}s"))

                    # ---- 각 후보를 창 내부 z-score 후 채널에 추가 ----
                    for arr, tag in candidates:
                        if arr is None:
                            continue
                        m = float(np.nanmean(arr))
                        s = float(np.nanstd(arr))
                        if not np.isfinite(s) or s < eps:
                            s = 1.0
                        z = (arr - m) / s
                        channel_data.append(z.astype(np.float32))
                        channel_tags.append(tag)

                # ✅ (추가) physiology cross-modality 조합 채널
                if enable_feature_expansion:
                    # 1) 곱셈 기반 조합
                    for a, b, name in combo_pairs:
                        if a in series_map and b in series_map:
                            seg_a = series_map[a][start:end]
                            seg_b = series_map[b][start:end]
                            arr = seg_a * seg_b
                            m = float(np.nanmean(arr))
                            s = float(np.nanstd(arr))
                            if not np.isfinite(s) or s < eps:
                                s = 1.0
                            z = (arr - m) / s
                            channel_data.append(z.astype(np.float32))
                            channel_tags.append(name)

                    # 2) ratio 기반 조합
                    for a, b, name in ratio_pairs:
                        if a in series_map and b in series_map:
                            seg_a = series_map[a][start:end]
                            seg_b = series_map[b][start:end]
                            arr = seg_a / (np.abs(seg_b) + eps_div)
                            m = float(np.nanmean(arr))
                            s = float(np.nanstd(arr))
                            if not np.isfinite(s) or s < eps:
                                s = 1.0
                            z = (arr - m) / s
                            channel_data.append(z.astype(np.float32))
                            channel_tags.append(name)

                X = np.stack(channel_data, axis=0)   # [C, T]
                y = float(np.nanmean(anxiety_z[start:end]))

                # 메타
                widx = widx_counter[(pid, sc_name)]
                widx_counter[(pid, sc_name)] += 1

                X_list.append(X)
                y_list.append(y)
                pid_list.append(pid)
                scene_list.append(sc_name)
                windex_list.append(widx)

        except Exception as e:
            print(f"[{pid}] 처리 실패: {e}")
            continue

    if len(X_list) == 0:
        print("⚠️ 생성된 윈도우가 없습니다. scene 필터/컬럼 존재 여부를 확인하세요.")
        return

    # 배열화 & 저장
    X_array = np.asarray(X_list, dtype=np.float32)         # [N, C, T]
    y_array = np.asarray(y_list, dtype=np.float32)         # [N]
    pid_array = np.asarray(pid_list)                       # [N]
    scene_array = np.asarray(scene_list)                   # [N]
    windex_array = np.asarray(windex_list, dtype=np.int32) # [N]
    feature_tags = np.array(channel_tags, dtype="U128")    # 마지막 창의 태그(동일 구성 가정)

    np.save(os.path.join(output_path, "X_array.npy"), X_array)
    np.save(os.path.join(output_path, "y_array.npy"), y_array)
    np.save(os.path.join(output_path, "pid_array.npy"), pid_array)
    np.save(os.path.join(output_path, "scene_array.npy"), scene_array)
    np.save(os.path.join(output_path, "windex_array.npy"), windex_array)
    np.save(os.path.join(output_path, "feature_tag_list.npy"), feature_tags)

    if save_meta:
        meta = {
            "sampling_rate": sampling_rate,
            "original_hz": original_hz,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "scenes": list(scenes_set) if scenes_set is not None else "ALL",
            "n_windows": int(len(X_array)),
            "n_participants": int(len(np.unique(pid_array))),
            "enable_target_smoothing": enable_target_smoothing,
            "target_smoothing_method": target_smoothing_method,
            "target_smoothing_steps": int(target_smoothing_steps),
            "smooth_before_zscore": smooth_before_zscore,
            "enable_feature_expansion": enable_feature_expansion,
            "fe_diff_orders": list(fe_diff_orders),
            "fe_ma_seconds": list(fe_ma_seconds),
            "fe_std_seconds": list(fe_std_seconds),
            "fe_enable_slope": fe_enable_slope,
            "fe_enable_iqr": fe_enable_iqr,
            "fe_enable_band_energy": fe_enable_band_energy,
            "feature_cols_base": base_cols,
            "feature_cols_final": feature_tags.tolist(),
            # 신규 baseline 설정 기록
            "baseline_mode_signals": baseline_mode_signals,
            "baseline_mode_target": baseline_mode_target,
            "baseline_first_seconds": baseline_first_seconds,
            "mad_c": mad_c
        }
        with open(os.path.join(output_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ 저장 완료:", output_path)
    print(f"📊 X shape: {X_array.shape} | y shape: {y_array.shape} | #PIDs: {len(np.unique(pid_array))}")
    print(f"🧩 Channels: {X_array.shape[1]} | (예: {feature_tags[:min(10,len(feature_tags))]})")
    print("📝 saved: scene_array.npy, windex_array.npy, feature_tag_list.npy" + (", meta.json" if save_meta else ""))

# def extract_raw_physio_windows(
#     data_path: str,
#     output_path: str = "./ml_processed_raw",
#     window_seconds: int = 5,     # 권장: 5초
#     stride_seconds: int = 5,     # 권장: 5초 (겹침 없음)
#     sampling_rate: int = 120,
#     scenes="Outside",            # None=전체, str 또는 list[str]
#     original_hz: int = 120,      # 원본 저장 주파수(기본 120Hz로 가정)
#     save_meta: bool = True,
#     # ---- (신규) 타깃 스무딩 옵션 (6) ----
#     enable_target_smoothing: bool = False,
#     target_smoothing_method: str = "ema",  # "ema" | "median"
#     target_smoothing_steps: int = 3,       # 3~5 권장 (샘플 단위; 120Hz면 3=25ms*3가 아님에 유의, 다운샘플 후 기준)
#     smooth_before_zscore: bool = True,
#     # ---- (신규) 피처 확장 옵션 (4) ----
#     enable_feature_expansion: bool = False,
#     fe_diff_orders=(1, 2),                 # 1차, 2차 차분 채널 추가
#     fe_ma_seconds=(2,),                    # 이동평균 초 단위 리스트 (예: (2,5))
#     fe_std_seconds=(5,),                   # 이동표준편차 초 단위 리스트
#     fe_enable_slope=True,                  # 창 전체 기울기 채널(상수채널)
#     fe_enable_iqr=True,                    # 창 전체 IQR 채널(상수채널)
#     fe_enable_band_energy=True,            # FFT 대역 에너지 채널(상수채널)
#     # ==== ✅ 신규: 저역 번들 옵션 ====
#     fe_enable_lowfreq_bundle: bool = False,  # 기본 OFF
#     fe_lowfreq_hop_seconds: float = None,    # None이면 stride_seconds로 간주
#     fe_lowfreq_targets: tuple = ("EDA_Tonic","EDA_Phasic","PPG_Rate","RSP_Rate","pupilL"),
#     fe_lowfreq_spec: dict = None,  # {"ema_taus":[10.0], "slope_secs":[10.0], "use_bandpower": False}
#     lf_ema_seconds: tuple = (10,),      # τ(s) 리스트 (예: (6,10,20))
#     lf_slope_seconds: tuple = (10,),    # Rolling slope 창 길이(sec)
#     # Nyquist=0.5*sampling_rate 기준으로 설정. hop=2s → fs=0.5Hz면 Nyq=0.25
#     lf_bands: tuple = ((0.00, 0.02, "LF"), (0.02, 0.08, "MF"), (0.08, 0.20, "HF")),
# ):
#     """
#     - 입력 폴더의 {pid}_Main.pkl 로부터 scene별로 원시 신호를 윈도잉.
#     - 출력: X_array [N,C,T], y_array [N], pid_array [N], scene_array [N], windex_array [N]
#     - feature_tag_list.npy: 사용된 채널 이름
#     - meta.json: 파라미터/요약 정보(옵션)

#     변경점:
#       • enable_target_smoothing: True면 y에 EMA/Median 필터 적용 (노이즈 완화)
#       • enable_feature_expansion: True면 각 채널에 시계열 파생/상수 특성 채널 추가
#         - 차분(1,2), 이동평균/표준편차, slope, IQR, FFT 대역에너지
#     """
#     os.makedirs(output_path, exist_ok=True)

#     # 윈도/스트라이드 샘플 수
#     window_size = int(window_seconds * sampling_rate)
#     stride_size = int(stride_seconds * sampling_rate)

#     # 사용 신호 컬럼 (파생 피처 위주)
#     signal_dict = {
#         "EDA":   ["EDA_Tonic", "EDA_Phasic", "SCR_Amplitude", "SCR_RiseTime"],
#         "PPG":   ["PPG_Rate"],  # HRV 주파수대역은 RR이 없으므로 PPG_Rate로 근사(주의)
#         "RSP":   ["RSP_Rate", "RSP_RVT", "RSP_Amplitude"],
#         "Pupil": ["pupilL", "pupilR", "pupil_mean"],
#     }
#     base_cols = sum(signal_dict.values(), [])  # 평탄화

#     # 스펙트럼 대역 정의 (모달리티별 권장치)
#     # - EDA tonic: 0–0.4Hz
#     # - RSP band: 0.2–0.5Hz
#     # - HRV 근사(LF/HF): 0.04–0.15, 0.15–0.4 (PPG_Rate 기반 근사)
#     band_map = {
#         "EDA_Tonic": [(0.0, 0.4, "EDA_0_0.4")],
#         "RSP_Rate":  [(0.2, 0.5, "RSP_0.2_0.5")],
#         "RSP_RVT":   [(0.2, 0.5, "RSP_0.2_0.5")],
#         "RSP_Amplitude": [(0.2, 0.5, "RSP_0.2_0.5")],
#         "PPG_Rate":  [(0.04, 0.15, "HRV_LF_approx"),
#                       (0.15, 0.40, "HRV_HF_approx")],
#         # pupil은 스펙트럼 기본 OFF (원하면 추가)
#     }

#     # 참가자 목록
#     participants = sorted([f.split("_")[0] for f in os.listdir(data_path) if f.endswith("_Main.pkl")])

#     # 결과 리스트
#     X_list, y_list, pid_list = [], [], []
#     scene_list, windex_list = [], []

#     # scenes 인자 정규화
#     if scenes is None:
#         scenes_set = None  # 모든 scene 허용
#     elif isinstance(scenes, str):
#         scenes_set = {scenes}
#     else:
#         scenes_set = set(scenes)

#     # 롤링 커널 크기 (샘플 단위) 준비
#     ma_ks = [max(1, int(round(s * sampling_rate))) for s in fe_ma_seconds]
#     std_ks = [max(1, int(round(s * sampling_rate))) for s in fe_std_seconds]

#     for pid in tqdm(participants, desc="Extracting Raw Signals"):
#         try:
#             df = pd.read_pickle(os.path.join(data_path, f"{pid}_Main.pkl"))

#             if 'scene' not in df.columns:
#                 df['scene'] = 'unknown'

#             # scene 필터링
#             if scenes_set is None:
#                 df_scene_all = df.copy()
#             else:
#                 df_scene_all = df[df['scene'].isin(scenes_set)].copy()

#             if df_scene_all.empty or "anxiety" not in df_scene_all.columns:
#                 continue

#             # pupil_mean 생성 (없으면)
#             if "pupil_mean" not in df_scene_all.columns and {"pupilL", "pupilR"}.issubset(df_scene_all.columns):
#                 df_scene_all["pupil_mean"] = df_scene_all[["pupilL", "pupilR"]].mean(axis=1)

#             # 필요한 컬럼만 유지 + 결측 제거 (scene 포함)
#             keep_cols = ["scene", "anxiety"] + [c for c in base_cols if c in df_scene_all.columns]
#             df_scene_all = df_scene_all[keep_cols].dropna().reset_index(drop=True)
#             if len(df_scene_all) < window_size:
#                 continue

#             # 다운샘플 (필요 시)
#             if sampling_rate < original_hz:
#                 df_scene_all = interpolate_downsample(
#                     df_scene_all, target_hz=sampling_rate, original_hz=original_hz
#                 )

#             # ---- 타깃 스무딩(옵션) ----
#             anxiety_raw = df_scene_all["anxiety"].to_numpy().astype(np.float32)
#             if enable_target_smoothing:
#                 k = max(1, int(target_smoothing_steps))
#                 if target_smoothing_method.lower() == "median":
#                     # 간단 median filter (길이 k, 홀수 강제)
#                     if k % 2 == 0: k += 1
#                     pad = k // 2
#                     xp = np.pad(anxiety_raw, (pad, pad), mode="reflect")
#                     sm = np.array([np.median(xp[i:i+k]) for i in range(len(xp)-k+1)], dtype=np.float32)
#                 else:
#                     # EMA
#                     alpha = 2.0 / (k + 1.0)
#                     sm = np.empty_like(anxiety_raw)
#                     acc = anxiety_raw[0]
#                     for i, v in enumerate(anxiety_raw):
#                         acc = alpha * v + (1 - alpha) * acc
#                         sm[i] = acc
#                 anxiety_for_norm = sm if smooth_before_zscore else anxiety_raw
#             else:
#                 anxiety_for_norm = anxiety_raw

#             # Z-score (씬 필터 후 전체 구간 기준)
#             a_mean, a_std = np.nanmean(anxiety_for_norm), np.nanstd(anxiety_for_norm)
#             a_std = a_std if a_std > 1e-6 else 1.0
#             anxiety_z = (anxiety_for_norm - a_mean) / a_std

#             # 모든 신호 컬럼 존재 확인(정책 유지: 전부 있어야 진행)
#             present_cols = [c for c in base_cols if c in df_scene_all.columns]
#             if len(present_cols) != len(base_cols):
#                 continue

#             # 참가자×scene별 윈도 인덱스 카운터
#             widx_counter = defaultdict(int)

#             n = len(df_scene_all)
#             scene_series = df_scene_all['scene'].to_numpy()

#             # 원본 시계열 캐시
#             series_map = {c: df_scene_all[c].to_numpy().astype(np.float32) for c in present_cols}

#             for start in range(0, n - window_size + 1, stride_size):
#                 end = start + window_size

#                 # scene 경계 안전: 창 내부에 서로 다른 scene이 섞이면 스킵
#                 window_scenes = scene_series[start:end]
#                 if np.any(window_scenes != window_scenes[0]):
#                     continue
#                 sc_name = str(window_scenes[0])

#                 channel_data = []
#                 channel_tags = []

#                 # ---- 채널별 표준화 이전에 파생 생성 (윈도 내부에서 z-score 적용) ----
#                 for col in present_cols:
#                     seg = series_map[col][start:end]  # 원본 창 (float32)

#                     # 기본 채널: seg (나중에 z-score)
#                     candidates = [(seg, col)]

#                     if enable_feature_expansion:
#                         # 1) 1·2차 차분 (길이 보존 위해 앞값 보간)
#                         for od in fe_diff_orders:
#                             d = _diff(seg, order=od)
#                             candidates.append((d, f"{col}_diff{od}"))

#                         # 2) 이동평균 / 이동표준편차 (길이 동일)
#                         for k in ma_ks:
#                             ma = _rolling_mean(seg, k)
#                             candidates.append((ma, f"{col}_ma{k}"))
#                         for k in std_ks:
#                             rs = _rolling_std(seg, k)
#                             candidates.append((rs, f"{col}_std{k}"))

#                         # 3) slope / IQR / band energy → 스칼라 → 상수 채널로 확장
#                         if fe_enable_slope:
#                             s = _slope_whole_window(seg)
#                             candidates.append((np.full_like(seg, s), f"{col}_slope"))
#                         if fe_enable_iqr:
#                             q = _iqr_whole_window(seg)
#                             candidates.append((np.full_like(seg, q), f"{col}_iqr"))
#                         if fe_enable_band_energy and col in band_map:
#                             for (flo, fhi, tag) in band_map[col]:
#                                 be = _band_energy_fft(seg, sampling_rate, flo, fhi)
#                                 candidates.append((np.full_like(seg, be), f"{col}_{tag}"))
#                         # === [PATCH 3/3] 저역 번들 (hop=2s 기준, 밴드파워 OFF 기본) ===================
#                         if fe_enable_lowfreq_bundle:
#                             # 3-1) hop seconds 결정: 지정 없으면 stride_seconds 그대로 사용
#                             _hop_sec = float(fe_lowfreq_hop_seconds) if fe_lowfreq_hop_seconds is not None else float(stride_seconds)

#                             # 3-2) 대상 컬럼만 적용 (존재하는 컬럼일 때만)
#                             apply_lowfreq = (col in fe_lowfreq_targets)

#                             # 3-3) spec 기본값
#                             _spec = dict(ema_taus=[10.0], slope_secs=[10.0], use_bandpower=False)
#                             if isinstance(fe_lowfreq_spec, dict):
#                                 _spec.update(fe_lowfreq_spec)

#                             if apply_lowfreq:
#                                 # (A) EMA(τ in seconds) : causal, hop기반
#                                 for tau in _spec.get("ema_taus", []) or []:
#                                     ema = _ema_causal_hop(seg, hop_seconds=_hop_sec, tau_seconds=float(tau))
#                                     candidates.append((ema.astype(np.float32), f"{col}__LF_EMA_{int(round(tau))}s"))

#                                 # (B) Rolling slope(window seconds) : causal, per-second slope
#                                 for wsec in _spec.get("slope_secs", []) or []:
#                                     rs = _rolling_slope_causal_hop(seg, hop_seconds=_hop_sec, window_seconds=float(wsec))
#                                     candidates.append((rs.astype(np.float32), f"{col}__LF_RSLOPE_{int(round(wsec))}s"))

#                                 # (C) 밴드파워/비율은 안전판에선 OFF (원하면 True로)
#                                 if bool(_spec.get("use_bandpower", False)):
#                                     # 밴드파워를 쓰는 경우라도 'fs_eff = 1/hop' 이어야 함.
#                                     # 구현은 의도적으로 생략: 초기 검증은 EMA/RSlope 2개만 권장.
#                                     pass
#                         # ================================================================================

#                     # 후보들을 각자 윈도 내 z-score 후 추가
#                     for arr, tag in candidates:
#                         m = float(arr.mean())
#                         s = float(arr.std())
#                         s = s if s > 1e-6 else 1.0
#                         channel_data.append(((arr - m) / s).astype(np.float32))
#                         channel_tags.append(tag)

#                 X = np.stack(channel_data, axis=0)     # [C, T]
#                 y = anxiety_z[start:end].mean()        # window 평균 anxiety (z)

#                 # 메타 기록
#                 widx = widx_counter[(pid, sc_name)]
#                 widx_counter[(pid, sc_name)] += 1

#                 X_list.append(X)
#                 y_list.append(y)
#                 pid_list.append(pid)
#                 scene_list.append(sc_name)
#                 windex_list.append(widx)

#         except Exception as e:
#             print(f"[{pid}] 처리 실패: {e}")
#             continue

#     if len(X_list) == 0:
#         print("⚠️ 생성된 윈도우가 없습니다. scene 필터/컬럼 존재 여부를 확인하세요.")
#         return

#     X_array = np.asarray(X_list, dtype=np.float32)         # [N, C, T]
#     y_array = np.asarray(y_list, dtype=np.float32)         # [N]
#     pid_array = np.asarray(pid_list)                       # [N]
#     scene_array = np.asarray(scene_list)                   # [N]
#     windex_array = np.asarray(windex_list, dtype=np.int32) # [N]

#     # feature_tag_list: 마지막 윈도에서의 channel_tags 사용 (모든 창 동일 구성 가정)
#     feature_tags = np.array(channel_tags, dtype="U128")

#     np.save(os.path.join(output_path, "X_array.npy"), X_array)
#     np.save(os.path.join(output_path, "y_array.npy"), y_array)
#     np.save(os.path.join(output_path, "pid_array.npy"), pid_array)
#     np.save(os.path.join(output_path, "scene_array.npy"), scene_array)
#     np.save(os.path.join(output_path, "windex_array.npy"), windex_array)
#     np.save(os.path.join(output_path, "feature_tag_list.npy"), feature_tags)

#     if save_meta:
#         meta = {
#             "sampling_rate": sampling_rate,
#             "original_hz": original_hz,
#             "window_seconds": window_seconds,
#             "stride_seconds": stride_seconds,
#             "scenes": list(scenes_set) if scenes_set is not None else "ALL",
#             "n_windows": int(len(X_array)),
#             "n_participants": int(len(np.unique(pid_array))),
#             "enable_target_smoothing": enable_target_smoothing,
#             "target_smoothing_method": target_smoothing_method,
#             "target_smoothing_steps": int(target_smoothing_steps),
#             "smooth_before_zscore": smooth_before_zscore,
#             "enable_feature_expansion": enable_feature_expansion,
#             "fe_diff_orders": list(fe_diff_orders),
#             "fe_ma_seconds": list(fe_ma_seconds),
#             "fe_std_seconds": list(fe_std_seconds),
#             "fe_enable_slope": fe_enable_slope,
#             "fe_enable_iqr": fe_enable_iqr,
#             "fe_enable_band_energy": fe_enable_band_energy,
#             "feature_cols_base": base_cols,
#             "feature_cols_final": feature_tags.tolist()
#         }
#         with open(os.path.join(output_path, "meta.json"), "w", encoding="utf-8") as f:
#             json.dump(meta, f, ensure_ascii=False, indent=2)

#     print("✅ 저장 완료:", output_path)
#     print(f"📊 X shape: {X_array.shape} | y shape: {y_array.shape} | #PIDs: {len(np.unique(pid_array))}")
#     print(f"🧩 Channels: {X_array.shape[1]} | (예: {feature_tags[:min(10,len(feature_tags))]})")
#     print("📝 saved: scene_array.npy, windex_array.npy, feature_tag_list.npy" + (", meta.json" if save_meta else ""))
# =========================
# [ADD] Stratified PID split with scene balance + test-size constraints
# Place this block at the end of ml_dataloader.py
# =========================
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import random
import json
import math

@dataclass
class SplitResult:
    train_m: np.ndarray
    val_m:   np.ndarray
    test_m:  np.ndarray
    info:    Dict

def _scene_distribution(mask: np.ndarray, scene: np.ndarray) -> Dict[str, float]:
    """윈도 마스크로 씬 분포(%) 계산."""
    sub = scene[mask]
    if sub.size == 0:
        return {}
    vals, cnts = np.unique(sub, return_counts=True)
    total = float(cnts.sum())
    return {str(v): float(c/total*100.0) for v, c in zip(vals, cnts)}

def _pid_windows(scene: np.ndarray, pid: np.ndarray) -> Dict:
    """PID별 씬별 윈도 카운트."""
    out = {}
    for p in np.unique(pid):
        m = (pid == p)
        vals, cnts = np.unique(scene[m], return_counts=True)
        out[p] = dict(zip([str(v) for v in vals], cnts.tolist()))
    return out

def _assign_by_pid_with_balance(
    pid: np.ndarray,
    scene: np.ndarray,
    val_ratio: float,
    *,
    min_test_pids: int,
    min_test_windows: int,
    scene_tolerance_pp: float,
    max_tries: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    PID 단위로 train/val/test를 할당.
    - 씬 분포가 각 split에서 전체 분포 대비 ±scene_tolerance_pp 이내 유지
    - test는 PID수/윈도수 하한을 만족
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    unique_p = np.unique(pid)
    # 테스트 비율은 'val_ratio와 같은 크기'로 시작(필요 시 자동 완화)
    test_ratio_init = val_ratio

    # 전체 씬 분포(윈도 기준)
    global_dist = _scene_distribution(np.ones_like(pid, dtype=bool), scene)

    pid2scene = _pid_windows(scene, pid)

    def build_masks(train_pids, val_pids, test_pids):
        train_m = np.isin(pid, train_pids)
        val_m   = np.isin(pid, val_pids)
        test_m  = np.isin(pid, test_pids)
        return train_m, val_m, test_m

    def scene_ok(train_m, val_m, test_m, tol_pp):
        for m in [train_m, val_m, test_m]:
            dist = _scene_distribution(m, scene)
            # 씬이 하나도 없을 수 있는 split은 실패 처리
            if len(dist) == 0:
                return False
            # 전체 분포와 편차 비교
            for sc, g_pct in global_dist.items():
                s_pct = dist.get(sc, 0.0)
                if abs(s_pct - g_pct) > tol_pp:
                    return False
        return True

    # 시도 루프
    tries = 0
    best = None
    # 완화 규칙 단계
    # 0: tol=scene_tolerance_pp,   min_test_pids as is
    # 1: tol=scene_tolerance_pp*2, min_test_pids-2
    # 2: tol=scene_tolerance_pp*2, min_test_pids-2, test_ratio += 0.05
    relax_stage = 0

    while tries < max_tries:
        tries += 1
        # 무작위 PID 셔플
        pids = unique_p.copy()
        np_rng.shuffle(pids)

        # 비율 설정
        test_ratio = test_ratio_init + (0.05 if relax_stage >= 2 else 0.0)

        n_total = len(pids)
        n_val   = max(1, int(round(n_total * val_ratio)))
        n_test  = max(min_test_pids, int(round(n_total * test_ratio)))
        n_val   = min(n_val, n_total - n_test - 1)  # train이 1보다 작아지지 않게
        n_train = n_total - n_val - n_test
        if n_train < 1:
            continue

        # 다-씬 PID 우선 배치(씬 분포 맞추기 쉬움)
        pid_scene_count = [(p, len(pid2scene.get(p, {}))) for p in pids]
        pid_scene_count.sort(key=lambda x: x[1], reverse=True)
        ordered_pids = np.array([p for p, _ in pid_scene_count])

        # 초기 배정: 단순 비율 컷
        test_pids = set(ordered_pids[:n_test].tolist())
        val_pids  = set(ordered_pids[n_test:n_test+n_val].tolist())
        train_pids= set(ordered_pids[n_test+n_val:].tolist())

        train_m, val_m, test_m = build_masks(train_pids, val_pids, test_pids)

        # 최소 윈도 조건 확인
        if test_m.sum() < min_test_windows:
            # 보류: 더 많은 PID를 test로 밀어 넣어본다
            need = min_test_windows - int(test_m.sum())
            # train에서 일부 이동
            if need > 0:
                move = min(need, len(train_pids))
                if move > 0:
                    mv = np_rng.choice(list(train_pids), size=move, replace=False)
                    for p in mv:
                        train_pids.remove(p)
                        test_pids.add(p)
                train_m, val_m, test_m = build_masks(train_pids, val_pids, test_pids)
                if test_m.sum() < min_test_windows:
                    # 여전히 부족 → 다음 시도
                    relax_stage = min(relax_stage+1, 2)
                    continue

        # 씬 분포 확인
        tol = scene_tolerance_pp if relax_stage == 0 else scene_tolerance_pp*2
        if not scene_ok(train_m, val_m, test_m, tol_pp=tol):
            relax_stage = min(relax_stage+1, 2)
            continue

        # 테스트 PID 하한 확인(완화 단계 1부터 -2 허용)
        min_test_pids_eff = min_test_pids if relax_stage == 0 else max(1, min_test_pids-2)
        if len(test_pids) < min_test_pids_eff:
            relax_stage = min(relax_stage+1, 2)
            continue

        # 성공
        info = {
            "tries": tries,
            "relax_stage": relax_stage,
            "global_scene_pct": global_dist,
            "train_scene_pct": _scene_distribution(train_m, scene),
            "val_scene_pct": _scene_distribution(val_m, scene),
            "test_scene_pct": _scene_distribution(test_m, scene),
            "n_pid": {
                "train": int(len(train_pids)),
                "val":   int(len(val_pids)),
                "test":  int(len(test_pids)),
                "total": int(n_total),
            },
            "n_windows": {
                "train": int(train_m.sum()),
                "val":   int(val_m.sum()),
                "test":  int(test_m.sum()),
                "total": int(len(pid)),
            },
            "pid_lists": {
                "train": sorted(list(map(int, train_pids))) if np.issubdtype(pid.dtype, np.integer) else sorted(list(train_pids)),
                "val":   sorted(list(map(int, val_pids)))   if np.issubdtype(pid.dtype, np.integer) else sorted(list(val_pids)),
                "test":  sorted(list(map(int, test_pids)))  if np.issubdtype(pid.dtype, np.integer) else sorted(list(test_pids)),
            }
        }
        best = (train_m, val_m, test_m, info)
        break

    if best is None:
        # 최종 실패 시, 가장 단순 비율로라도 반환(안정성)
        # (주의: 이 경우엔 scene balance를 보장하지 않음. 상위 레벨에서 로그로 알림)
        n_total = len(unique_p)
        n_val   = max(1, int(round(n_total * val_ratio)))
        n_test  = max(min_test_pids, int(round(n_total * test_ratio_init)))
        n_val   = min(n_val, n_total - n_test - 1)
        np_rng.shuffle(unique_p)
        test_pids = set(unique_p[:n_test].tolist())
        val_pids  = set(unique_p[n_test:n_test+n_val].tolist())
        train_pids= set(unique_p[n_test+n_val:].tolist())
        train_m   = np.isin(pid, list(train_pids))
        val_m     = np.isin(pid, list(val_pids))
        test_m    = np.isin(pid, list(test_pids))
        info = {
            "tries": tries,
            "relax_stage": "FAILED_FALLBACK",
            "global_scene_pct": _scene_distribution(np.ones_like(pid, dtype=bool), scene),
            "train_scene_pct": _scene_distribution(train_m, scene),
            "val_scene_pct": _scene_distribution(val_m, scene),
            "test_scene_pct": _scene_distribution(test_m, scene),
            "n_pid": {
                "train": int(len(train_pids)),
                "val":   int(len(val_pids)),
                "test":  int(len(test_pids)),
                "total": int(len(np.unique(pid))),
            },
            "n_windows": {
                "train": int(train_m.sum()),
                "val":   int(val_m.sum()),
                "test":  int(test_m.sum()),
                "total": int(len(pid)),
            },
            "pid_lists": {
                "train": sorted(list(map(int, train_pids))) if np.issubdtype(pid.dtype, np.integer) else sorted(list(train_pids)),
                "val":   sorted(list(map(int, val_pids)))   if np.issubdtype(pid.dtype, np.integer) else sorted(list(val_pids)),
                "test":  sorted(list(map(int, test_pids)))  if np.issubdtype(pid.dtype, np.integer) else sorted(list(test_pids)),
            }
        }
        best = (train_m, val_m, test_m, info)

    return best

def split_across_with_gap_stratified(
    pid: np.ndarray,
    scene: np.ndarray,
    widx: np.ndarray,
    *,
    val_ratio: float,
    gap_steps: int,
    min_test_pids: int = 10,
    min_test_windows: int = 1000,
    scene_tolerance_pp: float = 5.0,
    max_tries: int = 200,
    seed: int = 42
):
    """
    새 스플릿:
      1) PID 단위 분할(LOPO)
      2) 씬 분포 유지(±scene_tolerance_pp)
      3) 테스트 최소 규모 보장(min_test_pids, min_test_windows)

    주의: across-participant라면 PID 불교차이므로 gap은 사실상 무의미합니다.
         (형식 일관성을 위해 gap_steps는 메타에만 기록합니다.)
    """
    train_m, val_m, test_m, info = _assign_by_pid_with_balance(
        pid=pid, scene=scene, val_ratio=val_ratio,
        min_test_pids=min_test_pids, min_test_windows=min_test_windows,
        scene_tolerance_pp=scene_tolerance_pp, max_tries=max_tries, seed=seed
    )
    info["gap_steps"] = int(gap_steps)
    return SplitResult(train_m=train_m, val_m=val_m, test_m=test_m, info=info)
# === ml_dataloader.py or notebook ===
from sklearn.model_selection import GroupKFold
import numpy as np

def outer_pid_kfold_splits(pid, scene, widx, n_splits=5, seed=42):
    """
    PID 단위로 GroupKFold 수행 → 각 fold마다 train/val/test mask 반환
    - val은 train 내부에서 다시 20% 랜덤 분할
    """
    uniq_pids = np.unique(pid)
    gkf = GroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(seed)
    folds = []

    for i, (train_pid_idx, test_pid_idx) in enumerate(gkf.split(uniq_pids, groups=uniq_pids)):
        test_pids = uniq_pids[test_pid_idx]
        train_pids = uniq_pids[train_pid_idx]

        train_mask = np.isin(pid, train_pids)
        test_mask  = np.isin(pid, test_pids)

        # train 내부에서 val_ratio=0.2 랜덤 분할
        tr_idx = np.where(train_mask)[0]
        rng.shuffle(tr_idx)
        n_val = int(len(tr_idx) * 0.2)
        val_idx = tr_idx[:n_val]

        val_mask = np.zeros_like(pid, dtype=bool)
        val_mask[val_idx] = True
        train_mask[val_idx] = False   # val 제외

        folds.append(dict(
            train_m=train_mask,
            val_m=val_mask,
            test_m=test_mask,
            info={"fold": i+1, "n_train": train_mask.sum(),
                  "n_val": val_mask.sum(), "n_test": test_mask.sum()}
        ))
    return folds

def print_split_report(
    pid: np.ndarray,
    scene: np.ndarray,
    y: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    test_m: np.ndarray,
    title: str = "POST-LAG SPLIT (STRATIFIED)"
):
    """필수 요약 리포트: PID/윈도 수, 씬 분포, y 통계(IQR)"""
    def stats(mask):
        sub = y[mask]
        if sub.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "iqr": float("nan")}
        q75, q25 = np.percentile(sub, [75, 25])
        return {
            "n": int(sub.size),
            "mean": float(np.mean(sub)),
            "std": float(np.std(sub)),
            "iqr": float(q75 - q25),
            "n_pid": int(len(np.unique(pid[mask])))
        }
    tr = stats(train_m); va = stats(val_m); te = stats(test_m)
    tr_pct = _scene_distribution(train_m, scene)
    va_pct = _scene_distribution(val_m, scene)
    te_pct = _scene_distribution(test_m, scene)
    print(f"\n===== {title} =====")
    print(f"[PID counts] train={tr['n_pid']} | val={va['n_pid']} | test={te['n_pid']}")
    print(f"[Window counts] train={tr['n']} | val={va['n']} | test={te['n']}")
    print("[Scene %] train:", json.dumps(tr_pct, ensure_ascii=False))
    print("[Scene %]   val:", json.dumps(va_pct, ensure_ascii=False))
    print("[Scene %]  test:", json.dumps(te_pct, ensure_ascii=False))
    print(f"[y] train: mean={tr['mean']:.3f}, std={tr['std']:.3f}, IQR={tr['iqr']:.3f}")
    print(f"[y]   val: mean={va['mean']:.3f}, std={va['std']:.3f}, IQR={va['iqr']:.3f}")
    print(f"[y]  test: mean={te['mean']:.3f}, std={te['std']:.3f}, IQR={te['iqr']:.3f}\n")
