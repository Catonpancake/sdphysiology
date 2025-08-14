import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def process_physiology_data(
    data_path,
    output_path="./ml_processed",
    window_seconds=20,
    stride_seconds=2,
    sampling_rate=120,
):
    os.makedirs(output_path, exist_ok=True)

    window_size = sampling_rate * window_seconds
    stride_size = sampling_rate * stride_seconds

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

    participants = sorted([f.split("_")[0] for f in os.listdir(data_path) if f.endswith("_Main.pkl")])

    baseline_dict = {}
    anxiety_baseline_dict = {}
    all_features = []
    X_array = []
    y_array = []
    pid_array = []
    feature_tag_list = []

    for pid in tqdm(participants, desc="Processing"):
        try:
            df = pd.read_pickle(os.path.join(data_path, f"{pid}_Main.pkl"))
            df = df[df["scene"] == "Outside"].dropna().reset_index(drop=True)

            if "pupilL" in df.columns and "pupilR" in df.columns:
                df["pupil_mean"] = df[["pupilL", "pupilR"]].mean(axis=1)

            base = df.iloc[:sampling_rate * 10]
            baseline_dict[pid] = {
                col: (base[col].mean(), base[col].std() if base[col].std() > 1e-6 else 1.0)
                for mod, cols in valid_cols.items() for col in cols if col in base.columns
            }

            if "anxiety" in base.columns:
                mean = base["anxiety"].mean()
                std = base["anxiety"].std()
                anxiety_baseline_dict[pid] = (mean, std if std > 0.5 else 1.0)

            for start in range(0, len(df) - window_size + 1, stride_size):
                window = df.iloc[start:start + window_size].copy()
                if len(window) < window_size:
                    continue

                norm_window = window.copy()
                for mod, cols in valid_cols.items():
                    for col in cols:
                        if col in norm_window.columns and col in baseline_dict[pid]:
                            mean, std = baseline_dict[pid][col]
                            norm_window[col] = (norm_window[col] - mean) / std

                if "PPG_Clean" in norm_window.columns:
                    quality = nk.ppg_quality(norm_window["PPG_Clean"].values, sampling_rate=sampling_rate)
                    if np.nanmean(quality) < 0.5:
                        continue

                try:
                    peaks = np.where(window["PPG_Peaks"].values == 1)[0]
                    if len(peaks) >= 4:
                        ibi = np.diff(peaks) / sampling_rate
                        cv = np.std(ibi) / np.mean(ibi)
                        if cv > 0.5:
                            raise ValueError(f"High HRV CV: {cv:.2f}")
                        hrv = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False, method="time")
                        hrv_features = hrv[["HRV_RMSSD", "HRV_SDNN", "HRV_pNN50"]].iloc[0].to_dict()
                    else:
                        hrv_features = {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_pNN50": np.nan}
                except Exception:
                    hrv_features = {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_pNN50": np.nan}

                row = {"participant": pid, "start_idx": start}
                if "anxiety" in window.columns and pid in anxiety_baseline_dict:
                    mean, std = anxiety_baseline_dict[pid]
                    z_scored = (window["anxiety"] - mean) / std
                    row["anxiety"] = z_scored.mean()
                    y_array.append(z_scored.mean())

                feature_sequence = []
                feature_tags = []

                for mod, cols in valid_cols.items():
                    for col in cols:
                        if col in norm_window.columns:
                            clipped = norm_window[col].clip(-clip_dict[col], clip_dict[col])
                            row[f"{col}_mean"] = clipped.mean()
                            row[f"{col}_std"] = clipped.std()
                            row[f"{col}_max"] = clipped.max()
                            row[f"{col}_slope"] = np.polyfit(np.arange(len(clipped)), clipped, 1)[0]
                            feature_sequence.append(clipped.values)
                            feature_tags.append(f"{col}")

                if feature_sequence:
                    X_array.append(np.stack(feature_sequence, axis=1))  # [T, C]
                    pid_array.append(pid)
                    feature_tag_list.append(feature_tags)

                row.update(hrv_features)
                all_features.append(row)

        except Exception as e:
            print(f"[{pid}] 전체 처리 오류: {e}")
            continue

    df_feat = pd.DataFrame(all_features)
    X_array = np.array(X_array)
    y_array = np.array(y_array)
    pid_array = np.array(pid_array)
    feature_tag_list = feature_tag_list[0] if feature_tag_list else []

    np.save(os.path.join(output_path, "X_array.npy"), X_array)
    np.save(os.path.join(output_path, "y_array.npy"), y_array)
    np.save(os.path.join(output_path, "pid_array.npy"), pid_array)
    np.save(os.path.join(output_path, "feature_tag_list.npy"), feature_tag_list)
    df_feat.to_csv(os.path.join(output_path, "df_feat.csv"), index=False)

    print("✅ 저장 완료:", output_path)
    print(f"📊 X shape: {X_array.shape} | y shape: {y_array.shape} | feature dim: {len(feature_tag_list)}")

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
    # ---- (신규) 타깃 스무딩 옵션 (6) ----
    enable_target_smoothing: bool = False,
    target_smoothing_method: str = "ema",  # "ema" | "median"
    target_smoothing_steps: int = 3,       # 3~5 권장 (샘플 단위; 120Hz면 3=25ms*3가 아님에 유의, 다운샘플 후 기준)
    smooth_before_zscore: bool = True,
    # ---- (신규) 피처 확장 옵션 (4) ----
    enable_feature_expansion: bool = False,
    fe_diff_orders=(1, 2),                 # 1차, 2차 차분 채널 추가
    fe_ma_seconds=(2,),                    # 이동평균 초 단위 리스트 (예: (2,5))
    fe_std_seconds=(5,),                   # 이동표준편차 초 단위 리스트
    fe_enable_slope=True,                  # 창 전체 기울기 채널(상수채널)
    fe_enable_iqr=True,                    # 창 전체 IQR 채널(상수채널)
    fe_enable_band_energy=True,            # FFT 대역 에너지 채널(상수채널)
):
    """
    - 입력 폴더의 {pid}_Main.pkl 로부터 scene별로 원시 신호를 윈도잉.
    - 출력: X_array [N,C,T], y_array [N], pid_array [N], scene_array [N], windex_array [N]
    - feature_tag_list.npy: 사용된 채널 이름
    - meta.json: 파라미터/요약 정보(옵션)

    변경점:
      • enable_target_smoothing: True면 y에 EMA/Median 필터 적용 (노이즈 완화)
      • enable_feature_expansion: True면 각 채널에 시계열 파생/상수 특성 채널 추가
        - 차분(1,2), 이동평균/표준편차, slope, IQR, FFT 대역에너지
    """
    os.makedirs(output_path, exist_ok=True)

    # 윈도/스트라이드 샘플 수
    window_size = int(window_seconds * sampling_rate)
    stride_size = int(stride_seconds * sampling_rate)

    # 사용 신호 컬럼 (파생 피처 위주)
    signal_dict = {
        "EDA":   ["EDA_Tonic", "EDA_Phasic", "SCR_Amplitude", "SCR_RiseTime"],
        "PPG":   ["PPG_Rate"],  # HRV 주파수대역은 RR이 없으므로 PPG_Rate로 근사(주의)
        "RSP":   ["RSP_Rate", "RSP_RVT", "RSP_Amplitude"],
        "Pupil": ["pupilL", "pupilR", "pupil_mean"],
    }
    base_cols = sum(signal_dict.values(), [])  # 평탄화

    # 스펙트럼 대역 정의 (모달리티별 권장치)
    # - EDA tonic: 0–0.4Hz
    # - RSP band: 0.2–0.5Hz
    # - HRV 근사(LF/HF): 0.04–0.15, 0.15–0.4 (PPG_Rate 기반 근사)
    band_map = {
        "EDA_Tonic": [(0.0, 0.4, "EDA_0_0.4")],
        "RSP_Rate":  [(0.2, 0.5, "RSP_0.2_0.5")],
        "RSP_RVT":   [(0.2, 0.5, "RSP_0.2_0.5")],
        "RSP_Amplitude": [(0.2, 0.5, "RSP_0.2_0.5")],
        "PPG_Rate":  [(0.04, 0.15, "HRV_LF_approx"),
                      (0.15, 0.40, "HRV_HF_approx")],
        # pupil은 스펙트럼 기본 OFF (원하면 추가)
    }

    # 참가자 목록
    participants = sorted([f.split("_")[0] for f in os.listdir(data_path) if f.endswith("_Main.pkl")])

    # 결과 리스트
    X_list, y_list, pid_list = [], [], []
    scene_list, windex_list = [], []

    # scenes 인자 정규화
    if scenes is None:
        scenes_set = None  # 모든 scene 허용
    elif isinstance(scenes, str):
        scenes_set = {scenes}
    else:
        scenes_set = set(scenes)

    # 롤링 커널 크기 (샘플 단위) 준비
    ma_ks = [max(1, int(round(s * sampling_rate))) for s in fe_ma_seconds]
    std_ks = [max(1, int(round(s * sampling_rate))) for s in fe_std_seconds]

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

            # pupil_mean 생성 (없으면)
            if "pupil_mean" not in df_scene_all.columns and {"pupilL", "pupilR"}.issubset(df_scene_all.columns):
                df_scene_all["pupil_mean"] = df_scene_all[["pupilL", "pupilR"]].mean(axis=1)

            # 필요한 컬럼만 유지 + 결측 제거 (scene 포함)
            keep_cols = ["scene", "anxiety"] + [c for c in base_cols if c in df_scene_all.columns]
            df_scene_all = df_scene_all[keep_cols].dropna().reset_index(drop=True)
            if len(df_scene_all) < window_size:
                continue

            # 다운샘플 (필요 시)
            if sampling_rate < original_hz:
                df_scene_all = interpolate_downsample(
                    df_scene_all, target_hz=sampling_rate, original_hz=original_hz
                )

            # ---- 타깃 스무딩(옵션) ----
            anxiety_raw = df_scene_all["anxiety"].to_numpy().astype(np.float32)
            if enable_target_smoothing:
                k = max(1, int(target_smoothing_steps))
                if target_smoothing_method.lower() == "median":
                    # 간단 median filter (길이 k, 홀수 강제)
                    if k % 2 == 0: k += 1
                    pad = k // 2
                    xp = np.pad(anxiety_raw, (pad, pad), mode="reflect")
                    sm = np.array([np.median(xp[i:i+k]) for i in range(len(xp)-k+1)], dtype=np.float32)
                else:
                    # EMA
                    alpha = 2.0 / (k + 1.0)
                    sm = np.empty_like(anxiety_raw)
                    acc = anxiety_raw[0]
                    for i, v in enumerate(anxiety_raw):
                        acc = alpha * v + (1 - alpha) * acc
                        sm[i] = acc
                anxiety_for_norm = sm if smooth_before_zscore else anxiety_raw
            else:
                anxiety_for_norm = anxiety_raw

            # Z-score (씬 필터 후 전체 구간 기준)
            a_mean, a_std = np.nanmean(anxiety_for_norm), np.nanstd(anxiety_for_norm)
            a_std = a_std if a_std > 1e-6 else 1.0
            anxiety_z = (anxiety_for_norm - a_mean) / a_std

            # 모든 신호 컬럼 존재 확인(정책 유지: 전부 있어야 진행)
            present_cols = [c for c in base_cols if c in df_scene_all.columns]
            if len(present_cols) != len(base_cols):
                continue

            # 참가자×scene별 윈도 인덱스 카운터
            widx_counter = defaultdict(int)

            n = len(df_scene_all)
            scene_series = df_scene_all['scene'].to_numpy()

            # 원본 시계열 캐시
            series_map = {c: df_scene_all[c].to_numpy().astype(np.float32) for c in present_cols}

            for start in range(0, n - window_size + 1, stride_size):
                end = start + window_size

                # scene 경계 안전: 창 내부에 서로 다른 scene이 섞이면 스킵
                window_scenes = scene_series[start:end]
                if np.any(window_scenes != window_scenes[0]):
                    continue
                sc_name = str(window_scenes[0])

                channel_data = []
                channel_tags = []

                # ---- 채널별 표준화 이전에 파생 생성 (윈도 내부에서 z-score 적용) ----
                for col in present_cols:
                    seg = series_map[col][start:end]  # 원본 창 (float32)

                    # 기본 채널: seg (나중에 z-score)
                    candidates = [(seg, col)]

                    if enable_feature_expansion:
                        # 1) 1·2차 차분 (길이 보존 위해 앞값 보간)
                        for od in fe_diff_orders:
                            d = _diff(seg, order=od)
                            candidates.append((d, f"{col}_diff{od}"))

                        # 2) 이동평균 / 이동표준편차 (길이 동일)
                        for k in ma_ks:
                            ma = _rolling_mean(seg, k)
                            candidates.append((ma, f"{col}_ma{k}"))
                        for k in std_ks:
                            rs = _rolling_std(seg, k)
                            candidates.append((rs, f"{col}_std{k}"))

                        # 3) slope / IQR / band energy → 스칼라 → 상수 채널로 확장
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

                    # 후보들을 각자 윈도 내 z-score 후 추가
                    for arr, tag in candidates:
                        m = float(arr.mean())
                        s = float(arr.std())
                        s = s if s > 1e-6 else 1.0
                        channel_data.append(((arr - m) / s).astype(np.float32))
                        channel_tags.append(tag)

                X = np.stack(channel_data, axis=0)     # [C, T]
                y = anxiety_z[start:end].mean()        # window 평균 anxiety (z)

                # 메타 기록
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

    X_array = np.asarray(X_list, dtype=np.float32)         # [N, C, T]
    y_array = np.asarray(y_list, dtype=np.float32)         # [N]
    pid_array = np.asarray(pid_list)                       # [N]
    scene_array = np.asarray(scene_list)                   # [N]
    windex_array = np.asarray(windex_list, dtype=np.int32) # [N]

    # feature_tag_list: 마지막 윈도에서의 channel_tags 사용 (모든 창 동일 구성 가정)
    feature_tags = np.array(channel_tags, dtype="U128")

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
            "feature_cols_final": feature_tags.tolist()
        }
        with open(os.path.join(output_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ 저장 완료:", output_path)
    print(f"📊 X shape: {X_array.shape} | y shape: {y_array.shape} | #PIDs: {len(np.unique(pid_array))}")
    print(f"🧩 Channels: {X_array.shape[1]} | (예: {feature_tags[:min(10,len(feature_tags))]})")
    print("📝 saved: scene_array.npy, windex_array.npy, feature_tag_list.npy" + (", meta.json" if save_meta else ""))
