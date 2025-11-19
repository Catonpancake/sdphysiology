import os
import gc
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import psutil
from neurokit2.signal import signal_rate, signal_sanitize
from neurokit2.misc import as_vector
from Dataloader import read_acqknowledge_with_markers
## pair with physiology_batch_process.py and batchrunner.py. 
# This script processes physiology data for each participant in a sub-batch, due to memory issues.

def memory_debug(label=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"[{label}] 📋 {mem:.1f}MB")


def save_current_figure(participant_id, condition, signal_type, save_root="Physiology/figure"):
    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(save_root, exist_ok=True)
    filename = f"{participant_id}_{condition}_{signal_type}.png"
    fig.savefig(os.path.join(save_root, filename), dpi=300)
    plt.clf()
    plt.close(fig)
    del fig
    gc.collect()


def detect_abnormal_ppg(signals_ppg):
    low_hr = (signals_ppg["PPG_Rate"] < 40).sum()
    high_hr = (signals_ppg["PPG_Rate"] > 150).sum()
    total = len(signals_ppg)
    abnormal_ratio = (low_hr + high_hr) / total if total > 0 else 1.0
    return {
        "Low_HR_Count": low_hr,
        "High_HR_Count": high_hr,
        "Total": total,
        "Abnormal_Ratio": abnormal_ratio,
        "exclude_flag_ppg": abnormal_ratio > 0.05
    }


def detect_abnormal_eda(signals_eda, eda_info, sampling_rate):
    from scipy.signal import butter, filtfilt
    duration_min = len(signals_eda) / sampling_rate / 60
    scr_count = np.sum(np.array(eda_info["SCR_Peaks"]) > 0)
    scr_per_min = scr_count / duration_min if duration_min > 0 else 0
    scr_amplitudes = np.array(eda_info["SCR_Amplitude"])
    scr_amplitude_max = scr_amplitudes.max() if len(scr_amplitudes) > 0 else 0
    if len(scr_amplitudes) > 0:
        q1, q3 = np.percentile(scr_amplitudes, [25, 75])
        iqr = q3 - q1
        upper_cut = q3 + 3 * iqr
        scr_outlier_count = np.sum(scr_amplitudes > upper_cut)
    else:
        scr_outlier_count = 0
    tonic = signals_eda.get("EDA_Tonic")
    if tonic is not None and len(tonic) > 1:
        tonic_std = tonic.std()
        tonic_drift = (tonic.iloc[-1] - tonic.iloc[0]) / len(tonic) * sampling_rate / 60
    else:
        tonic_std = np.nan
        tonic_drift = np.nan
    raw_signal = signals_eda.get("EDA_Raw")
    if raw_signal is not None and len(raw_signal) > 10:
        b, a = butter(4, 1 / (0.5 * sampling_rate), btype='high')
        highfreq = filtfilt(b, a, raw_signal)
        noise_ratio = np.var(highfreq) / np.var(raw_signal) if np.var(raw_signal) != 0 else np.nan
    else:
        noise_ratio = np.nan
    exclude_eda = (
        scr_per_min < 0.2
        # scr_per_min > 10 or
        # tonic_std < 0.05 
        # abs(tonic_drift) > 0.5 or
        # noise_ratio > 0.3
    )
    return {
        "scr_count": scr_count,
        "scr_per_min": scr_per_min,
        "scr_amplitude_max": scr_amplitude_max,
        "scr_outlier_count": scr_outlier_count,
        "tonic_std": tonic_std,
        "tonic_drift_per_min": tonic_drift,
        "eda_noise_ratio": noise_ratio,
        "exclude_flag_eda": exclude_eda
    }


def detect_abnormal_rsp(signals_rsp, info_rsp, sampling_rate):
    troughs = info_rsp.get("RSP_Troughs", [])
    intervals = np.diff(troughs) / sampling_rate if len(troughs) > 1 else np.array([np.nan])

    # 안전한 amplitude 계산
    if len(troughs) > 0:
        amplitudes = signals_rsp["RSP_Amplitude"].iloc[troughs].dropna()
        amplitudes = amplitudes[amplitudes > 0]
    else:
        amplitudes = np.array([np.nan])

    cv_int = np.std(intervals) / np.mean(intervals) if np.mean(intervals) != 0 else np.nan
    cv_amp = np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) != 0 else np.nan
    rsp_quality = 1 - np.mean([cv_int, cv_amp]) if np.isfinite(cv_int) and np.isfinite(cv_amp) else 0
    rsp_quality = max(0, rsp_quality)  # 음수 방지

    rate = signals_rsp.get("RSP_Rate", [])
    if len(rate) > 0:
        abnormal_rsp_ratio = ((rate < 8) | (rate > 25)).sum() / len(rate)
        mean_rsp_rate = rate.mean()
        std_rsp_rate = rate.std()
    else:
        abnormal_rsp_ratio = 1.0
        mean_rsp_rate = 0
        std_rsp_rate = 0

    # exclude_rsp = rsp_quality < 0.5 or abnormal_rsp_ratio > 0.40  # 완화 기준 반영
    exclude_rsp = rsp_quality < 0.3 or abnormal_rsp_ratio > 0.40  # 완화 기준 반영
    
    return {
        "mean_rsp_rate": mean_rsp_rate,
        "std_rsp_rate": std_rsp_rate,
        "abnormal_rsp_ratio": abnormal_rsp_ratio,
        "rsp_quality_score": rsp_quality,
        "exclude_flag_rsp": exclude_rsp
    }
import numpy as np
import pandas as pd

def detect_abnormal_rsp_patch(data_or_signals, sampling_rate, info=None, min_troughs=3, rate_bounds=(8, 25)):
    """
    RSP 품질/이상 여부 진단 함수 (단일 DF 입력 지원 + 과거 인터페이스 호환)

    Parameters
    ----------
    data_or_signals : pd.DataFrame | dict
        - 권장: RSP 관련 모든 컬럼을 가진 DataFrame (예: RSP_Amplitude, RSP_Rate, RSP_Troughs 등)
        - 과거 호환: signals_rsp(dict/DF) 를 넣고 info는 별도 dict로 전달
    sampling_rate : float
        샘플링 레이트 (Hz)
    info : dict | None
        과거 방식에서의 info_rsp (예: {"RSP_Troughs": [...]})
    min_troughs : int
        최소 트러프 개수 기준
    rate_bounds : (low, high)
        비정상 호흡수 판정 기준 (bpm): 기본 (8, 25)

    Returns
    -------
    dict
        진단 결과 딕셔너리
    """

    # ---- 헬퍼: 객체에서 안전하게 컬럼/키를 Series/ndarray로 뽑기 ----
    def _get_series(obj, key):
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj[key] if key in obj.columns else None
        if isinstance(obj, dict):
            v = obj.get(key, None)
            if isinstance(v, pd.Series):
                return v
            if isinstance(v, (list, np.ndarray)):
                return pd.Series(v)
            if v is not None and np.isscalar(v):
                return pd.Series([v])
        return None

    # ---- 입력 해석: 단일 DF/딕셔너리 또는 과거 signals+info 구성 모두 지원 ----
    # 단일 DF 혹은 dict가 들어온 경우 -> data로 쓰고, info는 같은 곳에서 추출 시도
    data = data_or_signals
    info_obj = info

    # troughs: info 우선, 없으면 data에서 추출
    troughs_ser = _get_series(info_obj, "RSP_Troughs")
    if troughs_ser is None:
        troughs_ser = _get_series(data, "RSP_Troughs")

    if troughs_ser is not None:
        # 결측/음수/범위 밖 제거 + int 캐스팅
        arr = troughs_ser.to_numpy().astype(float)

        # case 1: 0/1 mask → 인덱스로 변환
        unique_vals = set(np.nan_to_num(arr, nan=0.0))
        if len(arr) == len(data) and unique_vals.issubset({0.0, 1.0}):
            troughs = np.flatnonzero(arr).tolist()

        # case 2: 인덱스 리스트 형태(예전 방식) → 그대로 사용
        else:
            troughs = troughs_ser.dropna().astype(int).tolist()
        if isinstance(data, pd.DataFrame):
            n = len(data)
            troughs = [t for t in troughs if 0 <= t < n]
        else:
            troughs = [t for t in troughs if t >= 0]
    else:
        troughs = []

    # intervals (sec)
    if len(troughs) > 1:
        intervals = np.diff(troughs) / float(sampling_rate)
    else:
        intervals = np.array([np.nan], dtype=float)

    # amplitude: 데이터에서 직접 추출 (권장 경로)
    amp_ser = _get_series(data, "RSP_Amplitude")
    if amp_ser is not None and len(troughs) > 0:
        valid_idx = [t for t in troughs if 0 <= t < len(amp_ser)]
        amplitudes = amp_ser.iloc[valid_idx].dropna().to_numpy()
        amplitudes = amplitudes[amplitudes > 0] if amplitudes.size > 0 else np.array([np.nan])
    else:
        amplitudes = np.array([np.nan], dtype=float)

    # --- CV 계산 (interval / amplitude) ---
    mean_int = np.nanmean(intervals); std_int = np.nanstd(intervals)
    cv_int = (std_int / mean_int) if (np.isfinite(mean_int) and mean_int != 0) else np.nan

    mean_amp = np.nanmean(amplitudes); std_amp = np.nanstd(amplitudes)
    cv_amp = (std_amp / mean_amp) if (np.isfinite(mean_amp) and mean_amp != 0) else np.nan

    # 품질 점수 (엄격/완화)
    if np.isfinite(cv_int) and np.isfinite(cv_amp):
        rsp_quality_strict = 1.0 - float((cv_int + cv_amp) / 2.0)
    else:
        rsp_quality_strict = 0.0
    rsp_quality_strict = max(0.0, rsp_quality_strict)

    vals = [v for v in [cv_int, cv_amp] if np.isfinite(v)]
    rsp_quality_relaxed = (1.0 - float(np.mean(vals))) if len(vals) > 0 else np.nan
    if np.isfinite(rsp_quality_relaxed):
        rsp_quality_relaxed = max(0.0, rsp_quality_relaxed)

    # --- rate 기반 비정상 비율 ---
    rate_ser = _get_series(data, "RSP_Rate")
    low, high = rate_bounds
    if rate_ser is not None and len(rate_ser) > 0:
        rate_arr = np.asarray(rate_ser, dtype=float)
        # 결측/비유한 제거는 분모 반영 위해 False 처리 (이상 비율 계산 시 NaN은 제외)
        valid = np.isfinite(rate_arr)
        if valid.any():
            denom = valid.sum()
            abnormal_rsp_ratio = float(((rate_arr[valid] < low) | (rate_arr[valid] > high)).sum() / denom)
            mean_rsp_rate = float(np.nanmean(rate_arr))
            std_rsp_rate  = float(np.nanstd(rate_arr))
        else:
            abnormal_rsp_ratio = 1.0
            mean_rsp_rate = 0.0
            std_rsp_rate  = 0.0
    else:
        # rate가 없을 때는 보수적으로 이상으로 둠
        abnormal_rsp_ratio = 1.0
        mean_rsp_rate = 0.0
        std_rsp_rate  = 0.0

    # --- 플래그 & exclude ---
    flag_low_quality = (rsp_quality_strict < 0.3)   # 배제 판단은 엄격판
    flag_rate_out    = (abnormal_rsp_ratio > 0.40)
    flag_trough_fail = (len(troughs) < min_troughs)

    exclude_rsp = (flag_low_quality or flag_rate_out)

    return {
        "mean_rsp_rate": float(mean_rsp_rate),
        "std_rsp_rate": float(std_rsp_rate),
        "abnormal_rsp_ratio": float(abnormal_rsp_ratio),
        "rsp_quality_strict": float(rsp_quality_strict),   # 배제 판단용
        "rsp_quality_relaxed": float(rsp_quality_relaxed) if np.isfinite(rsp_quality_relaxed) else np.nan,
        "cv_interval": float(cv_int) if np.isfinite(cv_int) else np.nan,
        "cv_amplitude": float(cv_amp) if np.isfinite(cv_amp) else np.nan,
        "n_troughs": int(len(troughs)),
        "flag_low_quality": bool(flag_low_quality),
        "flag_rate_out": bool(flag_rate_out),
        "flag_trough_fail": bool(flag_trough_fail),
        "exclude_flag_rsp": bool(exclude_rsp),
    }


# 파일 존재 여부에 따라 header 저장 여부 설정
import os
import gc
import psutil
import pandas as pd
import neurokit2 as nk
from neurokit2.signal import signal_rate, signal_sanitize
from neurokit2.misc import as_vector
from Dataloader import read_acqknowledge_with_markers
from physiologycheck import (
    
    
    memory_debug, detect_abnormal_ppg, detect_abnormal_eda, detect_abnormal_rsp,
    save_current_figure
)

def physiologyprocess(data_root='D:/LabRoom/Projects/SD Physiology/Processed/main',
                      output_dir='D:/LabRoom/Projects/SD Physiology/Processed/processed_individual',
                      log_dir='./logs',
                      selected_participants=None,
                      figure=True):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    def append_abnormal_dataframe(df, signal_type):
        log_file = os.path.join(log_dir, f"abnormal_{signal_type}.csv")
        header = not os.path.exists(log_file)
        df.to_csv(log_file, mode='a', header=header, index=False)

    participant_folders = sorted(os.listdir(data_root))
    if selected_participants is not None:
        participant_folders = [p for p in participant_folders if p in selected_participants]
    total_participants = len(participant_folders)

    for i, participant_folder in enumerate(participant_folders):
        print(f"[🔹 Processing ({i+1}/{total_participants}) {participant_folder}]", flush=True)
        memory_debug("Start")
        participant_path = os.path.join(data_root, participant_folder)

        for file in os.listdir(participant_path):
            file_path = os.path.join(participant_path, file)
            if not file.endswith(".acq") or "BT" in file:
                continue

            print(f"📄 {file_path}", flush=True)
            result, _, _ = read_acqknowledge_with_markers(file_path)
            result.columns = result.columns.str.strip()
            result.rename(columns={
                "EDA, Y, PPGED-R": "EDA",
                "RSP, X, RSPEC-R": "RSP",
                "PPG, X, PPGED-R": "PPG",
            }, inplace=True)
            result["Subject"] = participant_folder

            if "VR" in file:
                result["marker"] = result.get("marker", "Ongoing")
                result["scene"] = result.get("scene", "Unknown")
                result = result[result["marker"] == "Ongoing"]
            elif "Survey" in file or "Suvey" in file:
                continue

            sampling_rate = 2000

            # === Clean and extract signals ===
            ppg_signal = result["PPG"].values.copy()
            eda_signal = result["EDA"].values.copy()
            rsp_signal = result["RSP"].values.copy()

            # PPG
            methods_ppg = nk.ppg_methods(sampling_rate=sampling_rate, method="elgendi", method_quality="zhao2018",
                                         window=1.0, threshold=0.7)
            methods_ppg["kwargs_quality"]["approach"] = None
            ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate,
                                       method=methods_ppg["method_cleaning"], **methods_ppg["kwargs_cleaning"])
            peak_signal_ppg, info_ppg = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate,
                                                     method=methods_ppg["method_peaks"], correct_artifacts=True,
                                                     **methods_ppg["kwargs_peaks"])
            info_ppg["sampling_rate"] = sampling_rate
            ppg_rate = signal_rate(info_ppg["PPG_Peaks"], sampling_rate=sampling_rate, desired_length=len(ppg_cleaned))
            ppg_quality = nk.ppg_quality(ppg_cleaned, info_ppg["PPG_Peaks"], sampling_rate=sampling_rate,
                                         method="templatematch", approach=None)
            signals_ppg = pd.DataFrame({
                "PPG_Raw": ppg_signal,
                "PPG_Clean": ppg_cleaned,
                "PPG_Rate": ppg_rate,
                "PPG_Quality": ppg_quality,
                "PPG_Peaks": peak_signal_ppg["PPG_Peaks"].values
            })

            # EDA
            eda_signal = signal_sanitize(eda_signal)
            eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate, method="biosppy")
            eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate, method="highpass", cutoff=0.05)
            peak_signal_eda, info_eda = nk.eda_peaks(eda_decomposed["EDA_Phasic"].values, sampling_rate=sampling_rate,
                                                     method="neurokit", amplitude_min=0.1)
            info_eda["sampling_rate"] = sampling_rate
            signals_eda = pd.concat([
                pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned}),
                eda_decomposed, peak_signal_eda
            ], axis=1)

            # RSP
            rsp_signal = as_vector(rsp_signal)
            methods_rsp = nk.rsp_methods(sampling_rate=sampling_rate, method="khodadad2018", method_rvt="harrison2021")
            rsp_cleaned = nk.rsp_clean(rsp_signal, sampling_rate=sampling_rate,
                                       method=methods_rsp["method_cleaning"], **methods_rsp["kwargs_cleaning"])
            peak_signal_rsp, info_rsp = nk.rsp_peaks(rsp_cleaned, sampling_rate=sampling_rate,
                                                     method=methods_rsp["method_peaks"], amplitude_min=0.1,
                                                     **methods_rsp["kwargs_peaks"])
            info_rsp["sampling_rate"] = sampling_rate
            phase = nk.rsp_phase(peak_signal_rsp, desired_length=len(rsp_signal))
            amplitude = nk.rsp_amplitude(rsp_cleaned, peak_signal_rsp)
            rate = signal_rate(info_rsp["RSP_Troughs"], sampling_rate=sampling_rate, desired_length=len(rsp_signal))
            symmetry = nk.rsp_symmetry(rsp_cleaned, peak_signal_rsp)
            rvt = nk.rsp_rvt(rsp_cleaned, method=methods_rsp["method_rvt"], sampling_rate=sampling_rate, silent=True)
            signals_rsp = pd.concat([
                pd.DataFrame({
                    "RSP_Raw": rsp_signal,
                    "RSP_Clean": rsp_cleaned,
                    "RSP_Amplitude": amplitude,
                    "RSP_Rate": rate,
                    "RSP_RVT": rvt
                }),
                phase, symmetry, peak_signal_rsp
            ], axis=1)

            # === Quality Check + Immediate Save ===
            condition = file.split("_")[1].split(".")[0]

            ppg_metrics = detect_abnormal_ppg(signals_ppg.copy())
            ppg_metrics.update({"Participant": participant_folder, "File": file})
            append_abnormal_dataframe(pd.DataFrame([ppg_metrics]), "ppg")

            eda_metrics = detect_abnormal_eda(signals_eda.copy(), info_eda, sampling_rate)
            eda_metrics.update({"Participant": participant_folder, "File": file})
            append_abnormal_dataframe(pd.DataFrame([eda_metrics]), "eda")

            rsp_metrics = detect_abnormal_rsp(signals_rsp.copy(), info_rsp, sampling_rate)
            rsp_metrics.update({"Participant": participant_folder, "File": file})
            append_abnormal_dataframe(pd.DataFrame([rsp_metrics]), "rsp")

            # === Visualization & Save ===
            if figure == True:
                nk.ppg_plot(signals_ppg, info_ppg)
                save_current_figure(participant_folder, condition, "PPG")

                nk.eda_plot(signals_eda, info_eda)
                save_current_figure(participant_folder, condition, "EDA")

                nk.rsp_plot(signals_rsp, info_rsp)
                save_current_figure(participant_folder, condition, "RSP")

            # Memory Cleanup
            del (result, ppg_signal, ppg_cleaned, peak_signal_ppg, info_ppg, ppg_rate, ppg_quality, signals_ppg,
                 eda_signal, eda_cleaned, eda_decomposed, peak_signal_eda, info_eda, signals_eda,
                 rsp_signal, rsp_cleaned, peak_signal_rsp, info_rsp, phase, amplitude, symmetry, rvt, signals_rsp)
            gc.collect()
            memory_debug("After del and gc")

    print("\n✅ All data processed and saved.")
    
import os
import gc
import pandas as pd
import neurokit2 as nk
from bioread import read_file
from physiologycheck import detect_abnormal_rsp, signal_rate
import matplotlib.pyplot as plt


def process_rsp_only_all_channels(data_root, output_dir, log_dir='./logs', selected_participants=None, figure=True):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    def append_log(df):
        os.makedirs(log_dir, exist_ok=True)  # ensure directory
        log_file = os.path.join(log_dir, f"abnormal_rsp_allchannels.csv")
        header = not os.path.exists(log_file)

        if not df.empty:
            df.to_csv(log_file, mode='a', header=header, index=False)
            print(f"✅ Logged: {log_file}")
        else:
            print("⚠️ Warning: Empty dataframe, nothing logged.")

    # 🔄 파일 기반 처리 (폴더가 아닌)
    files = [f for f in os.listdir(data_root) if f.endswith(".acq") and "BT" not in f]
    if selected_participants:
        files = [f for f in files if any(p in f for p in selected_participants)]

    for file in sorted(files):
        try:
            acq_path = os.path.join(data_root, file)
            participant = file.split("_")[0]  # 파일명에서 participant ID 추정
            raw = read_file(acq_path)
            sampling_rate = 2000

            for idx, ch in enumerate(raw.channels):
                if not any(keyword in ch.name for keyword in ["RSP", "RSPEC"]):
                    continue

                rsp_signal = ch.data
                if len(rsp_signal) < 1000:  # 너무 짧은 채널 제외
                    continue

                # Cleaning & Features
                methods_rsp = nk.rsp_methods(sampling_rate=sampling_rate)
                rsp_cleaned = nk.rsp_clean(rsp_signal, sampling_rate=sampling_rate,
                                           method=methods_rsp["method_cleaning"], **methods_rsp["kwargs_cleaning"])
                peaks, info = nk.rsp_peaks(rsp_cleaned, sampling_rate=sampling_rate,
                                           method=methods_rsp["method_peaks"], amplitude_min=0.1,
                                           **methods_rsp["kwargs_peaks"])
                info["sampling_rate"] = sampling_rate
                amplitude = nk.rsp_amplitude(rsp_cleaned, peaks)
                rate = signal_rate(info["RSP_Troughs"], sampling_rate=sampling_rate, desired_length=len(rsp_signal))
                signals_rsp = pd.DataFrame({
                    "RSP_Raw": rsp_signal,
                    "RSP_Clean": rsp_cleaned,
                    "RSP_Amplitude": amplitude,
                    "RSP_Rate": rate
                }).join(peaks)

                # Quality check
                metrics = detect_abnormal_rsp(signals_rsp.copy(), info, sampling_rate)
                metrics.update({
                    "Participant": participant,
                    "File": file,
                    "RSP_ChannelIndex": idx,
                    "ChannelName": ch.name
                })
                append_log(pd.DataFrame([metrics]))

                # Optional figure
                if figure:
                    nk.rsp_plot(signals_rsp, info)
                    condition = file.split("_")[1].split(".")[0] if "_" in file else "Unknown"
                    fig_name = f"{participant}_{condition}_Ch{idx}.png"
                    fig_path = os.path.join(log_dir, "figs", fig_name)
                    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                    plt.savefig(fig_path)
                    plt.close()

                # Cleanup
                del rsp_signal, rsp_cleaned, peaks, info, amplitude, rate, signals_rsp
                gc.collect()

        except Exception as e:
            print(f"❌ Error with {file}: {e}")
            continue
