# behavior_utils.py
# ---------------------------------------------------------
# Behavior feature 전처리용 기본 함수 모음
#  - 모든 함수는 "이미 정렬된 per-window 시계열"을 입력으로 받아
#    1개의 윈도우에 대한 scalar feature들을 dict로 반환
#  - Main/Agent DataFrame에서 이 시계열을 뽑아오는 부분은
#    나중에 별도 함수에서 처리 (여긴 pure numpy/pandas 유틸)
# ---------------------------------------------------------

from typing import Dict, Sequence, Optional, Tuple
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 0. 공통 설정: 거리 zone
# ---------------------------------------------------------

PERSONAL_ZONES_DEFAULT = {
    "intimate": 0.45,
    "personal": 1.20,
    "social":   3.50,
    "public":   7.60,
}


# ---------------------------------------------------------
# 1. Distance 기반 기본 유틸
#    - 입력: distances: (T, A) or (T,)  [m 단위]
# ---------------------------------------------------------

def _as_2d(arr: np.ndarray) -> np.ndarray:
    """(T,), (T,1)을 모두 (T, A)로 강제."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def summarize_series_basic(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    1D 시계열 x에 대해 mean/std/min/max/median 계산.
    NaN은 자동으로 무시.
    """
    x = np.asarray(x, dtype=float)
    feats = {}

    if np.all(np.isnan(x)):
        feats[f"{prefix}_mean"]   = np.nan
        feats[f"{prefix}_std"]    = np.nan
        feats[f"{prefix}_min"]    = np.nan
        feats[f"{prefix}_max"]    = np.nan
        feats[f"{prefix}_median"] = np.nan
        return feats

    feats[f"{prefix}_mean"]   = float(np.nanmean(x))
    feats[f"{prefix}_std"]    = float(np.nanstd(x))
    feats[f"{prefix}_min"]    = float(np.nanmin(x))
    feats[f"{prefix}_max"]    = float(np.nanmax(x))
    feats[f"{prefix}_median"] = float(np.nanmedian(x))
    return feats


def compute_distance_features(
    distances: np.ndarray,
    *,
    zones: Dict[str, float] = PERSONAL_ZONES_DEFAULT,
    clip_max: Optional[float] = None,
):
    """
    [Distance Features]
    - distances: shape (T, A) or (T,)
        → t 시점에서 각 agent까지의 유클리드 거리 [m]
        → NaN이면 해당 frame/agent는 무시
    
    출력 (예시):
        - dist_min
        - dist_min_clipped
        - dist_mean_clipped
        - count_intimate / count_personal / ...
        - prop_intimate / ...
    """
    D = _as_2d(distances)  # (T, A)
    feats: Dict[str, float] = {}

    # --- (1) 프레임별 최소 거리 ---
    d_min = np.nanmin(D, axis=1)  # (T,)
    feats.update(summarize_series_basic(d_min, "dist_min"))

    # --- (2) clipping (public 영역 이상은 동일하게 취급) ---
    if clip_max is None:
        clip_max = zones.get("public", None)
    if clip_max is not None:
        d_min_clip = np.minimum(d_min, clip_max)
        feats.update(summarize_series_basic(d_min_clip, "dist_min_clip"))

    # --- (3) zone별 count / 비율 (프레임 × agent 기준) ---
    # zones: {"intimate":0.45, "personal":1.2, ...}
    #   → 각 zone 이하인 sample 수 (T×A 기준)
    for name, thr in zones.items():
        m_zone = (D <= thr)
        count = int(np.nansum(m_zone))
        total = int(np.isfinite(D).sum())
        prop = float(count / total) if total > 0 else np.nan
        feats[f"count_{name}"] = count
        feats[f"prop_{name}"] = prop

    return feats


def compute_distance_dynamics_features(
    distances: np.ndarray,
    *,
    dt: float,
    clip_max: Optional[float] = None,
):
    """
    [Distance Dynamics]
    - distances: (T, A) or (T,)
    - d_min(t)의 1차 차분을 이용해 '접근/이탈 속도' 특징 추출.

    출력:
        - dmin_slope_mean  (음수면 평균적으로 다가오는 경향)
        - dmin_slope_std
        - dmin_slope_min
        - dmin_slope_max
    """
    D = _as_2d(distances)
    d_min = np.nanmin(D, axis=1)  # (T,)

    if clip_max is not None:
        d_min = np.minimum(d_min, clip_max)

    # 차분 (T-1,)
    diff = np.diff(d_min) / float(dt)
    feats = summarize_series_basic(diff, "dmin_slope")
    return feats


# ---------------------------------------------------------
# 2. Intention 기반 유틸
#    - 입력: intentions: (T, A) array-like of str/int
#    - 예: "direct", "inside", "unrelated"
# ---------------------------------------------------------

def compute_intention_features(
    intentions: Sequence[Sequence],
    *,
    labels: Sequence[str] = ("direct", "inside", "unrelated"),
):
    """
    [Intention Features]
    - intentions: (T, A) 형태의 리스트/ndarray or DataFrame
        각 원소는 intention label (str or int)
        예: "direct", "inside", "unrelated", None/np.nan 등
    - labels: 집계할 label 목록

    출력:
        - count_intent_<label>
        - prop_intent_<label>
    """
    arr = np.asarray(intentions, dtype=object)
    feats: Dict[str, float] = {}

    valid_mask = pd.notna(arr)
    total = int(valid_mask.sum())

    for lab in labels:
        m = (arr == lab)
        count = int((m & valid_mask).sum())
        prop = float(count / total) if total > 0 else np.nan
        safe_lab = str(lab).replace(" ", "_")
        feats[f"count_intent_{safe_lab}"] = count
        feats[f"prop_intent_{safe_lab}"] = prop

    return feats


# ---------------------------------------------------------
# 3. Self-motion (플레이어 이동/회전)
#    - 속도/가속도 시계열에서 summary 추출
# ---------------------------------------------------------

def compute_self_motion_features(
    player_speed: Optional[np.ndarray] = None,
    player_accel: Optional[np.ndarray] = None,
    head_rot_speed: Optional[np.ndarray] = None,
):
    """
    [Self-motion Features]
    - player_speed: (T,)  선속도 [m/s] (없으면 None)
    - player_accel: (T,)  가속도 근사 [m/s^2] (옵션)
    - head_rot_speed: (T,) 머리 yaw 속도 [deg/s] 등

    입력이 None인 경우 해당 feature는 생성하지 않음.
    """
    feats: Dict[str, float] = {}

    if player_speed is not None:
        feats.update(summarize_series_basic(player_speed, "player_speed"))

    if player_accel is not None:
        feats.update(summarize_series_basic(player_accel, "player_accel"))

    if head_rot_speed is not None:
        feats.update(summarize_series_basic(head_rot_speed, "head_rot_speed"))

    return feats


# ---------------------------------------------------------
# 4. Gaze 기반 유틸 (시선 좌표로부터 displacement/jitter/entropy)
#    - gaze_xy: (T, 2) [시선 포인트 스크린/월드 좌표]
# ---------------------------------------------------------

def compute_gaze_features(
    gaze_xy: np.ndarray,
    *,
    n_bins: int = 16,
):
    """
    [Gaze Features]
    - gaze_xy: (T, 2) (예: 시선 방향 벡터의 x,z, 혹은 스크린 좌표)
    
    출력:
        - gaze_disp_mean / std : 프레임 간 이동량 크기
        - gaze_jitter_std      : 고주파 noise 정도 (diff의 std)
        - gaze_var_x / gaze_var_y
        - gaze_entropy         : 2D hist 기반 Shannon entropy (bit 단위)
    """
    G = np.asarray(gaze_xy, dtype=float)
    feats: Dict[str, float] = {}

    if G.ndim != 2 or G.shape[1] != 2:
        # 형태 안 맞으면 모두 NaN
        feats["gaze_disp_mean"] = np.nan
        feats["gaze_disp_std"] = np.nan
        feats["gaze_jitter_std"] = np.nan
        feats["gaze_var_x"] = np.nan
        feats["gaze_var_y"] = np.nan
        feats["gaze_entropy"] = np.nan
        return feats

    # --- (1) displacement magnitude ---
    dG = np.diff(G, axis=0)       # (T-1, 2)
    disp = np.linalg.norm(dG, axis=1)  # (T-1,)
    feats.update(summarize_series_basic(disp, "gaze_disp"))

    # jitter: displacement의 std 정도로 정의
    feats["gaze_jitter_std"] = float(np.nanstd(disp)) if disp.size > 0 else np.nan

    # --- (2) variance in each axis ---
    feats["gaze_var_x"] = float(np.nanvar(G[:, 0])) if G.size > 0 else np.nan
    feats["gaze_var_y"] = float(np.nanvar(G[:, 1])) if G.size > 0 else np.nan

    # --- (3) entropy (rough 2D occupancy) ---
    try:
        # 2D histogram
        H, _, _ = np.histogram2d(
            G[:, 0], G[:, 1],
            bins=n_bins
        )
        p = H.astype(float)
        p /= p.sum() if p.sum() > 0 else 1.0
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum()) if p.size > 0 else np.nan
    except Exception:
        entropy = np.nan

    feats["gaze_entropy"] = entropy
    return feats


# ---------------------------------------------------------
# 5. Time-to-collision (TTC) 관련 유틸 (옵션)
#    - 상대 위치/속도로부터 최소 TTC 계산
# ---------------------------------------------------------

def compute_ttc_min(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    *,
    radius: float = 0.8,
    ttc_max: float = 10.0,
):
    """
    [TTC Features]
    - rel_pos: (T, A, 2)   # agent - player 상대 위치 (x,z)
    - rel_vel: (T, A, 2)   # agent - player 상대 속도
    - radius: '충돌'로 간주할 거리 (예: personal zone or 그 근처)
    - ttc_max: 이보다 크면 '충돌 없음' 으로 간주 (clipping)

    간단한 선형 모델:
        || p_rel + v_rel * t || = radius
    를 만족하는 t >= 0 중 최소값을 프레임마다 계산,
    그 중 최소/평균/분산 등을 feature로 반환.
    """
    P = np.asarray(rel_pos, dtype=float)
    V = np.asarray(rel_vel, dtype=float)
    feats: Dict[str, float] = {}

    if P.ndim != 3 or V.ndim != 3 or P.shape != V.shape:
        feats["ttc_min"] = np.nan
        feats["ttc_mean"] = np.nan
        feats["ttc_std"] = np.nan
        return feats

    T, A, _ = P.shape
    ttc_list = []

    for t in range(T):
        for a in range(A):
            p = P[t, a]
            v = V[t, a]
            if np.any(np.isnan(p)) or np.any(np.isnan(v)):
                continue

            # 상대 속도가 거의 0이면 TTC 정의 안 함
            v_norm2 = float(np.dot(v, v))
            if v_norm2 < 1e-8:
                continue

            # ||p + v*t||^2 = r^2 -> (v·v)t^2 + 2(p·v)t + (p·p - r^2) = 0
            pv = float(np.dot(p, v))
            pp = float(np.dot(p, p)) - radius**2
            a_q = v_norm2
            b_q = 2.0 * pv
            c_q = pp

            disc = b_q**2 - 4*a_q*c_q
            if disc <= 0:
                continue

            sqrt_disc = np.sqrt(disc)
            t1 = (-b_q - sqrt_disc) / (2*a_q)
            t2 = (-b_q + sqrt_disc) / (2*a_q)

            # t >= 0 중 최소값
            candidates = [t for t in (t1, t2) if t >= 0]
            if not candidates:
                continue

            tt = min(candidates)
            if tt <= ttc_max:
                ttc_list.append(tt)

    if len(ttc_list) == 0:
        feats["ttc_min"] = np.nan
        feats["ttc_mean"] = np.nan
        feats["ttc_std"] = np.nan
    else:
        arr = np.array(ttc_list, dtype=float)
        feats["ttc_min"] = float(np.nanmin(arr))
        feats["ttc_mean"] = float(np.nanmean(arr))
        feats["ttc_std"] = float(np.nanstd(arr))

    return feats


# ---------------------------------------------------------
# 6. 윈도우 단위 behavior feature 종합 함수 (wrapper)
#    - 실제 behavior_features에서는 이 함수를 호출해서
#      window별 feature dict를 만든 다음, 컬럼으로 펼치면 됨
# ---------------------------------------------------------

def summarize_behavior_window(
    *,
    distances: Optional[np.ndarray] = None,        # (T, A) or (T,)
    intentions: Optional[Sequence[Sequence]] = None,  # (T, A)
    player_speed: Optional[np.ndarray] = None,    # (T,)
    player_accel: Optional[np.ndarray] = None,    # (T,)
    head_rot_speed: Optional[np.ndarray] = None,  # (T,)
    gaze_xy: Optional[np.ndarray] = None,         # (T, 2)
    rel_pos: Optional[np.ndarray] = None,         # (T, A, 2)  for TTC
    rel_vel: Optional[np.ndarray] = None,         # (T, A, 2)
    dt: float = 1/60.0,
    zones: Dict[str, float] = PERSONAL_ZONES_DEFAULT,
    clip_max_distance: Optional[float] = None,
    include_ttc: bool = False,
):
    """
    모든 behavior 시계열을 한 윈도우에 대해 한번에 요약하는 wrapper.
    - 필요한 항목만 넘기면, 없는 항목 관련 feature는 생성하지 않는다.
    """
    feats: Dict[str, float] = {}

    if distances is not None:
        feats.update(
            compute_distance_features(
                distances, zones=zones, clip_max=clip_max_distance
            )
        )
        feats.update(
            compute_distance_dynamics_features(
                distances, dt=dt, clip_max=clip_max_distance
            )
        )

    if intentions is not None:
        feats.update(compute_intention_features(intentions))

    if (player_speed is not None) or (player_accel is not None) or (head_rot_speed is not None):
        feats.update(
            compute_self_motion_features(
                player_speed=player_speed,
                player_accel=player_accel,
                head_rot_speed=head_rot_speed,
            )
        )

    if gaze_xy is not None:
        feats.update(compute_gaze_features(gaze_xy))

    if include_ttc and (rel_pos is not None) and (rel_vel is not None):
        feats.update(
            compute_ttc_min(
                rel_pos, rel_vel,
                radius=zones.get("personal", 1.2),
            )
        )

    return feats
