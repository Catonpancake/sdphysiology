# ============================================
# behavior_features.py
# --------------------------------------------
# Per-frame → per-window behavior feature 추출
# (Main.pkl + Agent.pkl 기준)
# ============================================

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from behavior_utils import compute_gaze_features

# --------------------------------------------
# 0. 설정: 개인공간/시야각 파라미터
# --------------------------------------------

PERSONAL_ZONES_DEFAULT: Dict[str, float] = {
    "intimate": 0.45,   # 0.45 m
    "personal": 1.20,   # 1.2 m
    "social":   3.60,   # 3.6 m
    "public":   7.60,   # 7.6 m
}

FOV_DEFAULT_DEG = 110.0   # HMD 수평 시야각(대략)


# --------------------------------------------
# 1. Vector / geometry helpers
# --------------------------------------------

def _safe_norm2d(x: np.ndarray, z: np.ndarray, eps: float = 1e-8):
    """2D 벡터 (x,z)의 L2 norm."""
    return np.sqrt(x**2 + z**2 + eps)


def _unit_vectors(x: np.ndarray, z: np.ndarray, eps: float = 1e-8):
    """(x,z) → 단위벡터."""
    r = _safe_norm2d(x, z, eps=eps)
    return x / r, z / r


def _angle_between(vx1: np.ndarray, vz1: np.ndarray,
                   vx2: np.ndarray, vz2: np.ndarray,
                   eps: float = 1e-8):
    """2D 벡터 (vx1,vz1)와 (vx2,vz2) 사이 각도(도 단위)."""
    ux1, uz1 = _unit_vectors(vx1, vz1, eps=eps)
    ux2, uz2 = _unit_vectors(vx2, vz2, eps=eps)
    dot = ux1 * ux2 + uz1 * uz2
    dot = np.clip(dot, -1.0, 1.0)
    ang_rad = np.arccos(dot)
    return np.degrees(ang_rad)


def _finite_diff(x: np.ndarray, dt: float):
    """1차 차분 기반 속도/가속도 계산용 helper."""
    dx = np.diff(x, prepend=x[0])
    return dx / dt


# --------------------------------------------
# 2. Player/Agent per-frame feature 계산
# --------------------------------------------

@dataclass
class ColumnMapping:
    def __init__(self):
        # 공통
        self.scene   = "scene"
        self.frame   = "Frame"

        # Player (Main.pkl)
        self.player_x     = "X_pos"
        self.player_y     = "Y_pos"
        self.player_z     = "Z_pos"
        self.player_y_rot = "Y_rot"   # yaw

        # Agent (Agent.pkl)
        self.agent_id = "AgentName"
        self.agent_x  = "X_pos"
        self.agent_y  = "Y_pos"
        self.agent_z  = "Z_pos"

        # (gaze: Left/Right 시선 방향 벡터)
        #  - Main.pkl에 실제 존재하는 컬럼 이름에 맞게 설정
        self.gazeL_x = "gazeL_X"
        self.gazeL_y = "gazeL_Y"
        self.gazeR_x = "gazeR_X"
        self.gazeR_y = "gazeR_Y"

        # 아래 두 개는 "L/R 평균"을 저장할 가상 컬럼 이름 (실제 컬럼은 함수에서 생성)
        self.gaze_x = "gaze_mean_x"
        self.gaze_y = "gaze_mean_y"


import numpy as np
import pandas as pd

def _augment_behavior_dynamics(
    df: pd.DataFrame,
    cols: "ColumnMapping",
    dt: float,
):
    """
    이미 compute_agent_player_relations에서 만든 per-frame feature들에
    '변화량/패턴' 기반 feature들을 추가해서 반환합니다.

    - speed / accel / head_rot_speed → 변화량(derivative) + 절댓값
    - dist_min / dist_mean / dist_std → 변화량
    - count_* (agents / zones / approach / fov) → 증가/감소량
    - trajectory (X_pos, Z_pos, Y_rot) → 전진/옆걸음/후진 비율
    """
    df = df.copy()

    # --------------------------------------------------
    # 1) speed / accel / 회전 속도 동역학
    # --------------------------------------------------
    if "speed" in df.columns:
        dspeed = df["speed"].diff().fillna(0.0)
        df["speed_diff"] = dspeed
        df["speed_diff_abs"] = dspeed.abs()
        df["speed_sq"] = df["speed"] ** 2

    if "accel" in df.columns:
        df["accel_abs"] = df["accel"].abs()

    if "head_rot_speed" in df.columns:
        drot = df["head_rot_speed"].diff().fillna(0.0)
        df["head_rot_speed_abs"] = df["head_rot_speed"].abs()
        df["head_rot_accel"] = drot
        df["head_rot_accel_abs"] = drot.abs()

    # --------------------------------------------------
    # 2) 거리 동역학 (누가 더 가까워지는지 / 멀어지는지)
    # --------------------------------------------------
    for c in ["dist_min", "dist_mean", "dist_std"]:
        if c in df.columns:
            base = (
                df[c]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
            dval = base.diff().fillna(0.0)
            df[f"{c}_diff"] = dval
            df[f"{c}_diff_abs"] = dval.abs()

    # --------------------------------------------------
    # 3) 인원/zone count 동역학 (증가/감소)
    # --------------------------------------------------
    count_cols = [
        "count_agents",
        "count_fov",
        "count_intimate",
        "count_personal",
        "count_social",
        "count_public",
        "count_approach",
    ]
    for c in count_cols:
        if c in df.columns:
            dc = df[c].diff().fillna(0.0)
            df[f"{c}_diff"] = dc
            # 증가/감소를 분리해서 event-like feature로 사용
            df[f"{c}_inc"] = dc.clip(lower=0)
            df[f"{c}_dec"] = (-dc).clip(lower=0)

    # --------------------------------------------------
    # 4) Trajectory: 전진/옆걸음/후진 비율
    #     - 플레이어의 yaw(Y_rot)를 기준으로 local frame으로 회전
    # --------------------------------------------------
    try:
        x_col = getattr(cols, "x_pos", "X_pos")
        z_col = getattr(cols, "z_pos", "Z_pos")
        yaw_col = getattr(cols, "y_rot", "Y_rot")

        if x_col in df.columns and z_col in df.columns and yaw_col in df.columns:
            dx = df[x_col].diff().fillna(0.0).to_numpy(dtype=float)
            dz = df[z_col].diff().fillna(0.0).to_numpy(dtype=float)
            yaw_rad = np.deg2rad(df[yaw_col].to_numpy(dtype=float))

            # 월드 좌표 → 플레이어 local 좌표 (yaw 기준 회전)
            # local z: forward(+), local x: right(+)
            cos_y = np.cos(-yaw_rad)
            sin_y = np.sin(-yaw_rad)

            fwd = dx * sin_y + dz * cos_y      # forward/backward
            side = dx * cos_y - dz * sin_y     # left/right

            fwd_series = pd.Series(fwd, index=df.index)
            side_series = pd.Series(side, index=df.index)

            total = np.sqrt(fwd_series ** 2 + side_series ** 2) + 1e-6

            df["move_forward"] = fwd_series
            df["move_sideways"] = side_series
            df["move_forward_abs"] = fwd_series.abs()
            df["move_sideways_abs"] = side_series.abs()
            df["sideways_ratio"] = (side_series.abs() / total).astype(float)
            # 후진 여부 (freeze / 뒤로 물러나는 반응)
            df["backward_flag"] = (fwd_series < 0).astype(float)
    except Exception:
        # 좌표/각도 컬럼이 없으면 그냥 무시
        pass

    return df



def compute_player_heading(main_df: pd.DataFrame,
                           cols: ColumnMapping):
    """
    Player의 정면 unit vector (fx,fz)를 per-frame으로 계산.
    (Y_rot : degree, 0°가 정면이라고 가정)
    """
    df = main_df.copy()
    yaw_deg = df[cols.player_y_rot].to_numpy(dtype=float)
    yaw_rad = np.deg2rad(yaw_deg)
    df["front_x"] = np.cos(yaw_rad)
    df["front_z"] = np.sin(yaw_rad)
    return df


# def compute_agent_player_relations(
#     main_df: pd.DataFrame,
#     agent_df: pd.DataFrame,
#     cols: ColumnMapping | None = None,
#     zones: dict[str, float] = None,
#     fov_deg: float = 110.0,
#     dt: float = 1.0 / 120.0,
#     elevator_scenes: tuple[str, str] = ("Elevator1", "Elevator2"),
#     floor_dy_thresh: float = 2.0,
# ):
#     """
#     Main + Agent를 합쳐서 per-frame behavior feature time-series를 계산.

#     반환 컬럼(예시)
#     - scene, Frame
#     - X_pos, Z_pos, Y_rot
#     - speed, accel, head_rot_speed
#     - dist_min, dist_mean, dist_std  (반경 내 valid agent 기준, 없으면 7.6으로 채움)
#     - count_agents                   (반경 내 + 같은 층인 agent 수)
#     - count_fov                      (FOV 안에 있는 valid agent 수)
#     - count_approach                 (접근중인 valid agent 수)
#     - count_intimate / personal / social / public  (서로 배타적인 zone bin)
#     """
#     if cols is None:
#         cols = ColumnMapping()
#     if zones is None:
#         zones = PERSONAL_ZONES_DEFAULT

#     # -------------------------
#     # 1) Player heading & kinematics
#     # -------------------------
#     main_h = compute_player_heading(main_df, cols)

#     # agent 정보가 없으면 player 정보만 리턴
#     if agent_df is None or agent_df.empty:
#         return _compute_player_only_timeseries(main_h, cols, dt=dt)

#     # -------------------------
#     # 2) Agent dataframe 정리
#     # -------------------------
#     # 필요한 컬럼만 사용 (없으면 NaN으로 채움)
#     a_cols = [cols.scene, cols.frame, cols.agent_id, cols.agent_x, cols.agent_z]
#     if cols.agent_y in agent_df.columns:
#         a_cols.append(cols.agent_y)

#     a = agent_df[a_cols].copy()

#     # ✅ Player_VR / HeadCollider / EyetrackerRecording 은 진짜 agent가 아님
#     IGNORE_AGENT_NAMES = {"Player_VR", "HeadCollider", "EyetrackerRecording", "EyetrackerRecorder"}
#     if cols.agent_id in a.columns:
#         a = a[~a[cols.agent_id].isin(IGNORE_AGENT_NAMES)].copy()

#     # -------------------------
#     # 3) Main + Agent merge (scene, Frame 기준)
#     # -------------------------
#     m_cols = [cols.scene, cols.frame,
#               cols.player_x, cols.player_z, cols.player_y,
#               cols.player_y_rot, "front_x", "front_z"]

#     m = main_h[m_cols].copy()

#     merged = pd.merge(
#         m,
#         a,
#         how="left",
#         left_on=[cols.scene, cols.frame],
#         right_on=[cols.scene, cols.frame],
#         suffixes=("", "_agent"),
#     )

#     # -------------------------
#     # 4) Agent→Player 벡터, 거리
#     # -------------------------
#     dx = merged[cols.player_x] - merged[cols.agent_x]
#     dz = merged[cols.player_z] - merged[cols.agent_z]

#     dist = _safe_norm2d(dx.to_numpy(float), dz.to_numpy(float))  # shape (N_rows,)
#     merged["dist_raw"] = dist

#     # -------------------------
#     # 5) Elevator만 층(Y) 필터 적용
#     # -------------------------
#     if (cols.player_y in merged.columns) and (cols.agent_y in merged.columns):
#         dy = (merged[cols.player_y] - merged[cols.agent_y]).abs()
#         is_elev = merged[cols.scene].isin(elevator_scenes)
#         invalid_floor = is_elev & dy.gt(floor_dy_thresh)
#     else:
#         invalid_floor = np.zeros(len(merged), dtype=bool)

#     merged["invalid_floor"] = invalid_floor

#     # -------------------------
#     # 6) 최대 반경(=public zone) 기준으로 valid agent 정의
#     # -------------------------
#     # zone 값들 중 가장 큰 반경 (보통 7.6)
#     max_public = max(zones.values())

#     has_agent = merged[cols.agent_id].notna() & ~merged[cols.agent_id].isin(IGNORE_AGENT_NAMES)

#     # "유효한" agent: 반경 이내 + (Elevator면 같은 층) + 실제 agent row
#     valid_agent = has_agent & (~invalid_floor) & (dist <= max_public)


#     merged["valid_agent"] = valid_agent

#     # -------------------------
#     # 7) 거리 통계 (valid agent 기준)
#     #    - agent가 하나도 없으면 dist_min/mean= max_public, dist_std=0으로 채움
#     # -------------------------
#     dist_for_stats = np.where(valid_agent, dist, np.nan)
#     merged["dist_for_stats"] = dist_for_stats

#     g = merged.groupby([cols.scene, cols.frame], sort=False)

#     df_dist = g["dist_for_stats"].agg(["min", "mean", "std"]).reset_index()
#     df_dist.rename(
#         columns={"min": "dist_min", "mean": "dist_mean", "std": "dist_std"},
#         inplace=True,
#     )

#     # NaN (유효 agent 없음) → 클리핑 반경으로 채움
#     df_dist["dist_min"] = df_dist["dist_min"].fillna(max_public)
#     df_dist["dist_mean"] = df_dist["dist_mean"].fillna(max_public)
#     df_dist["dist_std"] = df_dist["dist_std"].fillna(0.0)

#     # -------------------------
#     # 8) count_agents (valid agent 수)
#     # -------------------------
#     df_cnt = g["valid_agent"].sum().reset_index(name="count_agents")

#     # -------------------------
#     # 9) FOV / approaching 계산 (valid agent에 대해서만)
#     # -------------------------
#     # agent→player unit vector
#     # (dx, dz는 player-agent, 우리는 agent->player가 필요해서 부호 반전)
#     vx = -dx.to_numpy(float)
#     vz = -dz.to_numpy(float)
#     v_norm = _safe_norm2d(vx, vz)
#     vx_unit = np.divide(vx, v_norm, out=np.zeros_like(vx), where=v_norm > 0)
#     vz_unit = np.divide(vz, v_norm, out=np.zeros_like(vz), where=v_norm > 0)

#     # FOV: front vector와의 각도
#     fx = merged["front_x"].to_numpy(float)
#     fz = merged["front_z"].to_numpy(float)
#     f_norm = _safe_norm2d(fx, fz)
#     fx_unit = np.divide(fx, f_norm, out=np.zeros_like(fx), where=f_norm > 0)
#     fz_unit = np.divide(fz, f_norm, out=np.zeros_like(fz), where=f_norm > 0)

#     dot = fx_unit * vx_unit + fz_unit * vz_unit
#     dot = np.clip(dot, -1.0, 1.0)
#     ang = np.degrees(np.arccos(dot))

#     in_fov = valid_agent & (ang <= (fov_deg / 2.0))

#     merged["in_fov"] = in_fov

#     # approaching: agent가 player 쪽으로 다가오는지 (간단한 기준)
#     # 여기선 frame-wise diff로 거리 감소 여부만 사용
#     dist_series = pd.Series(dist, index=merged.index)
#     dist_diff = dist_series.groupby([merged[cols.scene], merged[cols.frame]]).diff()
#     # 음수면 가까워지고 있는 방향 (valid agent만 고려)
#     approaching = valid_agent & (dist_diff < 0)
#     merged["approach"] = approaching

#     df_fov = g["in_fov"].sum().reset_index(name="count_fov")
#     df_app = g["approach"].sum().reset_index(name="count_approach")

#     # -------------------------
#     # 10) zone별 count (intimate/personal/social/public, 서로 배타적)
#     # -------------------------
#     # zone은 거리 구간을 누적이 아니라 disjoint하게 나눔
#     # 예: [0,0.45), [0.45,1.2), [1.2,3.6), [3.6,7.6]
#     zone_defs = sorted(zones.items(), key=lambda x: x[1])  # radius 기준 정렬
#     prev_r = 0.0

#     zone_frames = []

#     for z_name, r in zone_defs:
#         mask = valid_agent & (dist > prev_r) & (dist <= r)
#         col_name = f"zflag_{z_name}"
#         merged[col_name] = mask
#         zone_frames.append(col_name)
#         prev_r = r

#     df_zone = g[zone_frames].sum().reset_index()
#     # 이름을 count_XXX로 변경
#     rename_dict = {col: f"count_{col.replace('zflag_', '')}" for col in zone_frames}
#     df_zone.rename(columns=rename_dict, inplace=True)

#     # -------------------------
#     # 11) player kinematics (speed, accel, head_rot_speed)
#     # -------------------------
#     df_player = _compute_player_only_timeseries(main_h, cols, dt=dt)

#     # -------------------------
#     # 12) 최종 merge
#     # -------------------------
#     df = df_player.merge(df_dist, on=[cols.scene, cols.frame], how="left")
#     df = df.merge(df_cnt, on=[cols.scene, cols.frame], how="left")
#     df = df.merge(df_fov, on=[cols.scene, cols.frame], how="left")
#     df = df.merge(df_app, on=[cols.scene, cols.frame], how="left")
#     df = df.merge(df_zone, on=[cols.scene, cols.frame], how="left")

#     # NaN count는 0으로
#     count_cols = [c for c in df.columns if c.startswith("count_")]
#     df[count_cols] = df[count_cols].fillna(0).astype(int)
    
#     df = _augment_behavior_dynamics(df, cols=cols, dt=dt)

#     # -----------------------------------------
#     # ⭐ NEW: Main에 있는 gaze(L/R) → 평균 gaze(x,y) 붙이기
#     # -----------------------------------------
#     try:
#         has_gaze = (
#             (cols.gazeL_x in main_df.columns)
#             and (cols.gazeL_y in main_df.columns)
#             and (cols.gazeR_x in main_df.columns)
#             and (cols.gazeR_y in main_df.columns)
#         )
#     except AttributeError:
#         has_gaze = False

#     if has_gaze:
#         gaze = main_df[[cols.scene, cols.frame,
#                         cols.gazeL_x, cols.gazeL_y,
#                         cols.gazeR_x, cols.gazeR_y]].copy()

#         # L/R 평균 → gaze_mean_x / gaze_mean_y
#         gaze[cols.gaze_x] = gaze[[cols.gazeL_x, cols.gazeR_x]].mean(axis=1)
#         gaze[cols.gaze_y] = gaze[[cols.gazeL_y, cols.gazeR_y]].mean(axis=1)

#         gaze = gaze[[cols.scene, cols.frame, cols.gaze_x, cols.gaze_y]]

#         # per-frame behavior TS(df)에 merge
#         df = df.merge(gaze, on=[cols.scene, cols.frame], how="left")

#     return df
def compute_agent_player_relations(
    main_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    cols: ColumnMapping | None = None,
    zones: dict[str, float] = None,
    fov_deg: float = 110.0,
    dt: float = 1.0 / 120.0,
    elevator_scenes: tuple[str, str] = ("Elevator1", "Elevator2"),
    floor_dy_thresh: float = 2.0,
):
    """
    Main + Agent를 합쳐서 per-frame behavior feature time-series를 계산.

    반환 컬럼(예시)
    - scene, Frame
    - X_pos, Z_pos, Y_rot
    - speed, accel, head_rot_speed
    - dist_min, dist_mean, dist_std  (반경 내 valid agent 기준, 없으면 max_public(보통 7.6)으로 채움)
    - count_agents                   (반경 내 + 같은 층인 agent 수)
    - count_fov                      (FOV 안에 있는 valid agent 수)
    - count_approach                 (접근중인 valid agent 수)
    - count_intimate / personal / social / public  (서로 배타적인 zone bin)
    """
    if cols is None:
        cols = ColumnMapping()
    if zones is None:
        zones = PERSONAL_ZONES_DEFAULT

    # -------------------------
    # 1) Player heading & kinematics
    # -------------------------
    main_h = compute_player_heading(main_df, cols)

    # agent 정보가 없으면 player 정보만 리턴
    if agent_df is None or agent_df.empty:
        return _compute_player_only_timeseries(main_h, cols, dt=dt)

    # -------------------------
    # 2) Agent dataframe 정리
    # -------------------------
    a_cols = [cols.scene, cols.frame, cols.agent_id, cols.agent_x, cols.agent_z]
    if cols.agent_y in agent_df.columns:
        a_cols.append(cols.agent_y)

    a = agent_df[a_cols].copy()

    # Player_VR / HeadCollider / EyetrackerRecording 은 진짜 agent가 아님
    IGNORE_AGENT_NAMES = {"Player_VR", "HeadCollider", "EyetrackerRecording", "EyetrackerRecorder"}
    if cols.agent_id in a.columns:
        a = a[~a[cols.agent_id].isin(IGNORE_AGENT_NAMES)].copy()

    # -------------------------
    # 3) Main + Agent merge (scene, Frame 기준)
    # -------------------------
    m_cols = [cols.scene, cols.frame,
              cols.player_x, cols.player_z, cols.player_y,
              cols.player_y_rot, "front_x", "front_z"]

    m = main_h[m_cols].copy()

    merged = pd.merge(
        m,
        a,
        how="left",
        left_on=[cols.scene, cols.frame],
        right_on=[cols.scene, cols.frame],
        suffixes=("", "_agent"),
    )

    # -------------------------
    # 4) Agent→Player 벡터, 거리
    # -------------------------
    dx = merged[cols.player_x] - merged[cols.agent_x]
    dz = merged[cols.player_z] - merged[cols.agent_z]

    dist = _safe_norm2d(dx.to_numpy(float), dz.to_numpy(float))  # shape (N_rows,)
    merged["dist_raw"] = dist

    # -------------------------
    # 5) Elevator만 층(Y) 필터 적용
    # -------------------------
    if (cols.player_y in merged.columns) and (cols.agent_y in merged.columns):
        dy = (merged[cols.player_y] - merged[cols.agent_y]).abs()
        is_elev = merged[cols.scene].isin(elevator_scenes)
        invalid_floor = is_elev & dy.gt(floor_dy_thresh)
    else:
        invalid_floor = np.zeros(len(merged), dtype=bool)

    merged["invalid_floor"] = invalid_floor

    # -------------------------
    # 6) 최대 반경(=public zone) 기준으로 valid agent 정의
    # -------------------------
    max_public = max(zones.values())
    # 위에서 IGNORE 이름은 이미 삭제했으므로 notna만 체크하면 됨
    has_agent = merged[cols.agent_id].notna()

    # "유효한" agent: 반경 이내 + (Elevator면 같은 층) + 실제 agent row
    valid_agent = has_agent & (~invalid_floor) & (dist <= max_public)
    merged["valid_agent"] = valid_agent

    # 거리 통계용 값
    dist_for_stats = np.where(valid_agent, dist, np.nan)
    merged["dist_for_stats"] = dist_for_stats

    # -------------------------
    # 7) FOV / approaching 계산 (valid agent에 대해서만)
    # -------------------------
    # agent→player unit vector
    # (dx, dz는 player-agent, 우리는 agent->player가 필요해서 부호 반전)
    vx = -dx.to_numpy(float)
    vz = -dz.to_numpy(float)
    v_norm = _safe_norm2d(vx, vz)
    vx_unit = np.divide(vx, v_norm, out=np.zeros_like(vx), where=v_norm > 0)
    vz_unit = np.divide(vz, v_norm, out=np.zeros_like(vz), where=v_norm > 0)

    # FOV: front vector와의 각도
    fx = merged["front_x"].to_numpy(float)
    fz = merged["front_z"].to_numpy(float)
    f_norm = _safe_norm2d(fx, fz)
    fx_unit = np.divide(fx, f_norm, out=np.zeros_like(fx), where=f_norm > 0)
    fz_unit = np.divide(fz, f_norm, out=np.zeros_like(fz), where=f_norm > 0)

    dot = fx_unit * vx_unit + fz_unit * vz_unit
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))

    in_fov = valid_agent & (ang <= (fov_deg / 2.0))
    merged["in_fov"] = in_fov

    # approaching: agent가 player 쪽으로 다가오는지 (간단한 기준)
    # 기존: dist_series.groupby([scene, frame]).diff()
    # → "같은 (scene, frame) 내에서만 diff"라는 의미이므로
    #    explicit groupby 없이 numpy로 동일 동작 구현

    scene_arr = merged[cols.scene].to_numpy()
    frame_arr = merged[cols.frame].to_numpy()
    dist_arr  = dist  # 이미 np.ndarray

    dist_diff = np.zeros_like(dist_arr, dtype=float)
    dist_diff[:] = np.nan  # 기본값 NaN

    # 이웃한 row가 같은 (scene, frame)이면 diff, 아니면 NaN 유지
    same_group = (scene_arr[1:] == scene_arr[:-1]) & (frame_arr[1:] == frame_arr[:-1])
    dist_diff[1:][same_group] = dist_arr[1:][same_group] - dist_arr[:-1][same_group]

    # 음수면 가까워지고 있는 방향 (valid agent만 고려)
    approaching = valid_agent & (dist_diff < 0)
    merged["approach"] = approaching

    # -------------------------
    # 8) zone별 count (intimate/personal/social/public, 서로 배타적)
    # -------------------------
    zone_defs = sorted(zones.items(), key=lambda x: x[1])  # radius 기준 정렬
    prev_r = 0.0
    zone_frames = []

    for z_name, r in zone_defs:
        mask = valid_agent & (dist > prev_r) & (dist <= r)
        col_name = f"zflag_{z_name}"
        merged[col_name] = mask
        zone_frames.append(col_name)
        prev_r = r

    # -------------------------
    # 9) 단일 groupby + agg (거리 통계 + count + FOV + zone count)
    # -------------------------
    agg_dict = {
        "dist_for_stats": ["min", "mean", "std"],
        "valid_agent": "sum",
        "in_fov": "sum",
        "approach": "sum",
    }
    for zcol in zone_frames:
        agg_dict[zcol] = "sum"

    g = merged.groupby([cols.scene, cols.frame], sort=False)
    agg = g.agg(agg_dict)

    # MultiIndex 컬럼을 flat name으로 변환
    agg.columns = [
        "_".join([c for c in col if c]) for col in agg.columns.to_flat_index()
    ]
    agg = agg.reset_index()

    # 기존 결과와 동일한 컬럼 이름으로 rename
    rename_map = {
        "dist_for_stats_min": "dist_min",
        "dist_for_stats_mean": "dist_mean",
        "dist_for_stats_std": "dist_std",
        "valid_agent_sum": "count_agents",
        "in_fov_sum": "count_fov",
        "approach_sum": "count_approach",
    }
    for zcol in zone_frames:
        rename_map[f"{zcol}_sum"] = f"count_{zcol.replace('zflag_', '')}"

    agg = agg.rename(columns=rename_map)

    # NaN (유효 agent 없음) → 클리핑 반경 / 0으로 채움 (기존 정책 유지)
    agg["dist_min"] = agg["dist_min"].fillna(max_public)
    agg["dist_mean"] = agg["dist_mean"].fillna(max_public)
    agg["dist_std"] = agg["dist_std"].fillna(0.0)

    # -------------------------
    # 10) player kinematics (speed, accel, head_rot_speed)
    # -------------------------
    df_player = _compute_player_only_timeseries(main_h, cols, dt=dt)

    # -------------------------
    # 11) 최종 merge
    # -------------------------
    df = df_player.merge(agg, on=[cols.scene, cols.frame], how="left")

    # NaN count는 0으로
    count_cols = [c for c in df.columns if c.startswith("count_")]
    df[count_cols] = df[count_cols].fillna(0).astype(int)

    # 동역학 파생 feature
    df = _augment_behavior_dynamics(df, cols=cols, dt=dt)

    # -----------------------------------------
    # ⭐ Main에 있는 gaze(L/R) → 평균 gaze(x,y) 붙이기
    # -----------------------------------------
    try:
        has_gaze = (
            (cols.gazeL_x in main_df.columns)
            and (cols.gazeL_y in main_df.columns)
            and (cols.gazeR_x in main_df.columns)
            and (cols.gazeR_y in main_df.columns)
        )
    except AttributeError:
        has_gaze = False

    if has_gaze:
        gaze = main_df[[cols.scene, cols.frame,
                        cols.gazeL_x, cols.gazeL_y,
                        cols.gazeR_x, cols.gazeR_y]].copy()

        # L/R 평균 → gaze_mean_x / gaze_mean_y
        gaze[cols.gaze_x] = gaze[[cols.gazeL_x, cols.gazeR_x]].mean(axis=1)
        gaze[cols.gaze_y] = gaze[[cols.gazeL_y, cols.gazeR_y]].mean(axis=1)

        gaze = gaze[[cols.scene, cols.frame, cols.gaze_x, cols.gaze_y]]

        # per-frame behavior TS(df)에 merge
        df = df.merge(gaze, on=[cols.scene, cols.frame], how="left")

    return df




def _compute_player_only_timeseries(main_h: pd.DataFrame,
                                    cols: ColumnMapping,
                                    dt: float = 1.0 / 120.0):
    """
    Player 하나만 있을 때 self-motion features 계산.
    (position, speed, accel, head_rot_speed)
    """
    df = main_h[[cols.scene, cols.frame, cols.player_x, cols.player_z, cols.player_y_rot]].copy()
    df = df.sort_values(by=[cols.scene, cols.frame]).reset_index(drop=True)

    x = df[cols.player_x].to_numpy(float)
    z = df[cols.player_z].to_numpy(float)
    yaw = df[cols.player_y_rot].to_numpy(float)

    # position → speed / accel
    dist_pos = _safe_norm2d(np.diff(x, prepend=x[0]), np.diff(z, prepend=z[0]))
    speed = dist_pos / dt
    accel = _finite_diff(speed, dt)

    # head rotation speed (deg/s)
    dyaw = np.diff(yaw, prepend=yaw[0])
    # -180~180 범위로 unwrap (optional)
    dyaw = ((dyaw + 180.0) % 360.0) - 180.0
    head_rot_speed = dyaw / dt

    df["speed"] = speed
    df["accel"] = accel
    df["head_rot_speed"] = head_rot_speed

    return df


# --------------------------------------------
# 3. Per-window behavior feature 집계
# --------------------------------------------
def make_behavior_windows(
    df_ts: pd.DataFrame,
    cols: ColumnMapping,
    window_seconds: float,
    stride_seconds: float,
    sampling_rate: float,
    pid_value: str,
    scene_filter: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    ce_df: Optional[pd.DataFrame] = None,   # ⭐ NEW: Customevent 함께 처리
):
    """
    per-frame behavior time series(DataFrame) → window-level feature로 요약.

    - df_ts : Main/Agent/physio 기반 per-frame TS
    - ce_df : 같은 participant의 Customevent TS (scene, Frame 기준)

    반환:
      X_beh   : (N_window, F) behavior+CE+gaze feature
      pid_arr : (N_window,)
      scene_arr : (N_window,)
      widx_arr  : (N_window,)
    """
    # -------------------------
    # 0) scene 필터링
    # -------------------------
    df = df_ts.copy()
    if scene_filter is not None:
        df = df[df[cols.scene].isin(scene_filter)].copy()

    if df.empty:
        return (
            np.zeros((0, 0), dtype=float),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=int),
        )

    # -------------------------
    # 1) Customevent → per-frame feature 붙이기
    # -------------------------
    if ce_df is not None and not ce_df.empty:
        df = _attach_ce_features_framewise(df, ce_df, cols)
    else:
        print("⚠️ make_behavior_windows: ce_df가 없거나 비어있음. Customevent feature는 사용하지 않습니다.")

    # -------------------------
    # 2) Gaze 사용 여부 확인
    #    (cols.gaze_x / gaze_y는 네가 ColumnMapping에서 설정)
    # -------------------------
    use_gaze = False
    if getattr(cols, "gaze_x", None) is not None and getattr(cols, "gaze_y", None) is not None:
        if cols.gaze_x in df.columns and cols.gaze_y in df.columns:
            use_gaze = True

    # -------------------------
    # 3) feature_cols 기본값 설정
    #    - scene / frame / y_cont / raw gaze 컬럼은 제외
    # -------------------------
    if feature_cols is None:
        exclude = {cols.scene, cols.frame, "y_cont"}
        if getattr(cols, "gaze_x", None) is not None:
            exclude.add(cols.gaze_x)
        if getattr(cols, "gaze_y", None) is not None:
            exclude.add(cols.gaze_y)
        feature_cols = [c for c in df.columns if c not in exclude]
    else:
        feature_cols = list(feature_cols)

    win_len = int(window_seconds * sampling_rate)
    hop     = int(stride_seconds * sampling_rate)
    if win_len <= 0 or hop <= 0:
        raise ValueError("window_seconds / stride_seconds / sampling_rate 설정을 확인하세요.")

    X_list: List[np.ndarray] = []
    pid_list: List[str] = []
    scene_list: List[str] = []
    widx_list: List[int] = []

    # gaze feature 이름 (순서 고정)
    gaze_feat_keys = [
        "gaze_disp_mean",
        "gaze_disp_std",
        "gaze_disp_min",
        "gaze_disp_max",
        "gaze_disp_median",
        "gaze_jitter_std",
        "gaze_var_x",
        "gaze_var_y",
        "gaze_entropy",
    ]

    # -------------------------
    # 4) scene별 sliding window
    # -------------------------
    for sc in df[cols.scene].drop_duplicates().tolist():
        df_sc = df[df[cols.scene] == sc].copy()
        if df_sc.empty:
            continue

        df_sc = df_sc.sort_values(cols.frame).reset_index(drop=True)
        n = len(df_sc)

        start = 0
        widx  = 0

        while start + win_len <= n:
            end = start + win_len
            seg = df_sc.iloc[start:end]

            # --- base feature: 지정된 컬럼 평균 ---
            base_vals = seg[feature_cols].to_numpy(dtype=float).mean(axis=0)
            feats = [base_vals]

            # --- gaze feature 추가 ---
            if use_gaze:
                gaze_xy = seg[[cols.gaze_x, cols.gaze_y]].to_numpy(dtype=float)
                gaze_dict = compute_gaze_features(gaze_xy)
                gaze_vals = [float(gaze_dict.get(k, np.nan)) for k in gaze_feat_keys]
                feats.append(np.asarray(gaze_vals, dtype=float))

            x_vec = np.concatenate(feats)

            X_list.append(x_vec)
            pid_list.append(pid_value)
            scene_list.append(sc)
            widx_list.append(widx)

            start += hop
            widx  += 1

    if not X_list:
        F = len(feature_cols) + (len(gaze_feat_keys) if use_gaze else 0)
        return (
            np.zeros((0, F), dtype=float),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=int),
        )

    X_beh    = np.vstack(X_list)
    pid_arr  = np.array(pid_list, dtype=object)
    scene_arr = np.array(scene_list, dtype=object)
    widx_arr  = np.array(widx_list, dtype=int)

    return X_beh, pid_arr, scene_arr, widx_arr



# --------------------------------------------
# 4. QC / 시각화 helpers (ipynb용)
# --------------------------------------------
def plot_behavior_timeseries(
    df_ts: pd.DataFrame,
    cols: ColumnMapping,
    feature_cols,
    max_points: int = 3000,
    title: str = "Behavior time-series (sample)",
):
    # scene, frame 기준 정렬
    df_plot = df_ts.sort_values(by=[cols.scene, cols.frame]).reset_index(drop=True)

    # 🔍 1) agent가 실제로 등장하는 첫 index 찾기
    has_agent = None
    for key in ["count_agents", "count_intimate", "count_personal",
                "count_social", "count_public"]:
        if key in df_plot.columns:
            m = df_plot[key] > 0
            has_agent = m if has_agent is None else (has_agent | m)

    if has_agent is not None and has_agent.any():
        first_idx = int(np.argmax(has_agent.to_numpy()))
    else:
        # 진짜로 전체에 agent가 하나도 없으면 맨 앞에서부터
        first_idx = 0

    # 🔍 2) first_idx ~ first_idx+max_points 범위만 사용
    if max_points is not None and len(df_plot) > max_points:
        end_idx = min(first_idx + max_points, len(df_plot))
        df_plot = df_plot.iloc[first_idx:end_idx].reset_index(drop=True)

    t = np.arange(len(df_plot))

    plt.figure(figsize=(10, 4 + 1.5 * len(feature_cols)))
    for i, f in enumerate(feature_cols):
        ax = plt.subplot(len(feature_cols), 1, i + 1)
        ax.plot(t, df_plot[f].to_numpy(dtype=float))
        ax.set_ylabel(f)
        if i == 0:
            ax.set_title(title)
    plt.xlabel("Frame index (sample)")
    plt.tight_layout()
    plt.show()

# def plot_behavior_timeseries(
#     df_ts: pd.DataFrame,
#     cols: ColumnMapping,
#     feature_cols: Sequence[str],
#     max_points: int = 3000,
#     title: str = "Behavior time-series (sample)",
# ):
#     """
#     특정 feature들의 per-frame time series를 간단히 라인플롯으로 확인.
#     (max_points 이후는 잘라서 속도 확보)
#     """
#     df_plot = df_ts.sort_values(by=[cols.scene, cols.frame]).reset_index(drop=True)
#     if len(df_plot) > max_points:
#         df_plot = df_plot.iloc[:max_points]

#     t = np.arange(len(df_plot))

#     plt.figure(figsize=(10, 4 + 1.5 * len(feature_cols)))
#     for i, f in enumerate(feature_cols):
#         ax = plt.subplot(len(feature_cols), 1, i + 1)
#         ax.plot(t, df_plot[f].to_numpy(dtype=float))
#         ax.set_ylabel(f)
#         if i == 0:
#             ax.set_title(title)
#     plt.xlabel("Frame index (sample)")
#     plt.tight_layout()
#     plt.show()


def plot_behavior_histograms(
    df_ts: pd.DataFrame,
    feature_cols: Sequence[str],
    bins: int = 50,
    title: str = "Behavior feature histograms",
):
    """
    feature 분포를 빠르게 확인하기 위한 히스토그램 플롯.
    """
    k = len(feature_cols)
    n_cols = min(3, k)
    n_rows = int(math.ceil(k / n_cols))

    plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    for i, f in enumerate(feature_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        vals = df_ts[f].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins)
        ax.set_title(f)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd

def filter_agent_records(df, cols,
                         max_dist=7.6,
                         y_diff_thresh=2.0,
                         require_same_floor=True,
                         floor_col="Floor",   # Optional
                         default_floor=None):
    """
    Main_df와 Agent_df merge 후,
    agent 관련 잘못된 기록을 필터링/정제하는 a robust filter.

    Parameters
    ----------
    df : merged DataFrame (scene, Frame 기준으로 merge된 상태)
    cols : ColumnMapping()
    max_dist : float
        Distance > max_dist → clip or mark as no-agent
    y_diff_thresh : float
        Player Y_pos와 Agent Y_pos 차이가 이 값보다 크면 → 다른 층
    require_same_floor : bool
        True이면 floor 정보가 있으면 반드시 match해야 함
    floor_col : str
        Customevent or Main에서 생성한 Floor 정보 컬럼명
    default_floor : Optional[int]
        Floor 정보가 없다면 y_pos 기반으로 간단한 floor 추정 가능
    """

    out = df.copy()

    # --- 1) distance 계산 (평면 distance) ---
    dx = out[cols.player_x] - out[cols.agent_x]
    dz = out[cols.player_z] - out[cols.agent_z]
    dist = np.sqrt(dx**2 + dz**2)

    # clip
    dist_clipped = np.minimum(dist, max_dist)
    out["dist"] = dist_clipped

    # --- 2) 층(y) 필터링 ---
    if cols.player_y and cols.agent_y in out.columns:
        dy = np.abs(out[cols.player_y] - out[cols.agent_y])
        out.loc[dy > y_diff_thresh, "dist"] = max_dist
        out["y_mismatch"] = (dy > y_diff_thresh).astype(int)
    else:
        out["y_mismatch"] = 0

    # --- 3) Floor matching ---
    if require_same_floor:
        if floor_col in out.columns:
            out["floor_mismatch"] = (out[floor_col] != out[floor_col + "_agent"]).astype(int)
            out.loc[out["floor_mismatch"] == 1, "dist"] = max_dist
        else:
            # Floor 정보가 없다면 y_pos 기반 pseudo-floor 추정
            if default_floor is not None:
                out["floor_mismatch"] = (default_floor != default_floor).astype(int)
            else:
                out["floor_mismatch"] = 0

    # --- 4) No-agent 처리: dist==max_dist 라면 count=0 ---
    out["valid_agent"] = (out["dist"] < max_dist).astype(int)

    return out

import os
_CROWD_MAP_PATH = os.path.join(os.path.dirname(__file__), "elevator_crowd_mapping.csv")
try:
    ELEVATOR_CROWD_MAP = pd.read_csv(_CROWD_MAP_PATH)
except FileNotFoundError:
    ELEVATOR_CROWD_MAP = None
    
    
def _attach_ce_features_framewise(
    df_ts: pd.DataFrame,
    ce_df: pd.DataFrame,
    cols: ColumnMapping,
    *,
    goal_name: str = "GoalZone",
    player_keyword: str = "Player",
):
    """
    df_ts (Main+Agent 기반 per-frame TS)에 Customevent 정보를
    per-frame feature로 붙인다.

    - CE_goal_visible : 해당 scene에서 GoalZone Frame 이후면 1, 그 전은 0
    - CE_floor        : Player의 CurrentFloor 이벤트를 Frame 기준으로 backward-fill
    """
    
    # 🔹 엘리베이터 crowd 매핑 CSV 로드

    df = df_ts.copy()

    # 기본값 초기화
    df["CE_goal_visible"] = 0.0
    # df["CE_floor"] = np.nan

    if ce_df is None or ce_df.empty:
        return df

    ce = ce_df.copy()

    # 컬럼 정리
    if "Name" in ce.columns:
        ce["Name"] = ce["Name"].astype(str).str.strip()
    else:
        ce["Name"] = ""

    if "EventType" in ce.columns:
        ce["EventType"] = ce["EventType"].astype(str).str.strip()
    else:
        ce["EventType"] = ""

    if "Frame" not in ce.columns:
        # Frame 없으면 붙일 수 없음
        return df

    ce["Frame"] = pd.to_numeric(ce["Frame"], errors="coerce")
    ce = ce.dropna(subset=["Frame"])

    if "Frame" not in df.columns:
        raise ValueError("df_ts에 Frame 컬럼이 없어 CE를 정렬할 수 없습니다.")

    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")

    # scene별로 처리
    for sc in df[cols.scene].drop_duplicates().tolist():
        idx_ts = (df[cols.scene] == sc)
        sub_ts = df.loc[idx_ts].copy()
        if sub_ts.empty:
            continue

        sub_ce = ce[ce[cols.scene] == sc].copy()
        if sub_ce.empty:
            continue

        # ----------------------------
        # 1) GoalZone 시간 → CE_goal_visible
        #    - Name 좌우 공백 제거 후 검색
        # ----------------------------
        name_stripped = sub_ce["EventType"].astype(str).str.strip()
        goal_rows = sub_ce[name_stripped.str.contains(goal_name, na=False)]

        if not goal_rows.empty:
            goal_frame = float(goal_rows["Frame"].min())
            mask_goal = (idx_ts & (df["Frame"] >= goal_frame))
            df.loc[mask_goal, "CE_goal_visible"] = 1.0

    #     # ----------------------------
    #     # 2) Player CurrentFloor → CE_floor
    #     # ----------------------------
    #     floor_rows = sub_ce[
    #         (sub_ce["EventType"] == "CurrentFloor")
    #         & (sub_ce["Name"].astype(str).str.strip().str.contains(player_keyword, na=False))
    #     ].copy()


    #     if not floor_rows.empty:
    #         # 층수 파싱
    #         def _parse_floor(row):
    #             for col in ["customevent_col_4", "customevent_col_5",
    #                         "customevent_col_6", "customevent_col_7"]:
    #                 if col not in floor_rows.columns:
    #                     continue
    #                 s = str(row.get(col, "")).strip()
    #                 # 숫자만 뽑기 (예: "Floor_3" → 3)
    #                 import re
    #                 m = re.search(r"(-?\d+)", s)
    #                 if m:
    #                     try:
    #                         return float(int(m.group(1)))
    #                     except Exception:
    #                         continue
    #             return np.nan

    #         floor_rows["floor_val"] = floor_rows.apply(_parse_floor, axis=1)
    #         floor_rows = floor_rows.dropna(subset=["floor_val"]).copy()
    #         if not floor_rows.empty:
    #             floor_rows = floor_rows.sort_values("Frame")

    #             # ✅ floor_df 준비 + Frame dtype 통일
    #             floor_df = floor_rows[["Frame", "floor_val"]].drop_duplicates(subset=["Frame"]).copy()
    #             floor_df["Frame"] = pd.to_numeric(floor_df["Frame"], errors="coerce").astype("int64")

    #             # ✅ ts_frames 준비 + Frame dtype 통일
    #             ts_frames = sub_ts["Frame"].to_frame("Frame").sort_values("Frame").copy()
    #             ts_frames["Frame"] = pd.to_numeric(ts_frames["Frame"], errors="coerce").astype("int64")

    #             aligned = pd.merge_asof(
    #                 ts_frames,
    #                 floor_df,
    #                 on="Frame",
    #                 direction="backward"
    #             )

    #             # 원래 sub_ts index 순서에 맞춰 재정렬
    #             aligned = aligned.reindex(sub_ts["Frame"].index)
    #             df.loc[idx_ts, "CE_floor"] = aligned["floor_val"].to_numpy()
    #     # --- 여기까지 scene loop 끝난 직후 ---

    # # 🔹 CE_floor NaN 보정: scene별로 앞/뒤 채우기
    # if "CE_floor" in df.columns:
    #     df["CE_floor"] = (
    #         df.groupby(cols.scene)["CE_floor"]
    #           .ffill()
    #           .bfill()
    #     )
            
                
    # # ------------------------------------------------
    # # 🔹 NEW: CE_floor → phase / delta_crowd / baseline 변환
    # # ------------------------------------------------
    # if ELEVATOR_CROWD_MAP is not None and "CE_floor" in df.columns:
    #     map_df = ELEVATOR_CROWD_MAP.copy()
    #     map_df = map_df.rename(columns={"scene": "scene_map"})

    #     # ✅ CE_floor / to_floor 모두 정수형으로 맞추기
    #     df["CE_floor_int"] = pd.to_numeric(df["CE_floor"], errors="coerce").round().astype("Int64")
    #     map_df["to_floor"] = pd.to_numeric(map_df["to_floor"], errors="coerce").astype("Int64")

    #     df = df.merge(
    #         map_df,
    #         left_on=[cols.scene, "CE_floor_int"],
    #         right_on=["scene_map", "to_floor"],
    #         how="left",
    #     )

    #     df.rename(
    #         columns={
    #             "phase_index": "CE_floor_phase",
    #             "delta_crowd": "CE_crowd_delta",
    #             "is_baseline": "CE_floor_is_baseline",
    #         },
    #         inplace=True,
    #     )

    #     df.drop(
    #         columns=["scene_map", "from_floor", "to_floor", "CE_floor_int"],
    #         errors="ignore",
    #         inplace=True,
    #     )

    #     for c in ["CE_floor_phase", "CE_crowd_delta", "CE_floor_is_baseline"]:
    #         if c in df.columns:
    #             df[c] = df[c].fillna(0.0)



    return df
