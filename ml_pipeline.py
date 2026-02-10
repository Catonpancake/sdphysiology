import numpy as np
import torch
import gc
from ml_utils import set_seed, train_model,train_test_split, get_model, evaluate_and_save, mask, run_feature_ablation,grid_search_model




def load_and_split_data(
    path="ml_processed",
    seed=42,
    split_ratio=(0.8, 0.1, 0.1),
    mode="across",
    target_pid=None,
    stride_seconds=2,
    window_seconds=20,
    sampling_rate=120
):
    set_seed(seed)

    # ---------- Load ----------
    X_array = np.load(f"{path}/X_array.npy")  # (N, T, C)
    y_array = np.load(f"{path}/y_array.npy")
    pid_array = np.load(f"{path}/pid_array.npy")
    feature_tag_list = np.load(f"{path}/feature_tag_list.npy").tolist()
    if X_array.shape[1] < X_array.shape[2]:  # 현재 [N, C, T]라면
        X_array = X_array.transpose(0, 2, 1)  # → [N, T, C]

    if mode == "across":
        unique_pids = np.unique(pid_array)
        np.random.default_rng(seed).shuffle(unique_pids)
        n = len(unique_pids)
        p_train = unique_pids[:int(n * split_ratio[0])]
        p_val   = unique_pids[int(n * split_ratio[0]):int(n * (split_ratio[0] + split_ratio[1]))]
        p_test  = unique_pids[int(n * (split_ratio[0] + split_ratio[1])):]

        X_train, y_train, pid_train = mask(X_array, y_array, pid_array, p_train)
        X_val, y_val, pid_val       = mask(X_array, y_array, pid_array, p_val)
        X_test, y_test, pid_test    = mask(X_array, y_array, pid_array, p_test)

        return (
            X_train, y_train, pid_train,
            X_val, y_val, pid_val,
            X_test, y_test, pid_test,
            feature_tag_list
        )

    elif mode == "within":
        if target_pid is None:
            raise ValueError("target_pid must be specified for within-participant mode.")

        # ---------- Pretrain from other participants ----------
        mask_pretrain = pid_array != target_pid
        mask_target   = pid_array == target_pid

        X_pre, y_pre = X_array[mask_pretrain], y_array[mask_pretrain]
        X_train, X_val, y_train, y_val = train_test_split(X_pre, y_pre, test_size=0.1, random_state=seed)
        pid_train = np.array(["pretrain"] * len(y_train))
        pid_val   = np.array(["val"] * len(y_val))

        # ---------- Fine-tune & Test split from target participant ----------
        X_target, y_target = X_array[mask_target], y_array[mask_target]

        cut = int(len(X_target) * 0.5)
        window_size = window_seconds * sampling_rate
        stride_size = stride_seconds * sampling_rate

        # 겹침을 방지하기 위해 fine-tune 마지막 window와 test 첫 window 간 margin을 둠
        overlap_margin = int(window_size // stride_size)

        X_finetune = X_target[:cut]
        y_finetune = y_target[:cut]

        X_test = X_target[cut + overlap_margin:]
        y_test = y_target[cut + overlap_margin:]
        pid_test = np.array([target_pid] * len(y_test))

        return (
            X_train, y_train, pid_train,
            X_val, y_val, pid_val,
            X_test, y_test, pid_test,
            X_finetune, y_finetune,
            feature_tag_list
        )

    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'across' or 'within'.")




def run_ablation(X_train, y_train, pid_train, X_val, y_val, pid_val, feature_tag_list,
                 model_type="CNN", fixed_params=None, seed=42, num_epochs=10,
                 save_path="ablation_result.csv",
                 patience=10, min_delta=1e-6, criterion=torch.nn.MSELoss(reduction="mean")
                 ):
    df_result = run_feature_ablation(
        X_train, y_train, pid_train,
        X_val, y_val, pid_val,
        feature_tags=feature_tag_list,
        model_type=model_type,
        fixed_params=fixed_params,
        num_epochs=num_epochs,
        seed=seed,
        patience=patience,        # ✅ 전달
        min_delta=min_delta,       # ✅ 전달
        criterion=criterion  # ✅ 추가
    )
    df_result.to_csv(save_path, index=False)
    print(f"✅ Ablation 결과 저장 완료 → {save_path}")
    return df_result


def run_grid_search(
    X_train, y_train, pid_train,
    model_type, search_space,
    seed_list=(42, 43, 44),          # ✅ multi-seed 권장 기본
    num_epochs=20,
    use_internal_split=False,        # ✅ 외부 val 고정 권장
    external_val_data=None,          # (X_val, y_val) 필수 when False
    patience=10, min_delta=1e-6, criterion=torch.nn.MSELoss(reduction="mean"),
    **kwargs
):
    if use_internal_split is False and external_val_data is None:
        raise ValueError("use_internal_split=False면 external_val_data=(X_val, y_val)을 넘겨주세요.")

    best_params, grid_results = grid_search_model(
        X_train, y_train, pid_train,
        model_type=model_type,
        search_space=search_space,
        num_epochs=num_epochs,
        seed_list=seed_list,                 # ✅ 전달
        use_internal_split=use_internal_split,
        external_val_data=external_val_data, # ✅ 전달
        patience=patience, min_delta=min_delta, criterion=criterion,
        **kwargs
    )
    print("✅ Grid Search 완료!")
    return best_params, grid_results


# def train_and_evaluate_seeds(
#     X_trainval, y_trainval, pid_trainval,
#     X_test, y_test,
#     model_type, best_params,
#     device,
#     num_seeds=10, num_epochs=20,
#     patience=3, min_delta=1e-3,
#     use_internal_split=True,             # ✅ 추가
#     external_val_data=None,              # ✅ (X_val, y_val) 튜플 또는 None
#     deterministic=True,                   # ✅ 선택: 결정성 제어
#     criterion=torch.nn.MSELoss(reduction="mean"),
#     internal_split_mode="train_val",      # {"train_val","train_only","train_val_test"}
#     internal_val_ratio=0.20,
# ):
#     """
#     Seed 앙상블 학습/평가 함수 (CNN, GRU/LSTM 등 공용).
#     - 입력 텐서 형태는 (N, T, C) 가정.
#     - CNN: params['input_channels'] = C 로 보정
#     - RNN류(GRU/LSTM/Transformer): params['input_size'] = C 로 보정
#     - train_model()은 내부에서 best state를 적용한 모델을 반환한다고 가정.

#     Returns:
#         all_train_losses: [seed별 train loss curve(list or np.array)]
#         all_val_losses  : [seed별 val loss curve(list or np.array)]
#         all_test_scores : [seed별 (r2, rmse, mae)]
#     """
#     import gc
#     import torch
#     from ml_utils import set_seed, get_model, evaluate_and_save, train_model

#     # ---------- 기본 체크 ----------
#     assert X_trainval.ndim == 3 and X_test.ndim == 3, "Expect inputs shaped (N, T, C)."
#     assert len(X_trainval) == len(y_trainval) == len(pid_trainval), \
#         "Length mismatch among X_trainval, y_trainval, pid_trainval."

#     C_train = X_trainval.shape[-1]
#     C_test  = X_test.shape[-1]
#     mt = str(model_type).upper()

#     # ---------- 파라미터 주입(보정) ----------
#     # (예전처럼 '사전 assert'로 None을 검사하지 말고, 먼저 주입 → 이후 일치성 검사)
#     _best_params = dict(best_params)  # 원본 보존
#     if mt == "CNN":
#         if _best_params.get("input_channels") is None:
#             _best_params["input_channels"] = C_train
#         # 최종 일치성 확인
#         assert _best_params["input_channels"] == C_train, \
#             f"[CNN] input_channels({ _best_params['input_channels'] }) != C_train({ C_train })"
#         assert C_test == _best_params["input_channels"], \
#             f"[CNN] C_test({ C_test }) must equal input_channels({ _best_params['input_channels'] })"
#     else:
#         # GRU / LSTM / Transformer 등
#         if _best_params.get("input_size") is None:
#             _best_params["input_size"] = C_train
#         assert _best_params["input_size"] == C_train, \
#             f"[RNN] input_size({ _best_params['input_size'] }) != C_train({ C_train })"
#         assert C_test == _best_params["input_size"], \
#             f"[RNN] C_test({ C_test }) must equal input_size({ _best_params['input_size'] })"

#     # ---------- 루프 준비 ----------
#     all_train_losses, all_val_losses, all_test_scores = [], [], []
#     all_train_scores, all_val_scores = [], []

#     for seed in range(num_seeds):
#         print(f"\n🟢 SEED {seed} 시작\n")
#         set_seed(seed)

#         # ---- 학습 (best state가 적용된 model 반환 가정) ----
#         model, train_losses, val_losses, val_r2, val_rmse, val_mae, train_idx, val_idx, train_r2, train_rmse, train_mae = train_model(
#             X_trainval, y_trainval,
#             params=_best_params,
#             model_type=model_type,
#             num_epochs=num_epochs,
#             seed=seed,
#             pid_array=pid_trainval,
#             return_curve=True,
#             patience=patience,
#             min_delta=min_delta,
#             use_internal_split=use_internal_split,     # ✅ 바깥 인자 그대로 전달
#             external_val_data=external_val_data,       # ✅ 바깥 인자 그대로 전달
#             deterministic=deterministic,                # ✅ 선택
#             criterion=criterion,                          # ✅ 추가
#             internal_split_mode=internal_split_mode,      # ✅ 내부 모드/비율 전달
#             internal_val_ratio=internal_val_ratio,
#         )
#         # 커브 저장 (없으면 빈 리스트)
#         all_train_losses.append(train_losses if train_losses is not None else [])
#         all_val_losses.append(val_losses if val_losses is not None else [])
#         all_train_scores.append((float(train_r2), float(train_rmse), float(train_mae)))
#         all_val_scores.append((float(val_r2), float(val_rmse), float(val_mae)))

#         # ---- 테스트용 fresh 모델 생성 & 가중치 로드 ----
#         if mt == "CNN":
#             test_model = get_model(model_type, input_size=_best_params["input_channels"], params=_best_params).to(device)
#         else:
#             test_model = get_model(model_type, input_size=_best_params["input_size"], params=_best_params).to(device)

#         test_model.load_state_dict(model.state_dict())

#         # ---- 평가 & 저장 ----
#         filename = f"{model_type.lower()}_test_predictions_seed{seed}.npz"
#         test_r2, test_rmse, test_mae, _ = evaluate_and_save(
#             test_model, (X_test, y_test), device, filename, model_type=model_type
#         )
#         all_test_scores.append((test_r2, test_rmse, test_mae))

#         # ---- 정리 ----
#         del model, test_model
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()

#     return all_train_losses, all_val_losses, all_test_scores, all_train_scores, all_val_scores

def train_and_evaluate_seeds(
    X_trainval, y_trainval, pid_trainval,
    X_test, y_test,
    model_type, best_params,
    device,
    num_seeds=10, num_epochs=20,
    patience=3, min_delta=1e-3,
    use_internal_split=True,
    external_val_data=None,              # (X_val, y_val) 또는 (X_val, y_val, pid_val)
    deterministic=True,
    criterion=torch.nn.MSELoss(reduction="mean"),
    internal_split_mode="train_val",      # {"train_val","train_only","train_val_test","two_stage"} 등
    internal_val_ratio=0.20,
):
    """
    Seed 앙상블 학습/평가 함수 (CNN, GRU/LSTM 등 공용).

    ✅ 추가 동작 (시그니처 변경 없이):
      - external_val_data가 (X_val, y_val, pid_val) 형태면,
        Stage-1: TRAIN 학습 + external VAL로 best_epoch 결정
        Stage-2: TRAIN+VAL 전체로 best_epoch만큼 검증 없이 재학습(final retrain)
        → 누수 없이 정석 final model로 TEST 평가

      - external_val_data가 (X_val, y_val)만 주어지면 기존 방식 그대로 동작(=Stage-2 스킵).
    """
    import gc
    import numpy as np
    import torch
    from ml_utils import set_seed, get_model, evaluate_and_save, train_model

    # ---------- 기본 체크 ----------
    assert X_trainval.ndim == 3 and X_test.ndim == 3, "Expect inputs shaped (N, T, C)."
    assert len(X_trainval) == len(y_trainval) == len(pid_trainval), \
        "Length mismatch among X_trainval, y_trainval, pid_trainval."

    C_train = X_trainval.shape[-1]
    C_test  = X_test.shape[-1]
    mt = str(model_type).upper()

    # ---------- external_val_data 파싱 (시그니처 유지) ----------
    # 허용:
    #   - None
    #   - (X_val, y_val)
    #   - (X_val, y_val, pid_val)  -> final retrain 발동용
    pid_val_ext = None
    external_val_xy = external_val_data

    if external_val_data is not None and isinstance(external_val_data, (tuple, list)):
        if len(external_val_data) == 3:
            X_val_ext, y_val_ext, pid_val_ext = external_val_data
            external_val_xy = (X_val_ext, y_val_ext)
        elif len(external_val_data) == 2:
            X_val_ext, y_val_ext = external_val_data
            external_val_xy = (X_val_ext, y_val_ext)
        else:
            raise ValueError("external_val_data must be None, (X_val,y_val), or (X_val,y_val,pid_val).")
    else:
        X_val_ext = y_val_ext = None  # not used unless tuple/list

    # ---------- 파라미터 주입(보정) ----------
    _best_params = dict(best_params)  # 원본 보존
    if mt == "CNN":
        if _best_params.get("input_channels") is None:
            _best_params["input_channels"] = C_train
        assert _best_params["input_channels"] == C_train, \
            f"[CNN] input_channels({_best_params['input_channels']}) != C_train({C_train})"
        assert C_test == _best_params["input_channels"], \
            f"[CNN] C_test({C_test}) must equal input_channels({_best_params['input_channels']})"
    else:
        if _best_params.get("input_size") is None:
            _best_params["input_size"] = C_train
        assert _best_params["input_size"] == C_train, \
            f"[RNN] input_size({_best_params['input_size']}) != C_train({C_train})"
        assert C_test == _best_params["input_size"], \
            f"[RNN] C_test({C_test}) must equal input_size({_best_params['input_size']})"

    # ---------- 루프 준비 ----------
    all_train_losses, all_val_losses, all_test_scores = [], [], []
    all_train_scores, all_val_scores = [], []

    for seed in range(num_seeds):
        print(f"\n🟢 SEED {seed} 시작\n")
        set_seed(seed)

        # ==========================
        # Stage-1: TRAIN 학습 + external VAL로 best_state 선택
        # ==========================
        model, train_losses, val_losses, val_r2, val_rmse, val_mae, train_idx, val_idx, train_r2, train_rmse, train_mae = train_model(
            X_trainval, y_trainval,
            params=_best_params,
            model_type=model_type,
            num_epochs=num_epochs,
            seed=seed,
            pid_array=pid_trainval,
            return_curve=True,
            patience=patience,
            min_delta=min_delta,
            use_internal_split=use_internal_split,
            external_val_data=external_val_xy,   # ✅ 항상 (X_val,y_val) 형태로만 전달
            deterministic=deterministic,
            criterion=criterion,
            internal_split_mode=internal_split_mode,
            internal_val_ratio=internal_val_ratio,
        )

        # 커브/점수 저장
        all_train_losses.append(train_losses if train_losses is not None else [])
        all_val_losses.append(val_losses if val_losses is not None else [])
        all_train_scores.append((float(train_r2), float(train_rmse), float(train_mae)))
        all_val_scores.append((float(val_r2), float(val_rmse), float(val_mae)))

        # ==========================
        # Stage-2 (옵션): TRAIN+VAL로 "best_epoch만큼" 검증 없이 final retrain
        #  - pid_val_ext가 제공된 경우만 발동
        # ==========================
        if (pid_val_ext is not None) and (X_val_ext is not None) and (y_val_ext is not None):
            # best_epoch를 val_losses의 최소 지점으로 추정 (Stage-1 기준)
            if val_losses is not None and len(val_losses) > 0:
                best_epoch_ext = int(np.argmin(np.asarray(val_losses)) + 1)
            else:
                # fallback: train_losses 길이 or num_epochs
                best_epoch_ext = int(len(train_losses) if train_losses is not None and len(train_losses) > 0 else num_epochs)
            best_epoch_ext = max(1, best_epoch_ext)

            X_final = np.concatenate([X_trainval, X_val_ext], axis=0)
            y_final = np.concatenate([y_trainval, y_val_ext], axis=0)
            pid_final = np.concatenate([pid_trainval, pid_val_ext], axis=0)

            # 검증 없이 고정 epoch만큼 재학습 (누수 없음)
            model_final, *_ = train_model(
                X_final, y_final,
                params=_best_params,
                model_type=model_type,
                num_epochs=best_epoch_ext,
                seed=seed,
                pid_array=pid_final,
                return_curve=False,
                use_internal_split=True,
                external_val_data=None,
                # train_only라 early stop 개념이 없지만 안전하게 크게 둠
                patience=999999,
                min_delta=0.0,
                deterministic=deterministic,
                criterion=criterion,
                internal_split_mode="train_only",
                internal_val_ratio=0.0,
            )
            model_to_test = model_final
            print(f"[FINAL] retrained on (train+val) for {best_epoch_ext} epochs (no-val).")
        else:
            model_to_test = model  # 기존 동작 유지

        # ---- 테스트용 fresh 모델 생성 & 가중치 로드 ----
        if mt == "CNN":
            test_model = get_model(model_type, input_size=_best_params["input_channels"], params=_best_params).to(device)
        else:
            test_model = get_model(model_type, input_size=_best_params["input_size"], params=_best_params).to(device)

        test_model.load_state_dict(model_to_test.state_dict())

        # ---- 평가 & 저장 ----
        filename = f"{model_type.lower()}_test_predictions_seed{seed}.npz"
        test_r2, test_rmse, test_mae, _ = evaluate_and_save(
            test_model, (X_test, y_test), device, filename, model_type=model_type
        )
        all_test_scores.append((test_r2, test_rmse, test_mae))

        # ---- 정리 ----
        del model, test_model
        if 'model_final' in locals():
            try:
                del model_final
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return all_train_losses, all_val_losses, all_test_scores, all_train_scores, all_val_scores


def summarize_test_results(all_test_scores):
    test_r2s = [r2 for r2, _, _ in all_test_scores]
    print(f"\n📊 평균 Test R²: {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")
import numpy as np
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, List, Union

def select_features_by_ablation(
    df_result: pd.DataFrame,
    feature_tag_list: Union[Sequence[str], np.ndarray, pd.Series],
    top_k: int = None,
    threshold: float = None,
    strict: bool = True,         # True: 누락 태그 발견 시 에러, False: 조용히 스킵
    allow_duplicates: bool = True  # True: 동일 이름 여러 개면 첫 번째 인덱스만 사용(경고), False: 에러
) -> Tuple[List[str], List[int]]:
    """
    Ablation 결과를 기반으로 중요 feature를 선택합니다.

    Parameters:
    - df_result: run_feature_ablation 결과(열: ['feature_removed','val_r2'] 포함)
    - feature_tag_list: 전체 feature 이름 시퀀스(list/ndarray/Series 모두 OK)
    - top_k: 선택할 feature 수
    - threshold: drop_in_r2 기준값 (선택적으로 사용)
    - strict: 선택된 feature가 tag 리스트에 없으면 에러(True) 또는 스킵(False)
    - allow_duplicates: feature_tag_list 내 중복 이름 허용 여부

    Returns:
    - selected_features: 중요 feature 이름 리스트
    - selected_indices:  중요 feature 인덱스 리스트(채널 차원 슬라이스용)
    """
    # ---------- 타입 방어 ----------
    if isinstance(feature_tag_list, (np.ndarray, pd.Series, tuple)):
        feature_tag_list = list(feature_tag_list)
    elif not isinstance(feature_tag_list, list):
        feature_tag_list = list(feature_tag_list)

    # ---------- 기본 검증 ----------
    required_cols = {"feature_removed", "val_r2"}
    missing_cols = required_cols - set(df_result.columns)
    if missing_cols:
        raise ValueError(f"❌ df_result에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    if "None (baseline)" not in set(df_result["feature_removed"]):
        raise ValueError("❌ Ablation 결과에 'None (baseline)' 항목이 없습니다.")

    # ---------- Drop-in R² 계산 ----------
    baseline_r2 = df_result.loc[df_result["feature_removed"] == "None (baseline)", "val_r2"].iloc[0]
    ablation_only = df_result.loc[df_result["feature_removed"] != "None (baseline)"].copy()
    ablation_only["drop_in_r2"] = baseline_r2 - ablation_only["val_r2"]

    # 중요도 높은 순서로 정렬 (drop_in_r2가 클수록 제거 시 성능 하락 → 원래 중요)
    ablation_only = ablation_only.sort_values("drop_in_r2", ascending=False)

    # ---------- 선택 규칙 ----------
    if top_k is not None:
        selected = ablation_only.head(top_k)
    elif threshold is not None:
        selected = ablation_only[ablation_only["drop_in_r2"] >= threshold]
    else:
        selected = ablation_only

    selected_features = selected["feature_removed"].tolist()

    # ---------- 인덱스 매핑(안전/고속) ----------
    # 동일 이름 중복 여부 체크
    name2idxs = {}
    for i, name in enumerate(feature_tag_list):
        name2idxs.setdefault(name, []).append(i)

    if not allow_duplicates:
        dups = {k: v for k, v in name2idxs.items() if len(v) > 1}
        if dups:
            sample = {k: v[:3] for k, v in dups.items()}
            raise ValueError(f"❌ feature_tag_list에 중복 이름이 있습니다(allow_duplicates=False): {sample}")

    selected_indices = []
    missing = []
    for f in selected_features:
        if f not in name2idxs:
            missing.append(f)
            if not strict:
                continue
        else:
            # 중복이면 첫 번째 인덱스 사용 (경우에 따라 정책 변경 가능)
            selected_indices.append(name2idxs[f][0])

    if missing and strict:
        raise ValueError(f"❌ 선택된 feature가 feature_tag_list에 없습니다: {missing}")
    elif missing and not strict:
        print(f"⚠️ 다음 feature는 tag 리스트에 없어 스킵했습니다: {missing}")

    # ---------- 로그 ----------
    print(f"📌 선택된 feature 수: {len(selected_indices)} / {len(feature_tag_list)}")
    print(f"📌 feature_indices: {selected_indices}")

    return [feature_tag_list[i] for i in selected_indices], selected_indices


# -*- coding: utf-8 -*-
import os
import gc
import json

import numpy as np
import pandas as pd
import torch

from ml_utils import (
    to_NTC_strict,
    hv_mask_from_train_x,
    hv_mask_from_train_y,
    center_from_train_split,
    _fit_scene_stats,
    _transform_scenewise,
)

from ml_pipeline import train_and_evaluate_seeds  # 이미 같은 파일이면 이 임포트는 빼도 됨


def run_lopo_cnn(
    data_dir: str,
    out_dir: str,
    best_cnn_params: dict,
    *,
    hv_mode: str = "y_train",        # {"none","y_train","x_variance"}
    hv_quantile: float = 0.3,
    num_seeds: int = 5,
    base_seed: int = 42,
    num_epochs: int = 50,
    internal_val_ratio: float = 0.2,
    patience: int = 7,
    min_delta: float = 1e-3,
    device: torch.device | None = None,
):
    """
    Leave-One-Participant-Out (LOPO) for CNN (regression 전용).

    - 한 번만 X/y/pid/scene/feature_tag_list 로드
    - 각 PID를 test로 두고, 나머지를 train(+internal val)으로 사용
    - HV mask, y-centering, scene-wise zscore는 fold마다
      "train subset" 기준으로 다시 계산 (leakage-free)
    - CNN hyperparameter는 best_cnn_params로 고정, seed만 여러 개 반복

    data_dir에는 다음 파일이 있다고 가정:
        X_array.npy           # (N,T,C) 또는 (N,C,T)
        y_array.npy           # (N,)
        pid_array.npy         # (N,)
        scene_array.npy       # (N,)
        windex_array.npy      # (N,)   ← 여기서는 직접 안 써도 됨 (옵션)
        feature_tag_list.npy  # list[str], 길이 = C

    out_dir에는:
        lopo_detail_cnn.csv   # (target_pid × seed) 단위 점수
        lopo_summary_cnn.csv  # PID별 mean±std (r2, rmse, mae)
        meta_lopo_cnn.json    # 전체 설정 및 요약
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[LOPO] data_dir={data_dir}")
    print(f"[LOPO] out_dir={out_dir}")
    print(f"[LOPO] device={device}")

    # ----------------------------------------------------
    # 1) 데이터 로드
    # ----------------------------------------------------
    X = np.load(os.path.join(data_dir, "X_array.npy"))          # (N,*,*)
    y = np.load(os.path.join(data_dir, "y_array.npy")).astype(np.float32)
    pid = np.load(os.path.join(data_dir, "pid_array.npy"))
    scene = np.load(os.path.join(data_dir, "scene_array.npy"))
    widx = np.load(os.path.join(data_dir, "windex_array.npy"))

    feature_tag_list = np.load(
        os.path.join(data_dir, "feature_tag_list.npy"),
        allow_pickle=True,
    ).tolist()

    # (N,C,T)/(N,T,C) → (N,T,C)
    X = to_NTC_strict(X, feature_tag_list)
    N, T, C = X.shape
    print(f"[LOPO] Loaded X: (N,T,C)=({N},{T},{C}), y={y.shape}, pid={pid.shape}")

    # NaN / Inf 제거 (X 또는 y에 하나라도 있으면 해당 윈도우 제거)
    finite_mask = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
    num_bad = int((~finite_mask).sum())
    if num_bad > 0:
        print(f"[CLEAN] Dropping {num_bad} / {len(finite_mask)} windows with NaN/Inf in X or y")
        X = X[finite_mask]
        y = y[finite_mask]
        pid = pid[finite_mask]
        scene = scene[finite_mask]
        widx = widx[finite_mask]
    else:
        print("[CLEAN] No NaN/Inf detected in X/y after load")

    # PID 목록
    unique_pids = np.unique(pid)
    print(f"[LOPO] Unique PIDs: {len(unique_pids)} participants")

    # ----------------------------------------------------
    # 2) CNN 파라미터 고정 세팅
    # ----------------------------------------------------
    base_params = dict(best_cnn_params)   # 원본 보호
    base_params["input_channels"] = C     # CNN은 채널 수 필요 (N,T,C → C)
    # batch_size가 best_cnn_params 안에 이미 있다면 그대로 사용

    # seed 리스트
    seed_list = [base_seed + i for i in range(num_seeds)]

    # 결과를 쌓을 리스트
    rows_detail = []

    # LOPO 전체 요약용: PID별 평균 r2 등을 나중에 집계
    # (여기서는 바로 DataFrame으로 한 번에 처리)

    # ----------------------------------------------------
    # 3) PID 루프 (LOPO core)
    # ----------------------------------------------------
    for idx, tgt_pid in enumerate(unique_pids):
        print("\n" + "=" * 60)
        print(f"[LOPO] {idx + 1}/{len(unique_pids)}  Target PID = {tgt_pid}")
        print("=" * 60)

        mask_test = (pid == tgt_pid)
        mask_tr   = ~mask_test

        if not np.any(mask_test):
            print(f"[WARN] PID={tgt_pid} 에 해당하는 윈도우가 없습니다. 스킵.")
            continue

        # ---------- 3-1) HV mask (TRAIN 기준) ----------
        if hv_mode == "none":
            keep_all = np.ones_like(y, dtype=bool)
        elif hv_mode == "x_variance":
            keep_all = hv_mask_from_train_x(X, train_mask=mask_tr, q=hv_quantile)
        elif hv_mode == "y_train":
            keep_all = hv_mask_from_train_y(
                y_all=y,
                pid_all=pid,
                scene_all=scene,
                train_mask=mask_tr,
                q=hv_quantile,
            )
        else:
            raise ValueError(f"Unknown hv_mode={hv_mode}")

        # HV 마스크 + train/test 분리
        train_mask_final = mask_tr & keep_all
        test_mask_final  = mask_test & keep_all

        if not np.any(train_mask_final):
            print(f"[WARN] PID={tgt_pid} train_mask_final is empty. 스킵.")
            continue
        if not np.any(test_mask_final):
            print(f"[WARN] PID={tgt_pid} test_mask_final is empty after HV mask. 스킵.")
            continue

        X_tr_raw = X[train_mask_final]
        y_tr_raw = y[train_mask_final]
        pid_tr   = pid[train_mask_final]
        scene_tr = scene[train_mask_final]

        X_te_raw = X[test_mask_final]
        y_te_raw = y[test_mask_final]
        scene_te = scene[test_mask_final]

        print(f"[LOPO] PID={tgt_pid} | train={len(y_tr_raw)}, test={len(y_te_raw)} (after HV mask)")

        # ---------- 3-2) y centering (train 기준) ----------
        center_fn, stat_y = center_from_train_split(
            y_tr_raw,
            pid_tr,
            scene_tr,
        )
        y_tr = center_fn(y_tr_raw, pid_tr, scene_tr)
        y_te = center_fn(y_te_raw, pid[test_mask_final], scene_te)

        # ---------- 3-3) X scene-wise zscore (train 기준) ----------
        scene_stats, global_stats = _fit_scene_stats(X_tr_raw, scene_tr)
        X_tr = _transform_scenewise(X_tr_raw, scene_tr, scene_stats, global_stats)
        X_te = _transform_scenewise(X_te_raw, scene_te, scene_stats, global_stats)

        # ---------- 3-4) Huber loss delta 설정 (train y 기준) ----------
        iqr = float(np.subtract(*np.percentile(y_tr, [75, 25])))
        delta = float(max(0.1, min(iqr, 5.0)))
        criterion = torch.nn.HuberLoss(delta=delta)

        # ---------- 3-5) train_and_evaluate_seeds 호출 ----------
        # 내부에서 train/val split을 하도록 설정
        all_train_losses, all_val_losses, all_test_scores, all_train_scores, all_val_scores = (
            train_and_evaluate_seeds(
                X_trainval=X_tr,
                y_trainval=y_tr,
                pid_trainval=pid_tr,
                X_test=X_te,
                y_test=y_te,
                model_type="CNN",
                best_params=base_params,
                device=device,
                num_seeds=num_seeds,
                num_epochs=num_epochs,
                patience=patience,
                min_delta=min_delta,
                use_internal_split=True,
                external_val_data=None,
                deterministic=True,
                criterion=criterion,
                internal_split_mode="train_val",
                internal_val_ratio=internal_val_ratio,
            )
        )

        # all_test_scores: [(r2, rmse, mae), ...] 구조를 가정
        for s_idx, (r2, rmse, mae) in enumerate(all_test_scores):
            row = {
                "target_pid": tgt_pid,
                "seed_index": s_idx,
                "seed": seed_list[s_idx] if s_idx < len(seed_list) else np.nan,
                "test_r2": float(r2) if r2 is not None else np.nan,
                "test_rmse": float(rmse) if rmse is not None else np.nan,
                "test_mae": float(mae) if mae is not None else np.nan,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
            }
            rows_detail.append(row)

        # 메모리 정리
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 4) 결과 정리 및 저장
    # ----------------------------------------------------
    if len(rows_detail) == 0:
        print("[LOPO] 결과가 비어 있습니다. PID/마스크 조건을 확인하세요.")
        return

    df_detail = pd.DataFrame(rows_detail)
    detail_path = os.path.join(out_dir, "lopo_detail_cnn.csv")
    df_detail.to_csv(detail_path, index=False, encoding="utf-8")
    print(f"[LOPO] Saved LOPO detail metrics to: {detail_path}")

    # PID별 요약 (mean ± std)
    df_summary = (
        df_detail
        .groupby("target_pid")
        .agg(
            n_seeds=("seed", "count"),
            n_train=("n_train", "max"),
            n_test=("n_test", "max"),
            r2_mean=("test_r2", "mean"),
            r2_std=("test_r2", "std"),
            rmse_mean=("test_rmse", "mean"),
            rmse_std=("test_rmse", "std"),
            mae_mean=("test_mae", "mean"),
            mae_std=("test_mae", "std"),
        )
        .reset_index()
    )
    summary_path = os.path.join(out_dir, "lopo_summary_cnn.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[LOPO] Saved LOPO per-PID summary to: {summary_path}")

    # 전체 평균 한 번 더 출력
    print("\n===== LOPO overall (평균 over PIDs) =====")
    print(
        f"R² mean over PIDs:  {df_summary['r2_mean'].mean():.4f} "
        f"(SD={df_summary['r2_mean'].std():.4f})"
    )
    print(
        f"RMSE mean over PIDs: {df_summary['rmse_mean'].mean():.4f}, "
        f"MAE mean over PIDs: {df_summary['mae_mean'].mean():.4f}"
    )

    # meta 정보도 같이 저장해두면 나중에 논문 쓸 때 편함
    meta = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "model_type": "CNN",
        "best_cnn_params": best_cnn_params,
        "hv_mode": hv_mode,
        "hv_quantile": hv_quantile,
        "num_seeds": num_seeds,
        "base_seed": base_seed,
        "num_epochs": num_epochs,
        "internal_val_ratio": internal_val_ratio,
        "patience": patience,
        "min_delta": min_delta,
        "device": str(device),
        "n_pids": int(len(unique_pids)),
        "N_total_windows": int(len(y)),
        "detail_csv": os.path.basename(detail_path),
        "summary_csv": os.path.basename(summary_path),
    }
    meta_path = os.path.join(out_dir, "meta_lopo_cnn.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[LOPO] Saved meta to: {meta_path}")
