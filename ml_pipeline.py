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
                 patience=10, min_delta=1e-6   # ✅ 추가
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
        min_delta=min_delta       # ✅ 전달
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
    patience=10, min_delta=1e-6,
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
        patience=patience, min_delta=min_delta,
        **kwargs
    )
    print("✅ Grid Search 완료!")
    return best_params, grid_results


def train_and_evaluate_seeds(
    X_trainval, y_trainval, pid_trainval,
    X_test, y_test,
    model_type, best_params,
    device,
    num_seeds=10, num_epochs=20,
    patience=3, min_delta=1e-3
):
    """
    Seed 앙상블 학습/평가 함수 (CNN, GRU/LSTM 등 공용).
    - 입력 텐서 형태는 (N, T, C) 가정.
    - CNN: params['input_channels'] = C 로 보정
    - RNN류(GRU/LSTM/Transformer): params['input_size'] = C 로 보정
    - train_model()은 내부에서 best state를 적용한 모델을 반환한다고 가정.

    Returns:
        all_train_losses: [seed별 train loss curve(list or np.array)]
        all_val_losses  : [seed별 val loss curve(list or np.array)]
        all_test_scores : [seed별 (r2, rmse, mae)]
    """
    import gc
    import torch
    from ml_utils import set_seed, get_model, evaluate_and_save, train_model

    # ---------- 기본 체크 ----------
    assert X_trainval.ndim == 3 and X_test.ndim == 3, "Expect inputs shaped (N, T, C)."
    assert len(X_trainval) == len(y_trainval) == len(pid_trainval), \
        "Length mismatch among X_trainval, y_trainval, pid_trainval."

    C_train = X_trainval.shape[-1]
    C_test  = X_test.shape[-1]
    mt = str(model_type).upper()

    # ---------- 파라미터 주입(보정) ----------
    # (예전처럼 '사전 assert'로 None을 검사하지 말고, 먼저 주입 → 이후 일치성 검사)
    _best_params = dict(best_params)  # 원본 보존
    if mt == "CNN":
        if _best_params.get("input_channels") is None:
            _best_params["input_channels"] = C_train
        # 최종 일치성 확인
        assert _best_params["input_channels"] == C_train, \
            f"[CNN] input_channels({ _best_params['input_channels'] }) != C_train({ C_train })"
        assert C_test == _best_params["input_channels"], \
            f"[CNN] C_test({ C_test }) must equal input_channels({ _best_params['input_channels'] })"
    else:
        # GRU / LSTM / Transformer 등
        if _best_params.get("input_size") is None:
            _best_params["input_size"] = C_train
        assert _best_params["input_size"] == C_train, \
            f"[RNN] input_size({ _best_params['input_size'] }) != C_train({ C_train })"
        assert C_test == _best_params["input_size"], \
            f"[RNN] C_test({ C_test }) must equal input_size({ _best_params['input_size'] })"

    # ---------- 루프 준비 ----------
    all_train_losses, all_val_losses, all_test_scores = [], [], []

    for seed in range(num_seeds):
        print(f"\n🟢 SEED {seed} 시작\n")
        set_seed(seed)

        # ---- 학습 (best state가 적용된 model 반환 가정) ----
        model, train_losses, val_losses, *_ = train_model(
            X_trainval, y_trainval,
            params=_best_params,
            model_type=model_type,
            num_epochs=num_epochs,
            seed=seed,
            pid_array=pid_trainval,
            return_curve=True,
            patience=patience,
            min_delta=min_delta
        )
        # 커브 저장 (없으면 빈 리스트)
        all_train_losses.append(train_losses if train_losses is not None else [])
        all_val_losses.append(val_losses if val_losses is not None else [])

        # ---- 테스트용 fresh 모델 생성 & 가중치 로드 ----
        if mt == "CNN":
            test_model = get_model(model_type, input_size=_best_params["input_channels"], params=_best_params).to(device)
        else:
            test_model = get_model(model_type, input_size=_best_params["input_size"], params=_best_params).to(device)

        test_model.load_state_dict(model.state_dict())

        # ---- 평가 & 저장 ----
        filename = f"{model_type.lower()}_test_predictions_seed{seed}.npz"
        test_r2, test_rmse, test_mae, _ = evaluate_and_save(
            test_model, (X_test, y_test), device, filename, model_type=model_type
        )
        all_test_scores.append((test_r2, test_rmse, test_mae))

        # ---- 정리 ----
        del model, test_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return all_train_losses, all_val_losses, all_test_scores



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


