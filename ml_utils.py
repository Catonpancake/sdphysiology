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
def create_dataloaders(X_array, y_array, pid_array, 
                       batch_size=32, seed=42, mode="train_val_test", model_type="CNN",
                       input_channels=None):
    unique_pids = np.unique(pid_array)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    n_total = len(unique_pids)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)

    if mode == "train_val_test":
        pids_train = unique_pids[:n_train]
        pids_val   = unique_pids[n_train:n_train+n_val]
        pids_test  = unique_pids[n_train+n_val:]
    elif mode == "full_train_test":
        pids_train = unique_pids[:n_train + n_val]
        pids_val   = []
        pids_test  = unique_pids[n_train + n_val:]
    else:
        raise ValueError("mode must be 'train_val_test' or 'full_train_test'")

    def select(pids):
        mask = np.isin(pid_array, pids)
        return X_array[mask], y_array[mask]

    X_train, y_train = select(pids_train)
    X_val,   y_val   = select(pids_val) if len(pids_val) > 0 else (None, None)
    X_test,  y_test  = select(pids_test)

    print(f"🎯 create_dataloaders X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # CNN 채널 수 명시(없으면 데이터에서 추론)
    if model_type.upper() == "CNN":
        if input_channels is None:
            input_channels = X_train.shape[-1]  # (N,T,C) 가정

    train_loader = to_loader(X_train, y_train, model_type, batch_size, shuffle=True,
                             input_channels=input_channels if model_type.upper()=="CNN" else None)
    val_loader   = (to_loader(X_val, y_val, model_type, batch_size, shuffle=False,
                              input_channels=input_channels) if X_val is not None else None)
    test_loader  = to_loader(X_test, y_test, model_type, batch_size, shuffle=False,
                             input_channels=input_channels if model_type.upper()=="CNN" else None)
    return train_loader, val_loader, test_loader

# --------------------- 평가 함수 ---------------------
# ← 반드시 열 0에 위치 (클래스 밖, 전역)
import torch

@torch.no_grad()
def evaluate(model, loader, device, model_type="CNN", return_arrays=True):
    """
    Safe evaluation:
      - Always normalizes preds / targets to 1-D (B,) before accumulation
      - Avoids 0-D scalar issues when batch size == 1
      - Returns (r2, rmse, mae, avg_loss, y_true, y_pred)
    """
    model.eval()
    criterion = torch.nn.MSELoss(reduction="mean")

    total_loss, total_n = 0.0, 0
    all_y_true, all_y_pred = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # --- forward ---
        preds = model(X_batch)

        # --- shape normalization: ALWAYS 1-D (B,) ---
        if preds.dim() == 2 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)   # (B,1) -> (B,)
        preds = preds.reshape(-1)        # safeguard: (B,) no matter what
        y_batch = y_batch.reshape(-1)    # targets also (B,)

        # --- loss ---
        loss = criterion(preds, y_batch)
        bs = y_batch.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        # --- collect ---
        if return_arrays:
            all_y_true.append(y_batch.detach().view(-1).cpu())
            all_y_pred.append(preds.detach().view(-1).cpu())

    avg_loss = total_loss / max(1, total_n)

    if return_arrays:
        y_true = torch.cat(all_y_true, dim=0).cpu().numpy() if all_y_true else None
        y_pred = torch.cat(all_y_pred, dim=0).cpu().numpy() if all_y_pred else None
        r2, rmse, mae = compute_metrics(y_true, y_pred) if (y_true is not None and y_pred is not None) else (float("nan"), float("nan"), float("nan"))
    else:
        y_true = y_pred = None
        r2 = rmse = mae = float("nan")

    return r2, rmse, mae, avg_loss, y_true, y_pred



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
    """
    from ml_utils import evaluate, save_predictions, to_loader, ensure_nct_for_cnn

    # 1) 데이터 → DataLoader 통일
    if isinstance(test_data, torch.utils.data.DataLoader):
        test_loader = test_data

    else:
        # tuple or list -> (X, y) 추출
        if isinstance(test_data, tuple):
            X, y = test_data
        elif isinstance(test_data, list):
            # [(X, y)] 형태 허용
            if len(test_data) == 0:
                raise ValueError("evaluate_and_save: empty test_data list.")
            if isinstance(test_data[0], tuple):
                X, y = test_data[0]
            else:
                raise TypeError(f"evaluate_and_save: unsupported list element type: {type(test_data[0])}")
        else:
            raise TypeError(f"evaluate_and_save: unsupported test_data type: {type(test_data)}")

        # 기본 검증
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(f"evaluate_and_save expects X as 3D np.ndarray, got {type(X)} with ndim={getattr(X,'ndim',None)}")
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        # CNN: (N,C,T) 강제 + input_channels 지정
        if model_type.upper() == "CNN":
            # input_channels 추론 우선순위: 인자 > 모델.conv1.in_channels > X.shape[-1](=C)
            if input_channels is None:
                input_channels = getattr(getattr(model, "conv1", None), "in_channels", None)
            if input_channels is None:
                input_channels = X.shape[-1]  # (N,T,C)일 가능성 고려

            # (N,C,T) 강제 정렬
            X_nct = ensure_nct_for_cnn(X, input_channels=input_channels)  # (N,C,T)
            # DataLoader 구성
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

    # 3) 저장
    save_predictions(y_true, y_pred, filename)
    print(f"📊 Test R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}  → saved to {filename}")

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
    grad_clip_norm=1.0,
    amp=True,
    deterministic=True
):
    """
    Drop-in replacement.

    ✅ 변경 요약(로직 최소 변경):
      - 학습 내부 동작은 그대로 유지
      - 리턴 시 train/val 분할 인덱스를 함께 반환
        * DataLoader의 sampler/dataset에 indices가 있을 때만 절대 인덱스 복구 가능
        * 없으면 None 반환
      - return_curve=True:
            (model, train_losses, val_losses, val_r2, val_rmse, val_mae, train_indices, val_indices)
        False:
            (model, val_r2, val_rmse, val_mae, train_indices, val_indices)
    """
    import gc
    import torch
    from ml_utils import set_seed, create_dataloaders, to_loader, get_model, evaluate  # ✅ evaluate 임포트 유지

    def _get_indices_from_loader(loader):
        """
        가능한 경우, 원본 X 기준의 절대 인덱스를 복원.
        - SubsetRandomSampler: loader.sampler.indices
        - torch.utils.data.Subset: loader.dataset.indices
        - 위가 없으면 None (외부에서 평가 시 None 처리)
        """
        idx = None
        # (1) sampler.indices
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "indices"):
            idx = loader.sampler.indices
        # (2) dataset.indices (Subset)
        elif hasattr(loader, "dataset") and hasattr(loader.dataset, "indices"):
            idx = loader.dataset.indices
        # numpy/torch 텐서로 정규화
        if idx is not None:
            try:
                import numpy as np
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

    # ✅ params 우선 준비
    p = dict(params)

    # ✅ 입력 크기 설정(데이터는 변형하지 않음 — 변형은 로더에서)
    if model_type.upper() == "CNN":
        p["input_channels"] = p.get("input_channels", X.shape[-1] if X.ndim == 3 else None)
        input_size = p["input_channels"]
    else:
        p["input_size"] = p.get("input_size", X.shape[2] if X.ndim == 3 else None)
        input_size = p["input_size"]

    if input_size is None:
        raise ValueError("❌ Unable to infer input_size/input_channels from X; please provide in params.")

    # --------- Build loaders ---------
    if use_internal_split:
        if pid_array is None:
            raise ValueError("❌ pid_array must be provided when use_internal_split=True")
        train_loader, val_loader, _ = create_dataloaders(
            X, y, pid_array=pid_array,
            batch_size=p["batch_size"],
            seed=seed,
            model_type=model_type,
            input_channels=p.get("input_channels", None)  # ✅ CNN 채널 전달
        )
    else:
        if external_val_data is None:
            raise ValueError("❌ external_val_data must be provided when use_internal_split=False")
        X_val, y_val = external_val_data

        if len(X) == 0 or len(y) == 0:
            raise ValueError("❌ Empty training data passed.")
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("❌ Empty validation data passed.")

        # ❌ 여기에서 X/X_val transpose 금지 — 로더가 처리
        train_loader = to_loader(X, y, model_type, batch_size=p["batch_size"], shuffle=True,
                                 input_channels=p.get("input_channels", None))
        val_loader   = to_loader(X_val, y_val, model_type, batch_size=p["batch_size"], shuffle=False,
                                 input_channels=p.get("input_channels", None))

    # 🔎 분할 인덱스(가능할 경우만) 확보
    train_indices = _get_indices_from_loader(train_loader)
    val_indices   = _get_indices_from_loader(val_loader)

    # --------- Model / Optim / Scheduler ---------
    model = get_model(model_type, input_size=input_size, params=p).to(device)

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

    criterion = torch.nn.MSELoss(reduction="mean")

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

    # --------- Train loop ---------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_n = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                preds = model(X_batch)
                if preds.dim() == 2 and preds.size(-1) == 1:
                    preds = preds.squeeze(-1)
                preds   = preds.reshape(-1)
                y_batch = y_batch.reshape(-1)
                loss = criterion(preds, y_batch)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            bs = y_batch.size(0)
            epoch_loss += loss.item() * bs
            total_n += bs

        train_loss = epoch_loss / max(1, total_n)
        train_losses.append(train_loss)

        # ---- Validation
        _, _, _, val_loss, _, _ = evaluate(model, val_loader, device, model_type=model_type, return_arrays=False)
        val_losses.append(val_loss)

        # ---- Scheduler step
        if scheduler_name == "plateau":
            scheduler.step(val_loss)
        elif scheduler_name == "cosine":
            scheduler.step()

        # ---- Early stopping
        if delta_is_relative:
            improved = (best_val_loss == float("inf")) or ((best_val_loss - val_loss) / max(1e-12, best_val_loss) > min_delta)
        else:
            improved = (val_loss + min_delta) < best_val_loss

        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            best_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping at epoch {epoch + 1} (best @ {best_epoch}, val={best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ✅ 여기서 Train/Val 평가를 둘 다 수행
    #    - train_loader는 shuffle=True여도 지표 계산에는 영향 없음 (평균/오차 기반)
    train_r2, train_rmse, train_mae, _, _, _ = evaluate(
        model, train_loader, device, model_type=model_type
    )
    val_r2, val_rmse, val_mae, _, _, _ = evaluate(
        model, val_loader, device, model_type=model_type
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ✅ 리턴만 확장 (indices는 앞서 잡은 값; None일 수 있음)
    if return_curve:
        return (
            model,                # 0
            train_losses,         # 1
            val_losses,           # 2
            val_r2, val_rmse, val_mae,   # 3,4,5 (기존)
            train_indices, val_indices,   # 6,7 (있으면 np.ndarray, 없으면 None)
            train_r2, train_rmse, train_mae  # 8,9,10 🔥 추가
        )
    else:
        return (
            model,
            val_r2, val_rmse, val_mae,
            train_indices, val_indices,
            train_r2, train_rmse, train_mae
        )
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
    min_delta=1e-6
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
                *_, val_r2, val_rmse, val_mae = train_model(
                    X, y, {**param_dict, "input_size": X.shape[-1]},  # 🔒 input_size 보장
                    model_type=model_type,
                    num_epochs=num_epochs,
                    seed=s,
                    pid_array=pid_array,
                    use_internal_split=use_internal_split,
                    external_val_data=external_val_data,
                    patience=patience,
                    min_delta=min_delta
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
    min_delta=1e-6
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
                min_delta=min_delta
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
    random_state: int = 42,
    *,
    lag_unit: str = "seconds",            # "seconds" | "frames"
    sampling_rate_hz: float = None,       # lag_unit="frames"일 때 필요(없으면 windex/stride로 추정)
    epsilon_flat: float = 1e-6            # 평탄 곡선 판정 임계값(최대-최소 < epsilon이면 flat)
):
    """
    반환 DF는 기존 컬럼에 더해:
      - 'lag_frames': 프레임 단위 라그
      - 'flat_curve': 해당 라그 스윕이 (max-min)<epsilon 인지 여부(한 번만 기록; 라그별로 동일)
    """
    # 샘플링레이트 추정 (필요 시)
    if lag_unit == "frames":
        if sampling_rate_hz is None:
            if len(windex) >= 2:
                step_frames = int(np.median(np.diff(windex)))
                sampling_rate_hz = step_frames / float(stride_seconds)
            else:
                raise ValueError("sampling_rate_hz가 필요합니다 (windex로 추정 불가).")
        lag_seconds_grid = [int(l)/float(sampling_rate_hz) for l in lag_grid]
        lag_frames_grid  = list(lag_grid)
    else:
        lag_seconds_grid = list(lag_grid)
        # seconds→frames(보고용)
        if len(windex) >= 2:
            step_frames = int(np.median(np.diff(windex)))
            sr_hz = step_frames / float(stride_seconds)
            lag_frames_grid = [int(np.round(s * sr_hz)) for s in lag_seconds_grid]
        else:
            lag_frames_grid = [None for _ in lag_seconds_grid]

    def _build_features(X_):
        """
        간단 평균 대신 채널별 통계로 피처 강화:
        - mean_t: 시간축 평균 (N, C)
        - std_t : 시간축 표준편차 (N, C)
        - slope : 1차차분 평균(시간 변화율) (N, C)
        합쳐서 (N, 3C)
        """
        if X_.ndim != 3:
            raise ValueError("X must be 3D")
        # (N, T, C) 또는 (N, C, T) 모두 지원
        if X_.shape[1] >= X_.shape[2]:  # (N, T, C)
            Taxis, Caxis = 1, 2
        else:                           # (N, C, T)
            Taxis, Caxis = 2, 1
        mean_t = X_.mean(axis=Taxis)
        std_t  = X_.std(axis=Taxis)
        diff   = np.diff(X_, axis=Taxis)
        slope  = diff.mean(axis=Taxis)
        feats  = np.concatenate([mean_t, std_t, slope], axis=1)
        return feats.astype(np.float32)

    def cv_score(feats_, y_, groups_arr):
        g = np.asarray(groups_arr)
        n_splits = max(2, min(5, len(np.unique(g))))
        if n_splits < 2:
            return np.nan, np.nan
        gkf = GroupKFold(n_splits=n_splits)
        scores = []
        # 표준화 + ElasticNet 파이프라인
        pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=6000, random_state=random_state)
        )
        for tr, va in gkf.split(feats_, y_, groups=g):
            pipe.fit(feats_[tr], y_[tr])
            p = pipe.predict(feats_[va])
            scores.append(r2_score(y_[va], p))
        return float(np.mean(scores)), float(np.std(scores))

    rows = []
    for lag_s, lag_f in zip(lag_seconds_grid, lag_frames_grid):
        X2, y2, pid2, scene2, widx2 = apply_lag_timeseries(
            X, y, pid, scene, windex,
            stride_seconds=stride_seconds,
            lag_seconds=lag_s
        )
        if len(y2) < 10:
            rows.append({"lag_s": lag_s, "lag_frames": lag_f, "cv_r2_mean": np.nan, "cv_r2_std": np.nan, "n_samples": int(len(y2))})
            continue
        feats2 = _build_features(X2)
        g_arr = pid2 if groups == "pid" else scene2
        m, s = cv_score(feats2, y2, g_arr)
        rows.append({"lag_s": lag_s, "lag_frames": lag_f, "cv_r2_mean": m, "cv_r2_std": s, "n_samples": int(len(y2))})

    df = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False, na_position="last").reset_index(drop=True)
    # 평탄성 플래그(보고/후처리용)
    if len(df):
        span = (df["cv_r2_mean"].max() - df["cv_r2_mean"].min())
        df["flat_curve"] = bool(span < epsilon_flat)
    return df

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
    _, _, scores_neg = train_and_evaluate_seeds(
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
    _, _, scores_perm = train_and_evaluate_seeds(
        X_trv_perm, y_trainval, pid_trainval,
        X_te_perm,  y_test,
        model_type=model_type,
        best_params=best_params,
        device=device,
        num_seeds=2, num_epochs=5, patience=3, min_delta=1e-3
    )
    r2_perm = float(np.mean([s[0] for s in scores_perm]))
    print(f"[QC] Time-order destroyed R² ≈ {r2_perm:.4f} (낮아야 정상)")

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
    fixed_params_base=None, search_space=None
):
    from ml_pipeline import (
        run_ablation, select_features_by_ablation, run_grid_search,
        train_and_evaluate_seeds, summarize_test_results
    )

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
        df_ablation = run_ablation(
            X_tr, y_tr_c, pid_tr,
            X_va, y_va_c, pid_va,
            feature_tag_list,
            model_type=model_type,
            fixed_params=fixed_params,
            seed=seed_master,
            num_epochs=ABLATION_EPOCHS,
            save_path=os.path.join(OUT_DIR, f"ablation_{HV_MODE}.csv"),
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

    train_losses, val_losses, test_scores = train_and_evaluate_seeds(
        X_trv, y_trv, pid_trv,
        X_te, y_te_c,
        model_type=model_type,
        best_params=best_params,
        device=device,
        num_seeds=NUM_SEEDS_FINAL,
        num_epochs=EPOCHS_FINAL,
        patience=patience_train,
        min_delta=min_delta_train
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
