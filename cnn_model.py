import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import itertools
import gc

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# CNN Model
class CNNRegressor(nn.Module):
    def __init__(self, input_dim, kernel_size=7, channels=64, pooling='max', dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(60) if pooling == 'avg' else nn.AdaptiveMaxPool1d(60)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels * 60, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # BCT format
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Training function
def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        if val_loader is not None:
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss = criterion(model(X_batch), y_batch)
                    val_epoch_loss += val_loss.item()
            val_losses.append(val_epoch_loss / len(val_loader))
        else:
            val_losses.append(None)

    return model, train_losses, val_losses

# Hyperparameter search
def run_hyperparameter_search(X_train, y_train, X_val, y_val, input_dim, device='cpu', dropout=0.3, epochs=10):
    best_score = -np.inf
    best_result = None

    for ks, ch, pool, bs, lr in itertools.product([5, 7], [32, 64], ['max', 'avg'], [32, 64], [1e-3, 3e-4, 5e-4]):
        pin = device == 'cuda'
        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=bs, shuffle=True, num_workers=2, pin_memory=pin)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=bs, shuffle=False, num_workers=2, pin_memory=pin)

        model = CNNRegressor(input_dim=input_dim, kernel_size=ks, channels=ch, pooling=pool, dropout=dropout)
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

        y_preds, y_trues = [], []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                preds = model(X_batch).cpu().numpy()
                y_preds.append(preds)
                y_trues.append(y_batch.cpu().numpy())

        y_pred = np.concatenate(y_preds).flatten()
        y_true = np.concatenate(y_trues).flatten()
        r2 = r2_score(y_true, y_pred)

        if r2 > best_score:
            if best_result is not None and 'model' in best_result:
                del best_result['model']
            best_score = r2
            best_result = {
                'kernel_size': ks,
                'channels': ch,
                'pooling': pool,
                'batch_size': bs,
                'lr': lr,
                'r2': r2,
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
        else:
            del model
        torch.cuda.empty_cache()
        gc.collect()

    return best_result

# Evaluation
def evaluate_on_test(model, X_test, y_test, batch_size=64, device='cpu'):
    loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            y_preds.append(preds)
            y_trues.append(y_batch.numpy())
    y_pred = np.concatenate(y_preds).flatten()
    y_true = np.concatenate(y_trues).flatten()
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'y_pred': y_pred,
        'y_true': y_true
    }

# Leave-one-participant-out Split
def split_LOPO(X_ts, y_ts, pid_ts, leave_out_pid, val_ratio=0.2):
    test_mask = pid_ts == leave_out_pid
    train_val_mask = pid_ts != leave_out_pid

    X_test = X_ts[test_mask]
    y_test = y_ts[test_mask]

    X_trainval = X_ts[train_val_mask]
    y_trainval = y_ts[train_val_mask]
    pid_trainval = pid_ts[train_val_mask]

    unique_pids = np.unique(pid_trainval)
    val_pids = np.random.choice(unique_pids, size=int(len(unique_pids) * val_ratio), replace=False)

    val_mask = np.isin(pid_trainval, val_pids)
    train_mask = ~val_mask

    return X_trainval[train_mask], y_trainval[train_mask], X_trainval[val_mask], y_trainval[val_mask], X_test, y_test
