import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error




class CNNRegressor(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)   # deterministic
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):               # x: (B, C, T)
        # 🛡️ shape guard
        assert x.dim() == 3, f"Expected (B,C,T), got {tuple(x.shape)}"
        assert x.size(1) == self.conv1.in_channels, \
            f"C={x.size(1)} != in_channels={self.conv1.in_channels}"
        assert x.size(2) >= self.conv1.kernel_size[0], \
            f"T={x.size(2)} < kernel={self.conv1.kernel_size[0]}"

        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)    # (B, C)
        x = self.dropout(x)
        x = self.fc(x)                  # (B,1)
        return x.squeeze(-1)            # (B,)

    
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.gru(x)              # (B, T, H)
        logit = self.fc(out[:, -1, :])    # (B, 1)
        return logit.squeeze(-1)          # (B,)  ⬅️ 축 지정해서 스칼라 붕괴 방지



class GRUAttentionRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn_fc = nn.Linear(hidden_size, 1)  # Attention score 계산용
        self.out_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        gru_out, _ = self.gru(x)  # (B, T, H)
        attn_weights = torch.softmax(self.attn_fc(gru_out), dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (B, H)
        output = self.out_fc(context).squeeze(1)  # (B,)
        return output
    
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = output[:, -1, :]  # 마지막 시점의 hidden state
        out = self.fc(out)
        return out.squeeze(-1)  # (B, 1) → (B,)
