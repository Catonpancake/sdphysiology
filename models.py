import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error




# class CNNRegressor(nn.Module):
#     def __init__(self, input_channels, num_filters, kernel_size, dropout):
#         super().__init__()
#         self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size)
#         self.act = nn.ReLU()
#         self.pool = nn.AdaptiveAvgPool1d(1)   # deterministic
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(num_filters, 1)

#     def forward(self, x):               # x: (B, C, T)
#         # 🛡️ shape guard
#         assert x.dim() == 3, f"Expected (B,C,T), got {tuple(x.shape)}"
#         assert x.size(1) == self.conv1.in_channels, \
#             f"C={x.size(1)} != in_channels={self.conv1.in_channels}"
#         assert x.size(2) >= self.conv1.kernel_size[0], \
#             f"T={x.size(2)} < kernel={self.conv1.kernel_size[0]}"

#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.pool(x).squeeze(-1)    # (B, C)
#         x = self.dropout(x)
#         x = self.fc(x)                  # (B,1)
#         return x.squeeze(-1)            # (B,)

# class CNNRegressor(nn.Module):
#     def __init__(self, input_channels: int, num_filters: int = 32,
#                  kernel_size: int = 3, dropout: float = 0.5):
#         super().__init__()
#         padding = kernel_size // 2

#         self.conv1 = nn.Conv1d(
#             in_channels=input_channels,
#             out_channels=num_filters,
#             kernel_size=kernel_size,
#             padding=padding
#         )
#         self.act   = nn.ReLU()
#         self.drop  = nn.Dropout(dropout)
#         self.pool  = nn.AdaptiveAvgPool1d(output_size=1)
#         self.out   = nn.Linear(num_filters, 1)

#         # 🔹 추가: 출력 스케일 제어용
#         self.out_scale = 5.0

#     def forward(self, x):
#         # x: (B, C, T)
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.pool(x)          # (B, num_filters, 1)
#         x = x.squeeze(-1)         # (B, num_filters)
#         x = self.out(x)           # (B, 1)

#         # 🔹 추가: [-out_scale, +out_scale]로 강제
#         x = torch.tanh(x) * self.out_scale

#         return x.squeeze(-1)      # (B,)

# models.py 안의 CNNRegressor 클래스를 아래로 "통째로 교체"하세요.
# models.py
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRegressor(nn.Module):
    """
    CNN + (chunk mean pooling to K bins) + Flatten + MLP head (regression)
    Input:  x (B, C, T)
    Output: y_hat (B,)
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,

        pool_bins: int = 50,
        mlp_depth: int = 3,
        mlp_hidden: int = 512,
        mlp_dropout: float = None,

        output_activation: str = "none",
        out_scale: float = 5.0,
    ):
        super().__init__()
        assert pool_bins >= 1
        assert mlp_depth >= 0

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.pool_bins = int(pool_bins)

        # head
        if mlp_dropout is None:
            mlp_dropout = dropout
        self.mlp_dropout = float(mlp_dropout)

        head_in = num_filters * self.pool_bins
        layers = []
        cur = head_in

        for _ in range(int(mlp_depth)):
            layers.append(nn.Linear(cur, int(mlp_hidden)))
            layers.append(nn.ReLU())
            if self.mlp_dropout > 0:
                layers.append(nn.Dropout(self.mlp_dropout))
            cur = int(mlp_hidden)

        layers.append(nn.Linear(cur, 1))
        self.head = nn.Sequential(*layers)

        self.output_activation = (output_activation or "none").lower()
        self.out_scale = float(out_scale)

    def _chunk_mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, T) -> (B, F, K)
        K = self.pool_bins
        If T not divisible by K, pad on the right to make divisible.
        """
        B, F_, T = x.shape
        K = self.pool_bins
        if T % K != 0:
            pad = K - (T % K)
            x = F.pad(x, (0, pad))   # pad on time axis (right)
            T = T + pad
        x = x.view(B, F_, K, T // K).mean(dim=-1)  # (B, F, K)
        return x

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv1(x)        # (B, F, T)
        x = self.act(x)
        x = self.drop(x)

        x = self._chunk_mean_pool(x)   # (B, F, K)
        x = x.flatten(1)               # (B, F*K)
        x = self.head(x)               # (B, 1)

        if self.output_activation in ("none", "linear", "identity", ""):
            pass
        elif self.output_activation in ("tanh", "clamp_tanh"):
            x = torch.tanh(x) * self.out_scale
        elif self.output_activation in ("sigmoid", "bounded"):
            x = torch.sigmoid(x) * self.out_scale
        elif self.output_activation in ("tanh01", "bounded_tanh01"):
            x = (torch.tanh(x) + 1.0) * (self.out_scale / 2.0)
        else:
            raise ValueError(f"Unknown output_activation={self.output_activation}")

        return x.squeeze(-1)



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
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        out = output[:, -1, :]  # 마지막 시점의 hidden state
        out = self.fc(out)
        return out.squeeze(-1)  # (B, 1) → (B,)
