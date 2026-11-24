"""
CNN-BiLSTM with attention for speech emotion recognition on log-mel spectrograms.

Input:  (batch, 1, n_mels, time)
Flow:   Conv2D blocks -> collapse freq axis -> BiLSTM over time -> attention
        pooling -> dense -> softmax over emotions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class Attention(nn.Module):
    """Additive attention over the BiLSTM's time steps: lets the model
    weight the most emotionally salient frames instead of averaging all
    of them equally."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, time, hidden_dim)
        scores = self.attn(x).squeeze(-1)          # (batch, time)
        weights = F.softmax(scores, dim=1)          # (batch, time)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_mels=128, num_classes=NUM_CLASSES,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 32),     # n_mels/2, T/2
            ConvBlock(32, 64),    # n_mels/4, T/4
            ConvBlock(64, 128),   # n_mels/8, T/8
        )

        freq_after_conv = n_mels // 8
        self.lstm_input_size = 128 * freq_after_conv

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attention = Attention(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x, return_attention=False):
        # x: (batch, 1, n_mels, time)
        x = self.conv_blocks(x)                       # (batch, 128, n_mels/8, T/8)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)  # (batch, time, feat)

        x, _ = self.lstm(x)                            # (batch, time, hidden*2)
        context, attn_weights = self.attention(x)       # (batch, hidden*2)

        context = self.dropout(context)
        logits = self.fc(context)                       # (batch, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits
