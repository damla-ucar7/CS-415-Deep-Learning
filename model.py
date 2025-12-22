import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SingleConvResidual(nn.Module):
    """
    LIGHTWEIGHT Residual Block.
    Only 1 Convolution per block. Fast but stable.
    """

    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()

        # 1. Main Operation (Just 1 Conv)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

        # 2. Skip Connection (Shortcut)
        # If dimensions change, we need a cheap 1x1 conv to resize 'x' so we can add it.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 1)) if pool else nn.Identity()

    def forward(self, x):
        # Calculate Main Path
        out = self.conv(x)
        out = self.bn(out)

        # Calculate Shortcut
        shortcut = self.shortcut(x)

        # RESIDUAL ADD (The magic happens here)
        out += shortcut

        # Activation & Pooling
        out = self.relu(out)
        out = self.pool(out)
        return out


class TranscriptionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 1. CNN (Fast & Deep) ---
        self.cnn = nn.Sequential(
            # Layer 1
            SingleConvResidual(1, 32, pool=False),  # [B, 32, 229, T]
            # Layer 2
            SingleConvResidual(32, 64, pool=True),  # [B, 64, 114, T]
            # Layer 3
            SingleConvResidual(64, 128, pool=True),  # [B, 128, 57, T]
            # Layer 4
            SingleConvResidual(128, 256, pool=True),  # [B, 256, 28, T]
            # Layer 5
            SingleConvResidual(256, 512, pool=False),  # [B, 512, 28, T]
            # Layer 6 (The extra layer you wanted)
            SingleConvResidual(512, 512, pool=True),  # [B, 512, 14, T]
        )

        # Output: 512 channels * 14 freq bins = 7168
        cnn_out_size = 512 * 14
        d_model = 512

        # --- 2. BRIDGE ---
        self.projection = nn.Linear(cnn_out_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)

        # --- 3. TRANSFORMER (Standard Width) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,  # Kept at 1024 for speed
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- 4. CLASSIFIER (No Bottleneck) ---
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),  # 512 -> 512
            nn.GELU(),  # Modern activation
            nn.Dropout(0.3),
            nn.Linear(d_model, 88),  # 512 -> 88
        )

    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 14, T]

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, T, C, F]
        x = x.reshape(b, t, c * f)  # [B, T, 7168]

        x = self.projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = self.fc(x)  # [B, T, 88]
        x = x.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 88, T]
        return x
