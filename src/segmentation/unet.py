import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# DoubleConv con Dropout
# ---------------------------
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2 + optional Dropout"""
    def __init__(self, in_channels: int, out_channels: int, p_dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# ---------------------------
# Downscaling
# ---------------------------
class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p_dropout: float = 0.2):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, p_dropout=p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

# ---------------------------
# Upscaling
# ---------------------------
class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, p_dropout: float = 0.2):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, p_dropout=p_dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, p_dropout=p_dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ---------------------------
# Attention block
# ---------------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ---------------------------
# UpAttention
# ---------------------------
class UpAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, p_dropout: float = 0.2):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, p_dropout=p_dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, p_dropout=p_dropout)

        self.att = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = self.att(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ---------------------------
# Output conv
# ---------------------------
class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# ---------------------------
# Full UNet
# ---------------------------
class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = False, p_dropout: float = 0.2):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64, p_dropout=p_dropout)
        self.down1 = Down(64, 128, p_dropout=p_dropout)
        self.down2 = Down(128, 256, p_dropout=p_dropout)
        self.down3 = Down(256, 512, p_dropout=p_dropout)
        self.down4 = Down(512, 512, p_dropout=p_dropout)
        self.up1 = UpAttention(1024, 256, bilinear, p_dropout=p_dropout)
        self.up2 = UpAttention(512, 128, bilinear, p_dropout=p_dropout)
        self.up3 = UpAttention(256, 64, bilinear, p_dropout=p_dropout)
        self.up4 = UpAttention(128, 64, bilinear, p_dropout=p_dropout)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ---------------------------
# Ottimizzatore con weight decay maggiore
# ---------------------------
# Esempio di utilizzo:
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
