import torch
from torch import nn

from .common import ResidualUnit

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, S):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=2 * S, stride=S, padding=S // 2
        )
        self.res = nn.Sequential(
            ResidualUnit(out_ch, 1), ResidualUnit(out_ch, 3), ResidualUnit(out_ch, 9)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        return x

class Decoder(nn.Module):
    """SoundStream decoder
    Args:
        C: base channel (32)
        D: RVQ dimension (512)
    """
    def __init__(self, C=32, D=512):
        super().__init__()
        self.in_proj = nn.Conv1d(D, 16 * C, kernel_size=7, padding=3)
        self.blocks = nn.Sequential(
            DecoderBlock(16 * C, 8 * C, 5),
            DecoderBlock(8 * C,  4 * C, 5),
            DecoderBlock(4 * C,  2 * C, 4),
            DecoderBlock(2 * C,  C,     2),
        )
        self.out = nn.Conv1d(C, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out(x)
