import torch
from torch import nn

from .common import ResidualUnit

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, S):
        super().__init__()
        self.res_units = nn.Sequential(
            ResidualUnit(in_ch, d=1), ResidualUnit(in_ch, d=3), ResidualUnit(in_ch, d=9)
        )
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=2 * S, stride=S, padding=S // 2
        )

    def forward(self, x):
        return self.conv(self.res_units(x))


class Encoder(nn.Module):
    """SoundStream encoder
    Args:
        C: base channel (32)
        D: RVQ dimension (512)
    """
    def __init__(self, C=32, D=512):
        super().__init__()
        self.stem = nn.Conv1d(1, C, 7, padding=3)
        self.blocks = nn.Sequential(
            EncoderBlock(C,      2 * C, 2),
            EncoderBlock(2 * C,  4 * C, 4),
            EncoderBlock(4 * C,  8 * C, 5),
            EncoderBlock(8 * C, 16 * C, 5),
        )
        self.out = nn.Conv1d(16 * C, D, 3, padding=1)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.out(x)
