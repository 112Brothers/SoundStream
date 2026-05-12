import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveSubDiscriminator(nn.Module):
    """Waveform discriminator"""
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1,    16,   15, padding=7),
            nn.Conv1d(16,   64,   41, stride=4, padding=20, groups=4),
            nn.Conv1d(64,   256,  41, stride=4, padding=20, groups=16),
            nn.Conv1d(256,  1024, 41, stride=4, padding=20, groups=64),
            nn.Conv1d(1024, 1024, 41, stride=4, padding=20, groups=256),
            nn.Conv1d(1024, 1024, 5,  padding=2),
        ])
        self.final = nn.Conv1d(1024, 1, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        feats = []
        for conv in self.convs:
            x = self.act(conv(x))
            feats.append(x)
        logit = self.final(x)
        feats.append(logit)
        return feats, logit


class MultiScaleDiscriminator(nn.Module):
    """WaveSubDiscriminators"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([WaveSubDiscriminator() for _ in range(3)])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=4, padding=2),
        ])

    def forward(self, x):
        all_feats, all_logits = [], []
        for pool, disc in zip(self.pools, self.discriminators):
            feats, logit = disc(pool(x))
            all_feats.append(feats)
            all_logits.append(logit)
        return all_feats, all_logits


class STFTSubDiscriminator(nn.Module):
    """STFT discriminator with ELU
    """

    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        self.convs = nn.ModuleList([
            nn.Conv2d(2,  32, (3, 9), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
            nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
        ])
        self.final = nn.Conv2d(32, 1, (3, 3), padding=(1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        spec = torch.stft(
            x.squeeze(1), self.n_fft, self.hop_length,
            window=self.window, return_complex=True,
        )  # [B,F,T_frames]
        spec = torch.stack([spec.real, spec.imag], dim=1)  # [B,2,F,T_frames]
        feats = []
        h = spec
        for conv in self.convs:
            h = self.act(conv(h))
            feats.append(h)
        logit = self.final(h)
        feats.append(logit)
        return feats, logit


class MultiResSTFTDiscriminator(nn.Module):
    """STFTSubDiscriminators"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTSubDiscriminator(512,  128),
            STFTSubDiscriminator(1024, 256),
            STFTSubDiscriminator(2048, 512),
        ])

    def forward(self, x):
        all_feats, all_logits = [], []
        for disc in self.discriminators:
            feats, logit = disc(x)
            all_feats.append(feats)
            all_logits.append(logit)
        return all_feats, all_logits

class Discriminator(nn.Module):
    """MSD+MRSTFTD discriminator"""

    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mrstftd = MultiResSTFTDiscriminator()

    def forward(self, x):
        msd_feats,  msd_logits  = self.msd(x)
        stft_feats, stft_logits = self.mrstftd(x)
        return msd_feats + stft_feats, msd_logits + stft_logits
