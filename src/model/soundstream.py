import torch
from torch import nn

from .encoder import Encoder
from .rvq import RVQ
from .decoder import Decoder

class SoundStream(nn.Module):
    """SoundStream codec pipeline (encoder->RVQ->Decoder)
    Args:
        C: base channel count (32)
        D: RVQ embedding dimension (512)
        K: RVQ codebook size (1024)
        num_q: number of RVQ quantizers (8)
    """
    def __init__(self, C=32, D=512, K=1024, num_q=8):
        super().__init__()
        self.encoder = Encoder(C=C, D=D)
        self.rvq     = RVQ(num_q, K, d=D)
        self.decoder = Decoder(C=C, D=D)

    def forward(self, x):
        T = x.shape[-1]
        z = self.encoder(x)
        z_q, codes, commit, perplexities = self.rvq(z)
        x_hat = self.decoder(z_q)[..., :T]
        return x_hat, z, z_q, commit, perplexities
