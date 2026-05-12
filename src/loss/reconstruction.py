import torch
import torch.nn.functional as F


def reconstruction_loss(x, x_hat):
    """L1 waveform reconstruction loss."""
    min_len = min(x.shape[-1], x_hat.shape[-1])
    return F.l1_loss(x_hat[..., :min_len], x[..., :min_len])

def commitment_loss(x, q):
    return (x - q.detach()).pow(2).mean()


_SPECTRAL_CACHE: dict = {}

def spectral_loss(x, x_hat):
    """Multiscale mel spectrogram loss"""
    min_len = min(x.shape[-1], x_hat.shape[-1])
    x     = x[..., :min_len]
    x_hat = x_hat[..., :min_len]
    scales  = [(512, 128), (1024, 256), (2048, 512)]
    buffers = _get_spectral_buffers(x.device)
    loss = 0.0
    for (n_fft, hop), (window, mel_fb) in zip(scales, buffers):
        X = torch.stft(
            x.squeeze(1).float(), n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True,
        )
        X_hat = torch.stft(
            x_hat.squeeze(1).float(), n_fft=n_fft, hop_length=hop,
            window=window, return_complex=True,
        )
        mel     = torch.matmul(mel_fb, X.abs())
        mel_hat = torch.matmul(mel_fb, X_hat.abs())
        loss = loss + F.l1_loss(torch.log(mel_hat + 1e-8), torch.log(mel + 1e-8))
    return loss / len(scales)

def _get_spectral_buffers(device):
    key = str(device)
    if key not in _SPECTRAL_CACHE:
        scales = [(512, 128, 64), (1024, 256, 128), (2048, 512, 128)]
        _SPECTRAL_CACHE[key] = [
            (torch.hann_window(n_fft, device=device),
             _mel_filterbank(n_fft, n_mels, sr=16000, device=device))
            for n_fft, _, n_mels in scales
        ]
    return _SPECTRAL_CACHE[key]

def _mel_filterbank(n_fft, n_mels, sr=16000, fmin=0.0, fmax=None, device="cpu"):
    """Return a [n_mels, n_fft//2+1] mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2.0, n_freqs, device=device)
    mel_min = _hz_to_mel(torch.tensor(fmin, device=device))
    mel_max = _hz_to_mel(torch.tensor(fmax, device=device))
    mel_points = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2, device=device)
    hz_points  = _mel_to_hz(mel_points)
    fb = torch.zeros(n_mels, n_freqs, device=device)
    for m in range(1, n_mels + 1):
        f_m_minus = hz_points[m - 1]
        f_m       = hz_points[m]
        f_m_plus  = hz_points[m + 1]
        up   = (freqs - f_m_minus) / (f_m - f_m_minus + 1e-8)
        down = (f_m_plus - freqs)  / (f_m_plus - f_m + 1e-8)
        fb[m - 1] = torch.clamp(torch.min(up, down), min=0.0)
    return fb

def _hz_to_mel(hz):
    return 2595.0 * torch.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
