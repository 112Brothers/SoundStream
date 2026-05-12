import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16_000
CROP_SAMPLES = 8_000  # 0.5s, 16kHz

class AudioFolderDataset(Dataset):
    def __init__(self, root, crop_samples=CROP_SAMPLES, random_crop=True):
        self.files = sorted(Path(root).rglob("*.flac")) + sorted(Path(root).rglob("*.wav"))
        if not self.files:
            raise ValueError(f"No .flac or .wav files found under {root}")
        self.crop_samples = crop_samples
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if self.crop_samples is None:
            return waveform
        T = waveform.shape[-1]
        if T < self.crop_samples:
            pad = self.crop_samples - T
            waveform = torch.nn.functional.pad(waveform, (0, pad), mode="replicate")
        T = waveform.shape[-1]
        if self.random_crop:
            start = random.randint(0, T - self.crop_samples)
        else:
            start = 0
        return waveform[:, start : start + self.crop_samples]


def _pad_collate(batch):
    """Pad variable-length waveforms to the longest in the batch."""
    max_len = max(x.shape[-1] for x in batch)
    return torch.stack([
        torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in batch
    ])


def get_loader(root, batch_size=12, num_workers=4, crop_samples=CROP_SAMPLES,
               random_crop=True, shuffle=True, drop_last=True, **kwargs):
    dataset = AudioFolderDataset(root, crop_samples=crop_samples, random_crop=random_crop)
    print(f"Found {len(dataset)} audio files under {root}")
    collate = _pad_collate if crop_samples is None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate
    )
