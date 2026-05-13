import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# fixtures

@pytest.fixture
def generator():
    from model import SoundStream
    return SoundStream(C=16, D=64, K=64, num_q=4)

@pytest.fixture
def discriminator():
    from model import Discriminator
    return Discriminator()

@pytest.fixture
def batch():
    return torch.randn(2, 1, 8000)

# model

def test_soundstream_output_shape(generator, batch):
    x_hat, z, z_q, commit, perps = generator(batch)
    assert x_hat.shape[0] == 2
    assert x_hat.shape[1] == 1
    assert x_hat.shape[-1] <= batch.shape[-1]

def test_soundstream_commit_loss_scalar(generator, batch):
    _, _, _, commit, _ = generator(batch)
    assert commit.shape == ()

def test_soundstream_perplexities_count(generator, batch):
    _, _, _, _, perps = generator(batch)
    assert len(perps) == 4  # num_q

def test_discriminator_output_count(discriminator, batch):
    feats, logits = discriminator(batch)
    assert len(logits) == 6  # 3 MSD + 3 STFT

def test_discriminator_feature_maps(discriminator, batch):
    feats, _ = discriminator(batch)
    assert len(feats) == 6
    assert all(isinstance(f, list) for f in feats)

def test_encoder_stride(generator, batch):
    z = generator.encoder(batch)
    # total stride = 2*4*5*5 = 200;
    assert abs(z.shape[-1] - batch.shape[-1] / 200) <= 1

def test_rvq_num_quantizers(generator, batch):
    generator.eval()
    with torch.no_grad():
        z = generator.encoder(batch)
        _, inds, _, perps = generator.rvq(z)
    assert len(inds) == 4
    assert len(perps) == 4


# loss

def test_reconstruction_loss_zero_on_equal():
    from loss import reconstruction_loss
    x = torch.randn(2, 1, 8000)
    assert reconstruction_loss(x, x).item() < 1e-6

def test_reconstruction_loss_length_mismatch():
    from loss import reconstruction_loss
    x = torch.randn(2, 1, 8000)
    x_hat = torch.randn(2, 1, 7848)
    loss = reconstruction_loss(x, x_hat)
    assert loss.shape == ()

def test_spectral_loss_positive():
    from loss import spectral_loss
    x = torch.randn(2, 1, 8000)
    x_hat = torch.randn(2, 1, 8000)
    assert spectral_loss(x, x_hat).item() > 0

def test_adversarial_d_loss_real_higher():
    from loss import adversarial_d_loss
    real = [torch.ones(2, 1, 10)]
    fake = [-torch.ones(2, 1, 10)]
    loss = adversarial_d_loss(real, fake)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)

def test_adversarial_g_loss_scalar():
    from loss import adversarial_g_loss
    logits = [torch.randn(2, 1, 10) for _ in range(6)]
    loss = adversarial_g_loss(logits)
    assert loss.shape == ()

def test_feature_matching_loss_zero_on_equal():
    from loss import feature_matching_loss
    feats = [[torch.randn(2, 32, 100), torch.randn(2, 32, 50)]]
    loss = feature_matching_loss(feats, feats)
    assert loss.item() < 1e-6


# data

from unittest.mock import patch

def _make_dataset(tmp_path, n=4, dur=16000):
    for i in range(n):
        (tmp_path / f"{i}.wav").touch()
    return patch("torchaudio.load", return_value=(torch.randn(1, dur), 16000))

def test_audio_dataset_crop(tmp_path):
    from datasets import AudioFolderDataset
    with _make_dataset(tmp_path, n=1, dur=24000):
        ds = AudioFolderDataset(str(tmp_path), crop_samples=8000)
        assert ds[0].shape == (1, 8000)

def test_audio_dataset_pad_short(tmp_path):
    from datasets import AudioFolderDataset
    with _make_dataset(tmp_path, n=1, dur=4000):
        ds = AudioFolderDataset(str(tmp_path), crop_samples=8000)
        assert ds[0].shape == (1, 8000)