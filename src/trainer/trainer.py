import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gc

import torch
from torch import amp
from tqdm import tqdm

from loss import (
    adversarial_d_loss, adversarial_g_loss,
    feature_matching_loss, reconstruction_loss, spectral_loss
)
from metrics import MetricCollection

def _get(cfg, key, default):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


class SoundStreamTrainer:
    """Trainer for soundstream (generator+discriminator GAN)
    Args:
        generator:      soundstream model
        discriminator:  Discriminator model
        config:         namespace / dict-like with training params
        device:         "cuda"/"cpu"
        dataloaders:    {"train": DataLoader, "val": DataLoader/None}
        experiment:     Comet ML experiment (optional)
        model_cfg:      dict with model params for checkpoint
    """

    def __init__(self, generator, discriminator, config, device,
                 dataloaders, experiment=None, model_cfg=None):
        self.generator     = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config        = config
        self.device        = device
        self.dataloaders   = dataloaders
        self.experiment    = experiment
        self.model_cfg     = model_cfg
        lr = _get(config, "lr", 1e-4)
        self.opt_g = torch.optim.Adam(generator.parameters(),     lr=lr, betas=(0.5,0.9))
        self.opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.9))
        _amp_on = not getattr(config, "no_amp", False) and device.startswith("cuda")
        self.amp_enabled = _amp_on
        self.scaler_g = amp.GradScaler("cuda", enabled=_amp_on)
        self.scaler_d = amp.GradScaler("cuda", enabled=_amp_on)
        self.step = 0
        self._best_stoi = 0.0

    def train(self):
        cfg = self.config
        steps      = _get(cfg, "steps", 45000)
        ckpt_dir   = _get(cfg, "ckpt_dir", "checkpoints")
        ckpt_every = _get(cfg, "ckpt_every", 5000)
        nisqa_every = _get(cfg, "nisqa_every", 5)
        accum_steps = _get(cfg, "accum_steps", _get(cfg, "accum", 1))
        resume     = _get(cfg, "resume", None)
        if resume:
            self._resume_checkpoint(resume)
        loader = self.dataloaders["train"]
        epoch = 0
        while self.step < steps:
            print(f"\nEpoch {epoch}  (step {self.step}/{steps})")
            self._train_epoch(epoch, loader, accum_steps)
            self.step += len(loader)
            epoch += 1
            eval_loader = self.dataloaders.get("val")
            if eval_loader is not None:
                use_nisqa = epoch % nisqa_every == 0
                results = self._evaluation_epoch(eval_loader, use_nisqa=use_nisqa)
                val_stoi = results.get("val_stoi", 0.0)
                if val_stoi > self._best_stoi and ckpt_dir:
                    self._best_stoi = val_stoi
                    self._save_checkpoint(self.step, ckpt_dir, name="ckpt_best_stoi.pt")
                    print(f"New best STOI={self._best_stoi:.4f}")
            torch.cuda.empty_cache()
            if ckpt_dir and self.step % ckpt_every < len(loader):
                self._save_checkpoint(self.step, ckpt_dir)
        if ckpt_dir:
            self._save_checkpoint(self.step, ckpt_dir, name="ckpt_final.pt")
            print("Saved final checkpoint.")

    def _train_epoch(self, epoch, loader, accum_steps=1):
        self.generator.train()
        self.discriminator.train()

        scale = 1.0 / accum_steps
        pbar = tqdm(loader)
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)

        for local_step, x in enumerate(pbar):
            step = self.step + local_step
            pbar.set_description(f"step {step}")
            x = x.to(self.device, non_blocking=True)
            is_update = ((local_step + 1)%accum_steps == 0) or (local_step + 1 == len(loader))
            with amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                x_hat, z, z_q, commit, perplexities = self.generator(x)
            x_real = x[..., :x_hat.shape[-1]]
            with amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                real_feats, real_logits = self.discriminator(x_real)
                real_feats_det = [[f.detach() for f in fs] for fs in real_feats]
                _, fake_logits_d = self.discriminator(x_hat.detach())
                loss_d = adversarial_d_loss(real_logits, fake_logits_d) * scale
            self.scaler_d.scale(loss_d).backward()
            del real_feats, real_logits
            if is_update:
                self.scaler_d.step(self.opt_d)
                self.scaler_d.update()
                self.opt_d.zero_grad(set_to_none=True)
            for p in self.discriminator.parameters():
                p.requires_grad_(False)
            with amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                fake_feats_g, fake_logits_g = self.discriminator(x_hat)
                loss_adv  = adversarial_g_loss(fake_logits_g)
                loss_feat = feature_matching_loss(real_feats_det, fake_feats_g)
                loss_rec  = reconstruction_loss(x, x_hat)
                loss_spec = spectral_loss(x, x_hat)
                loss_g = (loss_adv + 100.0 * loss_feat + loss_rec + loss_spec + commit) * scale
            self.scaler_g.scale(loss_g).backward()
            del real_feats_det, fake_feats_g
            if is_update:
                self.scaler_g.step(self.opt_g)
                self.scaler_g.update()
                self.opt_g.zero_grad(set_to_none=True)
            for p in self.discriminator.parameters():
                p.requires_grad_(True)
            loss_d_item = loss_d.item() / scale
            loss_g_item = loss_g.item() / scale
            if self.experiment is not None and is_update: # comet log
                metrics = {
                    "loss_d": loss_d_item, "loss_g": loss_g_item,
                    "loss_adv": loss_adv.item(), "loss_feat": loss_feat.item(),
                    "loss_rec": loss_rec.item(), "loss_spec": loss_spec.item(),
                    "commit": commit.item(),
                }
                for i, perp in enumerate(perplexities):
                    metrics[f"perplexity_q{i}"] = perp.item()
                self.experiment.log_metrics(metrics, step=step // accum_steps)
            pbar.set_postfix(
                rec=f"{loss_rec.item():.3f}",
                commit=f"{commit.item():.3f}",
                D=f"{loss_d_item:.3f}",
                G=f"{loss_g_item:.3f}",
                amp=int(self.amp_enabled),
            )

    @torch.no_grad()
    def _evaluation_epoch(self, loader, max_batches=None, use_nisqa=False, audio_log_samples=3):
        self.generator.eval()
        metrics_col = MetricCollection(use_nisqa=use_nisqa).to(self.device)
        total_rec = total_stoi = total_nisqa = 0.0
        total_batches = 0
        for batch_idx, x in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(self.device, non_blocking=True)
            x_hat, _, _, _, _ = self.generator(x)
            x_real = x[..., :x_hat.shape[-1]]
            if self.experiment is not None and batch_idx == 0 and audio_log_samples > 0:
                tag = f"step{self.step:06d}"
                n = min(audio_log_samples, x_real.shape[0])
                for i in range(n):
                    orig_np  = x_real[i, 0].cpu().float().numpy().clip(-1, 1)
                    recon_np = x_hat[i, 0].cpu().float().numpy().clip(-1, 1)
                    self.experiment.log_audio(orig_np,  sample_rate=16000, file_name=f"{tag}_orig_{i}.wav")
                    self.experiment.log_audio(recon_np, sample_rate=16000, file_name=f"{tag}_recon_{i}.wav")
            batch_metrics = metrics_col(x_hat, x_real)
            total_rec   += reconstruction_loss(x_real, x_hat).item()
            total_stoi  += batch_metrics["stoi"].item()
            if "nisqa" in batch_metrics:
                total_nisqa += batch_metrics["nisqa"].mean().item()
            total_batches += 1
        if total_batches == 0:
            raise RuntimeError("Evaluation loader produced no batches")
        results = {
            "val_rec":  total_rec  / total_batches,
            "val_stoi": total_stoi / total_batches,
        }
        if use_nisqa:
            results["val_nisqa"] = total_nisqa / total_batches
        del metrics_col
        gc.collect()
        torch.cuda.empty_cache()
        print("Eval:", ", ".join(f"{k}={v:.4f}" for k, v in results.items()))
        if self.experiment is not None:
            self.experiment.log_metrics(results, step=self.step)
        self.generator.train()
        return results

    def _save_checkpoint(self, step, ckpt_dir, name=None):
        os.makedirs(ckpt_dir, exist_ok=True)
        if name is None:
            name = f"ckpt_{step:06d}.pt"
        path = os.path.join(ckpt_dir, name)
        payload = {
            "step":          step,
            "generator":     self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g":         self.opt_g.state_dict(),
            "opt_d":         self.opt_d.state_dict(),
        }
        if self.model_cfg is not None:
            payload["model_cfg"] = self.model_cfg
        torch.save(payload, path)
        print(f"Saved checkpoint: {path}")

    def _resume_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.generator.load_state_dict(ckpt["generator"], strict=False)
        self.discriminator.load_state_dict(ckpt["discriminator"], strict=False)
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.step = ckpt["step"]
        print(f"Resumed from {path} at step {self.step}")
