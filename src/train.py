import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="soundstream")
def main(cfg: DictConfig) -> None:
    from model import SoundStream, Discriminator
    from datasets import get_loader
    from trainer import SoundStreamTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator     = SoundStream(C=cfg.model.C, D=cfg.model.D, K=cfg.model.K, num_q=cfg.model.num_q)
    discriminator = Discriminator()
    print(f"Generator: {sum(p.numel() for p in generator.parameters())/1e6:.1f}M params")

    if cfg.data.train is not None:
        loader = get_loader(cfg.data.train, batch_size=cfg.training.batch,
                            num_workers=cfg.data.workers, crop_samples=cfg.data.crop_samples)
        eval_loader = get_loader(cfg.data.eval, batch_size=cfg.training.eval_batch,
                                 num_workers=cfg.data.workers, crop_samples=None,
                                 random_crop=False, shuffle=False, drop_last=False,
                                 ) if cfg.data.eval else None
    else:
        print("data.train not set — using dummy data")
        dummy = torch.randn(16, 1, cfg.data.crop_samples)
        loader, eval_loader = DataLoader(dummy, batch_size=cfg.training.batch, drop_last=True), None

    experiment = None
    comet_key = cfg.logging.comet_key or os.environ.get("COMET_API_KEY")
    if comet_key:
        try:
            from comet_ml import Experiment
            experiment = Experiment(api_key=comet_key, project_name=cfg.logging.project_name)
            experiment.log_parameters(dict(cfg))
        except ImportError:
            print("comet_ml not installed — logging disabled")

    model_cfg = {k: getattr(cfg.model, k) for k in ("C", "D", "K", "num_q")}
    trainer = SoundStreamTrainer(
        generator=generator, discriminator=discriminator,
        config=cfg.training, device=device,
        dataloaders={"train": loader, "val": eval_loader},
        experiment=experiment, model_cfg=model_cfg,
    )
    trainer.train()

if __name__ == "__main__":
    main()