import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import torch
from tqdm import tqdm

from datasets import get_loader
from loss import reconstruction_loss
from metrics import STOIMetric, NISQAMetric


def load_model(checkpoint_path, device, C=32, D=512, K=1024, num_q=8):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    from model import SoundStream
    cfg = ckpt.get("model_cfg", {})
    C     = cfg.get("C",     C)
    D     = cfg.get("D",     D)
    K     = cfg.get("K",     K)
    num_q = cfg.get("num_q", num_q)
    model = SoundStream(C=C, D=D, K=K, num_q=num_q)
    model.load_state_dict(ckpt["generator"], strict=False)
    model.to(device).eval()
    step = ckpt.get("step", 0)
    print(f"Model: C={C}, D={D}, K={K}, num_q={num_q}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Generator params: {n_params:.2f}M")
    return model, step


@torch.no_grad()
def evaluate(model, loader, device, use_nisqa=True, max_batches=None):
    stoi_metric  = STOIMetric().to(device)
    nisqa_metric = NISQAMetric().to(device) if use_nisqa else None
    total_stoi  = 0.0
    total_nisqa = 0.0
    total_rec   = 0.0
    n = 0
    for i, x in enumerate(tqdm(loader, desc="Evaluating")):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        x_hat, *_ = model(x)
        x_real = x[..., :x_hat.shape[-1]]
        total_stoi += stoi_metric(x_hat, x_real).item()
        total_rec  += reconstruction_loss(x_real, x_hat).item()
        if nisqa_metric is not None:
            total_nisqa += nisqa_metric(x_hat).mean().item()
        n += 1
    results = {
        "stoi":      total_stoi / n,
        "rec":       total_rec  / n,
        "n_batches": n,
    }
    if use_nisqa:
        results["nisqa"] = total_nisqa / n
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint",    help="Path to checkpoint .pt file")
    parser.add_argument("--data",        required=True, help="Root dir with .flac/.wav files")
    parser.add_argument("--batch",       type=int, default=4)
    parser.add_argument("--workers",     type=int, default=2)
    parser.add_argument("--no-nisqa",    action="store_true")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of batches (default: full dataset)")
    parser.add_argument("--C",     type=int, default=32)
    parser.add_argument("--D",     type=int, default=512,
                        help="Bottleneck dimension (default: 512)")
    parser.add_argument("--K",     type=int, default=1024,
                        help="RVQ codebook size (default: 1024)")
    parser.add_argument("--num-q", type=int, default=8,
                        help="Number of RVQ quantizers (default: 8)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = get_loader(
        args.data,
        batch_size=args.batch,
        num_workers=args.workers,
        crop_samples=None,
        random_crop=False,
        shuffle=False,
        drop_last=False,
    )
    print(f"Eval set: {len(loader.dataset)} files  ({len(loader)} batches)")
    model, step = load_model(
        args.checkpoint, device,
        C=args.C, D=args.D, K=args.K, num_q=args.num_q,
    )
    print(f"Checkpoint step: {step}")
    results = evaluate(
        model, loader, device,
        use_nisqa=not args.no_nisqa,
        max_batches=args.max_batches,
    )

    print("Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f" {k}: {v:.4f}")
        else:
            print(f" {k}: {v}")

if __name__ == "__main__":
    main()