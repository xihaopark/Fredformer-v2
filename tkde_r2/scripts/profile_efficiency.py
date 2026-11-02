#!/usr/bin/env python3
"""Matched efficiency profile for full vs Nystrom channel attention (R2.6).

All variants share batch size, look-back, horizon, precision (fp32) and model
hyperparameters; only the attention type / number of landmarks changes.
Peak memory is torch.cuda.max_memory_allocated over training steps (tensor
allocations only, excluding the CUDA context and allocator cache), so it is
comparable across variants and hardware-independent up to kernel choices.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from argparse import Namespace
import os
from pathlib import Path

import torch

WORKTREE = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(WORKTREE))
from models.Fredformer import Model  # noqa: E402


# locked benchmark settings for ECL (321 channels) and Traffic (862 channels)
SIZES = {321: dict(cf_dim=256, cf_mlp=256, d_model=128), 862: dict(cf_dim=512, cf_mlp=512, d_model=256)}


def make_configs(c_in: int, horizon: int, use_nys: int, landmarks: int) -> Namespace:
    cfg = Namespace(
        enc_in=c_in, c_out=c_in, seq_len=96, pred_len=horizon, e_layers=3, n_heads=16, d_model=128,
        d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0.0, individual=0, patch_len=4, stride=4,
        padding_patch=None, revin=1, affine=0, subtract_last=0, use_nys=use_nys, num_landmarks=landmarks,
        pinv_iterations=6, cf_dim=256, cf_depth=2, cf_heads=8, cf_mlp=128, cf_head_dim=32,
        cf_drop=0.2, mlp_drop=0.2, ablation=0, use_freq_filter=False, cv_weighting_mode="direct",
        norm_plugin="none",
    )
    for k, v in SIZES.get(c_in, {}).items():
        setattr(cfg, k, v)
    cfg.cf_depth = 3
    return cfg


def profile(c_in: int, horizon: int, use_nys: int, landmarks: int, batch: int, steps: int) -> dict:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = Model(make_configs(c_in, horizon, use_nys, landmarks), global_mask=None).to(dev)
    model.cv_statistics = torch.rand(96, device=dev) + 0.5
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 96, c_in, device=dev)
    y = torch.randn(batch, horizon, c_in, device=dev)
    for _ in range(3):  # warm-up
        opt.zero_grad(); torch.nn.functional.mse_loss(model(x), y).backward(); opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.mse_loss(model(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) / steps * 1000
    train_peak = torch.cuda.max_memory_allocated() / 2**20
    model.eval(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(steps):
            model(x)
        torch.cuda.synchronize()
    infer_s = (time.perf_counter() - t0) / steps
    return dict(channels=c_in, horizon=horizon, attention="nystrom" if use_nys else "full",
                landmarks=landmarks if use_nys else "", batch=batch, params=params,
                train_ms_per_step=f"{train_ms:.1f}", train_peak_mb=f"{train_peak:.0f}",
                infer_peak_mb=f"{torch.cuda.max_memory_allocated() / 2**20:.0f}",
                infer_samples_per_s=f"{batch / infer_s:.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    fields = ["channels", "horizon", "attention", "landmarks", "batch", "params", "train_ms_per_step",
              "train_peak_mb", "infer_peak_mb", "infer_samples_per_s"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c_in in (321, 862):           # ECL, Traffic
            for use_nys, lm in ((0, 0), (1, 8), (1, 16), (1, 32), (1, 64)):
                try:
                    r = profile(c_in, 96, use_nys, lm, args.batch, args.steps)
                except torch.cuda.OutOfMemoryError:
                    r = dict(channels=c_in, horizon=96, attention="nystrom" if use_nys else "full",
                             landmarks=lm if use_nys else "", batch=args.batch, train_peak_mb="OOM")
                print(r, flush=True)
                w.writerow(r)
                f.flush()
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
