#!/usr/bin/env python3
"""Controlled synthetic frequency-bias study (R2.2).

Three sinusoids sit at fixed output bins k1<k2<k3 (periods 48, 12, 4 for a
96-step horizon). Two datasets swap the amplitude order -- (1, 0.3, 0.1) vs
(0.1, 0.3, 1) -- so that amplitude and frequency position are decoupled.

For every model and epoch we log, per key bin k on the test set:
  E_k      absolute spectral error |a_hat_k - a_Y_k|            (no denominator)
  rem_k    E_k / E_k(epoch 0): fraction of the component's own initial error left
  share_k  E_k^2 / sum_j E_j^2: share of the MSE gradient w.r.t. the output
           spectrum carried by bin k (by Parseval, dL/da_hat_k is proportional
           to a_hat_k - a_Y_k), i.e. the optimisation signal each component gets
  Delta_k  relative error E_k / |a_Y_k| (the paper's original metric)
Models: a time-domain patch Transformer (PatchTST-style), a channel/patch MLP
(no attention), a linear model, and Fredformer (frequency patches with
per-patch normalisation, FredNormer disabled).
"""

from __future__ import annotations

import argparse
import csv
import sys
from argparse import Namespace
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

WORKTREE = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(WORKTREE))
from models.Fredformer import Model as Fredformer  # noqa: E402

L = H = 96
PERIODS = (48, 12, 4)
BINS = tuple(H // p for p in PERIODS)  # 2, 8, 24
AMPS = {"decreasing": (1.0, 0.3, 0.1), "increasing": (0.1, 0.3, 1.0)}


def make_series(amps, n=10000, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = sum(a * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi)) for a, p in zip(amps, PERIODS))
    return (x + noise * rng.standard_normal(n)).astype(np.float32)


def windows(x):
    idx = np.arange(len(x) - L - H + 1)
    X = np.stack([x[i:i + L] for i in idx])[:, :, None]
    Y = np.stack([x[i + L:i + L + H] for i in idx])[:, :, None]
    return torch.from_numpy(X), torch.from_numpy(Y)


class PatchTransformer(nn.Module):
    def __init__(self, patch=8, d=64):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, L // patch, d))
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(d * (L // patch), H)

    def forward(self, x):  # x [B, L, 1]
        z = x[..., 0].unfold(1, self.patch, self.patch)
        z = self.enc(self.embed(z) + self.pos)
        return self.head(z.flatten(1))[..., None]


class MLP(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(L, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, H))

    def forward(self, x):
        return self.net(x[..., 0])[..., None]


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(L, H)

    def forward(self, x):
        return self.lin(x[..., 0])[..., None]


def fredformer():
    cfg = Namespace(enc_in=1, c_out=1, seq_len=L, pred_len=H, e_layers=2, n_heads=4, d_model=64, d_ff=128,
                    dropout=0.0, fc_dropout=0.0, head_dropout=0.0, individual=0, patch_len=8, stride=8,
                    padding_patch=None, revin=0, affine=0, subtract_last=0, use_nys=0, num_landmarks=8,
                    pinv_iterations=6, cf_dim=64, cf_depth=2, cf_heads=4, cf_mlp=128, cf_head_dim=16,
                    cf_drop=0.0, mlp_drop=0.0, ablation=0, use_freq_filter=False,
                    cv_weighting_mode="direct", norm_plugin="none")
    return Fredformer(cfg, global_mask=None)


MODELS = {"transformer": PatchTransformer, "mlp": MLP, "linear": Linear, "fredformer": fredformer}


@torch.no_grad()
def spectral_errors(model, X, Y, dev):
    model.eval()
    errs, amps = [], []
    for i in range(0, len(X), 512):
        p = model(X[i:i + 512])[..., 0]
        y = Y[i:i + 512][..., 0]
        fp, fy = torch.fft.rfft(p, dim=1), torch.fft.rfft(y, dim=1)
        errs.append((fp - fy).abs()[:, BINS].cpu())
        amps.append(fy.abs()[:, BINS].cpu())
    return torch.cat(errs).mean(0).numpy(), torch.cat(amps).mean(0).numpy()


def run(model_name, setting, seed, epochs, dev):
    torch.manual_seed(seed)
    x = make_series(AMPS[setting], seed=seed)
    n_train = int(len(x) * 0.7)
    Xtr, Ytr = windows(x[:n_train])
    Xte, Yte = windows(x[int(len(x) * 0.8):])
    Xtr, Ytr, Xte, Yte = (t.to(dev) for t in (Xtr, Ytr, Xte, Yte))
    model = MODELS[model_name]().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    e0 = None
    for epoch in range(epochs + 1):
        if epoch > 0:
            model.train()
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(Xtr[b]), Ytr[b])
                loss.backward()
                opt.step()
        E, A = spectral_errors(model, Xte, Yte, dev)
        e0 = E if e0 is None else e0
        share = E ** 2 / max((E ** 2).sum(), 1e-12)
        for j, k in enumerate(BINS):
            rows.append(dict(model=model_name, setting=setting, seed=seed, epoch=epoch, component=f"k{j + 1}",
                             bin=k, amplitude=f"{A[j]:.4f}", E=f"{E[j]:.6f}", rem=f"{E[j] / e0[j]:.6f}",
                             share=f"{share[j]:.6f}", Delta=f"{E[j] / A[j]:.6f}"))
    return rows


def main():
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    dev = torch.device("cuda")
    rows = []
    for setting in AMPS:
        for model_name in MODELS:
            for seed in map(int, args.seeds.split(",")):
                rows += run(model_name, setting, seed, args.epochs, dev)
                last = rows[-3:]
                print(setting, model_name, seed, [(r["component"], r["E"], r["rem"]) for r in last], flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
