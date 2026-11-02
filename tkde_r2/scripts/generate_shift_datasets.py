#!/usr/bin/env python3
"""Create test-only distribution-shift copies of benchmark datasets (R2.5 / R3).

Unlike generate_robustness_datasets.py, which perturbs the whole series, these
copies keep the training and validation segments untouched and perturb only the
test segment (including the look-back inputs of the first test windows). This
yields a controlled train-to-test shift.

Perturbations
    warp10 / warp25 : time-stretch the test segment by 1.10 / 1.25, so every
                      periodic component moves to a lower frequency (spectral drift).
    newfreq         : add a sinusoid whose period (17 steps) is absent from the
                      dominant training spectrum (unseen frequency pattern).
    noise10t        : Gaussian noise with 0.10 x channel std, test segment only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parents[2]  # repository root
DATASET = WORKTREE / "dataset"
SEQ_LEN = 96


def test_start(data_path: str, n_rows: int) -> int:
    """First row of the test split, following data_provider/data_loader.py."""
    if data_path.startswith("ETTh"):
        return 12 * 30 * 24 + 4 * 30 * 24
    if data_path.startswith("ETTm"):
        return (12 * 30 * 24 + 4 * 30 * 24) * 4
    num_test = int(n_rows * 0.2)
    return n_rows - num_test


def warp(seg: np.ndarray, factor: float) -> np.ndarray:
    n = seg.shape[0]
    src = np.arange(n) / factor
    idx = np.arange(n)
    return np.stack([np.interp(src, idx, seg[:, j]) for j in range(seg.shape[1])], axis=1)


def perturb(seg: np.ndarray, name: str, std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if name == "warp10":
        return warp(seg, 1.10)
    if name == "warp25":
        return warp(seg, 1.25)
    if name == "newfreq":
        t = np.arange(seg.shape[0], dtype=float)[:, None]
        phase = rng.uniform(0.0, 2 * np.pi, size=(1, seg.shape[1]))
        return seg + 0.3 * std * np.sin(2 * np.pi * t / 17.0 + phase)
    if name == "noise10t":
        return seg + rng.normal(0.0, 0.10, size=seg.shape) * std
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_paths", nargs="+")
    ap.add_argument("--perturbations", default="warp10,warp25,newfreq,noise10t")
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    for data_path in args.data_paths:
        df = pd.read_csv(DATASET / data_path)
        cols = [c for c in df.columns if c != "date"]
        values = df[cols].to_numpy(dtype=float)
        start = test_start(data_path, len(df)) - SEQ_LEN
        # channel scale from the rows before the test split
        std = np.nanstd(values[:start], axis=0, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0)
        for name in [p.strip() for p in args.perturbations.split(",") if p.strip()]:
            rng = np.random.default_rng(args.seed)
            out = values.copy()
            out[start:] = perturb(values[start:], name, std, rng)
            dst = DATASET / f"{Path(data_path).stem}_shift_{name}.csv"
            res = df.copy()
            res[cols] = out
            res.to_csv(dst, index=False)
            print(f"wrote {dst} test_rows_from={start}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
