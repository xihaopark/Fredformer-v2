#!/usr/bin/env python3
"""Test-time re-estimation of the FredNormer score S(k) under test-split shifts.

Compares, on identical checkpoints (reference configuration, seeds 2021-2023):
    w/o FN      model trained and tested without FredNormer (shift/ctrl results)
    train S     FredNormer with S estimated on the training split (evalchk re-evaluation)
    pooled S    S re-estimated once from all test look-back windows (inputs only)
    causal S    running estimate updated with the look-back windows seen so far
Writes tables/r2/online_cells.csv and latex/r2_online.tex.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
R2 = PROJECT / "tables" / "r2"
TEX = PROJECT / "latex"
DS = ["ETTh1", "ETTh2", "ETTm2", "weather"]
PERTS = ["clean", "noise10t", "newfreq", "warp10", "warp25"]
LABEL = {"clean": "clean", "noise10t": "noise", "newfreq": "unseen period", "warp10": r"warp $\times1.10$", "warp25": r"warp $\times1.25$"}
SEEDS = [2021, 2022, 2023]


def tidy(d: pd.DataFrame) -> pd.DataFrame:
    d = d.dropna(subset=["mse"]).copy()
    d["pert"] = d.data_path.str.extract(r"_shift_(\w+)\.csv")[0].fillna("clean")
    d["ds"] = d.data_path.str.replace(r"_shift_\w+", "", regex=True).str.replace(".csv", "", regex=False)
    return d[d.ds.isin(DS) & d.horizon.isin([96, 336]) & d.seed.isin(SEEDS)]


def main() -> None:
    on = tidy(pd.read_csv(R2 / "onlinec_results.csv"))
    sh = tidy(pd.read_csv(R2 / "shift_results.csv"))
    ct = tidy(pd.read_csv(R2 / "ctrl_results.csv"))
    none = pd.concat([sh[sh["mode"] == "none"], ct[ct["mode"] == "none"]])
    cells = pd.DataFrame({
        "w/o FN": none.groupby(["ds", "horizon", "pert"]).mse.mean(),
        "train S": on[on.tag == "evalchk"].groupby(["ds", "horizon", "pert"]).mse.mean(),
        "pooled S": on[on.tag == "onlinep"].groupby(["ds", "horizon", "pert"]).mse.mean(),
        "causal S": on[on.tag == "onlinec"].groupby(["ds", "horizon", "pert"]).mse.mean(),
    })
    wf = R2 / "onlinew_results.csv"
    if wf.exists():
        w = tidy(pd.read_csv(wf))
        for n0 in (256, 1024):
            cells[f"warm {n0}"] = w[w.tag == f"onlinew{n0}"].groupby(["ds", "horizon", "pert"]).mse.mean()
    cells.to_csv(R2 / "online_cells.csv")
    print(cells.round(4).to_string())
    full = cells.dropna()
    avg = full.groupby("pert").mean().loc[PERTS]
    wins_c = full.assign(w=full["causal S"] < full["train S"]).groupby("pert").w.sum().loc[PERTS]
    wins_n = full.assign(w=full["causal S"] < full["w/o FN"]).groupby("pert").w.sum().loc[PERTS]
    n = full.groupby("pert").size().loc[PERTS]
    print("\naverages\n", avg.round(4).to_string())
    print("\ncausal < train S:", dict(zip(PERTS, wins_c)), " causal < w/o FN:", dict(zip(PERTS, wins_n)), " n:", dict(zip(PERTS, n)))
    cols = list(avg.columns)
    head = {"w/o FN": "w/o FN", "train S": "train $S$", "pooled S": "pooled test $S$", "causal S": "causal online $S$",
            "warm 256": "warm start, $n_0{=}256$", "warm 1024": "warm start, $n_0{=}1024$"}
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Test-split shift & " + " & ".join(head[c] for c in cols) + r" \\", r"\midrule"]
    for p in PERTS:
        row = avg.loc[p]
        best = row.min()
        cells_tex = [(r"\textbf{%.4f}" % v) if abs(v - best) < 5e-5 else "%.4f" % v for v in row]
        lines.append(f"{LABEL[p]} & " + " & ".join(cells_tex) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "r2_online.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
