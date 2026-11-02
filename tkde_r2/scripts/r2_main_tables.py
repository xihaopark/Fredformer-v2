#!/usr/bin/env python3
"""Regenerate Table II (averages) and Table III (selected per-horizon results).

The "Ours" entries become seed means of the fixed public configurations
(no test-set selection); FAN and FreDF rows (iTransformer, our runs) are added to
Table II. Best/second-best markers are recomputed per column from the numbers.
"""

from __future__ import annotations

import glob
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get("TKDE_R2_ROOT", Path(__file__).resolve().parents[1]))  # tkde_r2/
R2 = PROJECT / "tables" / "r2"
TEX = PROJECT / "latex"
MSE_RE = re.compile(r"^mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", re.M)
HS = [96, 192, 336, 720]
T2_COLS = ["electricity", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "traffic", "weather", "Solar"]


def ours_runs() -> pd.DataFrame:
    rows = []
    c = pd.read_csv(R2 / "ctrl_results.csv").dropna(subset=["mse"])
    c = c[c["mode"] == "direct"]
    for r in c.itertuples():
        rows.append((r.data_path[:-4], int(r.horizon), int(r.seed), r.mse, r.mae))
    mb = R2 / "main_big_results.csv"
    if mb.exists():
        for r in pd.read_csv(mb).dropna(subset=["mse"]).itertuples():
            rows.append((r.data_path[:-4], int(r.horizon), int(r.seed), r.mse, r.mae))
    for ds in ("electricity", "traffic"):  # seed 2021 from the locked benchmark runs
        for h in HS:
            logs = sorted(glob.glob(str(PROJECT / "logs" / f"full_{ds}_96_{h}_seed2021_*.log")))
            for lg in reversed(logs):
                m = MSE_RE.findall(Path(lg).read_text(errors="ignore"))
                if m:
                    rows.append((ds, h, 2021, float(m[-1][0]), float(m[-1][1])))
                    break
    d = pd.DataFrame(rows, columns=["dataset", "horizon", "seed", "mse", "mae"])
    d = d.drop_duplicates(["dataset", "horizon", "seed"])
    d = d[~((d.dataset == "Solar") & (d.seed > 2023))]  # protocol: three seeds (2021-2023) for Solar
    # settings whose configuration was selected by validation loss (scripts/r2_valsel.py)
    sel_f, grid_f = R2 / "valsel_selection.csv", R2 / "grid_results.csv"
    if sel_f.exists() and grid_f.exists():
        sel = pd.read_csv(sel_f)
        grid = pd.read_csv(grid_f).dropna(subset=["mse"])
        for r in sel.itertuples():
            if r.selected == "reference":
                continue
            g = grid[(grid.data_path == r.dataset + ".csv") & (grid.horizon == r.horizon) & (grid.tag == "grid_" + r.selected)]
            if g.seed.nunique() >= 5:
                d = d[~((d.dataset == r.dataset) & (d.horizon == r.horizon))]
                d = pd.concat([d, pd.DataFrame(dict(dataset=r.dataset, horizon=r.horizon, seed=g.seed, mse=g.mse, mae=g.mae))])
            else:
                print(f"WARNING: selected config for {r.dataset}-{r.horizon} has only {g.seed.nunique()} seeds; using reference")
    return d


def base_runs(tag: str) -> pd.DataFrame:
    f = R2 / "base_results.csv"
    if not f.exists():
        return pd.DataFrame(columns=["dataset", "horizon", "seed", "mse", "mae"])
    d = pd.read_csv(f).dropna(subset=["mse"])
    k = d.key.str.split("|", expand=True)
    d = d.assign(dataset=k[1], horizon=k[2].astype(int), kind=k[4], seed=k[5].astype(int))
    return d[d.kind == tag][["dataset", "horizon", "seed", "mse", "mae"]]


def fmt_markers(values: list[float | None]) -> list[str]:
    """Return LaTeX strings with best bold and second underlined (ties share the mark)."""
    vals = [round(v, 3) if v is not None and not np.isnan(v) else None for v in values]
    uniq = sorted(set(v for v in vals if v is not None))
    best = uniq[0] if uniq else None
    second = uniq[1] if len(uniq) > 1 else None
    out = []
    for v in vals:
        if v is None:
            out.append("--")
        elif v == best:
            out.append(r"\textbf{%.3f}" % v)
        elif v == second:
            out.append(r"\underline{%.3f}" % v)
        else:
            out.append("%.3f" % v)
    return out


def num(cell: str) -> float | None:
    m = re.search(r"(\d+\.\d+)", cell)
    return float(m.group(1)) if m else None


def table2(ours: pd.DataFrame) -> None:
    src = (R2 / "orig_main_result1.tex").read_text()
    body_start = next(m.start() for m in re.finditer(r"^[ \t]*iTransformer  & \\multirow\{7\}", src, re.M))
    body_end = src.index("\\bottomrule", body_start)
    lines = src[body_start:body_end].split("\n")
    rows = []  # (name, prefix_cells, values[16], is_data_row, raw)
    for ln in lines:
        cells = [c.strip() for c in ln.split("&")]
        if len(cells) >= 18 and not ln.strip().startswith("%"):
            vals = [num(c) for c in cells[2:18]]
            rows.append(dict(name=cells[0], type=cells[1], vals=vals, raw=None))
        else:
            rows.append(dict(raw=ln))
    agg = ours.groupby(["dataset", "horizon"])[["mse", "mae"]].mean().groupby("dataset").mean()
    nseeds = ours.groupby("dataset").seed.nunique()
    ov = []
    for ds in T2_COLS:
        if ds in agg.index and ours[ours.dataset == ds].horizon.nunique() == 4:
            ov += [agg.loc[ds, "mse"], agg.loc[ds, "mae"]]
        else:
            ov += [None, None]
    extra = []
    ff = R2 / "fredfloss_results.csv"  # Fredformer-v2 trained with the FreDF loss (reference configuration, 3 seeds)
    if ff.exists():
        f = pd.read_csv(ff).dropna(subset=["mse"])
        f = f[f.tag == "fredfloss"].assign(dataset=lambda x: x.data_path.str[:-4])
        a = f.groupby(["dataset", "horizon"])[["mse", "mae"]].mean()
        v = []
        for ds in T2_COLS:
            sub = a.loc[ds] if ds in a.index.get_level_values(0) else None
            v += [sub.mse.mean(), sub.mae.mean()] if sub is not None and len(sub) == 4 else [None, None]
        extra.append(dict(name=r"\textbf{Ours} + FreDF loss", type="", vals=v, raw=None))
    for tag, label in (("fank2", "FAN (iTrans.)"), ("fredf", "FreDF (iTrans.)")):
        b = base_runs(tag)
        a = b.groupby(["dataset", "horizon"])[["mse", "mae"]].mean()
        v = []
        for ds in T2_COLS:
            sub = a.loc[ds] if ds in a.index.get_level_values(0) else None
            if sub is not None and len(sub) == 4:
                v += [sub.mse.mean(), sub.mae.mean()]
            else:
                v += [None, None]
        extra.append(dict(name=label, type="", vals=v, raw=None))
    data_rows = [r for r in rows if r.get("raw") is None]
    for r in data_rows:
        if "Ours" in r["name"]:
            r["vals"] = ov
    # insert FAN/FreDF after the Transformer block (before first \cmidrule after Ours)
    out_rows = []
    inserted = False
    for r in rows:
        out_rows.append(r)
        if r.get("raw") is None and "Ours" in r["name"] and not inserted:
            out_rows += extra
            inserted = True
    all_data = [r for r in out_rows if r.get("raw") is None]
    cols = list(zip(*[r["vals"] for r in all_data]))
    marked = [fmt_markers(list(c)) for c in cols]
    for i, r in enumerate(all_data):
        r["cells"] = [marked[j][i] for j in range(16)]
    text = []
    for r in out_rows:
        if r.get("raw") is not None:
            text.append(r["raw"].replace("\\multirow{7}", "\\multirow{10}"))
        else:
            name = r["name"] if "FreDF loss" in r["name"] else (r"\textbf{Ours}" if "Ours" in r["name"] else r["name"])
            typ = r["type"].replace("\\multirow{7}", "\\multirow{10}")
            text.append(f"        {name} & {typ} & " + " & ".join(r["cells"]) + r" & \\")
    new = src[:body_start] + "\n".join(text) + src[body_end:]
    new = new.replace(
        "Results are averaged from all prediction lengths.}",
        "Results are averaged over all prediction lengths. \\method (Ours) is the mean over five seeds "
        "(three for Solar and Traffic), using validation-selected configurations for ETT and Weather and the reference configurations otherwise; "
        "``Ours + FreDF loss'': \\method trained with the FreDF loss (reference configurations, three seeds, Traffic not run); FAN and FreDF are applied to iTransformer and averaged over three seeds; "
        "other baselines are the published results under the same protocol~\\cite{liu2023itransformer}. "
        "Best results are in \\textbf{bold} and second-best are \\underline{underlined}.}")
    new = new.replace("prediction lengths $S\\in", "prediction lengths $H\\in").replace("look-back length $T=96$", "look-back length $L=96$")
    (TEX / "main_result1.tex").write_text(new)
    print("Table II Ours:", [None if v is None else round(v, 3) for v in ov], "seeds:", nseeds.to_dict())


def table3(ours: pd.DataFrame) -> None:
    src = (R2 / "orig_main_result2.tex").read_text()
    names = {"ETTh2": "ETTh2", "ETTm1": "ETTm1", "Weather": "weather", "Solar": "Solar"}
    per = ours.groupby(["dataset", "horizon"])[["mse", "mae"]].mean()
    out, cur = [], None
    ours_col = 6  # 0-based model index of "Ours" among the 12 models
    for ln in src.split("\n"):
        m = re.search(r"\\textbf\{(ETTh2|ETTm1|Weather|Solar)", ln)
        if m:
            cur = names[m.group(1)]
        cells = ln.split("&")
        if cur and len(cells) >= 26 and not ln.strip().startswith("%") and sum(num(c) is not None for c in cells[2:26]) >= 10:
            hcell = cells[1].strip()
            vals = [num(c) for c in cells[2:26]]
            last = cells[25]
            tail = last[last.index(re.search(r"\d+\.\d+", last).group(0)) + 5:] if re.search(r"\d+\.\d+", last) else ""
            if hcell.isdigit() and (cur, int(hcell)) in per.index:
                vals[2 * ours_col] = per.loc[(cur, int(hcell)), "mse"]
                vals[2 * ours_col + 1] = per.loc[(cur, int(hcell)), "mae"]
            elif hcell.lower().startswith("avg") and cur in per.index.get_level_values(0):
                vals[2 * ours_col] = per.loc[cur].mse.mean()
                vals[2 * ours_col + 1] = per.loc[cur].mae.mean()
            mse_m = fmt_markers(vals[0::2])
            mae_m = fmt_markers(vals[1::2])
            cells_new = [x for pair in zip(mse_m, mae_m) for x in pair]
            ln = cells[0] + "& " + cells[1].strip() + " & " + " & ".join(cells_new) + " " + tail.strip()
        out.append(ln)
    text = "\n".join(out)
    text = text.replace("Long-term time series forecasting on ETTh2, ETTm1, Weather, and Solar-Energy datasets.}",
                        "Long-term time series forecasting on ETTh2, ETTm1, Weather, and Solar-Energy datasets. "
                        "\\method (Ours) is the mean over seeds (five seeds; three for Solar), with the configurations of Table~I in the main paper.}")
    (TEX / "main_result2.tex").write_text(text)
    print("Table III updated")


if __name__ == "__main__":
    o = ours_runs()
    print(o.groupby("dataset").agg(n=("mse", "size"), seeds=("seed", "nunique")))
    table2(o)
    table3(o)
