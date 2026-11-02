#!/usr/bin/env python3
"""Summarize R2 experiments into CSV summaries and LaTeX table fragments.

Statistics: seeds are averaged within each (dataset, horizon) cell first, so
the unit of comparison is a cell, not a seed. Paired two-sided Wilcoxon
signed-rank tests across cells compare each variant with the full model, and
Holm's step-down correction is applied over the family of variants.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT = Path(os.environ.get("TKDE_R2_ROOT", Path(__file__).resolve().parents[1]))  # tkde_r2/
R2 = PROJECT / "tables" / "r2"
TEX = PROJECT / "latex"

CTRL_LABEL = {
    "direct": "FredNormer (CV, full model)",
    "none": "w/o FredNormer",
    "constant": "Constant weight (mean CV)",
    "shuffled": "Permuted CV across frequencies",
    "inverse": "Inverse CV ($1/S$)",
    "learned_mask": "Learned mask (no $S$)",
    "monotone": "Learned monotone $w(S)$",
}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(R2 / name).dropna(subset=["mse"])
    d["dataset"] = d.data_path.str.extract(r"^([A-Za-z0-9]+?)(?:_shift_|\.csv)")[0]
    d["dataset"] = d["dataset"].replace({"weather": "Weather", "electricity": "ECL", "traffic": "Traffic"})
    d["pert"] = d.data_path.str.extract(r"_shift_([a-z0-9]+)")[0].fillna("clean")
    return d


def paired_tests(cell_means: pd.DataFrame, ref: str, variants: list[str]) -> pd.DataFrame:
    rows, pv = [], {}
    for v in variants:
        sub = cell_means[[ref, v]].dropna()
        diff = (sub[v] - sub[ref]) / sub[ref] * 100
        p = wilcoxon(sub[v], sub[ref]).pvalue if len(sub) >= 6 and (sub[v] != sub[ref]).any() else np.nan
        pv[v] = p
        rows.append(dict(variant=v, cells=len(sub), mean_rel_change_pct=diff.mean(),
                         ref_better_cells=int((sub[v] > sub[ref]).sum()), p_wilcoxon=p))
    out = pd.DataFrame(rows)
    adj = holm({k: v for k, v in pv.items() if not np.isnan(v)})
    out["p_holm"] = out.variant.map(adj)
    return out


def controls() -> None:
    d = load("ctrl_results.csv")
    cells = d.groupby(["dataset", "horizon", "mode"]).mse.agg(["mean", "std", "count"]).reset_index()
    full = cells.pivot_table(index=["dataset", "horizon"], columns="mode", values="count").min(axis=1)
    keep = full[full >= 3].index
    cm = cells.pivot_table(index=["dataset", "horizon"], columns="mode", values="mean").loc[keep]
    cs = cells.pivot_table(index=["dataset", "horizon"], columns="mode", values="std").loc[keep]
    cm.to_csv(R2 / "ctrl_cell_means.csv")
    variants = [m for m in CTRL_LABEL if m != "direct" and m in cm.columns]
    tests = paired_tests(cm, "direct", variants)
    tests.to_csv(R2 / "ctrl_tests.csv", index=False)
    # per-dataset averages over horizons
    ds = cm.groupby(level=0).mean()
    ds.loc["Avg."] = cm.mean()
    cols = [m for m in CTRL_LABEL if m in cm.columns]
    lines = [r"\begin{tabular}{l" + "c" * len(ds.index) + "cc}", r"\toprule",
             "Weighting & " + " & ".join(ds.index) + r" & Wins & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    tmap = tests.set_index("variant")
    for m in cols:
        vals = ds[m]
        best = ds[cols].min(axis=1)
        cells_txt = [(r"\textbf{%.4f}" if abs(v - b) < 5e-5 else "%.4f") % v for v, b in zip(vals, best)]
        if m == "direct":
            extra = r"-- & --"
        else:
            r = tmap.loc[m]
            extra = f"{int(r.ref_better_cells)}/{len(cm)} & {r.p_holm:.2f}"
        lines.append(f"{CTRL_LABEL[m]} & " + " & ".join(cells_txt) + f" & {extra} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "r2_controls.tex").write_text("\n".join(lines) + "\n")
    # full per-cell table for the supplement
    sl = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
          "Data & $H$ & " + " & ".join(CTRL_LABEL[m].split(" (")[0] for m in cols) + r" \\", r"\midrule"]
    for (dname, h), row in cm.iterrows():
        best = row[cols].min()
        sl.append(f"{dname} & {h} & " + " & ".join(
            (r"\textbf{%.3f}" if abs(row[m] - best) < 5e-5 else "%.3f") % row[m] + (r"\tiny{$\pm$%.3f}" % cs.loc[(dname, h), m])
            for m in cols) + r" \\")
    sl += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "r2_controls_full.tex").write_text("\n".join(sl) + "\n")
    print(tests.round(4).to_string())
    print(ds[cols].round(4).to_string())


def shifts() -> None:
    d = load("shift_results.csv")
    clean = load("ctrl_results.csv")
    clean = clean[clean.dataset.isin(d.dataset.unique()) & clean.horizon.isin([96, 336])
                  & clean["mode"].isin(["direct", "none", "constant", "inverse", "learned_mask"])
                  & clean.seed.isin([2021, 2022, 2023])]
    nc = load("normcmp_results.csv") if (R2 / "normcmp_results.csv").exists() else pd.DataFrame()
    if len(nc):
        # FAN with the official per-frequency top-k only (earlier runs used the default k=20)
        nc = nc[~(nc.extra.str.contains("fan") & ~nc.extra.str.contains("fan_topk"))]
        nc["mode"] = np.where(nc.extra.str.contains("fan"), "fan", "san")
    on = load("onlinec_results.csv") if (R2 / "onlinec_results.csv").exists() else pd.DataFrame()
    if len(on):  # causal online re-estimation of S at test time, same checkpoints as "direct"
        on = on[on.tag == "onlinec"].assign(mode="online")
    allr = pd.concat([d, clean, nc, on], ignore_index=True)
    elec = allr[allr.dataset == "ECL"]
    allr = allr[allr.dataset != "ECL"]
    cm = allr.groupby(["dataset", "horizon", "pert", "mode"]).mse.mean().unstack("mode")
    cm.to_csv(R2 / "shift_cell_means.csv")
    # degradation relative to the same model on clean data
    clean_ref = cm.xs("clean", level="pert")
    deg = cm.copy()
    for idx in deg.index:
        deg.loc[idx] = (cm.loc[idx] / clean_ref.loc[(idx[0], idx[1])] - 1) * 100
    deg.to_csv(R2 / "shift_degradation.csv")
    modes = [m for m in ["direct", "online", "none", "constant", "inverse", "learned_mask", "fan", "san"] if m in cm.columns]
    label = {"direct": "Fredformer-v2", "online": "Fredformer-v2, online $S$", "none": "w/o FredNormer", "constant": "Const.", "inverse": "Inv. CV",
             "learned_mask": "Learned mask", "fan": "w/o FN + FAN", "san": "w/o FN + SAN"}
    per = cm.groupby(level="pert").mean()[modes]
    lines = [r"\begin{tabular}{l" + "c" * len(modes) + "}", r"\toprule",
             "Test-time shift & " + " & ".join(label[m] for m in modes) + r" \\", r"\midrule"]
    names = {"clean": "None (clean)", "warp10": "Time warp $\\times$1.10", "warp25": "Time warp $\\times$1.25",
             "newfreq": "Unseen period (17)", "noise10t": "Noise 10\\%"}
    for p in ["clean", "warp10", "warp25", "newfreq", "noise10t"]:
        if p not in per.index:
            continue
        row = per.loc[p]
        best = row.min()
        lines.append(names[p] + " & " + " & ".join((r"\textbf{%.4f}" if abs(row[m] - best) < 5e-5 else "%.4f") % row[m]
                                                  for m in modes) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "r2_shift.tex").write_text("\n".join(lines) + "\n")
    print(per.round(4).to_string())
    wins = (cm["direct"] < cm["none"]).groupby(level="pert").agg(["sum", "count"])
    print("v2 better than w/o FN, by shift:\n", wins.to_string())
    if len(elec):
        print(elec.groupby(["pert", "mode"]).mse.mean().unstack().round(4).to_string())


def plugins() -> None:
    raw = pd.read_csv(R2 / "plug_results.csv").dropna(subset=["mse"])
    k = raw.key.str.split("|", expand=True)
    raw["dataset"], raw["horizon"], raw["model"], raw["cv"], raw["seed"] = k[1], k[2].astype(int), k[3], k[4], k[5]
    cm = raw.groupby(["model", "dataset", "horizon", "cv"]).mse.agg(["mean", "std", "count"]).unstack("cv")
    cm = cm[(cm["count"]["cv0"] >= 3) & (cm["count"]["cv1"] >= 3)]
    cm.to_csv(R2 / "plug_cell_means.csv")
    rows = []
    for model, g in cm.groupby(level=0):
        a, b = g["mean"]["cv0"], g["mean"]["cv1"]
        p = wilcoxon(a, b).pvalue if len(g) >= 6 else np.nan
        rows.append(dict(model=model, cells=len(g), plugin_better=int((b < a).sum()),
                         mean_rel_change_pct=((b - a) / a * 100).mean(), p_wilcoxon=p))
    t = pd.DataFrame(rows)
    adj = holm(dict(zip(t.model, t.p_wilcoxon.fillna(1.0))))
    t["p_holm"] = t.model.map(adj)
    t.to_csv(R2 / "plug_tests.csv", index=False)
    ds = cm["mean"].groupby(level=[0, 1]).mean()
    sd = raw.groupby(["model", "dataset", "cv", "seed"]).mse.mean().groupby(["model", "dataset", "cv"]).std().unstack("cv")
    models = list(ds.index.get_level_values(0).unique())
    datasets = list(ds.index.get_level_values(1).unique())
    lines = [r"\begin{tabular}{l" + "cc" * len(models) + "}", r"\toprule",
             "Data & " + " & ".join(r"\multicolumn{2}{c}{%s}" % m for m in models) + r" \\",
             " & " + " & ".join(["Ori & +FN"] * len(models)) + r" \\", r"\midrule"]
    for dname in datasets:
        cells = []
        for m in models:
            if (m, dname) not in ds.index:
                cells += ["--", "--"]
                continue
            a, b = ds.loc[(m, dname)]["cv0"], ds.loc[(m, dname)]["cv1"]
            sa, sb = sd.loc[(m, dname)]["cv0"], sd.loc[(m, dname)]["cv1"]
            fa = r"\textbf{%.3f}" if a <= b else "%.3f"
            fb = r"\textbf{%.3f}" if b < a else "%.3f"
            cells += [(fa % a) + r"\tiny{$\pm$%.3f}" % sa, (fb % b) + r"\tiny{$\pm$%.3f}" % sb]
        lines.append(DISPLAY.get(dname, dname) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", "Wins (cells) & " + " & ".join(
        r"\multicolumn{2}{c}{%d/%d}" % (r.plugin_better, r.cells) for r in t.itertuples()) + r" \\",
        r"$p_{\mathrm{Holm}}$ & " + " & ".join(r"\multicolumn{2}{c}{%.2f}" % r.p_holm for r in t.itertuples()) + r" \\",
        r"\bottomrule", r"\end{tabular}"]
    (TEX / "r2_plugin.tex").write_text("\n".join(lines) + "\n")
    print(t.round(4).to_string())


def _b(vals):
    """bold the minimum of a list of floats"""
    m = min(v for v in vals if v == v)
    return [(r"\textbf{%.3f}" if abs(v - m) < 5e-4 else "%.3f") % v if v == v else "--" for v in vals]


def shift_full() -> None:
    cm = pd.read_csv(R2 / "shift_cell_means.csv").set_index(["dataset", "horizon", "pert"])
    deg = pd.read_csv(R2 / "shift_degradation.csv").set_index(["dataset", "horizon", "pert"])
    modes = [m for m in ["direct", "online", "none", "constant", "inverse", "learned_mask", "fan", "san"] if m in cm.columns]
    label = {"direct": "Ours", "online": "Ours, online $S$", "none": "w/o FN", "constant": "Const.", "inverse": "Inv.", "learned_mask": "Mask",
             "fan": "+FAN", "san": "+SAN"}
    pn = {"clean": "clean", "warp10": "warp 1.10", "warp25": "warp 1.25", "newfreq": "period 17", "noise10t": "noise"}
    lines = [r"\begin{tabular}{lll" + "c" * len(modes) + "}", r"\toprule",
             "Data & $H$ & Shift & " + " & ".join(label[m] for m in modes) + r" \\", r"\midrule"]
    for (ds, h), g in cm.groupby(level=[0, 1]):
        for p in ["clean", "warp10", "warp25", "newfreq", "noise10t"]:
            if (ds, h, p) not in cm.index:
                continue
            row = cm.loc[(ds, h, p), modes].tolist()
            cells = _b(row)
            if p != "clean":
                cells = [c + r"\tiny{(%+.1f)}" % deg.loc[(ds, h, p), m] for c, m in zip(cells, modes)]
            lines.append(f"{ds} & {h} & {pn[p]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TEX / "r2_shift_full.tex").write_text("\n".join(lines) + "\n")


def plugin_full() -> None:
    cm = pd.read_csv(R2 / "plug_cell_means.csv", header=[0, 1], index_col=[0, 1, 2])
    mean = cm["mean"]
    models = list(mean.index.get_level_values(0).unique())
    lines = [r"\begin{tabular}{ll" + "cc" * len(models) + "}", r"\toprule",
             "Data & $H$ & " + " & ".join(r"\multicolumn{2}{c}{%s}" % m for m in models) + r" \\",
             " & & " + " & ".join(["Ori & +FN"] * len(models)) + r" \\", r"\midrule"]
    idx = sorted({(d, h) for _, d, h in mean.index}, key=lambda x: (x[0], int(x[1])))
    for d, h in idx:
        cells = []
        for m in models:
            if (m, d, h) in mean.index:
                cells += _b([mean.loc[(m, d, h), "cv0"], mean.loc[(m, d, h), "cv1"]])
            else:
                cells += ["--", "--"]
        lines.append(f"{DISPLAY.get(d, d)} & {h} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "r2_plugin_full.tex").write_text("\n".join(lines) + "\n")


def ext() -> None:
    d = pd.read_csv(R2 / "ext_results.csv").dropna(subset=["mse"])
    k = d.key.str.split("|", expand=True)
    d["ds"], d["h"], d["m"], d["v"] = k[1], k[2].astype(int), k[3], k[4]
    name = {("Fredformer", "direct"): "Ours", ("Fredformer", "none"): "Ours w/o FN"}
    d["col"] = [name.get((m, v), m + (" +FN" if v == "cv1" else "")) for m, v in zip(d.m, d.v)]
    order = ["Ours", "Ours w/o FN", "iTransformer", "iTransformer +FN", "PatchTST", "PatchTST +FN", "DLinear", "DLinear +FN"]
    cm = d.groupby(["ds", "h", "col"])[["mse", "mae"]].mean().unstack("col")
    lines = [r"\begin{tabular}{ll" + "cc" * len(order) + "}", r"\toprule",
             "Data & $H$ & " + " & ".join(r"\multicolumn{2}{c}{%s}" % c for c in order) + r" \\",
             " & & " + " & ".join(["MSE & MAE"] * len(order)) + r" \\", r"\midrule"]
    dn = {"exchange_rate": "Exchange", "national_illness": "Illness"}
    for ds, g in cm.groupby(level=0):
        rows = [(h, g.loc[(ds, h)]) for h in sorted(g.index.get_level_values(1))]
        rows.append(("Avg.", g.mean()))
        for h, r in rows:
            mse = _b([r[("mse", c)] for c in order])
            mae = _b([r[("mae", c)] for c in order])
            lines.append(f"{dn[ds]} & {h} & " + " & ".join(x for p in zip(mse, mae) for x in p) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TEX / "r2_ext.tex").write_text("\n".join(lines) + "\n")


def efficiency(main_only: bool = True) -> None:
    d = pd.read_csv(R2 / "efficiency.csv", dtype=str).fillna("")
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Data & Attention & Params (M) & Train (ms/step) & Train peak (GB) & Infer.\ peak (GB) & Infer.\ (samples/s) \\",
             r"\midrule"]
    names = {"321": "ECL", "862": "Traffic"}
    for c, g in d.groupby("channels", sort=False):
        for r in g.itertuples():
            if main_only and r.landmarks == "16":
                continue
            att = "Full" if r.attention == "full" else r"Nystr\"om, $\tilde m{=}%s$" % r.landmarks
            if r.train_peak_mb == "OOM":
                cells = [r"\multicolumn{5}{c}{out of memory (%s GB GPU)}" % GPU_GB]
            else:
                cells = ["%.1f" % (float(r.params) / 1e6), "%.0f" % float(r.train_ms_per_step),
                         "%.1f" % (float(r.train_peak_mb) / 1024), "%.1f" % (float(r.infer_peak_mb) / 1024),
                         r.infer_samples_per_s]
            lines.append(f"{names[c]} & {att} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TEX / ("r2_efficiency.tex" if main_only else "r2_efficiency_full.tex")).write_text("\n".join(lines) + "\n")
    print(d.to_string())
    if main_only:
        efficiency(main_only=False)


GPU_GB = 96
DISPLAY = {"weather": "Weather", "electricity": "ECL", "traffic": "Traffic"}


if __name__ == "__main__":
    import sys
    for part in (sys.argv[1:] or ["controls", "shifts", "plugins", "shift_full", "plugin_full", "ext"]):
        print(f"===== {part}")
        globals()[part]()
