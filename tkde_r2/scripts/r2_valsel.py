#!/usr/bin/env python3
"""Validation-based configuration selection for the main table.

For every ETT/Weather setting, choose among the reference configuration (seed
2021 run of the control batch) and the five grid variants (seed 2021) the one with
the lowest best-epoch validation MSE. Writes the selection table and an r2_runner
job file with seeds 2022-2025 for selected non-reference variants (seed 2021 of
the selected variant is reused from the grid run).
"""

import csv
import glob
import re
import os
from pathlib import Path

import pandas as pd

PROJECT = Path(os.environ.get("TKDE_R2_ROOT", Path(__file__).resolve().parents[1]))  # tkde_r2/
R2 = PROJECT / "tables" / "r2"
VL = re.compile(r"Vali Loss: ([0-9.]+) Test Loss")


def best_val(log):
    v = [float(x) for x in VL.findall(Path(log).read_text(errors="ignore"))]
    return min(v) if v else None


def main():
    grid = pd.read_csv(R2 / "grid_results.csv").dropna(subset=["mse"])
    rows, jobs = [], []
    for (ds, h), g in grid.groupby(["data_path", "horizon"]):
        stem = ds[:-4]
        ref_logs = sorted(glob.glob(str(PROJECT / "logs" / "r2" / f"r2_ctrl_{stem}_96_{h}_direct_seed2021_*.log")))
        cands = {}
        if ref_logs:
            ctrl = pd.read_csv(R2 / "ctrl_results.csv").dropna(subset=["mse"])
            ref = ctrl[(ctrl.data_path == ds) & (ctrl.horizon == h) & (ctrl["mode"] == "direct") & (ctrl.seed == 2021)]
            cands["reference"] = (best_val(ref_logs[-1]), float(ref.mse.iloc[0]) if len(ref) else None, "")
        for r in g.itertuples():
            cands[r.tag.replace("grid_", "")] = (best_val(r.log), r.mse, r.extra)
        cands = {k: v for k, v in cands.items() if v[0] is not None}
        sel = min(cands, key=lambda k: cands[k][0])
        rows.append(dict(dataset=stem, horizon=h, selected=sel, val=cands[sel][0], test_seed2021=cands[sel][1],
                         ref_val=cands.get("reference", (None,))[0], ref_test_seed2021=cands.get("reference", (None, None))[1],
                         n_candidates=len(cands)))
        if sel != "reference":
            for s in ("2022", "2023", "2024", "2025"):
                jobs.append(dict(tag="grid_" + sel, data_path=ds, horizon=h, seed=s, mode="direct", extra=cands[sel][2]))
    pd.DataFrame(rows).to_csv(R2 / "valsel_selection.csv", index=False)
    with open(R2 / "valsel_seed_jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "data_path", "horizon", "seed", "mode", "extra"])
        w.writeheader()
        w.writerows(jobs)
    print(pd.DataFrame(rows).round(4).to_string())
    print("jobs", len(jobs))


if __name__ == "__main__":
    main()
