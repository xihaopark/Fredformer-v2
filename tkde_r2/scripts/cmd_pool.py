#!/usr/bin/env python3
"""Run arbitrary experiment commands in a GPU slot pool and record MSE/MAE.

Job file: CSV with columns key,cwd,seed,cmd (cmd is a shell-quoted string).
The seed is exported as TKDE_SEED. Jobs already recorded with an MSE are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

MSE_RE = re.compile(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)")
LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "r2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    args = ap.parse_args()

    jobs = list(csv.DictReader(open(args.jobs)))
    out = Path(args.out)
    done = {r["key"] for r in csv.DictReader(out.open()) if r.get("mse")} if out.exists() else set()
    pending = [j for j in jobs if j["key"] not in done]
    print(f"jobs={len(jobs)} pending={len(pending)}", flush=True)
    if not out.exists():
        with out.open("w", newline="") as f:
            csv.writer(f).writerow(["key", "mse", "mae", "seconds", "log"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    slots = [g for g in args.gpus.split(",") for _ in range(args.slots_per_gpu)]
    running: dict[int, tuple] = {}
    while pending or running:
        for sid, gpu in enumerate(slots):
            if sid in running or not pending:
                continue
            job = pending.pop(0)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", job["key"])[:180]
            log_path = LOG_DIR / f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, TKDE_SEED=job["seed"], OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            log = log_path.open("w")
            log.write(f"# cwd: {job['cwd']}\n# command: {job['cmd']}\n")
            log.flush()
            cwd = job["cwd"].replace("<tslib>", os.environ.get("TSLIB_DIR", "<tslib>"))
            proc = subprocess.Popen(shlex.split(job["cmd"]), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[sid] = (proc, job, log_path, log, time.time())
        for sid in list(running):
            proc, job, log_path, log, t0 = running[sid]
            if proc.poll() is None:
                continue
            log.close()
            del running[sid]
            m = MSE_RE.findall(log_path.read_text(errors="ignore"))
            ok = proc.returncode == 0 and m
            with out.open("a", newline="") as f:
                csv.writer(f).writerow([job["key"], m[-1][0] if ok else "", m[-1][1] if ok else "",
                                        f"{time.time() - t0:.0f}", str(log_path)])
            print(("DONE " if ok else f"FAIL rc={proc.returncode} ") + job["key"], flush=True)
        time.sleep(5)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
