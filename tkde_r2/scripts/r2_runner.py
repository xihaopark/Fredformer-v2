#!/usr/bin/env python3
"""Second-round (R2) experiment runner.

Builds Fredformer commands from the locked reference parameters, runs them in a
GPU slot pool, and records MSE/MAE per job in a CSV. Re-running the same job
file skips jobs whose result row already exists.

Job file: CSV with columns
    tag,data_path,horizon,seed,mode,extra
where mode is a --cv_weighting_mode value or "none" (FredNormer disabled) and
extra is a space-separated string of additional run_longExp.py arguments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(os.environ.get("TKDE_R2_ROOT", Path(__file__).resolve().parents[1]))  # tkde_r2/
WORKTREE = PROJECT.parent  # repository root
PY = Path(os.environ.get("TKDE_PY", sys.executable))
REF = PROJECT / "tables" / "reference_params_original.csv"
LOG_DIR = PROJECT / "logs" / "r2"
MSE_RE = re.compile(r"^mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", re.M)
FIELDS = ["tag", "data_path", "horizon", "seed", "mode", "extra", "mse", "mae", "seconds", "log"]


def load_refs() -> dict[tuple[str, str], dict[str, str]]:
    return {(r["data_path"], r["pred_len"]): r for r in csv.DictReader(REF.open())}


def build_command(ref: dict[str, str], job: dict[str, str], gpu: str) -> list[str]:
    stem = Path(job["data_path"]).stem
    model_id = f"r2_{job['tag']}_{stem}_96_{job['horizon']}_{job['mode']}_seed{job['seed']}"
    if job.get("model_id"):  # evaluate an existing checkpoint of another job
        model_id = job["model_id"]
    elif job.get("extra"):
        model_id += "_x" + hashlib.md5(job["extra"].encode()).hexdigest()[:6]
    cmd = [
        str(PY), "-u", "run_longExp.py",
        "--random_seed", job["seed"],
        "--is_training", "1",
        "--root_path", ref.get("root_path_override") or "./dataset/",
        "--data_path", job["data_path"],
        "--model_id", model_id,
        "--model", "Fredformer",
        "--data", ref["dataset"],
        "--features", ref["features"],
        "--seq_len", ref["seq_len"],
        "--label_len", ref["label_len"],
        "--pred_len", job["horizon"],
        "--enc_in", ref["enc_in"],
        "--c_out", ref["c_out"],
        "--e_layers", ref["e_layers"],
        "--n_heads", ref["n_heads"],
        "--d_model", ref["d_model"],
        "--d_ff", ref["d_ff"],
        "--dropout", ref["dropout"],
        "--fc_dropout", ref["fc_dropout"],
        "--head_dropout", ref["head_dropout"],
        "--patch_len", ref["patch_len"],
        "--stride", ref["stride"],
        "--patience", ref["patience"],
        "--des", f"R2_{job['tag']}",
        "--train_epochs", ref["train_epochs"],
        "--itr", "1",
        "--batch_size", ref["batch_size"],
        "--num_workers", "0",
        "--skip_save_arrays",
        "--learning_rate", ref["learning_rate"],
        "--lradj", ref["lradj"],
        "--pct_start", ref["pct_start"],
        "--gpu", gpu,
        "--cf_dim", ref["cf_dim"],
        "--cf_depth", ref["cf_depth"],
        "--cf_heads", ref["cf_heads"],
        "--cf_mlp", ref["cf_mlp"],
        "--cf_head_dim", ref["cf_head_dim"],
        "--use_nys", ref["use_nys"],
        "--individual", ref["individual"],
    ]
    if job["mode"] == "none":
        cmd.append("--disable_cv_weighting")
    else:
        cmd += ["--cv_weighting_mode", job["mode"]]
    extra = shlex.split(job.get("extra") or "")
    if "--keep_arrays" in extra:  # save predictions for the spectral-error analysis
        extra.remove("--keep_arrays")
        cmd.remove("--skip_save_arrays")
    # later occurrences of an argument override earlier ones in argparse
    return cmd + extra


def job_key(job: dict[str, str]) -> tuple[str, ...]:
    return (job["tag"], job["data_path"], str(job["horizon"]), str(job["seed"]), job["mode"], job.get("extra") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--slots-per-gpu", type=int, default=3)
    ap.add_argument("--train-epochs", default=None, help="override train_epochs (smoke tests)")
    args = ap.parse_args()

    refs = load_refs()
    jobs = list(csv.DictReader(open(args.jobs)))
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {job_key(r) for r in csv.DictReader(out.open()) if r.get("mse")}
    pending = [j for j in jobs if job_key(j) not in done]
    print(f"jobs={len(jobs)} done={len(jobs) - len(pending)} pending={len(pending)}", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with out.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    slots = [g for g in args.gpus.split(",") for _ in range(args.slots_per_gpu)]
    running: dict[str, tuple] = {}
    failures = 0
    while pending or running:
        for slot_id, gpu in enumerate(slots):
            key = f"{slot_id}"
            if key in running or not pending:
                continue
            job = pending.pop(0)
            ref = dict(refs[(job["ref_data_path"] if job.get("ref_data_path") else job["data_path"], str(job["horizon"]))])
            if job.get("root_path"):
                ref["root_path_override"] = job["root_path"]
            if args.train_epochs:
                ref["train_epochs"] = args.train_epochs
            cmd = build_command(ref, job, "0")
            log_path = LOG_DIR / f"{cmd[cmd.index('--model_id') + 1]}_{job['tag']}_{time.strftime('%Y%m%d_%H%M%S')}_{len(running)}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[k] = "4"
            log = log_path.open("w")
            log.write("# command: " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[key] = (proc, job, log_path, log, time.time())
            print(f"START gpu={gpu} {job_key(job)}", flush=True)
        for key in list(running):
            proc, job, log_path, log, t0 = running[key]
            if proc.poll() is None:
                continue
            log.close()
            del running[key]
            text = log_path.read_text(errors="ignore")
            m = MSE_RE.findall(text)
            row = {k: job.get(k, "") for k in ("tag", "data_path", "horizon", "seed", "mode", "extra")}
            row.update(seconds=f"{time.time() - t0:.0f}", log=str(log_path))
            if proc.returncode == 0 and m:
                row.update(mse=m[-1][0], mae=m[-1][1])
                print(f"DONE mse={m[-1][0]} {job_key(job)}", flush=True)
            else:
                failures += 1
                row.update(mse="", mae="")
                print(f"FAIL rc={proc.returncode} {job_key(job)} log={log_path}", flush=True)
            with out.open("a", newline="") as f:
                csv.DictWriter(f, FIELDS).writerow(row)
        time.sleep(5)
    print(f"ALL_DONE failures={failures}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
