# Reproducing the TKDE revision experiments

This folder contains everything used for the experiments of the second-round revision of
*Fredformer-v2: Frequency-Debiased Transformer with Stationarity-Aware Frequency Weighting for Time Series Forecasting*:
job lists, result tables, the scripts that produce the paper tables, and the training logs.

```
tkde_r2/
  scripts/        experiment runners, table generators, profiling, synthetic study
  tables/r2/      *_jobs.csv (every run) and *_results.csv (MSE/MAE per run)
  tables/reference_params_original.csv   reference configuration per dataset/horizon
  tslib_plugin/   FredNormer plug-in and baselines for Time-Series-Library
  logs.tar.gz     training logs of all runs
  latex/          output folder of the table generators
```

Logs and result tables were post-processed only to remove local file-system paths and an internal code name.

## Setup

1. Install the repository requirements (`pip install -r requirements.txt`, PyTorch >= 2.0).
2. Place the benchmark CSV files (ETT, Weather, Electricity, Traffic, Solar, ExchangeRate, Illness) in `./dataset/`.
   They are the standard files distributed with iTransformer / Time-Series-Library.
3. For the plug-in, FAN/FreDF, and ExchangeRate/Illness runs, check out
   [Time-Series-Library](https://github.com/thuml/Time-Series-Library) at commit `4e938a1`, apply
   `tslib_plugin/tslib_4e938a1.patch`, copy `tslib_plugin/new_files/*.py` into its `layers/`,
   and set `TSLIB_DIR` to that checkout.

## FredNormer options (`run_longExp.py`)

| Option | Meaning |
|---|---|
| `--cv_weighting_mode direct` | FredNormer: multiply real and imaginary DFT coefficients by S(k) = sigma(k)/mu(k) (default) |
| `--disable_cv_weighting` | model without FredNormer |
| `--cv_weighting_mode constant / shuffled / inverse` | constant weight mean(S), permuted S, 1/S |
| `--cv_weighting_mode learned_mask / monotone` | learned frequency mask; learned w(S) = softplus(a) S + softplus(b) |
| `--norm_plugin fan / san` | FAN or SAN attached to the FredNormer-free model |
| `--use_nys 1 --num_landmarks m` | Nystrom channel attention |
| `--fredf_alpha 0.5` | train with the FreDF loss (1-alpha) MSE + alpha mean\|rfft(pred)-rfft(true)\| (Table I, "Ours + FreDF loss") |
| `--eval_only` | load the checkpoint of an identical training run and only run the test split |
| `--cv_reestimate_test causal` | online S(k): running amplitude statistics over the test look-back windows seen so far, updated before each batch is predicted (Table III, "online S"); `pooled` uses all test windows at once (reference only) |
| `--cv_prior_windows n0` | warm start of the causal estimate with the training statistics weighted as n0 windows (Supp. Table S10) |

## Paper tables

| Paper item | Job list | Runner | Results | Table script |
|---|---|---|---|---|
| Table I (main results) | `main_big_jobs`, `main_ecl_jobs`, `main_traffic_jobs`, `valsel_seed_jobs` | `r2_runner.py` | `main_big_results`, `ctrl_results`, `grid_results` | `r2_main_tables.py` |
| Table I, FAN / FreDF rows | `base_jobs*.csv` | `cmd_pool.py` | `base_results` | `r2_main_tables.py` |
| Table II (FredNormer controls) | `ctrl_jobs*.csv` | `r2_runner.py` | `ctrl_results` | `r2_summarize.py controls` |
| Table III (test-only shifts) | `shift_jobs`, `normcmp_jobs*` | `r2_runner.py` | `shift_results`, `normcmp_results` | `r2_summarize.py shifts` |
| Table IV (plug-in) | `plug_jobs` | `cmd_pool.py` | `plug_results` | `r2_summarize.py plugins` |
| Table I, "Ours + FreDF loss" row | `fredfloss_jobs*.csv` (tag `fredfloss`) | `r2_runner.py` | `fredfloss_results` | `r2_main_tables.py` |
| Table III, "online S" row; Supp. Table S10 | `onlinec_jobs`, `onlinew_jobs` (eval-only on the `shift_jobs` checkpoints) | `r2_runner.py` | `onlinec_results`, `onlinew_results`, `online_cells` | `r2_summarize.py shifts`, `r2_online_summary.py` |
| Table VII (efficiency) | -- | `profile_efficiency.py` | `efficiency.csv` | `r2_summarize.py efficiency` |
| Sec. VI-D, ECL with 16 landmarks | `fredfloss_jobs*.csv` (tag `nys16`) | `r2_runner.py` | `fredfloss_results` | -- |
| Fig. 3 (synthetic frequency bias) | -- | `synthetic_frequency_bias.py` | `synth_bias*.csv` | -- |
| Supp. Table S2 (validation selection) | `grid_jobs` | `r2_runner.py` | `grid_results` | `r2_valsel.py` |
| Supp. Table S6 (ExchangeRate, Illness) | `ext_jobs` | `cmd_pool.py` | `ext_results` | `r2_summarize.py ext` |

Example:

```bash
cd tkde_r2/scripts
python generate_shift_datasets.py ETTh1.csv ETTh2.csv ETTm2.csv weather.csv   # test-only shifted copies
python r2_runner.py ../tables/r2/ctrl_jobs.csv --out ../tables/r2/ctrl_results.csv --gpus 0,1 --slots-per-gpu 2
TSLIB_DIR=/path/to/Time-Series-Library python cmd_pool.py ../tables/r2/plug_jobs.csv --out ../tables/r2/plug_results.csv --gpus 0,1
python r2_summarize.py controls shifts plugins efficiency
```

A job row with a `model_id` column evaluates the checkpoint of that earlier job instead of training
(used by the eval-only / online-S job lists). `r2_runner.py` and `cmd_pool.py` skip jobs whose result row already exists, so the shipped result tables
regenerate the paper tables without retraining. Seeds are averaged within each dataset-horizon setting;
comparisons use paired two-sided Wilcoxon tests across settings with Holm correction.
Configurations are selected on the validation split only.
