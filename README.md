# Fredformer / Fredformer-v2

This repository contains the code of

* **Fredformer-v2: Frequency-Debiased Transformer with Stationarity-Aware Frequency Weighting for Time Series Forecasting** (journal extension, under review at IEEE TKDE), and
* **Fredformer: Frequency Debiased Transformer for Time Series Forecasting** (KDD 2024).

## What is new in Fredformer-v2

* **FredNormer** (`--cv_weighting_mode direct`, default): before training, the coefficient of variation
  S(k) = sigma(k)/mu(k) of the DFT amplitudes of instance-normalized training windows is computed once
  (pooled over windows and channels; S(k) = 0 if mu(k) <= 1e-8). The same real factor S(k) multiplies the real
  and imaginary DFT coefficients of every input window, so phase and conjugate symmetry are preserved.
  FredNormer has no trainable parameters. `--disable_cv_weighting` removes it.
* Control weightings (constant, permuted, inverse, learned mask, learned monotone), FAN/SAN plug-ins, and
  test-only distribution shifts used for the controlled evaluation.
* Nystrom channel attention for high-dimensional data (`--use_nys 1 --num_landmarks 64`).
* All experiments of the TKDE revision, with job lists, results, logs, and table scripts: see [`tkde_r2/`](tkde_r2/README.md).

# Dependencies
Fredformer is built based on PyTorch.
You can install PyTorch following the instructions in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

# About data:
We have prepared a dataset for this example: Weather. The CSV file is in the dataset folder.

# Custom Data Usage

If you are using your own data, please format it as a CSV file, with each row representing a sample and each column representing a channel. 
When selecting the data type "data_name," please choose "Custom" and ensure the CSV file is stored in the `dataset` folder.

# Some main parameters:

- `--patch_len` (int, default=16): Frequency patching length.
- `--cf_dim` (int, default=48): Feature dimension.
- `--cf_drop` (float, default=0.2): Dropout rate.
- `--cf_depth` (int, default=2): Number of Transformer layers.
- `--cf_heads` (int, default=6): Number of multi-heads.
- `--cf_mlp` (int, default=128): Feed-forward network dimension.
- `--cf_head_dim` (int, default=32): Dimension for each single head.
- `--use_nys` (int, default=0): Use Nyström method (0 = No, 1 = Yes).
- `--mlp_drop` (float, default=0.3): Dropout rate for MLP.


# Training. 
For Fredfromer:
The scripts for our are in the directory ```./scripts/Fredformer```.
You can run the following command, then open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/Fredformer/weather.sh
 ```
Log files will be generated and updated in  ```./logs/``` during training.

# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{Piao2024fredformer,
  title={Fredformer: Frequency Debiased Transformer for Time Series Forecasting},
  author={Xihao Piao and Zheng Chen and Taichi Murayama and Yasuko Matsubara and Yasushi Sakurai},
  booktitle={30th SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track},
  year={2024},
}

```

