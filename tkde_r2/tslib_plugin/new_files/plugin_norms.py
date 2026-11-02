"""Competing plug-in normalizers for the R2 robustness comparison.

Ported from the official implementations:
- FAN (Ye et al., NeurIPS 2024): https://github.com/wayne155/FAN
- SAN (Liu et al., NeurIPS 2023): https://github.com/icantnamemyself/SAN

Both wrap the Fredformer backbone (FredNormer disabled) as model-agnostic
normalize/denormalize steps. Their auxiliary losses are added to the forecasting
loss and trained jointly (SAN's original schedule pre-trains the statistics
predictor for a few epochs first; we train it jointly from the start).
"""

import torch
from torch import nn
import torch.nn.functional as F


def main_freq_part(x, k):
    """Split x [B, T, C] into its top-k rFFT components and the residual."""
    xf = torch.fft.rfft(x, dim=1)
    indices = torch.topk(xf.abs(), k, dim=1).indices
    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    x_filtered = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1).real.float()
    return x - x_filtered, x_filtered


class FAN(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20):
        super().__init__()
        self.freq_topk = freq_topk
        self.model_freq = nn.Sequential(nn.Linear(seq_len, 64), nn.ReLU())
        self.model_all = nn.Sequential(nn.Linear(64 + seq_len, 128), nn.ReLU(), nn.Linear(128, pred_len))

    def normalize(self, x):
        norm_input, x_filtered = main_freq_part(x, self.freq_topk)
        main_t, x_t = x_filtered.transpose(1, 2), x.transpose(1, 2)
        pred = self.model_all(torch.cat([self.model_freq(main_t), x_t], dim=-1))
        self.pred_main = pred.transpose(1, 2)
        return norm_input

    def denormalize(self, y_norm):
        self.pred_residual = y_norm
        return y_norm + self.pred_main

    def loss(self, true):
        residual, main = main_freq_part(true, self.freq_topk)
        return F.mse_loss(self.pred_main, main) + F.mse_loss(self.pred_residual, residual)


class _SANMLP(nn.Module):
    def __init__(self, seq_len_new, period_len, pred_len_new, mode):
        super().__init__()
        self.input = nn.Linear(seq_len_new, 512)
        self.input_raw = nn.Linear(seq_len_new * period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, pred_len_new)
        self.final_activation = nn.ReLU() if mode == 'std' else nn.Identity()

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = torch.cat([self.input(x), self.input_raw(x_raw)], dim=-1)
        return self.final_activation(self.output(self.activation(x))).permute(0, 2, 1)


class SAN(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, period_len=24):
        super().__init__()
        self.period_len = period_len
        self.channels = enc_in
        self.epsilon = 1e-5
        self.model = _SANMLP(seq_len // period_len, period_len, pred_len // period_len, 'mean')
        self.model_std = _SANMLP(seq_len // period_len, period_len, pred_len // period_len, 'std')
        self.weight = nn.Parameter(torch.ones(2, enc_in))

    def normalize(self, x):
        bs, length, dim = x.shape
        xr = x.reshape(bs, -1, self.period_len, dim)
        mean = torch.mean(xr, dim=-2, keepdim=True)
        std = torch.std(xr, dim=-2, keepdim=True)
        norm_input = (xr - mean) / (std + self.epsilon)
        mean_all = torch.mean(x, dim=1, keepdim=True)
        out_mean = self.model(mean.squeeze(2) - mean_all, x - mean_all) * self.weight[0] + mean_all * self.weight[1]
        out_std = self.model_std(std.squeeze(2), x)
        self.station_pred = torch.cat([out_mean, out_std], dim=-1)
        return norm_input.reshape(bs, length, dim)

    def denormalize(self, y_norm):
        bs, length, dim = y_norm.shape
        yr = y_norm.reshape(bs, -1, self.period_len, dim)
        mean = self.station_pred[:, :, :self.channels].unsqueeze(2)
        std = self.station_pred[:, :, self.channels:].unsqueeze(2)
        return (yr * (std + self.epsilon) + mean).reshape(bs, length, dim)

    def loss(self, true):
        bs, length, dim = true.shape
        yr = true.reshape(bs, -1, self.period_len, dim)
        target = torch.cat([torch.mean(yr, dim=2), torch.std(yr, dim=2)], dim=-1)
        return F.mse_loss(self.station_pred, target)
