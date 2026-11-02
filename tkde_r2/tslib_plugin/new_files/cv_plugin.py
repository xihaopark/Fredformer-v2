"""FredNormer frequency weighting as a model-agnostic plug-in (TKDE R2, Table IV).

Mirrors Fredformer-v2: per-frequency CV = std/mean of DFT amplitudes of
instance-normalized training windows, pooled over windows and channels.
Each input window is instance-normalized, its spectrum multiplied by the fixed
CV weights, transformed back, and de-normalized, so the backbone (and its own
normalization) is unchanged apart from the reweighted spectral shape.
"""

import torch
from torch import nn


def instance_norm(x, eps=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + eps)
    return (x - mean) / std, mean, std


@torch.no_grad()
def compute_cv(train_loader, device, max_batches=100):
    amps = []
    for i, (batch_x, *_rest) in enumerate(train_loader):
        if i >= max_batches:
            break
        xn, _, _ = instance_norm(batch_x.float().to(device))
        amps.append(torch.fft.fft(xn, dim=1).abs())
    amps = torch.cat(amps, dim=0)  # [N, L, C]
    mean = amps.mean(dim=(0, 2))
    std = amps.std(dim=(0, 2))
    return torch.where(mean > 1e-8, std / mean, torch.zeros_like(mean))


class CVPlugin(nn.Module):
    def __init__(self, model, cv):
        super().__init__()
        self.inner = model
        self.register_buffer('cv', cv.view(1, -1, 1))

    def forward(self, x_enc, *args, **kwargs):
        xn, mean, std = instance_norm(x_enc)
        xw = torch.fft.ifft(torch.fft.fft(xn, dim=1) * self.cv, dim=1).real
        return self.inner(xw * std + mean, *args, **kwargs)


class FANPlugin(nn.Module):
    """FAN (Ye et al., NeurIPS 2024) wrapped around a TSLib backbone; aux loss via .aux_loss(y)."""

    def __init__(self, model, seq_len, pred_len, enc_in, topk=20):
        super().__init__()
        from layers.plugin_norms import FAN
        self.inner = model
        self.fan = FAN(seq_len, pred_len, enc_in, topk)

    def forward(self, x_enc, *args, **kwargs):
        out = self.inner(self.fan.normalize(x_enc), *args, **kwargs)
        return self.fan.denormalize(out[:, -self.fan.model_all[-1].out_features:, :])

    def aux_loss(self, y):
        return self.fan.loss(y)
