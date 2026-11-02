"""
Fredformer with FredNormer-weighted Decoder (Original Complex Version with Frequency Filter)

This module implements the original Fredformer architecture with:
1. Frequency domain filtering at input stage
2. Complex patching and transformer backbone
3. Individual channel processing with Flatten_Head
4. Real/imaginary separation and recombination
"""

__all__ = ['Fredformer_backbone']

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from layers.RevIN import RevIN
from layers.cross_Transformer_nys import Trans_C as Trans_C_nys
from layers.cross_Transformer import Trans_C
from layers.frequency_filter import FrequencyDomainFilter

class Fredformer_backbone(nn.Module):
    def __init__(self, ablation:int, mlp_drop:float, use_nys:int, output:int, mlp_hidden:int, cf_dim:int, cf_depth:int, cf_heads:int, cf_mlp:int, cf_head_dim:int, cf_drop:float, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, d_model:int, 
                head_dropout=0, padding_patch=None, individual=False, revin=True, affine=True, subtract_last=False, use_freq_filter=False, global_mask_amp=None,
                num_landmarks=5, pinv_iterations=6, eps=1e-8, disable_channel_attention=False, **kwargs):
        
        super().__init__()
        self.use_nys = use_nys
        self.ablation = ablation
        self.use_freq_filter = use_freq_filter
        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.output = output

        # Frequency Domain Filter
        if self.use_freq_filter and global_mask_amp is not None:
            # Create args-like object for filter
            class Args:
                def __init__(self, enc_in, seq_len, pred_len):
                    self.enc_in = enc_in
                    self.seq_len = seq_len
                    self.pred_len = pred_len
            
            args = Args(c_in, context_window, target_window)
            self.freq_filter = FrequencyDomainFilter(args, global_mask_amp)
        else:
            self.freq_filter = None
            
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow = target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len)/stride + 1)
        self.norm = nn.LayerNorm(patch_len)
        
        # Backbone 
        self.re_attn = True
        if self.use_nys == 0:
            self.fre_transformer = Trans_C(dim=cf_dim, depth=cf_depth, heads=cf_heads, mlp_dim=cf_mlp, dim_head=cf_head_dim, dropout=cf_drop, patch_dim=patch_len*2, horizon=self.horizon*2, d_model=d_model*2,
                                           disable_channel_attention=disable_channel_attention)
        else:
            self.fre_transformer = Trans_C_nys(dim=cf_dim, depth=cf_depth, heads=cf_heads, mlp_dim=cf_mlp, dim_head=cf_head_dim, dropout=cf_drop, patch_dim=patch_len*2, horizon=self.horizon*2, d_model=d_model*2,
                                               num_landmarks=num_landmarks, pinv_iterations=pinv_iterations, eps=eps,
                                               disable_channel_attention=disable_channel_attention)
        
        # Head
        self.head_nf_f = d_model * 2 * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        
        self.ircom = nn.Linear(self.targetwindow*2, self.targetwindow)
        self.rfftlayer = nn.Linear(self.targetwindow*2-2, self.targetwindow)
        self.final = nn.Linear(self.targetwindow*2, self.targetwindow)

        # break up R&I:
        self.get_r = nn.Linear(d_model*2, d_model*2)
        self.get_i = nn.Linear(d_model*2, d_model*2)
        self.output1 = nn.Linear(target_window, target_window)

        # ablation
        self.input = nn.Linear(c_in, patch_len*2)
        self.outpt = nn.Linear(d_model*2, c_in)
        self.abfinal = nn.Linear(patch_len*patch_num, target_window)

    def forward(self, z, cv_statistics=None):  # z: [bs x nvars x seq_len]
        
        if self.revin: 
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
            
        if self.freq_filter is not None:
            z = self.freq_filter(z)
        
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # Apply CV-based weighting if provided
        if cv_statistics is not None:
            # Expand cv_statistics to match z1 and z2 dimensions
            # cv_statistics: [seq_len] -> [1, 1, seq_len]
            cv_weights = cv_statistics.unsqueeze(0).unsqueeze(0)
            # Apply weighting to both real and imaginary parts
            z1 = z1 * cv_weights
            z2 = z2 * cv_weights

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z2: [bs x nvars x patch_num x patch_len]

        # for channel-wise_1
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        # model shape
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_len = z1.shape[3]
        
        # proposed
        z1 = torch.reshape(z1, (batch_size*patch_num, c_in, z1.shape[-1]))  # z: [bs * patch_num, nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size*patch_num, c_in, z2.shape[-1]))  # z: [bs * patch_num, nvars, patch_len]

        z = self.fre_transformer(torch.cat((z1, z2), -1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)  # z1: [bs, nvars, patch_num, horizon]
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)  # z: [bs x nvars x target_window] 
        z2 = self.head_f2(z2)  # z: [bs x nvars x target_window]
        
        z = torch.fft.ifft(torch.complex(z1, z2))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # denorm
        if self.revin: 
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def compute_cv_statistics(self, train_loader, device):
        """
        Compute CV statistics from training data.
        
        Args:
            train_loader: Training data loader
            device: Device to compute on
            
        Returns:
            cv: [context_window] - CV statistics for each frequency
        """
        print("Computing CV statistics for frequency components...")
        
        # Collect frequency domain data
        freq_data = []
        with torch.no_grad():
            for batch_x, _, _, _ in train_loader:
                batch_x = batch_x.float().to(device)
                # [B, L, C] -> [B, C, L]
                batch_x = batch_x.permute(0, 2, 1)
                # Apply RevIN if enabled
                if self.revin:
                    batch_x = self.revin_layer(batch_x, 'norm')
                # Apply frequency filter if enabled
                if self.freq_filter is not None:
                    batch_x = self.freq_filter(batch_x)
                # Convert to frequency domain
                freq_batch = torch.fft.fft(batch_x)
                freq_data.append(freq_batch)
                # Limit the number of batches to avoid memory issues
                if len(freq_data) >= 100:  # Use first 100 batches
                    break
        # Concatenate all frequency data
        freq_data = torch.cat(freq_data, dim=0)  # [total_samples, nvars, seq_len]
        # Compute CV statistics for each frequency component
        # CV = std / mean for each frequency across all samples and variables
        freq_mean = torch.mean(torch.abs(freq_data), dim=(0, 1))  # [seq_len]
        freq_std = torch.std(torch.abs(freq_data), dim=(0, 1))    # [seq_len]
        # Avoid division by zero
        cv = torch.where(freq_mean > 1e-8, freq_std / freq_mean, torch.zeros_like(freq_mean))
        print(f"CV statistics computed: shape {cv.shape}")
        return cv

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x

class Flatten_Head_t(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout):
        super().__init__()
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x
        x = self.linear3(x)
        return x
