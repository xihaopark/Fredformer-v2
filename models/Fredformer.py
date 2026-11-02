"""
Fredformer Model with FredNormer-weighted Decoder (Ultra-Simplified Architecture)

This module implements the main Fredformer model with an ultra-simplified but theoretically-driven architecture:
- Fredformer backbone as a frequency debiasing feature extractor
- Simple feature-to-prediction mapping with CV-based weighting
- Ultra-simplified structure with clear theoretical foundation
"""

from typing import Callable, Optional, Union
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.Fredformer_backbone import Fredformer_backbone
from layers.plugin_norms import FAN, SAN


class Model(nn.Module):
    """
    Fredformer Model with FredNormer-weighted Decoder (Ultra-Simplified Version)
    
    This model implements an ultra-simplified but theoretically-driven architecture where:
    1. Fredformer backbone processes frequency spectrum with debiasing mechanism
    2. Simple feature-to-prediction mapping: flatten → linear projection
    3. CV-based weighting: MLP maps CV statistics to prediction weights
    4. Theoretical foundation: Stable frequency components (low CV) receive higher weights
    """
    
    def __init__(self, configs, global_mask, max_seq_len: Optional[int] = 1024, 
                 d_k: Optional[int] = None, d_v: Optional[int] = None, 
                 norm: str = 'BatchNorm', attn_dropout: float = 0.3, 
                 act: str = "gelu", key_padding_mask: Union[bool, str] = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, 
                 res_attention: bool = True, pre_norm: bool = False, 
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, 
                 pretrain_head: bool = False, head_type: str = 'flatten', 
                 verbose: bool = False, **kwargs):
        """
        Initialize Fredformer model with FredNormer-weighted decoder
        
        Args:
            configs: Configuration object containing model parameters
            global_mask: Global frequency mask for filtering (not used in simplified version)
            max_seq_len: Maximum sequence length
            d_k: Dimension of key vectors
            d_v: Dimension of value vectors
            norm: Normalization type
            attn_dropout: Attention dropout rate
            act: Activation function
            key_padding_mask: Key padding mask configuration
            padding_var: Padding variance
            attn_mask: Attention mask
            res_attention: Whether to use residual attention
            pre_norm: Whether to apply pre-normalization
            store_attn: Whether to store attention weights
            pe: Positional encoding type
            learn_pe: Whether to learn positional encoding
            pretrain_head: Whether to use pretrained head
            head_type: Type of output head
            verbose: Whether to print verbose information
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        
        # Extract configuration parameters
        self._extract_configs(configs)
        
        # Initialize Fredformer backbone with FredNormer-weighted decoder
        self.model = self._build_backbone(configs, d_k, d_v, norm, attn_dropout,
                                        act, key_padding_mask, padding_var, attn_mask,
                                        res_attention, pre_norm, store_attn, pe, 
                                        learn_pe, pretrain_head, head_type, verbose, **kwargs)
        
        # CV statistics for weighting (computed during training)
        self.cv_statistics = None

        # Competing plug-in normalizer (R2 robustness comparison)
        plugin = getattr(configs, 'norm_plugin', 'none')
        if plugin == 'fan':
            self.norm_plugin = FAN(self.context_window, self.target_window, self.c_in, getattr(configs, 'fan_topk', 20))
        elif plugin == 'san':
            self.norm_plugin = SAN(self.context_window, self.target_window, self.c_in, getattr(configs, 'san_period', 24))
        else:
            self.norm_plugin = None

        # Control variants for the frequency weighting (R2 ablations)
        self.cv_weighting_mode = getattr(configs, 'cv_weighting_mode', 'direct')
        n_half = self.context_window // 2 + 1
        if self.cv_weighting_mode == 'learned_mask':
            # Learnable frequency mask without S, initialised to identity
            self.freq_mask = nn.Parameter(torch.ones(n_half))
        elif self.cv_weighting_mode == 'monotone':
            # w = softplus(a) * CV + softplus(b): non-decreasing in CV by construction
            self.mono_a = nn.Parameter(torch.tensor([float(np.log(np.e - 1.0))]))
            self.mono_b = nn.Parameter(torch.tensor([-5.0]))

    def _extract_configs(self, configs):
        """Extract and store configuration parameters"""
        self.c_in = configs.enc_in
        self.context_window = configs.seq_len
        self.target_window = configs.pred_len
        self.n_layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.fc_dropout = configs.fc_dropout
        self.head_dropout = getattr(configs, 'head_dropout', 0)
        self.individual = getattr(configs, 'individual', False)
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch = getattr(configs, 'padding_patch', None)
        self.revin = getattr(configs, 'revin', True)
        self.affine = getattr(configs, 'affine', True)
        self.subtract_last = getattr(configs, 'subtract_last', False)
        self.use_nys = configs.use_nys
        self.num_landmarks = getattr(configs, 'num_landmarks', 5)
        self.pinv_iterations = getattr(configs, 'pinv_iterations', 6)
        self.eps = getattr(configs, 'eps', 1e-8)
        self.disable_channel_attention = getattr(configs, 'disable_channel_attention', False)
        self.ablation = getattr(configs, 'ablation', 0)
        
        # Cross-frequency transformer parameters
        self.cf_dim = configs.cf_dim
        self.cf_depth = configs.cf_depth
        self.cf_heads = configs.cf_heads
        self.cf_mlp = configs.cf_mlp
        self.cf_head_dim = configs.cf_head_dim
        self.cf_drop = getattr(configs, 'cf_drop', configs.dropout)
        self.mlp_hidden = configs.cf_mlp
        self.mlp_drop = getattr(configs, 'mlp_drop', configs.fc_dropout)

    def _build_backbone(self, configs, d_k, d_v, norm, attn_dropout, act, 
                       key_padding_mask, padding_var, attn_mask, res_attention, 
                       pre_norm, store_attn, pe, learn_pe, pretrain_head, 
                       head_type, verbose, **kwargs):
        """Build Fredformer backbone with original architecture"""
        return Fredformer_backbone(
            ablation=self.ablation,
            mlp_drop=self.mlp_drop,
            use_nys=self.use_nys,
            output=1,
            mlp_hidden=self.mlp_hidden,
            cf_dim=self.cf_dim,
            cf_depth=self.cf_depth,
            cf_heads=self.cf_heads,
            cf_mlp=self.cf_mlp,
            cf_head_dim=self.cf_head_dim,
            cf_drop=self.cf_drop,
            c_in=self.c_in,
            context_window=self.context_window,
            target_window=self.target_window,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            head_dropout=self.head_dropout,
            padding_patch=self.padding_patch,
            individual=self.individual,
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last,
            use_freq_filter=getattr(configs, 'use_freq_filter', False),
            global_mask_amp=getattr(configs, 'global_mask_amp', None),
            num_landmarks=self.num_landmarks,
            pinv_iterations=self.pinv_iterations,
            eps=self.eps,
            disable_channel_attention=self.disable_channel_attention,
            **kwargs
        )

    def compute_cv_statistics(self, train_loader, device):
        """
        Compute CV statistics from training data.
        
        Args:
            train_loader: Training data loader
            device: Device to compute on
            
        Returns:
            cv: [context_window] - CV statistics for each frequency
        """
        return self.model.compute_cv_statistics(train_loader, device)

    @staticmethod
    def mirror_half_spectrum(half, n):
        """Expand weights for bins 0..n//2 to a conjugate-symmetric length-n vector."""
        return torch.cat([half, half[1:(n + 1) // 2].flip(0)])

    def frequency_weights(self, cv_statistics):
        """Return the per-frequency weights applied to the DFT coefficients."""
        if self.cv_weighting_mode == 'learned_mask':
            return self.mirror_half_spectrum(self.freq_mask, self.context_window)
        if self.cv_weighting_mode == 'monotone' and cv_statistics is not None:
            return F.softplus(self.mono_a) * cv_statistics + F.softplus(self.mono_b)
        return cv_statistics

    def forward(self, x, cv_statistics=None):
        """
        Forward pass of Fredformer with FredNormer-weighted decoder
        
        Args:
            x: Input tensor of shape [Batch, Input_length, Channel]
            cv_statistics: Optional CV statistics for weighting. If None, uses stored statistics
            
        Returns:
            Output tensor of shape [Batch, Target_length, Channel]
        """
        # Use stored CV statistics if not provided
        if cv_statistics is None:
            cv_statistics = self.cv_statistics
        
        if self.norm_plugin is not None:
            x = self.norm_plugin.normalize(x)

        # Convert from [Batch, Input_length, Channel] to [Batch, Channel, Input_length]
        x = x.permute(0, 2, 1)
        
        # Pass through Fredformer backbone with FredNormer-weighted decoder
        output = self.model(x, self.frequency_weights(cv_statistics))
        
        # Convert back to [Batch, Target_length, Channel]
        output = output.permute(0, 2, 1)

        if self.norm_plugin is not None:
            output = self.norm_plugin.denormalize(output)
        
        return output
