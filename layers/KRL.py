import torch
import torch.nn as nn


class KernelizedReweightingLayer(nn.Module):
    """
    Kernelized Reweighting Layer (KRL) - Parallel Theoretical Analysis Engine
    
    This module serves as a parallel theoretical analysis engine:
    1. Receives amplitude-based stability statistic s and outputs λ guidance signals
    2. Uses separate neural networks to learn sin and cos differences from amplitude
    3. Does NOT participate in main data flow preprocessing
    4. Only provides theoretical guidance for the kernel-adaptive decoder
    """
    
    def __init__(self, num_frequencies, hidden_dim=128):
        super(KernelizedReweightingLayer, self).__init__()
        self.num_frequencies = num_frequencies
        
        # Two separate MLPs for real and imaginary parts
        # Both take amplitude-based statistics as input but learn different patterns
        self.mlp_cos = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.mlp_sin = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, s_statistic):
        """
        Forward pass of KRL - Parallel Analysis Engine
        
        Args:
            s_statistic: Amplitude-based stability statistic [Frequencies]
                        This should be computed as: sqrt(real^2 + imag^2)
            
        Returns:
            lambdas: Dictionary containing lambda weights for decoder guidance
        """
        # Ensure s_statistic has correct shape
        if s_statistic.dim() == 1:
            s_statistic = s_statistic.unsqueeze(-1)  # [Frequencies, 1]
        
        # Learn different lambda patterns for cos and sin from the same amplitude input
        # This allows the networks to learn how amplitude affects real vs imaginary components differently
        lambda_cos = self.mlp_cos(s_statistic).squeeze(-1)  # [Frequencies]
        lambda_sin = self.mlp_sin(s_statistic).squeeze(-1)  # [Frequencies]
        
        # Return lambda weights for decoder guidance
        lambdas = {
            'cos': lambda_cos,
            'sin': lambda_sin
        }
        
        return lambdas 