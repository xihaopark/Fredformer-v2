import torch
import torch.nn as nn


class ConditionalExpansionDecoder(nn.Module):
    """
    Conditional Expansion Decoder (Final Version)
    
    This is the final, perfected decoder design that implements the "expand first, modulate later" approach.
    It processes flattened transformer outputs and applies conditional affine modulation.
    
    Input: [B*N, C, D] where B=batch, N=num_patches, C=channels, D=model_dim
    Output: [B*N, C, H_patch] where H_patch = target_window // num_patches
    """
    
    def __init__(self, num_patches, channels, model_dim, target_patch_len, krl_lambda_len, hidden_dim=64):
        """
        Initialize Conditional Expansion Decoder
        
        Args:
            num_patches: Number of patches
            channels: Number of channels/variables
            model_dim: Model dimension (d_model)
            target_patch_len: Target output patch length (H_patch)
            krl_lambda_len: Length of lambda vector from KRL (frequency-level)
            hidden_dim: Hidden dimension for affine parameter generator
        """
        super().__init__()
        self.num_patches = num_patches
        self.channels = channels
        self.model_dim = model_dim
        self.target_patch_len = target_patch_len
        
        # Step 1: Dimensionality expansion layer per channel
        # Input: [C, D], Output: [H_patch] for each channel
        self.expansion_layer = nn.Linear(model_dim, target_patch_len)
        
        # Step 2: Affine parameter generator for conditional modulation
        # This MLP generates γ and β based on frequency-level λ guidance signals
        # Input: [H_patch] (frequency-level lambdas), Output: [H_patch * 2] (gamma and beta)
        self.affine_param_generator = nn.Sequential(
            nn.Linear(target_patch_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_patch_len * 2)
        )
        
        # Store the expected lambda length for validation
        self.expected_lambda_len = target_patch_len

    def forward(self, encoder_output_feat, lambdas):
        """
        Forward pass of Conditional Expansion Decoder
        
        Args:
            encoder_output_feat: Output from transformer [B*N, C, D]
                               where B*N=batch_size*num_patches, C=channels, D=model_dim
            lambdas: Lambda weights from KRL [L_freq] (frequency-level lambdas)
            
        Returns:
            final_decoded_patches: Conditionally modulated patches [B*N, C, H_patch]
        """
        # encoder_output_feat: [B*N, C, D]
        BN, C, D = encoder_output_feat.shape

        # Step 1: Dimensionality Expansion per channel
        preliminary_decoded = self.expansion_layer(encoder_output_feat)  # [B*N, C, H_patch]

        # Step 2: Generate and apply affine transformation using frequency-level lambdas
        lambda_cos = lambdas['cos']  # [L_freq]
        
        # Handle lambda length mismatch: KRL output vs expected patch total length
        expected_total_length = self.num_patches * self.target_patch_len
        actual_lambda_length = lambda_cos.shape[0]
        
        if actual_lambda_length < expected_total_length:
            # Pad lambda with the last value to match expected length
            padding_size = expected_total_length - actual_lambda_length
            lambda_cos = torch.cat([lambda_cos, lambda_cos[-1].repeat(padding_size)])
        elif actual_lambda_length > expected_total_length:
            # Truncate lambda to match expected length
            lambda_cos = lambda_cos[:expected_total_length]
        
        # Extract frequency-level lambdas for each patch
        # Reshape lambda to [N, H_patch] where each row contains frequency-level lambdas for that patch
        lambda_reshaped = lambda_cos.reshape(self.num_patches, self.target_patch_len)  # [N, H_patch]
        
        # Generate affine parameters for each patch using its frequency-level lambdas
        # Each patch gets its own gamma and beta based on its frequency components
        affine_params = self.affine_param_generator(lambda_reshaped)  # [N, H_patch * 2]
        gamma, beta = torch.chunk(affine_params, 2, dim=-1)  # Each is [N, H_patch]
        
        # Reshape for broadcasting over Batch and Channel dimensions
        # from [N, H_patch] -> [1, N, 1, H_patch] -> [B*N, 1, H_patch]
        B = BN // self.num_patches
        gamma = gamma.view(1, self.num_patches, self.target_patch_len).expand(B, -1, -1).reshape(BN, 1, self.target_patch_len)
        beta = beta.view(1, self.num_patches, self.target_patch_len).expand(B, -1, -1).reshape(BN, 1, self.target_patch_len)
        
        final_decoded_patches = preliminary_decoded * gamma + beta  # [B*N, C, H_patch]
        return final_decoded_patches


class KernelAdaptiveDecoder(nn.Module):
    """
    Kernel-Adaptive Decoder (Legacy version - kept for reference)
    
    This is the previous version of the decoder. It has been replaced by
    ConditionalExpansionDecoder for better design and performance.
    """
    
    def __init__(self, model_dim, num_patches, patch_len, hidden_dim=64):
        super(KernelAdaptiveDecoder, self).__init__()
        self.num_patches = num_patches
        self.patch_len = patch_len
        self.model_dim = model_dim
        
        # Affine parameter generator - a small MLP
        self.affine_param_generator = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, model_dim * 2)
        )
        
        # Final projection layer to convert features to predictions
        self.final_projection = nn.Linear(model_dim, patch_len)

    def forward(self, encoder_output, lambdas):
        """
        Forward pass of kernel-adaptive decoder (Legacy version)
        
        Args:
            encoder_output: Output from Fredformer encoder [B, C, N, D]
                           where D is the model dimension (d_model)
            lambdas: Lambda weights from KRL [L_freq] (e.g., lambdas['cos'])
            
        Returns:
            decoded_output: Transformed encoder output [B, C, N, patch_len]
        """
        B, C, N, D = encoder_output.shape
        
        # 1. Calculate average lambda per patch
        lambda_cos = lambdas['cos']
        
        # Ensure we have enough lambda values for all patches
        if lambda_cos.shape[0] < N:
            # Pad with the last value if needed
            padding_size = N - lambda_cos.shape[0]
            lambda_cos = torch.cat([lambda_cos, lambda_cos[-1].repeat(padding_size)])
        
        # Take the first N lambda values and calculate patch average
        lambda_patch_avg = lambda_cos[:N].unsqueeze(-1)  # [N, 1]
        
        # 2. Generate affine parameters
        affine_params = self.affine_param_generator(lambda_patch_avg)  # [N, D*2]
        gamma, beta = torch.chunk(affine_params, 2, dim=-1)  # Each is [N, D]
        
        # 3. Reshape for broadcasting and apply transform
        gamma = gamma.view(1, 1, N, D)  # [1, 1, N, D]
        beta = beta.view(1, 1, N, D)    # [1, 1, N, D]
        
        # Apply affine transformation: output = feature * γ + β
        transformed_features = encoder_output * gamma + beta  # [B, C, N, D]
        
        # 4. Final projection to prediction length
        # Reshape for linear projection: [B*C*N, D] -> [B*C*N, patch_len]
        batch_size, channels, patches, features = transformed_features.shape
        reshaped_features = transformed_features.view(-1, features)
        
        # Apply final projection
        predictions = self.final_projection(reshaped_features)  # [B*C*N, patch_len]
        
        # Reshape back to original dimensions
        decoded_output = predictions.view(batch_size, channels, patches, self.patch_len)
        
        return decoded_output 