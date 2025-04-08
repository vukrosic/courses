# lesson_4_llama4_feedforward_code.py

# %% [markdown]
# # Understanding the Llama 4 Feed-Forward Network (FFN)
#
# This tutorial explores the Feed-Forward Network (FFN) used in the Llama 4 architecture, specifically the MLP (Multi-Layer Perceptron) variant used in dense layers. The FFN is applied independently to each token position after the attention mechanism and residual connection. Its role is to further process the information aggregated by the attention layer, adding non-linearity and increasing the model's representational capacity.
#
# Llama models typically use a specific FFN structure involving gated linear units (like SwiGLU), which has shown strong performance. We will break down the `Llama4TextMLP` module and its surrounding components (like Layer Normalization) step by step.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# %% [markdown]
# ## Step 1: Setup and Configuration
#
# First, let's define configuration parameters relevant to the FFN and create sample input data. This input data represents the hidden state *after* the attention block and its residual connection, but *before* the post-attention layer normalization.

# %%
# Configuration (Simplified for clarity)
hidden_size = 128  # Dimensionality of the model's hidden states
# Intermediate size for the FFN. Often calculated based on hidden_size.
# A common pattern is around 2.67 * hidden_size, rounded up to a multiple (e.g., 256).
ffn_intermediate_ratio = 8 / 3
multiple_of = 32 # Common multiple for FFN intermediate size
intermediate_size = int(hidden_size * ffn_intermediate_ratio)
# This line of code adjusts the intermediate_size to be a multiple of 'multiple_of'.
# It does this by first adding 'multiple_of - 1' to 'intermediate_size', then performing integer division by 'multiple_of',
# and finally multiplying the result by 'multiple_of'. This effectively rounds up 'intermediate_size' to the nearest multiple of 'multiple_of'.
intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of

hidden_act = "silu" # Activation function (SiLU/Swish)
rms_norm_eps = 1e-5 # Epsilon for RMSNorm
ffn_bias = False # Whether to use bias in FFN linear layers

# Sample Input (Represents output of Attention + Residual)
batch_size = 2
sequence_length = 10
# This is the state before the post-attention LayerNorm
input_to_ffn_block = torch.randn(batch_size, sequence_length, hidden_size)

print("Configuration:")
print(f"  hidden_size: {hidden_size}")
print(f"  intermediate_size: {intermediate_size} (Calculated from ratio {ffn_intermediate_ratio:.2f}, multiple of {multiple_of})")
print(f"  hidden_act: {hidden_act}")
print(f"  rms_norm_eps: {rms_norm_eps}")

print("\nSample Input Shape (Before FFN Block Norm):")
print(f"  input_to_ffn_block: {input_to_ffn_block.shape}")

# %% [markdown]
# ## Step 2: Pre-Normalization (Post-Attention LayerNorm)
#
# Before passing the hidden state through the FFN, Llama applies a Layer Normalization step. Unlike standard Transformers that often use LayerNorm *after* the FFN and residual connection, Llama uses a pre-normalization approach. Here, it's the "post-attention" normalization (`post_attention_layernorm` in the original `Llama4TextDecoderLayer`). Llama typically uses RMSNorm.

# %%
# Simplified RMSNorm Implementation
class SimplifiedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # Learnable gain parameter
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32) # Calculate in float32 for stability
        # Calculate variance (mean of squares) across the hidden dimension
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Normalize: input / sqrt(variance + epsilon)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Apply learnable weight and cast back to original dtype
        return (self.weight * hidden_states).to(input_dtype)

# Instantiate and apply the normalization
post_attention_norm = SimplifiedRMSNorm(hidden_size, eps=rms_norm_eps)
normalized_hidden_states = post_attention_norm(input_to_ffn_block)

print("Shape after Post-Attention RMSNorm:")
print(f"  normalized_hidden_states: {normalized_hidden_states.shape}")

# %% [markdown]
# ## Step 3: The Feed-Forward Network (MLP with Gated Linear Unit)
#
# The core of the FFN in Llama's dense layers is an MLP using a gated mechanism, often referred to as SwiGLU (SiLU Gated Linear Unit). It consists of three linear projections:
#
# 1.  **`gate_proj`:** Projects the input to the `intermediate_size`.
# 2.  **`up_proj`:** Also projects the input to the `intermediate_size`.
# 3.  **`down_proj`:** Projects the result back down to the `hidden_size`.
#
# The calculation is: `down_proj( F.silu(gate_proj(x)) * up_proj(x) )`
#
# - The `gate_proj` output is passed through an activation function (SiLU/Swish).
# - This activated gate is element-wise multiplied by the `up_proj` output.
# - The result is then projected back to the original hidden dimension by `down_proj`.

# %%
# Define FFN layers
gate_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
up_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
down_proj = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)

# Define the activation function (SiLU/Swish)
# ACT2FN could be used here, but for simplicity, we directly use nn.SiLU
if hidden_act == "silu":
    activation_fn = nn.SiLU()
else:
    # Add other activations if needed, otherwise raise error
    raise NotImplementedError(f"Activation {hidden_act} not implemented in this example.")

# Apply the FFN layers to the *normalized* hidden states
gate_output = gate_proj(normalized_hidden_states)
up_output = up_proj(normalized_hidden_states)

# Apply activation to the gate and perform element-wise multiplication
activated_gate = activation_fn(gate_output)
gated_result = activated_gate * up_output

# Apply the final down projection
ffn_output = down_proj(gated_result)

print("\nShapes within FFN:")
print(f"  gate_output: {gate_output.shape}") # (batch, seq_len, intermediate_size)
print(f"  up_output: {up_output.shape}")     # (batch, seq_len, intermediate_size)
print(f"  gated_result: {gated_result.shape}") # (batch, seq_len, intermediate_size)
print(f"  ffn_output: {ffn_output.shape}")   # (batch, seq_len, hidden_size)


# %% [markdown]
# ## Step 4: Residual Connection
#
# Similar to the attention block, a residual connection is used around the FFN block. The output of the FFN (`ffn_output`) is added to the input that went *into* the FFN block (i.e., the output of the attention block + its residual, stored here as `input_to_ffn_block`).

# %%
# Add the FFN output to the input of the FFN block (before normalization)
final_output = input_to_ffn_block + ffn_output

print("\nShape after FFN Residual Connection:")
print(f"  final_output: {final_output.shape}") # Should be (batch, seq_len, hidden_size)

# %% [markdown]
# ## Step 5: Putting it Together (Simplified Llama4 FFN Block)
#
# Let's combine the normalization and MLP steps into a simplified module. Note that the residual connection is typically handled *outside* this specific module in the main `DecoderLayer`.

# %%
class SimplifiedLlama4FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.hidden_act = config['hidden_act']
        self.ffn_bias = config['ffn_bias']
        self.rms_norm_eps = config['rms_norm_eps']

        # Normalization Layer (applied before MLP)
        self.norm = SimplifiedRMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # MLP Layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.ffn_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.ffn_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.ffn_bias)

        # Activation
        if self.hidden_act == "silu":
            self.activation_fn = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {self.hidden_act} not implemented.")

    def forward(self, hidden_states):
        # 1. Apply pre-FFN normalization
        normalized_states = self.norm(hidden_states)

        # 2. Apply MLP (SwiGLU)
        gate = self.gate_proj(normalized_states)
        up = self.up_proj(normalized_states)
        down = self.down_proj(self.activation_fn(gate) * up)

        # This module returns *only* the MLP output.
        # The residual connection is applied outside.
        return down

# Instantiate and run the simplified module
ffn_config_dict = {
    'hidden_size': hidden_size,
    'intermediate_size': intermediate_size,
    'hidden_act': hidden_act,
    'ffn_bias': ffn_bias,
    'rms_norm_eps': rms_norm_eps,
}

simplified_ffn_module = SimplifiedLlama4FFN(ffn_config_dict)

# Run forward pass using the module
# Input is the state *before* the norm
mlp_output_from_module = simplified_ffn_module(input_to_ffn_block)

# Apply the residual connection externally
final_output_from_module = input_to_ffn_block + mlp_output_from_module

print("\nOutput shape from simplified FFN module (before residual):", mlp_output_from_module.shape)
print("Output shape after external residual connection:", final_output_from_module.shape)
# Verify that the manual calculation matches the module output (should be very close)
print("Outputs are close:", torch.allclose(final_output, final_output_from_module, atol=1e-6))


# %% [markdown]
# ## Conclusion
#
# The Llama 4 Feed-Forward Network block typically consists of:
# 1.  **Pre-Normalization:** An RMSNorm layer applied to the output of the previous (attention + residual) block.
# 2.  **Gated MLP (SwiGLU):** Two linear layers projecting to an intermediate dimension, combined using an activation (SiLU) and element-wise multiplication, followed by a projection back to the hidden dimension.
# 3.  **Residual Connection:** The output of the MLP is added back to the input of the normalization layer.
#
# This structure provides the necessary non-linearity and processing power for each token position within the transformer layer.

# %%