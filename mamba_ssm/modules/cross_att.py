
import torch
from torch import nn


class CrossAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(CrossAttention, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.attention = torch.nn.MultiheadAttention(d_model, num_heads)

  def forward(self, xa, xv):
    """
    Calculates cross attention between features xa and xv.

    Args:
      xa: First set of features, of shape (batch_size, seq_len_a, d_model).
      xv: Second set of features, of shape (batch_size, seq_len_v, d_model).

    Returns:
      The output of the cross attention layer, of shape (batch_size, seq_len_a, d_model).
    """

    # No need to explicitly define query, key, and value as they are inferred from xa in MultiHeadAttention
    attention_output, _ = self.attention(xa, xv, xv)  # _ is for unused weights

    return attention_output

# Example usage
d_model = 512  # Dimensionality of model
num_heads = 8  # Number of attention heads
batch_size = 16  # Batch size
seq_len_a = 10  # Sequence length of first set of features
seq_len_v = 10  # Sequence length of second set of features

# Create sample features (replace with your actual data)
xa = torch.randn(batch_size, seq_len_a, d_model)
xv = torch.randn(batch_size, seq_len_v, d_model)

# Create the cross attention layer
cross_attention_layer = CrossAttention(d_model, num_heads)

# Calculate cross attention
attention_output = cross_attention_layer(xa, xv)

print(attention_output.shape)  # Output shape will be (batch_size, seq_len_a, d_model)
