import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm
from residual import ResidualConnection

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask))
        x = self.norm1(x)
        x = self.residual2(x, lambda x: self.feed_forward(x))
        x = self.norm2(x)
        return x
