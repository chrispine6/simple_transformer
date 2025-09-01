import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm
from residual import ResidualConnection

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.norm1(x)
        x = self.residual2(x, lambda x: self.cross_attention(x, enc_output, enc_output, src_mask))
        x = self.norm2(x)
        x = self.residual3(x, lambda x: self.feed_forward(x))
        x = self.norm3(x)
        return x
