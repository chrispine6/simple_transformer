# the most simple transformer architecture
---
These files implement a minimal Transformer model with the following simplifications:

Single-head attention (n_heads=1) for simplicity, though the code supports multi-head attention.
Basic feed-forward network with one hidden layer and ReLU activation.
Layer normalization with learnable parameters (gamma and beta).
Residual connections with dropout for regularization.
Encoder and decoder with one layer each, including self-attention and (for the decoder) cross-attention.
