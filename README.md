# the most simple transformer architecture

---
These files implement a minimal Transformer model with the following simplifications:

Single-head attention (n_heads=1) for simplicity, though the code supports multi-head attention.
Basic feed-forward network with one hidden layer and ReLU activation.
Layer normalization with learnable parameters (gamma and beta).
Residual connections with dropout for regularization.
Encoder and decoder with one layer each, including self-attention and (for the decoder) cross-attention.

The transformer.py file implements a minimal Transformer model with the following features:

Input Embeddings: Uses nn.Embedding for source and target vocabularies, scaled by the square root of d_model as per the original Transformer paper.
Positional Encoding: Implements sinusoidal positional encodings to capture sequence order, precomputed for a maximum sequence length of 5000.
Encoder and Decoder Stacks: Supports multiple encoder and decoder layers (default num_layers=1 for simplicity) using the previously defined Encoder and Decoder classes.
Masking: Includes a create_mask method to handle padding and prevent attending to future tokens in the decoder (autoregressive masking).
Final Linear Layer: Maps the decoder output to the target vocabulary size for predictions.
Dropout: Applied to embeddings and positional encodings for regularization.

Example usage

```python
import torch
from transformer import Transformer

# Example parameters
src_vocab_size = 1000
tgt_vocab_size = 1000
model = Transformer(src_vocab_size, tgt_vocab_size, d_model=64, n_heads=1, d_ff=256, num_layers=1)

# Dummy input data
src = torch.randint(0, src_vocab_size, (32, 10))  # Batch size 32, sequence length 10
tgt = torch.randint(0, tgt_vocab_size, (32, 10))
src_mask, tgt_mask = model.create_mask(src, tgt, pad_idx=0)

# Forward pass
output = model(src, tgt, src_mask, tgt_mask)
```

```
