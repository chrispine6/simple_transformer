import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, n_heads=1, d_ff=256, num_layers=1, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_len=5000, d_model=d_model)
        
        self.encoders = nn.ModuleList([
            Encoder(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            Decoder(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Input embeddings and positional encoding
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        
        src = src + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt = tgt + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        
        src = self.dropout(src)
        tgt = self.dropout(tgt)
        
        # Encoder
        enc_output = src
        for encoder in self.encoders:
            enc_output = encoder(enc_output, src_mask)
        
        # Decoder
        dec_output = tgt
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output, src_mask, tgt_mask)
        
        # Final linear layer
        output = self.final_linear(dec_output)
        return output

    def create_mask(self, src, tgt, pad_idx=0):
        # Source mask: mask padding tokens
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Target mask: mask padding tokens and future tokens (for autoregressive decoding)
        tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nofuture_mask = torch.tril(torch.ones(seq_length, seq_length)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nofuture_mask
        
        return src_mask, tgt_mask
