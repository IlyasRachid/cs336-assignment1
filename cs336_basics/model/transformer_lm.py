import torch.nn as nn
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear

class Transformer(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 8,
        rope_theta: int | None = None,
        ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        embeddings = self.token_embeddings(input_ids)
        attention = embeddings
        for block in self.layers:
            attention = block(attention)

        attention_norm = self.ln_final(attention)
        weights = self.lm_head(attention_norm)

        return weights
