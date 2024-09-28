# This is modified from TransformerDecoderLayer in mlx.nn.layers.transformer.py

import math
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.activations import gelu
from mlx.nn.layers.base import Module
from mlx.nn.layers.dropout import Dropout
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import LayerNorm
from mlx.nn.utils import checkpoint
from mlx.core.fast import scaled_dot_product_attention

nn.MultiHeadAttention
class MultiHeadAttention(Module):
    # Similar to MultiHeadAttention in transformer.py, but with fast.scaled_dot_product_attention
    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads
        self.query_proj = Linear(dims, dims, bias=bias)
        self.key_proj = Linear(dims, dims, bias=bias)
        self.value_proj = Linear(dims, dims, bias=bias)
        self.out_proj = Linear(dims, dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None, stream=mx.gpu):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        queries = queries.reshape(B, L, num_heads, -1, stream=mx.gpu).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1, stream=mx.gpu).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1, stream=mx.gpu).transpose(0, 2, 1, 3)

        # print(keys.shape)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])

        values_hat = scaled_dot_product_attention(
            q=queries, 
            k=keys, 
            v=values, 
            scale=scale, 
            mask=mask,
            stream=mx.gpu
        )
        # print(values_hat.shape)
        # (12, 8, 512, 16)
        values_hat = values_hat.transpose(0, 2, 1, 3).reshape(B, L, -1, stream=mx.gpu)
        return self.out_proj(values_hat)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        # usually inf but 1e9 is as good and softmax(full(1e9)) != nan
        # TODO: Should replace this with finfo(dtype).min
        mask = mask.astype(dtype) * -1e9
        return mask

class GPTLayer(Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        dropout: float = 0.0,
        activation: Callable[[Any], Any] = gelu,
        bias: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        mlp_dims = dims * 4
        self.self_attention = MultiHeadAttention(dims, num_heads, bias=bias)
        self.ln1_weight = mx.ones((dims,))
        self.ln2_weight = mx.ones((dims,))
        self.linear1 = Linear(dims, mlp_dims, bias)
        self.linear2 = Linear(mlp_dims, dims, bias)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation

    def __call__(self, x, x_mask):
        y = mx.fast.layer_norm(x, weight=self.ln1_weight, bias=None, eps=1e-5, stream=mx.gpu)
        y = self.self_attention(y, y, y, x_mask, stream=mx.gpu)
        y = self.dropout1(y)
        x = x + y

        y = mx.fast.layer_norm(x, weight=self.ln2_weight, bias=None, eps=1e-5, stream=mx.gpu)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.dropout3(y)
        y = self.linear2(y)
        y = x + y

        return y


class GPT(Module):
    def __init__(
            self,
            dims: int,
            num_heads: int,
            vocab_size: int,
            block_size: int,
            num_layers: int,


            dropout: float = 0.0,
            activation: Callable[[Any], Any] = gelu,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dims)
        self.wpe = nn.Embedding(block_size, dims)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = [
            GPTLayer(dims=dims, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ]

        self.lm_head = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, idx):
        b, t = idx.shape
        pos = mx.arange(0, t, dtype=mx.uint32)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)

        x = self.dropout(tok_emb + pos_emb)
        x_mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1], dtype=mx.bfloat16)

        for l in self.layers:
            x = l(x, x_mask)

        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens, temp=1.0):
        for _ in range(max_new_tokens):
            b, t = idx.shape
            pos = mx.arange(0, t, dtype=mx.uint32)
            tok_emb = self.wte(idx)
            pos_emb = self.wpe(pos)

            x = self.dropout(tok_emb + pos_emb)
            x_mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1], dtype=mx.bfloat16)

            for l in self.layers:
                x = l(x, x_mask)
            
            y = self.lm_head(x[:, -1])
            y = mx.random.categorical(y * (1/temp))
            y = mx.reshape(y, (1, len(y)))

            idx = mx.concatenate([idx, y], 1)

        return idx