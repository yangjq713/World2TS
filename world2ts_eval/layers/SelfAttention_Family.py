from math import sqrt

import torch
from torch import nn


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        _, _, _, embedding_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(embedding_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        return output.contiguous(), attention if self.output_attention else None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        batch_size, query_length, _ = queries.shape
        _, key_length, _ = keys.shape
        queries = self.query_projection(queries).view(
            batch_size, query_length, self.n_heads, -1
        )
        keys = self.key_projection(keys).view(batch_size, key_length, self.n_heads, -1)
        values = self.value_projection(values).view(
            batch_size, key_length, self.n_heads, -1
        )
        output, attention = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        output = output.view(batch_size, query_length, -1)
        return self.out_projection(output), attention
