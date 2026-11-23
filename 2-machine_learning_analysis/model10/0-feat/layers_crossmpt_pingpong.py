import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_k = d_model // heads
        self.h = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        query, key, value = [
            lin(x).view(batch, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, ff_mult=4):
        super().__init__()
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.ff = PositionwiseFeedForward(d_model, d_model * ff_mult, dropout=dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, y, mask=None):
        x = self.sublayer[0](x, lambda x: self.attn(x, y, y, mask))
        x = self.sublayer[1](x, self.ff)
        return x


class BidirectionalCrossMPT(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, ff_mult=4):
        super().__init__()
        self.layer_drug = CrossAttentionBlock(d_model, heads, dropout, ff_mult)
        self.layer_prot = CrossAttentionBlock(d_model, heads, dropout, ff_mult)
        self.handoff_proj = nn.Linear(d_model, d_model)
        self.handoff_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat_drug, feat_prot, mask_drug=None, mask_prot=None):
        # Bengkel ablation: bypass CrossMPT/pingpong, then let the caller pool
        # each encoder independently and concatenate the pooled features.
        return feat_drug, feat_prot
