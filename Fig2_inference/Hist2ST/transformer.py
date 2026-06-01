#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super().__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = [
            t.reshape(B, N, self.heads, self.dim_head).transpose(1, 2)
            for t in qkv
        ]
        # q/k/v: [B, heads, N, dim_head]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)

        return self.to_out(out)


class attn_block(nn.Module):
    """
    Compatible with original Hist2ST call:
        attn_block(dim, heads, dim_head, mlp_dim, dropout)
    """

    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=1024, dropout=0.0):
        super().__init__()

        self.attn = PreNorm(
            dim,
            Attention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            ),
        )

        self.ff = PreNorm(
            dim,
            FeedForward(
                dim=dim,
                hidden_dim=mlp_dim,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn.functional as F

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super().__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

from anndata import AnnData
from torch import nn, einsum
from scipy.stats import pearsonr
from torch.autograd import Function
from torch.autograd.variable import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    # @get_local('attn')
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn=PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff=PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x
'''
