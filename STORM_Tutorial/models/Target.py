import os
import glob
from pathlib import Path
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from utils.logger import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from functools import partial
from .ViT_utils import TimmVisionTransformer
from .build import MODELS

# =============================================================================
# Preprocessing Module for Omics Data
# =============================================================================
class PreprocessOmics:
    """
    PreprocessOmics handles loading of the gene vocabulary, gene selection,
    and pre-processing of gene expression data.
    """

    def __init__(self, device='cpu'):
        """
        Initialize the preprocessing module.
        Loads the gene vocabulary and target gene names, and identifies valid genes.
        """
        self.device = device
        vocab_file = "/home/xhl/omicsft/scGPT/ckpt/vocab.json"
        self.vocab = self._load_vocab(vocab_file)
        self.pad_token = "<pad>"

        # Add special tokens if not present in the vocabulary
        special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab.append_token(token)

        # Load target gene names from CSV file
        gene_names_file = "/home/xhl/omicsft/select_target_gene.csv"
        self.gene_names = pd.read_csv(gene_names_file)['HGNC_symbol'].values.flatten()

        # Check which genes are present in the vocabulary
        gene_ids_meta = {
            "id_in_vocab": [1 if gene in self.vocab else -1 for gene in self.gene_names]
        }
        gene_ids_in_vocab = np.array(gene_ids_meta["id_in_vocab"])
        print(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in a vocabulary of size {len(self.vocab)}.")

        # Store valid gene indices and convert gene names to tensor of ids
        self.valid_gene_indices = np.where(gene_ids_in_vocab >= 0)[0]
        self.post_gene_names = self.gene_names[self.valid_gene_indices]
        self.gene_ids = torch.tensor(self.vocab(self.post_gene_names.tolist()), dtype=torch.long, device=self.device)

    def _load_vocab(self, vocab_file):
        """Load vocabulary from file."""
        import json
        with open(vocab_file, 'r') as f:
            vocab_dict = json.load(f)
        return Vocab(vocab_dict)

    def pre_process(self, data: torch.Tensor, normalize_total: float = 1e4):
        """
        Normalize gene expression data and apply log1p transformation.

        Args:
            data (torch.Tensor): Raw gene expression data with shape (num_cells, num_genes).
            normalize_total (float): Target total count for normalization (default: 1e4).

        Returns:
            tuple:
                - data (torch.Tensor): Filtered data with only valid genes.
                - normalized_data (torch.Tensor): Data normalized to the target total.
                - log1p_data (torch.Tensor): Log-transformed normalized data.
        """
        # Keep only valid gene expression data
        data = data[:, self.valid_gene_indices]

        # Compute total expression per cell (row sum)
        cell_sums = data.sum(dim=1, keepdim=True)

        # Normalize data (handle cells with zero total expression)
        normalized_data = torch.where(
            cell_sums == 0,
            torch.zeros_like(data),
            (data / cell_sums) * normalize_total
        )

        # Apply log1p transformation
        log1p_data = torch.log1p(normalized_data)
        return data, normalized_data, log1p_data

    def tokenize_batch(
        self,
        data: torch.Tensor,
        append_cls: bool = True,
        include_zero_gene: bool = False,
        cls_token: str = "<cls>"
    ):
        """
        Tokenize a batch of gene expression data.
        For each cell, creates a tuple of (gene_ids, gene_counts) for nonzero gene expressions.
        Optionally, prepends a class token.

        Args:
            data (torch.Tensor): Tensor with shape (batch_size, n_features).
            append_cls (bool): Whether to prepend the class token (default: True).
            include_zero_gene (bool): Whether to include genes with zero expression (default: False).
            cls_token (str): The class token string (default: "<cls>").

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]:
                A list where each element is a tuple of (gene_ids, gene_counts) for a cell.
        """
        cls_id = self.vocab[cls_token]
        if data.shape[1] != len(self.gene_ids):
            raise ValueError(
                f"Number of features in data ({data.shape[1]}) does not match number of gene_ids ({len(self.gene_ids)})."
            )

        tokenized_data = []
        batch_size = data.shape[0]

        for i in range(batch_size):
            row = data[i]
            if include_zero_gene:
                values = row
                genes = self.gene_ids
            else:
                idx = torch.nonzero(row, as_tuple=True)[0]
                values = row[idx]
                genes = self.gene_ids[idx]

            if append_cls:
                cls_tensor = torch.tensor([cls_id], device=self.device, dtype=torch.long)
                zero_tensor = torch.tensor([0.0], device=self.device, dtype=torch.float32)
                genes = torch.cat([cls_tensor, genes])
                values = torch.cat([zero_tensor, values])

            tokenized_data.append((genes, values))

        return tokenized_data

    def pad_batch(
        self,
        batch: list,
        max_len: int,
        pad_token: str = "<pad>",
        pad_value: int = -2,
        cls_appended: bool = True
    ) -> dict:
        """
        Pad a batch of tokenized gene data to a fixed maximum length.

        Args:
            batch (List[Tuple]): List of (gene_ids, gene_counts) tuples.
            max_len (int): Maximum sequence length.
            pad_token (str): Token used for padding (default: "<pad>").
            pad_value (int): Value used to pad the gene counts (default: -2).
            cls_appended (bool): Whether the class token is prepended (affects sampling).

        Returns:
            dict:
                - "genes": Padded tensor of gene IDs with shape (batch_size, max_len).
                - "values": Padded tensor of gene counts with shape (batch_size, max_len).
        """
        max_ori_len = max(len(item[0]) for item in batch)
        max_len = min(max_ori_len, max_len)

        pad_id = self.vocab[pad_token]
        gene_ids_list = []
        values_list = []

        for gene_ids, values in batch:
            if len(gene_ids) > max_len:
                # If sequence is longer than max_len, sample indices.
                if not cls_appended:
                    idx = np.random.choice(len(gene_ids), max_len, replace=False)
                else:
                    # Ensure the cls token (position 0) is preserved.
                    idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                    idx = idx + 1
                    idx = np.insert(idx, 0, 0)
                gene_ids = gene_ids[idx]
                values = values[idx]

            if len(gene_ids) < max_len:
                pad_len = max_len - len(gene_ids)
                gene_ids = torch.cat([
                    gene_ids,
                    torch.full((pad_len,), pad_id, dtype=gene_ids.dtype, device=self.device)
                ])
                values = torch.cat([
                    values,
                    torch.full((pad_len,), pad_value, dtype=values.dtype, device=self.device)
                ])

            gene_ids_list.append(gene_ids)
            values_list.append(values)

        batch_padded = {
            "genes": torch.stack(gene_ids_list, dim=0),
            "values": torch.stack(values_list, dim=0),
        }
        return batch_padded

    def tokenize_and_pad_batch(
        self,
        data: np.ndarray,
        max_len: int,
        pad_token: str = "<pad>",
        pad_value: int = -2,
        append_cls: bool = True,
        include_zero_gene: bool = False,
        cls_token: str = "<cls>"
    ) -> dict:
        """
        Combined tokenization and padding for a batch of gene expression data.

        Args:
            data (np.ndarray): Batch data with shape (batch_size, n_features).
            max_len (int): Maximum sequence length.
            pad_token (str): Padding token (default: "<pad>").
            pad_value (int): Padding value for gene counts (default: -2).
            append_cls (bool): Whether to prepend the class token.
            include_zero_gene (bool): Whether to include genes with zero expression.
            cls_token (str): The class token string.

        Returns:
            dict: Dictionary with padded "genes" and "values" tensors.
        """
        # Convert numpy array to torch tensor if needed
        if not torch.is_tensor(data):
            data = torch.tensor(data, device=self.device)

        tokenized_data = self.tokenize_batch(
            data,
            append_cls=append_cls,
            include_zero_gene=include_zero_gene,
            cls_token=cls_token
        )
        batch_padded = self.pad_batch(
            tokenized_data,
            max_len,
            pad_token=pad_token,
            pad_value=pad_value,
            cls_appended=append_cls
        )
        return batch_padded

# =============================================================================
# Model Components
# =============================================================================
class Residual(nn.Module):
    """
    Residual module that adds the input to the output of the given function.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function splits the input and applies a sigmoid gate.
    """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class Mlp(nn.Module):
    """
    Feed-forward network (MLP) with one hidden layer and dropout.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Self-attention module with query, key, and value projections.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block combining self-attention and feed-forward network with residual connections.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """
    Cross-attention module for fusing information from two modalities.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear projections for query, key, and value
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        """
        Args:
            query (torch.Tensor): Query tensor with shape (B, N, C).
            key_value (torch.Tensor): Key/value tensor with shape (B, S, C).

        Returns:
            torch.Tensor: Output tensor with shape (B, N, C).
        """
        B, N, C = query.shape
        _, S, _ = key_value.shape

        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)
        k = self.k(key_value).reshape(B, S, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)
        v = self.v(key_value).reshape(B, S, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, S]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# =============================================================================
# SpatialTranscriptomicsCoCa Model
# =============================================================================
@MODELS.register_module()
class SpatialTranscriptomicsCoCa(nn.Module):
    """
    A multimodal model that fuses image and gene expression data.
    It includes an image encoder, a gene expression encoder (scGPT), and a multimodal decoder.
    """

    def __init__(
        self,
        # preprocess,
        multimodal_decoder_depth: int = 4,
        # Image configuration
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 1024,
        input_channels: int = 3,
        depth: int = 24,
        num_heads: int = 8,
        vocab_size: int = 60697,
        embsize: int = 512,
        expr_num_heads: int = 8,
        d_hid: int = 512,
        nlayers: int = 12,
        # pad_token: str = "<pad>",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        mask_ratio: float = 0.50,
    ):
        """
        Initialize the SpatialTranscriptomicsCoCa model.

        Args:
            preprocess: Preprocessing module for gene expression data.
            multimodal_decoder_depth (int): Number of layers in the multimodal decoder.
            img_size (int): Input image size.
            patch_size (int): Patch size for image tokenization.
            embed_dim (int): Embedding dimension for image tokens.
            input_channels (int): Number of image channels.
            depth (int): Depth (number of transformer blocks) of the image encoder.
            num_heads (int): Number of attention heads in the image encoder.
            vocab_size (int): Vocabulary size for gene expressions.
            embsize (int): Embedding size for the scGPT model.
            expr_num_heads (int): Number of attention heads in the gene expression encoder.
            d_hid (int): Hidden dimension in the scGPT model.
            nlayers (int): Number of layers in the scGPT model.
            pad_token (int): Padding token id.
            vocab (dict): Gene vocabulary.
            mlp_ratio (float): Ratio for MLP hidden dimension.
            qkv_bias (bool): Whether to use bias in QKV projections.
            drop_rate (float): Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            mask_ratio (float): Ratio of image patches to mask.
        """
        super().__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_channels = input_channels

        # Preprocessing module for gene expression data
        # self.preprocess = preprocess

        # Initialize scGPT model for gene expression encoding
        # self.pad_token = pad_token
        # Image encoder (Vision Transformer)
        self.encoder = TimmVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=1.0
        )
    def load_model_from_ckpt(self, ckpt_path, log=True):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            '''
            for k in list(base_ckpt.keys()):
                if k.startswith('encoder'):
                    print("need loaded ",k,base_ckpt[k])
                    base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            '''
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if log:
                if incompatible.missing_keys:
                    print_log('missing_keys', logger='STORM')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='STORM'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='STORM')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='STORM'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='STORM')
        else:
            print_log('Training from scratch!!!', logger='STORM')
    
    def load_checkpoint_and_freeze(self, encoder_ckpt, scgpt_ckpt):
        """
        Load pretrained weights for encoder and scGPT, then freeze their parameters,
        except for specific layers in scGPT that should remain trainable.
        """
        encoder_state = torch.load(encoder_ckpt, map_location=torch.device('cpu'))
        scgpt_state = torch.load(scgpt_ckpt, map_location=torch.device('cpu'))

        # Load encoder checkpoint
        missing_keys_encoder, unexpected_keys_encoder = self.encoder.load_state_dict(encoder_state, strict=False)
        
        # Load scGPT checkpoint
        missing_keys_scgpt, unexpected_keys_scgpt = self.scGPT.load_state_dict(scgpt_state, strict=False)

        # Freeze all parameters in encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Freeze all parameters in scGPT except the specified layers
        for name, param in self.scGPT.named_parameters():
            if name not in [
                "value_encoder.linear1.weight", "value_encoder.linear1.bias",
                "value_encoder.linear2.weight", "value_encoder.linear2.bias",
                "value_encoder.norm.weight", "value_encoder.norm.bias",
                "linear_out.weight", "linear_out.bias"
            ]:
                param.requires_grad = False  # Freeze all other parameters

        return {
            "missing_keys_encoder": missing_keys_encoder,
            "unexpected_keys_encoder": unexpected_keys_encoder,
            "missing_keys_scgpt": missing_keys_scgpt,
            "unexpected_keys_scgpt": unexpected_keys_scgpt
        }


    def unfreeze(self):
        """
        Unfreeze parameters of the specified module ('encoder' or 'scgpt').
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        # print("ImageEncoder is now trainable.")
        for param in self.scGPT.parameters():
            param.requires_grad = True
        # print("scGPT model is now trainable.")

    def patchify(self, images):
        """
        Divide images into patches.

        Args:
            images (torch.Tensor): Input images with shape (B, C, H, W).

        Returns:
            torch.Tensor: Patches with shape (B, num_patches, patch_size**2 * C).
        """
        p = self.patch_size
        h = w = images.shape[2] // p
        x = images.reshape(images.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(images.shape[0], h * w, p**2 * 3)

    def forward_rgb(self, x,res):
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)

        for blk in self.encoder.blocks:
            x = blk(x)
        return x[:, 0]

# =======================
# 1. Attention Pool 模块
# =======================
class AttentionPooling(nn.Module):
    """
    将 (B, N, D) 池化为 (B, D) 的完整 Transformer Block 风格实现。
    包含：
      1) 可学习的 query token，用于多头注意力；
      2) LayerNorm；
      3) 多头注意力 (multi-head attention)；
      4) 残差连接 + 投影；
      5) 前馈网络 (FeedForward, FFN) + 残差连接；
    """

    def __init__(
        self,
        dim: int,            # 输入 token 的特征维度 D
        heads: int = 8,      # 多头注意力的头数
        dim_head: int = 64,  # 每个头的维度
        ff_mult: int = 4,    # FFN 内部层放大的倍数
        dropout: float = 0.2 # Dropout 比例
    ):
        """
        Args:
            dim (int): 输入序列中 token 的维度 (D)
            heads (int): 多头注意力的头数
            dim_head (int): 每个注意力头的维度
            ff_mult (int): FFN 的隐藏层放大倍数
            dropout (float): 注意力和部分投影的 dropout 概率
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        # 1) 可学习的单独 query token，相当于一个 learnable [CLS]
        self.query_token = nn.Parameter(torch.randn(1, 1, dim))

        # 2) LayerNorm，用于在注意力前对输入序列做归一化
        self.norm = nn.LayerNorm(dim)

        # 3) 多头注意力需要的线性层
        #    - to_q 只会作用在 query_token 上
        #    - to_kv 作用在整段输入序列 x 上
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 4) 前馈网络 (FeedForward, FFN)
        ff_inner_dim = ff_mult * dim
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_inner_dim),
            nn.GELU(),
            nn.Linear(ff_inner_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入序列，形状为 (B, N, D)

        Returns:
            torch.Tensor: 聚合后的单向量，形状为 (B, D)
        """
        b, n, d = x.shape

        # 1) 先做 LayerNorm
        x_ln = self.norm(x)      # (B, N, D)

        # 2) 准备 query、key、value
        #    - query 仅来自 learnable 的 query_token
        q_token = self.query_token.expand(b, -1, -1)  # (B, 1, D)

        #    - 线性投影到多头
        q = self.to_q(q_token)   # (B, 1, heads*dim_head)
        kv = self.to_kv(x_ln)    # (B, N, 2*heads*dim_head)
        k, v = kv.chunk(2, dim=-1)

        # 3) reshape => (B, heads, seq_len, dim_head)
        h = self.heads
        q = q.view(b, 1, h, self.dim_head).transpose(1, 2)  # (B, h, 1, dim_head)
        k = k.view(b, n, h, self.dim_head).transpose(1, 2)  # (B, h, N, dim_head)
        v = v.view(b, n, h, self.dim_head).transpose(1, 2)  # (B, h, N, dim_head)

        # 4) 注意力分数 + softmax
        scale = (self.dim_head ** -0.5)
        sim = torch.matmul(q, k.transpose(-1, -2)) * scale   # (B, h, 1, N)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 5) 注意力加权输出
        out = torch.matmul(attn, v)                          # (B, h, 1, dim_head)
        out = out.transpose(1, 2).reshape(b, 1, h*self.dim_head)  # (B, 1, inner_dim)
        out = self.proj_out(out)   # (B, 1, D)

        # 6) 残差连接到 query_token
        out = q_token + self.dropout(out)

        # 7) 前馈网络 (FFN) + 残差
        out = out + self.ff(out)   # (B, 1, D)

        return out.squeeze(1)      # (B, D)




@MODELS.register_module()
class SpatialTranscriptomicsCLIP(nn.Module):
    """
    MultiModel, CLIP, Contrastive Learning
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 1024,
        input_channels: int = 3,
        depth: int = 24,
        num_heads: int = 8,
        vocab_size: int = 60697,
        embsize: int = 512,
        expr_num_heads: int = 8,
        d_hid: int = 512,
        nlayers: int = 12,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        # 新增：对比学习特征维度
        proj_dim: int = 1024, 
        image_pooling: bool = True
    ):
        super().__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        # self.mask_ratio = mask_ratio

        # 初始化 scGPT
        self.scGPT = TransformerModel(
            ntoken=vocab_size,
            d_model=embsize,
            nhead=expr_num_heads,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=0.2,
        )

        # 初始化 Image Encoder (Vision Transformer)
        self.encoder = TimmVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=1.0
        )

        # 池化到 (B, 512)，作为表达数据的“CLS”表征
        self.expr_pool = AttentionPooling(dim=embsize)

        # 可以选择是否对图像也做 Attention Pool
        self.image_pooling = image_pooling
        self.img_pool = AttentionPooling(dim=embed_dim)

        # CLIP风格：将图像和表达投影到同一隐空间
        self.img_proj  = nn.Linear(embed_dim, proj_dim, bias=False)
        self.expr_proj = nn.Linear(embsize, proj_dim, bias=False)

        # CLIP风格里常见的可学习尺度 logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def load_checkpoint_and_freeze(self, encoder_ckpt, scgpt_ckpt):
        encoder_state = torch.load(encoder_ckpt, map_location=torch.device('cpu'))
        scgpt_state   = torch.load(scgpt_ckpt, map_location=torch.device('cpu'))

        missing_keys_encoder, unexpected_keys_encoder = self.encoder.load_state_dict(encoder_state, strict=False)
        missing_keys_scgpt, unexpected_keys_scgpt = self.scGPT.load_state_dict(scgpt_state, strict=False)

        for param in self.encoder.parameters():
            param.requires_grad = False

        for name, param in self.scGPT.named_parameters():
            if name not in [
                "value_encoder.linear1.weight", "value_encoder.linear1.bias",
                "value_encoder.linear2.weight", "value_encoder.linear2.bias",
                "value_encoder.norm.weight",   "value_encoder.norm.bias",
            ]:
                param.requires_grad = False

        return {
            "missing_keys_encoder": missing_keys_encoder,
            "unexpected_keys_encoder": unexpected_keys_encoder,
            "missing_keys_scgpt": missing_keys_scgpt,
            "unexpected_keys_scgpt": unexpected_keys_scgpt
        }

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.scGPT.parameters():
            param.requires_grad = True

    def forward(self, image, expr):
        """
        CLIP风格对比学习的 forward：
        1. 编码 image => (B, N, embed_dim) => attention pool => (B, embed_dim)
        2. 编码 expr  => (B, cell_num, embsize) => attention pool => (B, embsize)
        3. 分别投影到 CLIP 空间 => (B, proj_dim)
        4. 计算 InfoNCE / CLIP 的对比学习损失
        """
        # ========================
        # 1. 处理图像，得到 (B, embed_dim)
        # ========================
        # 输入图像 => ViT => 全部 token (含CLS)
        image_tokens = self.encoder.forward_features(image) # (B, N, embed_dim)，其中第0号是 cls token
        
        if self.image_pooling:
            # 方式B: 用 attention pool
            img_cls = self.img_pool(image_tokens)  # (B, embed_dim)
        else:
            # 方式A: 直接使用 cls token
            img_cls = image_tokens[:, 0, :]

        # ========================
        # 2. 处理表达数据 => scGPT => (B, cell_num, embsize)
        # ========================
        # 假设 expr 的 shape = (B, cell_num, gene_num)
        batch_size, cell_num, gene_num = expr["genes"].shape  # (B, cell_num, gene_num)

        # **1. reshape 合并 batch 维度和 cell 维度**
        flat_gene_ids = expr["genes"].reshape(batch_size * cell_num, gene_num)  # (B * cell_num, gene_num)
        flat_gene_values = expr["values"].reshape(batch_size * cell_num, gene_num)  # (B * cell_num, gene_num)
        src_key_padding_mask = flat_gene_ids.eq(60694)  # (B * cell_num, gene_num)

        # **2. 一次性处理整个 batch，消除 for 循环**
        output_dict = self.scGPT(flat_gene_ids, flat_gene_values, src_key_padding_mask=src_key_padding_mask)
        expr_tokens = output_dict['cell_emb']  # (B * cell_num, embed_dim)
        # **3. reshape 回 (B, cell_num, embed_dim)**
        expr_tokens_batch = expr_tokens.view(batch_size, cell_num, -1)  # (B, cell_num, embed_dim)

        # 用 AttentionPooling 得到 (B, embsize)
        expr_cls = self.expr_pool(expr_tokens_batch)

        # ========================
        # 3. 投影到同一隐空间
        # ========================
        img_feat  = self.img_proj(img_cls)      # (B, proj_dim)
        expr_feat = self.expr_proj(expr_cls)    # (B, proj_dim)
                
        # ========================
        # 4. CLIP风格对比学习损失
        # ========================
        # logit_scale = self.logit_scale.exp()
        # # 计算相似度矩阵 (B, B)
        # logits_per_image = logit_scale * img_feat @ expr_feat.t()  # (B, B)
        # logits_per_expr  = logits_per_image.t()                    # (B, B)

        # labels = torch.arange(batch_size, device=img_feat.device)
        # loss_i = F.cross_entropy(logits_per_image, labels)
        # loss_e = F.cross_entropy(logits_per_expr,  labels)
        # contrastive_loss = (loss_i + loss_e) * 0.5

        # 返回对比学习损失，以及图像/表达特征，方便需要时做评测
        return img_feat, expr_feat
    
    def forward_rgb(self, x,res):
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)

        for blk in self.encoder.blocks:
            x = blk(x)
        return x[:, 0]



@MODELS.register_module()

class PathOmicsCOCA(nn.Module):
    def __init__(
        self,
        config,
        gene_num: int = 4478,
        embed_dim: int = 1024,
        mask_ratio: float = 0.25,
        multimodal_decoder_depth: int = 4,
        img_size: int = 224,
        patch_size: int = 16,
        input_channels: int = 3,
        image_depth: int = 24,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        expr_depth: int = 6,
        init_values: float = 1.0
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.gene_num = gene_num

        # Image encoder (ViT)
        self.encoder = TimmVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            embed_dim=embed_dim,
            depth=image_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=init_values
        )
        #print("self.gene_num",self.gene_num)
        #print("self.embed_dim", self.embed_dim)
        # Expression encoder: 3-layer MLP
        self.expr_encoder = nn.Sequential(
            nn.Linear(self.gene_num, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Tile-level transformer encoder (4 layers + CLS token)
        self.expr_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.expr_blocks = nn.ModuleList()
        # import pdb;pdb.set_trace()
        for i in range(expr_depth):  # Replace 4 with a hyperparameter if needed
            block = TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads)
            self.expr_blocks.append(block)
        
        # Mask token for missing cells
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Decoder for expression reconstruction
        self.decoder = nn.ModuleList([
            nn.ModuleList([
                Residual(TransformerBlock(dim=embed_dim, 
                                          num_heads=num_heads)),
                Residual(CrossAttention(dim=embed_dim))
            ]) for _ in range(multimodal_decoder_depth)
        ])

        self.decoder_head = nn.Linear(embed_dim, gene_num)
        self.l1_loss = torch.nn.SmoothL1Loss(reduction='none')

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.expr_cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    def load_model_from_ckpt(self, ckpt_path, log=True):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            '''
            for k in list(base_ckpt.keys()):
                if k.startswith('encoder'):
                    print("need loaded ",k,base_ckpt[k])
                    base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            '''
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if log:
                if incompatible.missing_keys:
                    print_log('missing_keys', logger='STORM')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='STORM'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='STORM')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='STORM'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='STORM')
        else:
            print_log('Training from scratch!!!', logger='STORM') 
    def load_checkpoint_and_freeze(self, encoder_ckpt):
        """
        Load pretrained weights for encoder and scGPT, then freeze their parameters,
        except for specific layers in scGPT that should remain trainable.
        """
        encoder_state = torch.load(encoder_ckpt, map_location=torch.device('cpu'))
        
        # Load encoder checkpoint
        missing_keys_encoder, unexpected_keys_encoder = self.encoder.load_state_dict(encoder_state, strict=False)

        # Freeze all parameters in encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        return {
            "missing_keys_encoder": missing_keys_encoder,
            "unexpected_keys_encoder": unexpected_keys_encoder,
        }


    def unfreeze(self):
        """
        Unfreeze parameters of the specified module ('encoder').
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
    

    def random_masking(self, x):
        # import pdb;pdb.set_trace()
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.zeros([B, N], device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore) # 1表示保留，0表示mask
        # broadcast，mask.unsqueeze->(bs, 196, 1024)
        x_with_mask = x * mask.unsqueeze(-1) + self.mask_token * (1 - mask.unsqueeze(-1))
        
        return x_with_mask, mask, ids_restore

    def forward(self, image, expr):
        # import pdb;pdb.set_trace()
        B, N, G = expr.shape

        # 1. Encode image
        img_tokens = self.encoder.forward_features(image)  # [B, 197, 1024]
        img_cls, img_feat = img_tokens[:, 0, :], img_tokens[:, 1:, :]   # [B, 196, 1024]
        
        # 3. Random masking like MAE
        # 2. Encode expression (cell-level)
        expr_tokens = self.expr_encoder(expr) 
        expr_with_mask, mask, ids_restore = self.random_masking(expr_tokens)

        # [B, 196, 1024]
        expr_cls_token = self.expr_cls_token.expand(B, -1, -1)
        expr_with_cls = torch.cat([expr_cls_token, expr_with_mask], dim=1)
        # full_tokens = torch.cat([torch.zeros_like(cls_token), expr_tokens], dim=1)  # [B, 196+1, 1024]

        # 4. Tile-level Transformer
        for blk in self.expr_blocks:
            expr_with_cls = blk(expr_with_cls)
        
        # 8. Decoder with cross-attention to image
        decode_expr_tokens = expr_with_cls[:, 1:, :]
        for attn_block, cross_block in self.decoder:
            decode_expr_tokens = attn_block(decode_expr_tokens)
            decode_expr_tokens = cross_block(decode_expr_tokens, img_feat)
        
        # Reconstruction head: predict pixel values
        pred_expr = self.decoder_head(decode_expr_tokens)
        # target = self.patchify(image)

        # Compute alignment loss between image tokens and gene expression tokens
        visible_mask = mask
        unvisible_mask = 1 - visible_mask
        cosine_sim = F.cosine_similarity(img_feat, expr_with_cls[:, 1:, :], dim=-1)
        align_loss = 1 - cosine_sim
        align_loss = (align_loss * visible_mask).sum() / visible_mask.sum()

        # Compute reconstruction loss on masked patches
        expr_loss = (pred_expr - expr) ** 2
        expr_loss = expr_loss.mean(dim=-1)
        expr_loss = (expr_loss * unvisible_mask).sum() / unvisible_mask.sum()

        total_loss = align_loss + expr_loss
        return align_loss, expr_loss, total_loss 
        
        # encoded_tile = self.expr_tile_encoder(expr_with_cls)  # [B, 197, 1024]
        # 5. Similarity loss between image patch and visible expression patch
        # visible_img_patch = torch.gather(img_patch, 1, ids_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim))
        # visible_expr_patch = torch.gather(expr_tokens, 1, ids_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim))

        # align_loss = 1 - F.cosine_similarity(visible_expr_patch, visible_img_patch, dim=-1).mean()

        # 6. Symmetric contrastive loss with CLS
        # img_cls = img_feat[:, 0]
        # expr_cls = encoded_tile[:, 0]
        # sim_matrix = F.cosine_similarity(img_cls.unsqueeze(1), expr_cls.unsqueeze(0), dim=-1)
        # sym_loss = -torch.mean(torch.diag(sim_matrix))

        # 7. Prepare full input for decoder
        # B, N, D = expr_tokens.shape
        # mask_tokens = self.mask_token.expand(B, N, -1)
        # expr_tokens_full = expr_tokens.clone()
        # expr_tokens_full[mask] = mask_tokens[mask]  # fill masked positions

        # # 
        # x = expr_tokens_full
        # for self_attn, cross_attn in self.decoder:
        #     x = self_attn(x)
        #     x = cross_attn(x, img_patch)

        # # 9. Predict masked expr
        # pred_expr = self.decoder_head(x)  # [B, N, G]

        # gt_expr = expr[mask]  # [#masked, G]
        # pred_masked = pred_expr[mask]  # [#masked, G]
        # expr_loss = F.mse_loss(pred_masked, gt_expr)

        # return align_loss, expr_loss, align_loss + expr_loss
        # Multimodal decoder: alternate self-attention and cross-attention blocks
    def forward_rgb(self, x,res):
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)

        for blk in self.encoder.blocks:
            x = blk(x)
        return x[:, 0]
