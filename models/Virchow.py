import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
from typing import Callable, List, Tuple, Type ,Union
from timm.layers import PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from models.Storm import ABMIL,Attn_Net_Gated
from models.transformer import TransformerEncoder
from models.ViT_utils import TimmVisionTransformer
from functools import partial
from itertools import repeat
import collections.abc
import math

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class GluMlp(nn.Module):
    """MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Sigmoid,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features // 2)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nn.SiLU, gate_last=False)


@MODELS.register_module()
class Virchow(TimmVisionTransformer):
    """For a parameter description see ViTCellViT and TimmVisionTransformer"""

    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 224,
        patch_size: int = 14,
        depth: int = 32,
        num_heads: int = 16,
        embed_dim: int = 1280,
        num_classes: int = 0,
        init_values: float = 1e-5,
        dynamic_img_size: bool = True,
        reg_tokens: int = 0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_classes=0,
            init_values=init_values,
            dynamic_img_size=dynamic_img_size,
            global_pool="",
            act_layer=torch.nn.SiLU,
            mlp_layer=SwiGLUPacked,
            mlp_ratio=5.3375,
            reg_tokens=reg_tokens,
        )
        self.extract_layers = extract_layers
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)

        return x[:, 0], extracted_layers
    
    def load_model_from_ckpt(self, ckpt_path, log=True):
        """Load pretrained UNI from provided path

        Args:
            model_uni_path (str): Path to UNI (ViT-L foundation model)
        """
        if ckpt_path is None:
            print(f"No checkpoint provided!")
        else:
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            print(f"Loading checkpoint: {msg}")

    def forward_rgb(self, x,res):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
        return x[:, 0]
