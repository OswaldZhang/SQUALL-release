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
import math

@MODELS.register_module()
class UNI(TimmVisionTransformer):
    """For a parameter description see ViTCellViT and TimmVisionTransformer"""

    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 24,
        num_heads: int = 16,
        embed_dim: int = 1024,
        num_classes: int = 0,
        init_values: float = 1e-5,
        dynamic_img_size: bool = True,
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
        )
        self.extract_layers = extract_layers

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

        # x = self.forward_head(x)
        return x[:, 0], extracted_layers

@MODELS.register_module()
class UNI_ABMIL(ABMIL):
    """CellViT with UNI backbone settings

    Information about UNI:
        https://github.com/mahmoodlab/UNI
        https://www.nature.com/articles/s41591-024-02857-3

    Checkpoints must be downloaded from the HuggingFace model repository of UNI.

    Args:
        model_uni_path (Union[Path, str]): Path to UNI checkpoint
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
    """

    def __init__(
        self,config
    ):
        self.img_size = 224
        self.patch_size = 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 12
        self.qkv_bias = True
        self.extract_layers = [6, 12, 18, 24]
        self.input_channels = 3
        self.mlp_ratio = 4
        self.drop_rate = 0

        super().__init__(
            config
        )

        self.encoder = UNI(
            extract_layers=self.extract_layers
        )
        # qbw 11.14 gating attention
        fc = [nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()]
        dropout = True
        gate = True
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = Attn_Net_Gated(L=self.embed_dim, D=int(self.embed_dim/4), dropout=dropout, n_classes=1)
        else:
            attention_net = nn.Linear(self.embed_dim, 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        #x : ( B , 784 , config.cls_dim )
        
        self.classifier = nn.Linear(self.embed_dim, self.cls_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # init weight qzk
        self.apply(self._init_weights)
        self.build_loss_func()
        for name, param in self.named_parameters():
            print(f"{name}: {param.requires_grad}")

    def load_model_from_ckpt(self, ckpt_path, log=True):
        """Load pretrained UNI from provided path

        Args:
            model_uni_path (str): Path to UNI (ViT-L foundation model)
        """
        if ckpt_path is None:
            print(f"No checkpoint provided!")
        else:
            state_dict = torch.load(str(ckpt_path))
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loading checkpoint: {msg}")
    def forward(self,  rgb, res):
        rgb = rgb.permute(0, 3, 1, 2)  # 将维度调整为 (B, C, H, W)
        z,_ = self.encoder(rgb)
        features = z
        A, x = self.attention_net(features)
        
        ###from propoise
        A = torch.transpose(A, 1, 0)
        bag_feature = torch.mm(F.softmax(A, dim=1) , x)
        attn_scores = A
        #ablation
        #CHIEF implement
        
        '''
        #old version, attention no gradient 
        A = F.softmax(A, dim=1)
        
        bag_feature = A * x  # 元素级乘法，对 x 进行加权 => [batchsize, 1024]
        #print("bag_feature.shape",bag_feature.shape)
        bag_feature = (x * A).sum(dim=0, keepdim=True)  # 在行上做求和，得到 1x1024 的结果
        #print("bag_feature.shape",bag_feature.shape)
        attn_scores = A
        '''
        #CHIEF implement
        
        output = self.classifier(bag_feature)#.clone()
        
        return output,attn_scores
