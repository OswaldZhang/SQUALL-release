import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
from timm.layers import PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from models.transformer import TransformerEncoder
import math

class CEIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed_rgb = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.down_sample,
            embed_dim=config.embed_dim
        )
        self.expr_down_sample = config.down_sample // (config.img_size // config.expr_size)
        self.patch_embed_expr = nn.Sequential(
            nn.Conv2d(config.expr_chans, config.embed_dim // self.expr_down_sample, kernel_size=1),
            PatchEmbed(
                img_size=config.expr_size,
                patch_size=self.expr_down_sample,
                embed_dim=config.embed_dim,
                in_chans=config.embed_dim // self.expr_down_sample
            )
        )
        self.blocks_rgb = TransformerEncoder(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            if_rpb = config.if_rpb,
            PE_RPB_fixed = config.PE_RPB_fixed
        )
        self.blocks_expr = TransformerEncoder(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            if_rpb = config.if_rpb,
            PE_RPB_fixed = config.PE_RPB_fixed
        )

    def forward(self, rgb, expr, res):
        z_rgb = self.forward_rgb(rgb, res)
        z_expr = self.forward_expr(expr, res)
        return z_rgb, z_expr

    def forward_rgb(self, rgb, res):
        x = self.patch_embed_rgb(rgb)
        z = self.blocks_rgb(x, res)
        
        return z

    def forward_expr(self, expr, res):
        x = self.patch_embed_expr(expr)
        z = self.blocks_expr(x, res)
        
        return z


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, similarity):
        B = similarity.size(0)

        pos = torch.diagonal(similarity, dim1=-2, dim2=-1).unsqueeze(1)  # Shape: B x 1
        neg = similarity[~torch.eye(B, dtype=bool).to(similarity.device)].reshape(B, -1)  # Shape: B x (N-1)

        logits = torch.cat([pos, neg], dim=1)  # Shape: B x N
        labels = torch.zeros(B, dtype=torch.long).to(similarity.device)

        loss = self.criterion(logits / self.temperature, labels)

        return loss


# Pretrain model
@MODELS.register_module()
class CEIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[CEIP]', logger='CEIP')
        self.config = config
        self.img_size = config.img_size
        self.expr_size = config.expr_size
        self.encoder = CEIPEncoder(config)

        self.contrastive_head = InfoNCE(temperature=0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, expr, res):
        expr = expr.to_dense().float() #qbw10.16
        #print(expr.shape)
        size = int(math.sqrt(expr.shape[1]))
        expr = expr.reshape(expr.shape[0],size, size, expr.shape[-1])#qbw10.16
        expr = expr.permute(0, 3, 1, 2)#qbw10.16
        z_rgb, z_expr = self.encoder(rgb, expr, res)
        B, N, C = z_rgb.shape

        inter_contrastive_loss = 0.
        intra_contrastive_loss = 0.

        for i in range(N):
            z_rgb_inter = z_rgb[:, i]
            z_expr_inter = z_expr[:, i]
            z_rgb_inter = nn.functional.normalize(z_rgb_inter, dim=1)
            z_expr_inter = nn.functional.normalize(z_expr_inter, dim=1)
            similarity = torch.matmul(z_rgb_inter, z_expr_inter.permute(1, 0))
            inter_contrastive_loss = inter_contrastive_loss + self.contrastive_head(similarity) / N

        for i in range(B):
            z_rgb_intra = z_rgb[i]
            z_expr_intra = z_expr[i]
            z_rgb_intra = nn.functional.normalize(z_rgb_intra, dim=1)
            z_expr_intra = nn.functional.normalize(z_expr_intra, dim=1)
            similarity = torch.matmul(z_rgb_intra, z_expr_intra.permute(1, 0))
            intra_contrastive_loss = intra_contrastive_loss + self.contrastive_head(similarity) / B

        print_log(f'inter_contrastive_loss: {inter_contrastive_loss.item()} intra_contrastive_loss: {intra_contrastive_loss.item()}', logger='CEIP')
        return inter_contrastive_loss, intra_contrastive_loss

    def forward_rgb(self, rgb, res):
        z = self.encoder.forward_rgb(rgb, res)
        return z

    def forward_expr(self, expr, res):
        z = self.encoder.forward_expr(expr, res)
        return z


# Finetune model
@MODELS.register_module()
class CEIPClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed_rgb = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.down_sample,
            embed_dim=config.embed_dim
        )
        self.expr_down_sample = config.down_sample // (config.img_size // config.expr_size)
        self.patch_embed_expr = nn.Sequential(
            nn.Conv2d(config.expr_chans, config.embed_dim // self.expr_down_sample, kernel_size=1),
            PatchEmbed(
                img_size=config.expr_size,
                patch_size=self.expr_down_sample,
                embed_dim=config.embed_dim,
                in_chans=config.embed_dim // self.expr_down_sample
            )
        )
        self.blocks_rgb = TransformerEncoder(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate
        )
        self.blocks_expr = TransformerEncoder(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate
        )
        self.cls_token_rgb = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.cls_pos_rgb = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.cls_token_expr = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.cls_pos_expr = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(config.embed_dim, config.cls_dim)
        )
        self.apply(self._init_weights)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, ckpt_path, log=True):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('encoder'):
                    base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if log:
                if incompatible.missing_keys:
                    print_log('missing_keys', logger='CEIP')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='CEIP'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='CEIP')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='CEIP'
                    )

                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='CEIP')
        else:
            print_log('Training from scratch!!!', logger='CEIP')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_rgb(self, rgb, res):
        cls = self.cls_token_rgb.expand(rgb.shape[0], -1, -1) + self.cls_pos_rgb
        x = self.patch_embed_rgb(rgb)
        x = torch.cat([cls, x], dim=1)
        z = self.blocks_rgb.forward_rgb(x)

        return z[:, 0]

    def forward_expr(self, expr, res):
        cls = self.cls_token_expr.expand(expr.shape[0], -1, -1) + self.cls_pos_expr
        x = self.patch_embed_expr(expr)
        x = torch.cat([cls, x], dim=1)
        z = self.blocks.forward_expr(x)

        return z[:, 0]

    def classifier(self, x):
        return self.cls_head_finetune(x)
