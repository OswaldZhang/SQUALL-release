import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
from timm.layers import PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
import torch.distributed as dist
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from models.transformer import Attention, MultiWayMLP, TransformerDecoder 
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        gathered_x = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_x, x)
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        return torch.cat(gathered_x, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.chunk(ctx.world_size, dim=0)[ctx.rank]
        return grad_input


class MultiWayBlock(nn.Module):
    def __init__(self, layer_index, merge_layer_depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,if_rpb = False):
        super().__init__()
        self.norm = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,if_rpb = if_rpb)
        self.mlp = MultiWayMLP(layer_index, merge_layer_depth, in_features=dim, mlp_hidden_dim=mlp_hidden_dim,
                               act_layer=act_layer, norm_layer=norm_layer, drop=drop)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp(x, mask))
        return x

    def forward_rgb(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp.forward_rgb(x))
        return x

    def forward_expr(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp.forward_expr(x))
        return x
    
    def forward_all(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp.forward_all(x))
        return x
    

class LinearShuffle(nn.Module):
    def __init__(self, dim, end_dim, up_scale):
        super(LinearShuffle, self).__init__()
        self.relu = nn.ReLU()
        self.up_scale = up_scale
        self.fc1 = nn.Linear(dim, up_scale * dim)
        self.fc2 = nn.Linear(dim // up_scale, end_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.fc1(x)
        x = self.relu(x)
        x = x.reshape(B, H * self.up_scale, W * self.up_scale, -1)
        x = self.fc2(x)
        return x


class MultiWayEncoder(nn.Module):
    def __init__(self, merge_layer_depth=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,if_rpb = False):
        super().__init__()
        dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MultiWayBlock(
                layer_index=i, merge_layer_depth=merge_layer_depth, dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr_list[i],if_rpb = if_rpb
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x, mask):
        for _, block in enumerate(self.blocks):
            x = block(x, mask)
        x = self.norm(x)
        return x

    def forward_rgb(self, x):
        for _, block in enumerate(self.blocks):
            x = block.forward_rgb(x)
        x = self.norm(x)
        return x

    def forward_expr(self, x):
        for _, block in enumerate(self.blocks):
            x = block.forward_expr(x)
        x = self.norm(x)
        return x
    
    def forward_all(self, x):
        for _, block in enumerate(self.blocks):
            x = block.forward_all(x)
        x = self.norm(x)
        return x


class PositionEmbeding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_type = config.pos_type
        self.img_size = config.img_size
        self.down_sample = config.down_sample
        if self.pos_type == 'mlp':
            self.pos_embed = nn.Sequential(
                nn.Linear(2, config.embed_dim),
                nn.LayerNorm(config.embed_dim)
            )
        elif self.pos_type == 'learned':
            self.pos = nn.Parameter(torch.zeros(1, (self.img_size // self.down_sample) ** 2, config.embed_dim))
        elif self.pos_type == 'relative':
            pass
        else:
            raise NotImplementedError

    def forward(self, x, res):
        B = x.shape[0]
        if self.pos_type == 'mlp':
            H = W = self.img_size // self.down_sample
            grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).to(x.device)
            grid = grid.reshape(1, H * W, 2).repeat(B, 1, 1)
            res = torch.tensor(res).reshape(B, 1, 1)
            pos = self.pos_embed(grid * res)
            x = x + pos
        elif self.pos_type == 'learned':
            pos = self.pos.expand(B, -1, -1)
            x = x + pos
        elif self.pos_type == 'relative':
            pass
        return x


class SquallEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mask_ratio
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
        self.pos_embed = PositionEmbeding(config)
        self.blocks = MultiWayEncoder(
            merge_layer_depth=config.merge_layer_depth,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,if_rpb = config.if_rpb

        )

    def forward(self, rgb, expr, res):
        x_rgb = self.patch_embed_rgb(rgb)
        x_expr = self.patch_embed_expr(expr)
        B, N, _ = x_rgb.shape
        mask = torch.rand(B, N) < self.mask_ratio

        x = torch.zeros_like(x_rgb)
        x[mask] = x_rgb[mask]
        x[~mask] = x_expr[~mask]

        x = self.pos_embed(x, res)
        z = self.blocks(x, mask)
        
        return z, mask

    def forward_rgb(self, rgb, res):
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        z = self.blocks.forward_rgb(x)
        
        return z

    def forward_expr(self, expr, res):
        x = self.patch_embed_expr(expr)
        x = self.pos_embed(x, res)
        z = self.blocks.forward_expr(x)
        
        return z
    
    def forward_all(self, rgb, expr, res):
        x_rgb = self.patch_embed_rgb(rgb)
        x_expr = self.patch_embed_expr(expr)
        x_rgb = self.pos_embed(x_rgb, res)
        x_expr = self.pos_embed(x_expr, res)
        x = torch.cat([x_rgb, x_expr], dim=1)
        z = self.blocks.forward_all(x)
        
        return z


class SquallDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.expr_size = config.expr_size
        self.down_sample = config.down_sample
        self.expr_down_sample = config.down_sample // (config.img_size // config.expr_size)
        '''
        self.if_PE = config.if_PE
        if config.if_PE:
            self.pos_embed_rgb = PositionEmbeding(config)
            self.pos_embed_expr = PositionEmbeding(config)
        '''
        #print("config.if_PE",config.if_PE)
        #print("config.if_rpb",config.if_rpb)
        self.decoder_rgb = TransformerDecoder(
            embed_dim=config.embed_dim,
            depth=config.decoder_depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            if_rpb = config.if_rpb
        )
        self.decoder_expr = TransformerDecoder(
            embed_dim=config.embed_dim,
            depth=config.decoder_depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            if_rpb = config.if_rpb
        )
        self.increase_dim_rgb = LinearShuffle(config.embed_dim, config.expr_chans, self.expr_down_sample)
        self.increase_dim_expr = nn.Linear(config.embed_dim, 3 * config.down_sample ** 2)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, z, res):
        B = z.shape[0]
        H = W = self.img_size // self.down_sample
        '''
        if self.if_PE:
            z_rgb = self.pos_embed_rgb(z, res)
            z_expr = self.pos_embed_expr(z, res)
        else:
        '''
        z_rgb = z
        z_expr = z
        z_rgb = self.decoder_rgb(z_rgb).reshape(B, H, W, -1)
        z_expr = self.decoder_expr(z_expr).reshape(B, H, W, -1)

        pred_rgb = self.increase_dim_expr(z_expr).reshape(B, self.img_size, self.img_size, 3)
        pred_expr = self.increase_dim_rgb(z_rgb).reshape(B, self.expr_size, self.expr_size, self.config.expr_chans)
        #pred_rgb = torch.sigmoid(pred_rgb)
        #pred_expr = torch.sigmoid(pred_expr)

        return pred_rgb, pred_expr
    def forward_rgb_to_expr(self, z, res):
        B = z.shape[0]
        H = W = self.img_size // self.down_sample
        #print("forward_rgb_get_embedding z.shape",z.shape)
        z_rgb = self.decoder_rgb(z).reshape(B, H, W, -1)
        #print("z_rgb",z_rgb.shape)
        pred_expr = self.increase_dim_rgb(z_rgb).reshape(B, self.expr_size, self.expr_size, self.config.expr_chans)
        return pred_expr


# Pretrain model
@MODELS.register_module()
class Squall(nn.Module):
    def __init__(self, config):
        super().__init__()
        #print_log(f'[Squall]', logger='Squall')
        self.config = config
        self.img_size = config.img_size
        self.expr_size = config.expr_size
        self.encoder = SquallEncoder(config)
        self.decoder = SquallDecoder(config)

        self.l1_loss = torch.nn.SmoothL1Loss()
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

    def load_model_from_ckpt(self, ckpt_path, log=True):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('encoder'):
                    print("need loaded ",k,base_ckpt[k])
                    base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if log:
                if incompatible.missing_keys:
                    print_log('missing_keys', logger='Squall')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='Squall'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='Squall')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='Squall'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='Squall')
        else:
            print_log('Training from scratch!!!', logger='Squall')

    def forward(self, rgb, expr, res):
        expr = expr.to_dense().float() #qbw10.16
        #print(expr.shape)
        size = int(math.sqrt(expr.shape[1]))
        expr = expr.reshape(expr.shape[0],size, size, expr.shape[-1])#qbw10.16
        expr = expr.permute(0, 3, 1, 2)#qbw10.16

        z, mask = self.encoder(rgb, expr, res)
        B, N, _ = z.shape
        pred_rgb, pred_expr = self.decoder(z, res)
        #print("pred_rgb",pred_rgb)
        #print("pred_expr",pred_expr)

        mask = mask.reshape(B, 1, int(math.sqrt(N)), int(math.sqrt(N))).float()
        rgb_mask = F.interpolate(mask, size=(self.img_size, self.img_size), mode='nearest').bool().squeeze(1)
        expr_mask = F.interpolate(mask, size=(self.expr_size, self.expr_size), mode='nearest').bool().squeeze(1)
        rgb = rgb.permute(0, 2, 3, 1)
        expr = expr.permute(0, 2, 3, 1)
        rgb_loss = self.l1_loss(pred_rgb[~rgb_mask], rgb[~rgb_mask])
        expr_loss = self.l1_loss(pred_expr[expr_mask], expr[expr_mask]) #* (self.img_size / self.expr_size) ** 2 qbw 11.21
        print_log(f'rgb_loss: {rgb_loss.item()} expr_loss: {expr_loss.item()}', logger='Squall')
        return rgb_loss, expr_loss

    def forward_rgb(self, rgb, res):
        z = self.encoder.forward_rgb(rgb, res)
        return z

    def forward_expr(self, expr, res):
        z = self.encoder.forward_expr(expr, res)
        return z
    
    def forward_all(self, rgb, expr, res):
        z = self.encoder.forward_all(rgb, expr, res)
        return z

    def forward_rgb_to_expr(self, rgb, res):
        if rgb.shape[1] != 3:
            rgb = rgb.permute(0, 3, 1, 2)  # rgb -> (B, C, H, W)#qbw change 1.15
        #cls = self.cls_token.expand(rgb.shape[0], -1, -1) + self.cls_pos
        z = self.encoder.forward_rgb(rgb,res)
        #x = self.patch_embed_rgb(rgb)
        #x = self.pos_embed(x, res)
        #x = torch.cat([cls, x], dim=1)
        #z = self.blocks.forward_rgb(x)
        #print("forward_rgb_to_expr ,z.shape",z.shape)
        B, N, _ = z.shape
        pred_expr = self.decoder.forward_rgb_to_expr(z, res)

        return pred_expr
    def forward_all_to_expr(self, rgb,expr,res):
        '''
        if rgb.shape[1] != 3:
            rgb = rgb.permute(0, 3, 1, 2)  # rgb -> (B, C, H, W)#qbw change 1.15
        #cls = self.cls_token.expand(rgb.shape[0], -1, -1) + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        #x = torch.cat([cls, x], dim=1)
        z = self.blocks.forward_rgb(x)
        '''
        z = self.encoder.forward_all(rgb, expr, res)
        print("forward_rgb_to_expr ,z.shape",z.shape)
        B, N, _ = z.shape
        mid = N//2
        rgb_emb = z[:, :mid, :]
        pred_expr = self.decoder.forward_rgb_to_expr(rgb_emb, res)
        #expr_predicted_perm = pred_expr.permute(0, 3, 1, 2)
        # 2. 插值为 224x224
        #expr_predicted_upsampled = F.interpolate(expr_predicted_perm, size=(224, 224), mode='bilinear', align_corners=False)
        # 3. 转回 [B, 224, 224, 15757]
        #expr_predicted_upsampled = expr_predicted_upsampled.permute(0, 2, 3, 1)
        return pred_expr

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
class Squall_contrast(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Squall_contrast]', logger='Squall_contrast')
        self.config = config
        self.img_size = config.img_size
        self.expr_size = config.expr_size
        self.encoder = SquallEncoder(config)

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
                    print_log('missing_keys', logger='Squall')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='Squall_contrast'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='Squall')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='Squall_contrast'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='Squall')
        else:
            print_log('Training from scratch!!!', logger='Squall')

    def forward(self, rgb, expr, res):


        expr = expr.to_dense().float() #qbw10.16
        #print(expr.shape)
        size = int(math.sqrt(expr.shape[1]))
        expr = expr.reshape(expr.shape[0],size, size, expr.shape[-1])#qbw10.16
        expr = expr.permute(0, 3, 1, 2)#qbw10.16
        #self.encoder = SquallEncoder(config)
        #z_rgb , z_expr = self.encoder(rgb, expr, res)
        z_rgb = self.encoder.forward_rgb(rgb, res)
        z_expr = self.encoder.forward_expr(expr, res)
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

        print_log(f'inter_contrastive_loss: {inter_contrastive_loss.item()} intra_contrastive_loss: {intra_contrastive_loss.item()}', logger='Squall_contrast')
        return inter_contrastive_loss, intra_contrastive_loss

    def forward_rgb(self, rgb, res):
        z = self.encoder.forward_rgb(rgb, res)
        return z

    def forward_expr(self, expr, res):
        z = self.encoder.forward_expr(expr, res)
        return z




# Pretrain model
@MODELS.register_module()
class Squall_matching(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Squall_matching]', logger='Squall_matching')
        self.config = config
        self.img_size = config.img_size
        self.expr_size = config.expr_size
        self.encoder = SquallEncoder(config)

        self.contrastive_head = InfoNCE(temperature=0.1)
        self.bce_loss = nn.BCEWithLogitsLoss()
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
                    print_log('missing_keys', logger='Squall')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='Squall_contrast'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='Squall')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='Squall_contrast'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='Squall')
        else:
            print_log('Training from scratch!!!', logger='Squall')

    def forward(self, rgb, expr, res):


        expr = expr.to_dense().float() #qbw10.16
        #print(expr.shape)
        size = int(math.sqrt(expr.shape[1]))
        expr = expr.reshape(expr.shape[0],size, size, expr.shape[-1])#qbw10.16
        expr = expr.permute(0, 3, 1, 2)#qbw10.16
        #self.encoder = SquallEncoder(config)
        #z_rgb , z_expr = self.encoder(rgb, expr, res)
        z_rgb = self.encoder.forward_rgb(rgb, res)
        z_expr = self.encoder.forward_expr(expr, res)
        B, N, C = z_rgb.shape

        itm_loss = 0.0

        for i in range(N):
            z_rgb_inter = z_rgb[:, i]
            z_expr_inter = z_expr[:, i]
            z_rgb_inter = nn.functional.normalize(z_rgb_inter, dim=1)
            z_expr_inter = nn.functional.normalize(z_expr_inter, dim=1)
            similarity = torch.matmul(z_rgb_inter, z_expr_inter.permute(1, 0))
            labels = torch.eye(B, device=z_rgb.device)
            itm_loss += self.bce_loss(similarity, labels) / N
            dummy_loss = torch.zeros_like(itm_loss)

        print_log(f'itm_loss: {itm_loss.item()}', logger='Squall_matching')
        return itm_loss, dummy_loss

    def forward_rgb(self, rgb, res):
        z = self.encoder.forward_rgb(rgb, res)
        return z

    def forward_expr(self, expr, res):
        z = self.encoder.forward_expr(expr, res)
        return z






def calculate_auc(labels, logits, num_classes):
    """
    Calculate AUC-ROC for multi-class classification, ignoring classes not present in the dataset.

    Args:
        labels (Tensor): Ground truth labels, shape (batch_size,).
        logits (Tensor): Predicted logits, shape (batch_size, num_classes).
        num_classes (int): Total number of classes.

    Returns:
        float: Mean AUC-ROC score over present classes.
    """
    try:
        # One-hot encode the labels
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).cpu().numpy()
        
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        
        # Check which classes are present in the labels
        positive_samples_per_class = labels_one_hot.sum(axis=0)
        present_classes = np.where(positive_samples_per_class > 0)[0]
        
        if len(present_classes) == 0:
            print("No classes present in the test set.")
            return -1

        # Compute AUC for present classes only
        auc_list = []
        for c in present_classes:
            auc = roc_auc_score(labels_one_hot[:, c], probabilities[:, c])
            auc_list.append(auc)
        mean_auc = np.mean(auc_list)
        return mean_auc
    except ValueError as e:
        if labels_one_hot.shape[0]>1:
            print(f"ValueError during AUC-ROC calculation: {e}")
        return -1




# Finetune model
@MODELS.register_module()
class SquallClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
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
        self.pos_embed = PositionEmbeding(config)
        self.blocks = MultiWayEncoder(
            merge_layer_depth=config.merge_layer_depth,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,if_rpb = config.if_rpb

        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(config.embed_dim, config.cls_dim)
        )
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.ce_loss = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        # ret 是模型的 logits 输出，形状 [batch_size, num_classes]
        # label 是实际标签，形状 [batch_size]
        logits = ret.clone()  # logits 形状为 [batch_size, num_classes]
        #label = torch.where(label == -1, torch.tensor(12, device=label.device), label)  # used for cancer classification
        labels = gt.clone().view(-1).long()  # latten
        #print("labels ",labels)
        #print("logits ",logits)
        if logits.dim() == 3:
            logits = logits.squeeze(1)# for batch test
        loss = self.ce_loss(logits, labels)

        preds = logits.argmax(dim=-1)  # 
        #Top-1, Top-3, 和 Top-5 accuracy
        top1_correct = preds.eq(labels).sum().item()  # Top-1 accuracy
        topk_correct = []
        top1_acc = top1_correct / logits.size(0)
        if self.cls_dim > 2:  # 仅在类别数大于 2 时计算 Top-3
            for k in [3, 5]:
                # 获取 logits 的 top-k 预测
                topk_preds = torch.topk(logits, k=k, dim=1).indices
                correct_k = topk_preds.eq(labels.view(-1, 1)).sum().item()
                topk_correct.append(correct_k)
            top3_acc = topk_correct[0] / logits.size(0)
            top5_acc = topk_correct[1] / logits.size(0)
        else:
            top3_acc = 0  # 二分类任务中 Top-3 无意义
            top5_acc = 0

        # accuracy
        #accuracy = accuracy_score(labels.clone().detach().cpu().numpy(), preds.clone().detach().cpu().numpy())
        accuracy = [top1_acc,top3_acc,top5_acc]
        # calculate F1 score
        f1 = f1_score(labels.clone().detach().cpu().numpy(), preds.clone().detach().cpu().numpy(), average='weighted')
        # calculate AUC
        auc_roc = calculate_auc(labels, logits, num_classes=self.cls_dim)

        '''
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100
        '''
        return loss, (accuracy, f1, auc_roc)

    def load_model_from_ckpt(self, ckpt_path, log=True):
        print("its here!!!1 ")
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                print("check all ",k,base_ckpt[k])
                if k.startswith('encoder'):
                    print("need loaded ",k,base_ckpt[k])
                    base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if log:
                if incompatible.missing_keys:
                    print_log('missing_keys', logger='Squall')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='Squall'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='Squall')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='Squall'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='Squall')
        else:
            print_log('Training from scratch!!!', logger='Squall')

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


    def forward_all(self, rgb, expr, res):
        cls = self.cls_token.expand(rgb.shape[0], -1, -1) + self.cls_pos
        x_rgb = self.patch_embed_rgb(rgb)
        x_expr = self.patch_embed_expr(expr)
        x_rgb = self.pos_embed(x_rgb, res)
        x_expr = self.pos_embed(x_expr, res)
        x = torch.cat([cls, x_rgb, x_expr], dim=1)
        z = self.blocks.forward_all(x)

        return z[:, 0]

    def forward_rgb(self, rgb, res):
        #rgb = rgb.permute(0, 3, 1, 2)  # rgb -> (B, C, H, W)#qbw change 1.15
        cls = self.cls_token.expand(rgb.shape[0], -1, -1) + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        x = torch.cat([cls, x], dim=1)
        z = self.blocks.forward_rgb(x)

        return z[:, 0]

    def forward_expr(self, expr, res):
        x = self.patch_embed_expr(expr)
        x = self.pos_embed(x, res)
        z = self.blocks.forward_expr(x)

        return z[:, 0]
    
    def classifier(self, x):
        return self.cls_head_finetune(x)


@MODELS.register_module()
class ADMIL(nn.Module):
    def __init__(self, config):
        super(ADMIL, self).__init__()
        self.embed_dim = config.embed_dim
        self.cls_dim = config.cls_dim
        # Instance-level feature extractor
        self.instance_encoder = nn.Linear(self.embed_dim, self.embed_dim)
        # Attention mechanism
        self.attention = nn.Linear(self.embed_dim, 1)
        # Final classification layer
        self.classifier = nn.Linear(self.embed_dim, self.cls_dim)

        self.apply(self._init_weights)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

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

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, x):
        # Instance encoding: shape (B, N, hidden_dim)
        x = F.relu(self.instance_encoder(x))

        # Attention scores: shape (B, N, 1)
        attn_scores = torch.tanh(self.attention(x))
        attn_scores = F.softmax(attn_scores, dim=1)

        # Weighted sum of instance features: shape (B, hidden_dim)
        bag_feature = torch.sum(attn_scores * x, dim=1)

        # Final classification: shape (B, cls_num)
        output = self.classifier(bag_feature)
        return output


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        print("Attn_Net_Gated input ",L,D)
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.1))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class nll_loss(object):
    def __call__(self, hazards, S, Y, c, alpha=0.4, eps=1e-7, **kwargs):
        '''
        hazard : max predicted hazard
        time : survival times B*1
        '''
        batch_size = 1 if Y.numel() == 1 else Y.shape[0]

        Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k
        c = c.view(batch_size, 1).float()  # censorship status, 0 or 1

        # Calculate the cumulative survival probability, or S if S is empty
        if S is None:
            if batch_size == 1:
                S = torch.cumprod(1 - hazards, dim=0)  # gradient accumulation
            else:
                S = torch.cumprod(1 - hazards, dim=1)  # gradient accumulation

        S = S.view(batch_size, -1)  # [batch_size, num_intervals]

        S_padded = torch.cat([torch.ones((batch_size, 1), device=c.device), S], dim=1)
        hazards = hazards.view(batch_size, -1)  # hazards:  [batch_size, num_intervals]
        # calculate loss of  uncensored and censored samples
        uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) +
            torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
        )
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()

        return loss



class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CoxSurvLoss(object):
    def __call__(self, hazards, time, status, **kwargs):
        '''
        hazard : max predicted hazard
        time : survival times B*1
        '''
        hazards = torch.sigmoid(hazards)
        hazards = hazards.squeeze(1)
        #print("hazards.shape",hazards.shape)
        current_batch_len = len(time)
        #current_batch_len = 1
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)

        # 创建 R 矩阵，R[i,j] 表示 j 的生存时间是否大于等于 i
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = time[j] >= time[i]

        # 将 status 和 R 矩阵转换为 tensor，确保与 hazards 的 dtype 和 device 一致
        status = torch.tensor(status, dtype=hazards.dtype, device=hazards.device).unsqueeze(1)
        R_mat = torch.tensor(R_mat, dtype=hazards.dtype, device=hazards.device)

        # 计算 Cox 损失
        theta = hazards.view(current_batch_len, -1)  # 确保 theta 的形状为 [batch_size, num_outputs]
        exp_theta = torch.exp(theta)
        #print("theta",theta)
        #print("exp_theta",exp_theta)
        # 对每列风险值进行操作，并累加结果
        loss_cox = 0
        for t in range(theta.shape[1]):  # 遍历每一个时间点的风险值
            theta_t = theta[:, t]  # 当前时间点的 theta，形状为 [10]
            exp_theta_t = exp_theta[:, t]  # 当前时间点的 exp(theta)，形状为 [10]
            log_sum_exp_t = torch.log(torch.sum(exp_theta_t.unsqueeze(1) * R_mat, dim=1))  # 按列操作 R_mat
            loss_cox += -torch.mean((theta_t - log_sum_exp_t) * (1 - status.squeeze()))  # 汇总所有时间点的损失

        return loss_cox

class nll_loss(object):
    def __call__(self, hazards, S, Y, c, alpha=0.4, eps=1e-7, **kwargs):
        '''
        hazard : max predicted hazard
        time : survival times B*1
        '''
        # 确保 batch_size 为 1
        batch_size = 1 if Y.numel() == 1 else Y.shape[0]

        Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k，确保 Y 是 int64 类型
        c = c.view(batch_size, 1).float()  # censorship status, 0 or 1

        # 计算累计生存概率，如果 S 为空则计算 S
        if S is None:
            if batch_size == 1:
                S = torch.cumprod(1 - hazards, dim=0)  # 在 dim=0 维度上累计乘积
            else:
                S = torch.cumprod(1 - hazards, dim=1)  # 在 dim=0 维度上累计乘积

        
        '''
        if batch_size == 1:
            print("S:", S)
            print("c:", c)
            print("Y:", Y)
            print("hazards:", hazards)
        '''


        # 进行维度匹配，以便进行拼接
        S = S.view(batch_size, -1)  # 将 S 转换为二维张量 [batch_size, num_intervals]

        # 在前面添加一个值为 1 的元素，用于表示起始生存概率
        S_padded = torch.cat([torch.ones((batch_size, 1), device=c.device), S], dim=1)  # 在 dim=1 上拼接

        # 调整 hazards 的形状
        hazards = hazards.view(batch_size, -1)  # 确保 hazards 形状为 [batch_size, num_intervals]
        #print("S_padded:", S_padded)
        #print("hazards:", hazards)
        # 计算 uncensored 和 censored 的损失
        uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) +
            torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
        )
        #print("uncensored_loss",uncensored_loss)
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
        #print("censored_loss",censored_loss)
        # 计算总的负对数似然损失
        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        #print("loss calculate",loss)
        loss = loss.mean()

        return loss


def calculate_auc(labels, logits, num_classes):
    """
    Calculate AUC-ROC for multi-class classification, ignoring classes not present in the dataset.

    Args:
        labels (Tensor): Ground truth labels, shape (batch_size,).
        logits (Tensor): Predicted logits, shape (batch_size, num_classes).
        num_classes (int): Total number of classes.

    Returns:
        float: Mean AUC-ROC score over present classes.
    """
    try:
        # One-hot encode the labels
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).cpu().numpy()
        
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        
        # Check which classes are present in the labels
        positive_samples_per_class = labels_one_hot.sum(axis=0)
        present_classes = np.where(positive_samples_per_class > 0)[0]
        
        if len(present_classes) == 0:
            print("No classes present in the test set.")
            return -1

        # Compute AUC for present classes only
        auc_list = []
        for c in present_classes:
            auc = roc_auc_score(labels_one_hot[:, c], probabilities[:, c])
            auc_list.append(auc)
        mean_auc = np.mean(auc_list)
        return mean_auc
    except ValueError as e:
        if labels_one_hot.shape[0]>1:
            print(f"ValueError during AUC-ROC calculation: {e}")
        return -1

def bootstrap_evaluation(model, X_test, y_test, n_iter=1000):
    top1_acc = []
    top3_acc = []
    top5_acc = []
    auroc_scores = []
    weighted_f1_scores = []
    
    for _ in range(n_iter):
        # sampling with replacement
        X_resampled, y_resampled = resample(X_test, y_test, random_state=42)
        
        # prediction
        y_pred = model.predict(X_resampled)
        y_prob = model.predict_proba(X_resampled) 
        
        # Top-1 Accuracy
        top1 = np.mean(y_pred == y_resampled)
        top1_acc.append(top1)
        
        # Top-5 Accuracy
        top5 = np.mean([y_resampled[i] in np.argsort(y_prob[i])[-5:] for i in range(len(y_resampled))])
        top5_acc.append(top5)
        
        #  AUROC
        y_resampled_bin = label_binarize(y_resampled, classes=np.unique(y_resampled))
        auroc = roc_auc_score(y_resampled_bin, y_prob, average='macro', multi_class='ovr')
        auroc_scores.append(auroc)
        
        #  Weighted F1 Score
        weighted_f1 = f1_score(y_resampled, y_pred, average='weighted')
        weighted_f1_scores.append(weighted_f1)
    
    # 2.5%,  97.5% quantile
    top1_ci = np.percentile(top1_acc, [2.5, 97.5])
    top5_ci = np.percentile(top5_acc, [2.5, 97.5])
    auroc_ci = np.percentile(auroc_scores, [2.5, 97.5])
    weighted_f1_ci = np.percentile(weighted_f1_scores, [2.5, 97.5])
    
    return {
        "Top-1 Accuracy": (np.mean(top1_acc), top1_ci),
        "Top-5 Accuracy": (np.mean(top5_acc), top5_ci),
        "AUROC": (np.mean(auroc_scores), auroc_ci),
        "Weighted F1": (np.mean(weighted_f1_scores), weighted_f1_ci)
    }




def calculate_metrics_vectorized(label, ret):
    """
    使用矢量化操作计算 Pearson 相关系数、R² 分数和余弦相似度
    """
    # 筛选非全零基因
    non_zero_mask = ~(label == 0).all(axis=0)
    label = label[:, non_zero_mask]
    ret = ret[:, non_zero_mask]

    # 1. 计算 Pearson 相关系数
    label_mean = label.mean(axis=0)
    ret_mean = ret.mean(axis=0)
    label_std = label.std(axis=0)
    ret_std = ret.std(axis=0)
    cov = np.mean((label - label_mean) * (ret - ret_mean), axis=0)
    pearson_corrs = cov / (label_std * ret_std)

    # 2. 计算 R² 分数
    ss_total = np.sum((label - label_mean) ** 2, axis=0)
    ss_residual = np.sum((label - ret) ** 2, axis=0)
    r2_scores = 1 - (ss_residual / ss_total)

    # 3. 计算余弦相似度
    label_norm = np.linalg.norm(label, axis=0)
    ret_norm = np.linalg.norm(ret, axis=0)
    dot_product = np.sum(label * ret, axis=0)
    cos_similarities = dot_product / (label_norm * ret_norm)

    return (
        pearson_corrs,
        r2_scores,
        cos_similarities
    )







@MODELS.register_module()
class ABMIL(nn.Module):
    def __init__(self, config, gate=True, size_arg="large", dropout=True, n_classes=4):
        super(ABMIL, self).__init__()
        ###Squall parts
        self.config = config
        self.patch_embed_rgb = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.down_sample,
            embed_dim=config.embed_dim
        )
        ''''''
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
        
        self.pos_embed = PositionEmbeding(config)
        '''
        self.blocks = TransformerEncoder(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            if_rpb = config.if_rpb,
            PE_RPB_fixed = config.PE_RPB_fixed
        )
        '''
        self.blocks = MultiWayEncoder(
            merge_layer_depth=config.merge_layer_depth,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            if_rpb = config.if_rpb,
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        #self.cls_pos = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        #self.cls_head_finetune = nn.Sequential(nn.Linear(config.embed_dim, config.cls_dim))
        trunc_normal_(self.cls_token, std=.02)
        #trunc_normal_(self.cls_pos, std=.02)


        #no gating version
        self.embed_dim = config.embed_dim
        self.attention = nn.Linear(self.embed_dim, 1)
        self.instance_encoder = nn.Linear(self.embed_dim, self.embed_dim)
        self.cls_dim = config.cls_dim  #





        # qbw 11.14 gating attention
        fc = [nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = Attn_Net_Gated(L=self.embed_dim, D=int(self.embed_dim/2), dropout=dropout, n_classes=1)
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
        if "ceip" in ckpt_path:#not solid, use tempororiliy
            if ckpt_path is not None:
                ckpt = torch.load(ckpt_path)
                base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

                for k in list(base_ckpt.keys()):
                    if k.startswith('encoder'):
                        base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                        del base_ckpt[k]
                for k in list(base_ckpt.keys()):
                    if k.startswith('blocks_rgb'):
                        base_ckpt[k.replace("blocks_rgb","blocks")] = base_ckpt[k]
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
        else:
            if ckpt_path is not None:
                ckpt = torch.load(ckpt_path)
                base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
                #print("base_ckpt ",base_ckpt)
                for k in list(base_ckpt.keys()):
                    if k.startswith('encoder'):
                        #print("find encoder!",k)
                        base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                        del base_ckpt[k]
                    if 'relative_attention_bias' in k and 'encoder' in k:
                        print("relative_attention_bias ",base_ckpt[k[len('encoder.'):]])
                    if 'encoder' in k and 'lookup_table_weight' in k:
                        old_shape = base_ckpt[k[len('encoder.'):]].shape  # 检查旧参数的形状
                        new_shape = self.state_dict()[k[len('encoder.'):]].shape  # 获取当前模型需要的形状
                        print("old ",base_ckpt[k[len('encoder.'):]])
                        print("new ",self.state_dict()[k[len('encoder.'):]])
                        if old_shape[-1] == 49 and new_shape[-1] == 50:
                            print("start change load")
                            # 创建一个新的参数，增加额外的维度
                            new_param = torch.zeros(new_shape, device=base_ckpt[k[len('encoder.'):]].device, dtype=base_ckpt[k[len('encoder.'):]].dtype)
                            new_param[..., :-1] = base_ckpt[k[len('encoder.'):]]  # 将原始权重填充到新参数中，最后一列留空或为零
                            new_param[..., -1] =  0 # 将原始权重填充到新参数中，最后一列留空或为零
                            base_ckpt[k[len('encoder.'):]] = new_param  # 更新到 ckpt 参数中
                    if 'encoder' in k and 'lookup_table_bias' in k:
                        old_shape = base_ckpt[k[len('encoder.'):]].shape  # 检查旧参数的形状
                        new_shape = self.state_dict()[k[len('encoder.'):]].shape  # 获取当前模型需要的形状
                        print("old ",base_ckpt[k[len('encoder.'):]])
                        print("new ",self.state_dict()[k[len('encoder.'):]])
                        if old_shape[-1] == 49 and new_shape[-1] == 50:
                            print("start change load")
                            # 创建一个新的参数，增加额外的维度
                            new_param = torch.zeros(new_shape, device=base_ckpt[k[len('encoder.'):]].device, dtype=base_ckpt[k[len('encoder.'):]].dtype)
                            new_param[..., :-1] = base_ckpt[k[len('encoder.'):]]  # 将原始权重填充到新参数中，最后一列留空或为零
                            new_param[..., -1] =  0 # 将原始权重填充到新参数中，最后一列留空或为零
                            base_ckpt[k[len('encoder.'):]] = new_param  # 更新到 ckpt 参数中

                incompatible = self.load_state_dict(base_ckpt, strict=False)
                if log:
                    if incompatible.missing_keys:
                        print_log('missing_keys', logger='Squall')
                        print_log(
                            get_missing_parameters_message(incompatible.missing_keys),
                            logger='Squall'
                        )
                    if incompatible.unexpected_keys:
                        print_log('unexpected_keys', logger='Squall')
                        print_log(
                            get_unexpected_parameters_message(incompatible.unexpected_keys),
                            logger='Squall'
                        )

                    print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='Squall')
            else:
                print_log('Training from scratch!!!', logger='Squall')



    def build_loss_func(self):
        loss_dict = {"CE":nn.CrossEntropyLoss(),
                    "MSE":nn.MSELoss(),
                    "NLL":nll_loss(),
                    "COX":CoxSurvLoss(),

                        }
        self.loss_abmil = loss_dict[self.config.loss]

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

    def get_loss_acc(self, ret, label):
        if self.config.loss == "COX":
            time,status = label
            # NLL 生存损失函数
            loss = self.loss_abmil(ret,time, status)
            print("loss",loss)
            batch_size = 1 if time.numel() == 1 else time.shape[0]
            if batch_size > 1:
                #print("USE C-index calculate")
                # 使用 c-index 来衡量模型性能
                #print("ret.max()",ret.max())
                #pred_risk = -ret.max()[0]  # 选择 risk 最大的那一个作为预测
                pred_risk = -ret.clone().detach().max(dim=2)[0].cpu().numpy()#.detach()
                pred_risk = pred_risk.flatten()
                status_bool = status.detach().cpu().numpy().astype(bool).flatten()
                time_array = time.detach().cpu().numpy().flatten()
                #print("pred_risk",pred_risk)
                #print("status_bool",status_bool)
                #print("time_array",time_array)
                cindex= concordance_index_censored(
                    status_bool,
                    time_array,
                    pred_risk 
                )[0]

                survival_data = np.core.records.fromarrays([status_bool, time_array], names='event, time')
                times = np.array([0,1, 2, 3])
                assert 'event' in survival_data.dtype.names, "Missing 'event' field in survival_data"
                assert 'time' in survival_data.dtype.names, "Missing 'time' field in survival_data"

                #print("survival_data['time']",survival_data['time'])
                #print("survival_data['event']",survival_data['event'])
                #print("pred_risk",pred_risk)
                #print("times",times)
                dynamic_auc = cindex
                #dynamic_auc =  = cumulative_dynamic_auc(survival_data['time'], survival_data['event'], pred_risk, times)
            else:
                cindex  = 0 
                dynamic_auc = [0] * 4
            return loss, (cindex * 100,dynamic_auc,cindex * 100,)
        if self.config.loss == "NLL":
            time,status = label
            # NLL 生存损失函数
            loss = self.loss_abmil(ret,None, time, status)
            #print("loss",loss)
            batch_size = 1 if time.numel() == 1 else time.shape[0]
            if batch_size > 1:
                #print("USE C-index calculate")
                # 使用 c-index 来衡量模型性能
                #print("ret.max()",ret.max())
                #pred_risk = -ret.max()[0]  # 选择 risk 最大的那一个作为预测
                pred_risk = -ret.clone().detach().max(dim=2)[0].cpu().numpy()#.detach()
                pred_risk = pred_risk.flatten()
                status_bool = status.detach().cpu().numpy().astype(bool).flatten()
                time_array = time.detach().cpu().numpy().flatten()
                #print("pred_risk",pred_risk)
                #print("status_bool",status_bool)
                #print("time_array",time_array)
                cindex= concordance_index_censored(
                    status_bool,
                    time_array,
                    pred_risk 
                )[0]

                survival_data = np.core.records.fromarrays([status_bool, time_array], names='event, time')
                times = np.array([0,1, 2, 3])
                assert 'event' in survival_data.dtype.names, "Missing 'event' field in survival_data"
                assert 'time' in survival_data.dtype.names, "Missing 'time' field in survival_data"

                print("survival_data['time']",survival_data['time'])
                print("survival_data['event']",survival_data['event'])
                print("pred_risk",pred_risk)
                print("times",times)
                dynamic_auc = cumulative_dynamic_auc(survival_data['time'], survival_data['event'], pred_risk, times)
            else:
                cindex  = 0 
                dynamic_auc = [0] * 4
            return loss, (cindex * 100,dynamic_auc,cindex * 100,)
        if self.config.loss == "CE":
            # ret 是模型的 logits 输出，形状 [batch_size, num_classes]
            # label 是实际标签，形状 [batch_size]
            logits = ret.clone()  # logits 形状为 [batch_size, num_classes]
            #label = torch.where(label == -1, torch.tensor(12, device=label.device), label)  # used for cancer classification
            labels = label.clone().view(-1).long()  # latten
            #print("labels ",labels)
            #print("logits ",logits)
            if logits.dim() == 3:
                logits = logits.squeeze(1)# for batch test
            #print("logits",logits.shape)
            #print("labels",labels.shape)
            loss = self.ce_loss(logits, labels)

            preds = logits.argmax(dim=-1)  # 
            #Top-1, Top-3, 和 Top-5 accuracy
            top1_correct = preds.eq(labels).sum().item()  # Top-1 accuracy
            topk_correct = []
            top1_acc = top1_correct / logits.size(0)
            if self.cls_dim > 2:  # 仅在类别数大于 2 时计算 Top-3
                for k in [3, 5]:
                    # 获取 logits 的 top-k 预测
                    topk_preds = torch.topk(logits, k=k, dim=1).indices
                    correct_k = topk_preds.eq(labels.view(-1, 1)).sum().item()
                    topk_correct.append(correct_k)
                top3_acc = topk_correct[0] / logits.size(0)
                top5_acc = topk_correct[1] / logits.size(0)
            else:
                top3_acc = 0  # 二分类任务中 Top-3 无意义
                top5_acc = 0

            # accuracy
            #accuracy = accuracy_score(labels.clone().detach().cpu().numpy(), preds.clone().detach().cpu().numpy())
            accuracy = [top1_acc,top3_acc,top5_acc]
            # calculate F1 score
            f1 = f1_score(labels.clone().detach().cpu().numpy(), preds.clone().detach().cpu().numpy(), average='weighted')
            # calculate AUC
            auc_roc = calculate_auc(labels, logits, num_classes=self.cls_dim)


            return loss, (accuracy, f1, auc_roc)
        else:
            loss_fn = nn.MSELoss()
            loss = loss_fn(ret, label)  # MSELoss loss
            if label.dim() == 1:
                label = label.unsqueeze(0) 
            if ret.dim() == 3:
                ret = ret.squeeze(1)
            batch_size = label.shape[0]
            pearson_corrs = []
            for i in range(batch_size):
                corr, _ = pearsonr(label[i].clone().detach().cpu().numpy(), ret[i].clone().detach().cpu().numpy())  # 计算每个样本的皮尔森相关系数
                pearson_corrs.append(corr)
            mean_pearson_corr = np.mean(pearson_corrs)

            r2_scores = []
            for i in range(batch_size):
                r2 = r2_score(label[i].clone().detach().cpu().numpy(), ret[i].clone().detach().cpu().numpy())  # 计算每个样本的R²
                r2_scores.append(r2)
            mean_r2 = np.mean(r2_scores)

            # 3. 计算平均余弦相似度
            cos_similarities = []
            for i in range(batch_size):
                cos_sim = cosine_similarity(label[i].clone().detach().cpu().numpy().reshape(1, -1), ret[i].clone().detach().cpu().numpy().reshape(1, -1))[0][0]  # 计算每个样本的余弦相似度
                cos_similarities.append(cos_sim)
            mean_cos_sim = np.mean(cos_similarities)
            '''
            if batch_size == 1:
                #per sample
                pearson_corrs = []
                for i in range(batch_size):
                    corr, _ = pearsonr(label[i].clone().detach().cpu().numpy(), ret[i].clone().detach().cpu().numpy())  # 计算每个样本的皮尔森相关系数
                    pearson_corrs.append(corr)
                mean_pearson_corr = np.mean(pearson_corrs)

                r2_scores = []
                for i in range(batch_size):
                    r2 = r2_score(label[i].clone().detach().cpu().numpy(), ret[i].clone().detach().cpu().numpy())  # 计算每个样本的R²
                    r2_scores.append(r2)
                mean_r2 = np.mean(r2_scores)

                # 3. 计算平均余弦相似度
                cos_similarities = []
                for i in range(batch_size):
                    cos_sim = cosine_similarity(label[i].clone().detach().cpu().numpy().reshape(1, -1), ret[i].clone().detach().cpu().numpy().reshape(1, -1))[0][0]  # 计算每个样本的余弦相似度
                    cos_similarities.append(cos_sim)
                mean_cos_sim = np.mean(cos_similarities)
            else:
                num_genes = label.shape[1]  # 基因数量
                pearson_corrs = []
                r2_scores = []
                cos_similarities = []
                #label_gene = label[:, 0].clone().detach().cpu().numpy()
                #ret_gene = ret[:, 0].clone().detach().cpu().numpy()
                label_np = []
                ret_np = []
                for gene_idx in range(num_genes):
                    # 提取当前基因在所有样本中的预测值和标签值
                    label_gene = label[:, gene_idx].clone().detach().cpu().numpy()
                    ret_gene = ret[:, gene_idx].clone().detach().cpu().numpy()
                    if np.all(label_gene == 0):
                        continue
                    label_np.append(label_np)
                    ret_np.append(ret_gene)

                for gene_idx in range(num_genes):
                    # 提取当前基因在所有样本中的预测值和标签值
                    label_gene = label[:, gene_idx].clone().detach().cpu().numpy()
                    ret_gene = ret[:, gene_idx].clone().detach().cpu().numpy()
                    if np.all(label_gene == 0):
                        continue
                    #print("label_gene",label_gene.shape)
                    #print("ret_gene",ret_gene.shape)
                    
                    # 1. 计算 Pearson 相关系数
                    corr, _ = pearsonr(label_gene, ret_gene)
                    pearson_corrs.append(corr)
                    
                    # 2. 计算 R² 分数
                    r2 = r2_score(label_gene, ret_gene)
                    r2_scores.append(r2)
                    
                    # 3. 计算余弦相似度
                    cos_sim = cosine_similarity(label_gene.reshape(1, -1), ret_gene.reshape(1, -1))[0][0]
                    cos_similarities.append(cos_sim)
                
                mean_pearson_corr = np.mean(pearson_corrs)
                mean_r2 = np.mean(r2_scores)
                mean_cos_sim = np.mean(cos_similarities)

                # 计算每个统计量的平均值
                #print("pearson_corrs",pearson_corrs)
                #print("r2_scores",r2_scores)
                #print("cos_similarities",cos_similarities)
            '''
            # all metrice
            return loss, ( mean_pearson_corr,mean_r2, mean_cos_sim)


    def forward_ddp(self,  rgb, res):
        def print_gpu_memory():
            if torch.cuda.is_available():
                print(f"当前GPU占用内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            #print(f"模型计算图保留的内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        rgb = rgb.permute(0, 3, 1, 2)  # 将维度调整为 (B, C, H, W)
        cls = self.cls_token.expand(rgb.shape[0], -1, -1) + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        x = torch.cat([cls, x], dim=1)
        z = self.blocks.forward_rgb(x)
        print("forward_rgb")
        print_gpu_memory()
        x = z[:, 0]
        #x : B 1024 
        # MIL
        x = F.relu(self.instance_encoder(x))
        # Attention scores: shape (B, 1)
        attn_scores = torch.tanh(self.attention(x))
        attn_scores = F.softmax(attn_scores, dim=1)
        # Weighted sum of instance features: shape (B, hidden_dim)
        bag_feature = torch.sum(attn_scores * x, dim=0)
        print("bag_feature shape",bag_feature.shape)
        return bag_feature

    def forward_get_embedding(self,  rgb, res):
        rgb = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
        cls = self.cls_token.expand(rgb.shape[0], -1, -1)# + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        x = torch.cat([cls, x], dim=1)
        #x = torch.cat([x, cls], dim=1)# maybe for iRPB is work
        #z = self.blocks.forward(x, res)
        z = self.blocks.forward_rgb(x)

        features = z[:, 0]
        #features = z[:, -1]# maybe for iRPB is work
        features = features.contiguous()
        gathered_features = AllGatherWithGrad.apply(features)
        return gathered_features

    def forward_get_embedding_expr(self,  expr, res):
        def print_gpu_memory():
            if torch.cuda.is_available():
                print(f"当前GPU占用内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            #print(f"模型计算图保留的内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        expr = expr.to_dense().float() #qbw10.16
        size = int(math.sqrt(expr.shape[1]))
        expr = expr.reshape(expr.shape[0],size, size, expr.shape[-1])#qbw10.16
        expr = expr.permute(0, 3, 1, 2)#qbw10.16
        cls = self.cls_token.expand(expr.shape[0], -1, -1)# + self.cls_pos
        x = self.patch_embed_expr(expr)
        x = self.pos_embed(x, res)
        x = torch.cat([cls, x], dim=1)
        z = self.blocks.forward_expr(x)
        
        features = z[:, 0]
        features = features.contiguous()

        gathered_features = AllGatherWithGrad.apply(features)
        return gathered_features

    def resample_overlap(self,overlap_rate):
        pass

    def forward_from_embedding(self, gathered_features):
        #print("gathered_features",gathered_features.shape)
        A, x = self.attention_net(gathered_features)
        
        ###from propoise
        A = torch.transpose(A, 1, 0)
        bag_feature = torch.mm(F.softmax(A, dim=1) , x)
        attn_scores = A


        '''
        #old version, attention no gradient 
        A = F.softmax(A, dim=1)
        
        bag_feature = A * x  # 元素级乘法，对 x 进行加权 => [batchsize, 1024]
        #print("bag_feature.shape",bag_feature.shape)
        bag_feature = (x * A).sum(dim=0, keepdim=True)  # 在行上做求和，得到 1x1024 的结果
        #print("bag_feature.shape",bag_feature.shape)
        attn_scores = A
        '''
        output = self.classifier(bag_feature)#.clone()
        return output,attn_scores

    def forward_from_embedding_qzk(self, gathered_features):
        #Version.1 no gating by qzk
        x = F.relu(self.instance_encoder(gathered_features), inplace=False)
        # Attention scores: shape (B, 1)
        attn_scores = torch.tanh(self.attention(x))
        attn_scores = F.softmax(attn_scores, dim=0)#qbw 11.11 debug
        bag_feature = torch.sum(attn_scores * x, dim=0)#.clone()
        bag_feature = bag_feature.unsqueeze(0)
        output = self.classifier(bag_feature)#.clone()
        return output,attn_scores


    def forward(self,  rgb, res):
        def print_gpu_memory():
            if torch.cuda.is_available():
                print(f"当前GPU占用内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            #print(f"模型计算图保留的内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        rgb = rgb.permute(0, 3, 1, 2)  # 将维度调整为 (B, C, H, W)
        cls = self.cls_token.expand(rgb.shape[0], -1, -1)# + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        #x = torch.cat([x, cls], dim=1)# maybe for iRPB is work
        x = torch.cat([cls, x], dim=1)
        #z = self.blocks.forward(x, res)
        z = self.blocks.forward_rgb(x, res)
        features = z[:, 0]
        #eatures = z[:, -1]# maybe for iRPB is work
        features = features.contiguous()
        gathered_features = AllGatherWithGrad.apply(features)
        A, x = self.attention_net(gathered_features)
        
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


