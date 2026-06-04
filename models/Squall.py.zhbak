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
        z_rgb = self.decoder_rgb(z_rgb).reshape(B, H, W, -1)
        z_expr = self.decoder_expr(z_expr).reshape(B, H, W, -1)

        pred_rgb = self.increase_dim_expr(z_expr).reshape(B, self.img_size, self.img_size, 3)
        pred_expr = self.increase_dim_rgb(z_rgb).reshape(B, self.expr_size, self.expr_size, self.config.expr_chans)
        #pred_rgb = torch.sigmoid(pred_rgb)
        #pred_expr = torch.sigmoid(pred_expr)

        return pred_rgb, pred_expr




# Pretrain model
@MODELS.register_module()
class Squall(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Squall]', logger='Squall')
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
                    print_log('missing_keys', logger='SQUALL')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='SQUALL'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='SQUALL')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='SQUALL'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='SQUALL')
        else:
            print_log('Training from scratch!!!', logger='SQUALL')

    def forward(self, rgb, expr, res):
        expr = expr.to_dense().float() #qbw10.16
        #print(expr.shape)
        size = int(math.sqrt(expr.shape[1]))
        expr = expr.reshape(expr.shape[0],size, size, expr.shape[-1])#qbw10.16
        expr = expr.permute(0, 3, 1, 2)#qbw10.16

        z, mask = self.encoder(rgb, expr, res)
        B, N, _ = z.shape
        pred_rgb, pred_expr = self.decoder(z, res)

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
class SQUALL_contrast(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[SQUALL_contrast]', logger='SQUALL_contrast')
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
                    print_log('missing_keys', logger='SQUALL')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='SQUALL_contrast'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='SQUALL')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='SQUALL_contrast'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='SQUALL')
        else:
            print_log('Training from scratch!!!', logger='SQUALL')

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

        print_log(f'inter_contrastive_loss: {inter_contrastive_loss.item()} intra_contrastive_loss: {intra_contrastive_loss.item()}', logger='SQUALL_contrast')
        return inter_contrastive_loss, intra_contrastive_loss

    def forward_rgb(self, rgb, res):
        z = self.encoder.forward_rgb(rgb, res)
        return z

    def forward_expr(self, expr, res):
        z = self.encoder.forward_expr(expr, res)
        return z




# Pretrain model
@MODELS.register_module()
class SQUALL_matching(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[SQUALL_matching]', logger='SQUALL_matching')
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
                    print_log('missing_keys', logger='SQUALL')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='SQUALL_contrast'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='SQUALL')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='SQUALL_contrast'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='SQUALL')
        else:
            print_log('Training from scratch!!!', logger='SQUALL')

    def forward(self, rgb, expr, res):


        expr = expr.to_dense().float() #qbw10.16
        size = int(math.sqrt(expr.shape[1]))
        expr = expr.reshape(expr.shape[0],size, size, expr.shape[-1])#qbw10.16
        expr = expr.permute(0, 3, 1, 2)#qbw10.16
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

        print_log(f'itm_loss: {itm_loss.item()}', logger='SQUALLmatching')
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
        labels = gt.clone().view(-1).long()  # latten
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
        return loss, (accuracy, f1, auc_roc)

    def load_model_from_ckpt(self, ckpt_path, log=True):
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
                    print_log('missing_keys', logger='SQUALL')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='SQUALL'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='SQUALL')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='SQUALL'
                    )
                print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='SQUALL')
        else:
            print_log('Training from scratch!!!', logger='SQUALL')

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
        if rgb.shape[1] != 3:
            rgb = rgb.permute(0, 3, 1, 2)  # rgb -> (B, C, H, W)#qbw change 1.15
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

def partial_ll_loss(lrisks, survival_times, event_indicators):
    """
    implement from MMP faisal lab
    lrisks: log risks, B x 1
    survival_times: time bin, B x 1
    event_indicators: event indicator, B x 1
    """    
    num_uncensored = torch.sum(event_indicators, 0)
    if num_uncensored.item() == 0:
        return {'loss': torch.sum(lrisks) * 0}
    
    sindex = torch.argsort(-survival_times)
    survival_times = survival_times[sindex]
    event_indicators = event_indicators[sindex]
    lrisks = lrisks[sindex]

    log_risk_stable = torch.logcumsumexp(lrisks, 0)

    likelihood = lrisks - log_risk_stable
    uncensored_likelihood = likelihood * event_indicators
    logL = -torch.sum(uncensored_likelihood)
    # negative average log-likelihood
    return logL / num_uncensored



class CoxSurvLoss(object):
    def __call__(self, hazards, time, status, **kwargs):
        '''
        hazards: Predicted hazards (log risks), shape [batch_size, 1]
        time: Survival times, shape [batch_size, 1]
        status: Event indicators, shape [batch_size, 1]
        '''
        # 去掉多余的维度
        lrisks = hazards.squeeze(1)  # [batch_size]
        survival_times = time  # [batch_size]
        event_indicators = status # [batch_size]

        # 打印调试信息
        #print("hazards (log risks):", lrisks)
        #print("survival_times:", survival_times)
        #print("event_indicators:", event_indicators)

        # 调用 partial_ll_loss
        # partial_ll_loss 会自动根据 event_indicators 筛选样本是否对损失有贡献
        loss = partial_ll_loss(lrisks, survival_times, event_indicators)
        return loss

class nll_loss(object):
    def __call__(self, hazards, survival, Y, c, alpha=0.4, eps=1e-7, **kwargs):
        """
        implement from GPFM ,
        Continuous time scale divided into k discrete bins: T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
        Y = T_discrete is the discrete event time:
            - Y = -1 if T_cont \in (-inf, 0)
            - Y = 0 if T_cont \in [0, a_1)
            - Y = 1 if T_cont in [a_1, a_2)
            - ...
            - Y = k-1 if T_cont in [a_(k-1), inf)
        hazards = discrete hazards, hazards(t) = P(Y=t | Y>=t, X) for t = -1, 0, 1, 2, ..., k-1
        survival = survival function, survival(t) = P(Y > t | X)

        All patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
        -> hazards(-1) = 0
        -> survival(-1) = P(Y > -1 | X) = 1

        Summary:
            - neural network is hazard probability function, h(t) for t = 0, 1, 2, ..., k-1
            - h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
        """
        hazards = hazards.squeeze(1)
        #hazards
        batch_size = hazards.shape[0]
        #batch_size = len(Y)
        Y = Y.view(batch_size, 1).long()  # ground truth bin, 0, 1, 2, ..., k-1
        c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
        if survival is None:
            survival = torch.cumprod(
                1 - hazards, dim=1
            )  # survival is cumulative product of 1 - hazards
        #print("survival",survival)
        survival_padded = torch.cat(
            [torch.ones_like(c), survival], 1
        )  # survival(-1) = 1, all patients are alive from (-inf, 0) by definition
        # after padding, survival(t=-1) = survival[0], survival(t=0) = survival[1], survival(t=1) = survival[2], etc
        uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(survival_padded, 1, Y).clamp(min=eps))
            + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
        )
        censored_loss = -c * torch.log(
            torch.gather(survival_padded, 1, Y + 1).clamp(min=eps)
        )
        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
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
        ###For CEIP
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
        trunc_normal_(self.cls_token, std=.02)

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

                incompatible = self.load_state_dict(base_ckpt, strict=False)
                if log:
                    if incompatible.missing_keys:
                        print_log('missing_keys', logger='SQUALL')
                        print_log(
                            get_missing_parameters_message(incompatible.missing_keys),
                            logger='SQUALL'
                        )
                    if incompatible.unexpected_keys:
                        print_log('unexpected_keys', logger='SQUALL')
                        print_log(
                            get_unexpected_parameters_message(incompatible.unexpected_keys),
                            logger='SQUALL'
                        )

                    print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='SQUALL')
            else:
                print_log('Training from scratch!!!', logger='SQUALL')



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
                pred_risk = -ret.clone().detach().max(dim=2)[0].cpu().numpy()#.detach()
                pred_risk = pred_risk.flatten()
                status_bool = status.detach().cpu().numpy().astype(bool).flatten()
                time_array = time.detach().cpu().numpy().flatten()
                try:
                    cindex = concordance_index_censored(
                        status_bool,
                        time_array,
                        pred_risk 
                    )[0]
                except:
                    cindex = 0#some times no uncensored patients

                survival_data = np.core.records.fromarrays([status_bool, time_array], names='event, time')
                times = np.array([0,1, 2, 3])
                assert 'event' in survival_data.dtype.names, "Missing 'event' field in survival_data"
                assert 'time' in survival_data.dtype.names, "Missing 'time' field in survival_data"
                dynamic_auc = cindex
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
                pred_risk = -ret.clone().detach().max(dim=2)[0].cpu().numpy()#.detach()
                pred_risk = pred_risk.flatten()
                status_bool = status.detach().cpu().numpy().astype(bool).flatten()
                time_array = time.detach().cpu().numpy().flatten()
                try:
                    cindex= concordance_index_censored(
                        status_bool,
                        time_array,
                        pred_risk 
                    )[0]
                except:
                    cindex = 0 #some times no uncensored patients
                survival_data = np.core.records.fromarrays([status_bool, time_array], names='event, time')
                times = np.array([0,1, 2, 3])
                assert 'event' in survival_data.dtype.names, "Missing 'event' field in survival_data"
                assert 'time' in survival_data.dtype.names, "Missing 'time' field in survival_data"
                #dynamic_auc = cumulative_dynamic_auc(survival_data['time'], survival_data['event'], pred_risk, times)
                dynamic_auc = 0
            else:
                cindex  = 0 
                dynamic_auc = 0
            return loss, (cindex * 100,dynamic_auc,cindex * 100,)
        if self.config.loss == "CE":
            # ret :logits output，shape:[batch_size, num_classes]
            # label : shape: batch_size
            logits = ret.clone()  # logits 形状为 [batch_size, num_classes]
            labels = label.clone().view(-1).long()  # latten
            if logits.dim() == 3:
                logits = logits.squeeze(1)# for batch test
            loss = self.ce_loss(logits, labels)

            preds = logits.argmax(dim=-1)  # 
            #Top-1, Top-3, 和 Top-5 accuracy
            top1_correct = preds.eq(labels).sum().item()  # Top-1 accuracy
            topk_correct = []
            top1_acc = top1_correct / logits.size(0)
            if self.cls_dim > 4:  # 仅在类别数大于 2 时计算 Top-3
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
            r2_scores = []
            cos_similarities = []
            for i in range(batch_size):
                label_np = label[i].clone().detach().cpu().numpy()
                ret_np = ret[i].clone().detach().cpu().numpy()
                if np.all(label_np == 0) or np.all(ret_np == 0):
                    continue  # 跳过当前样本
                corr, _ = pearsonr(label_np, ret_np)
                pearson_corrs.append(corr)
                r2 = r2_score(label_np, ret_np)
                r2_scores.append(r2)
                cos_sim = cosine_similarity(label_np.reshape(1, -1), ret_np.reshape(1, -1))[0][0]
                cos_similarities.append(cos_sim)
            mean_pearson_corr = np.mean(pearson_corrs) if pearson_corrs else 0  # 避免列表为空时报错
            mean_r2 = np.mean(r2_scores) if r2_scores else 0
            mean_cos_sim = np.mean(cos_similarities) if cos_similarities else 0
            return loss, ( mean_pearson_corr,mean_r2, mean_cos_sim)


    def forward_ddp(self,  rgb, res):
        rgb = rgb.permute(0, 3, 1, 2)  # 将维度调整为 (B, C, H, W)
        cls = self.cls_token.expand(rgb.shape[0], -1, -1) + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        x = torch.cat([cls, x], dim=1)
        z = self.blocks.forward_rgb(x)
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
        z = self.blocks.forward_rgb(x)

        features = z[:, 0]
        features = features.contiguous()
        gathered_features = AllGatherWithGrad.apply(features)
        return gathered_features

    def forward_get_embedding_expr(self,  expr, res):
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
        #output = F.softmax(output, dim = 1)
        return output,attn_scores

    def forward_from_embedding_NoGate(self, gathered_features):
        #Version.1 no gating
        x = F.relu(self.instance_encoder(gathered_features), inplace=False)
        # Attention scores: shape (B, 1)
        attn_scores = torch.tanh(self.attention(x))
        attn_scores = F.softmax(attn_scores, dim=0)#qbw 11.11 debug
        bag_feature = torch.sum(attn_scores * x, dim=0)#.clone()
        bag_feature = bag_feature.unsqueeze(0)
        output = self.classifier(bag_feature)#.clone()
        return output,attn_scores

    def gather_features(self, features):
        """ 在 DataParallel (DP) 下，手动合并所有 GPU 计算的 features """
        if isinstance(features, list):
            # `DataParallel` 会让每张 GPU 计算不同部分的 batch，因此这里手动合并
            features = torch.cat(features, dim=0)  # 拼接所有 GPU 计算的 features
        return features

    def forward(self,  rgb, res):
        rgb = rgb.permute(0, 3, 1, 2)  # 将维度调整为 (B, C, H, W)
        cls = self.cls_token.expand(rgb.shape[0], -1, -1)# + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        #x = torch.cat([x, cls], dim=1)# maybe for iRPB is work
        x = torch.cat([cls, x], dim=1)
        #z = self.blocks.forward(x, res)
        z = self.blocks.forward_rgb(x)
        features = z[:, 0]
        #########
        #print("features.shape ",features.shape)
        #features = self.gather_features(features)
        #print("features_gather.shape ",features.shape)
        ########
        A, x = self.attention_net(features)
        
        ###from propoise
        A = torch.transpose(A, 1, 0)
        bag_feature = torch.mm(F.softmax(A, dim=1) , x)
        print("bag_feature.shape ",bag_feature.shape)
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
        '''
        print("output.shape ",output.shape)
        if isinstance(output, list):  # DP 模式会返回多个 GPU 的 output
            output = torch.stack(output, dim=0).mean(dim=0)  # 对所有 GPU 的 output 求平均
        print("output_concat.shape ",output.shape)
        '''
        
        return output,attn_scores


class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv



@MODELS.register_module()
class SQUALL_LoRA(nn.Module):
    def __init__(self, config, gate=True, size_arg="large", dropout=True, n_classes=4):
        super(SQUALL_LoRA, self).__init__()
        ###squall parts
        self.config = config
        self.patch_embed_rgb = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.down_sample,
            embed_dim=config.embed_dim
        )
        self.pos_embed = PositionEmbeding(config)
        self.blocks = MultiWayEncoder(
            merge_layer_depth=config.merge_layer_depth,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            if_rpb = config.if_rpb,
        )
        for param in self.patch_embed_rgb.parameters():
            param.requires_grad = False
        for param in self.pos_embed.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        


        r = 4
        alpha = 2*r
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        for t_layer_i, blk in enumerate(self.blocks.blocks):
            w_qkv_linear = blk.attn.qkv
            print(w_qkv_linear)
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )     
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        trunc_normal_(self.cls_token, std=.02)

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
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()


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
                    if 'attn.qkv.weight' in k and 'decoder' not in k:
                        #used for loRA
                        print("Found")
                        new_k = k.replace("attn.qkv.weight","attn.qkv.qkv.weight")
                        base_ckpt[new_k] = base_ckpt[k]
                        del base_ckpt[k]
                for k in list(base_ckpt.keys()):  
                    if 'relative_attention_bias' in k and 'encoder' in k:
                        print("relative_attention_bias ",base_ckpt[k[len('encoder.'):]])
                    if k.startswith('encoder'):
                        #print("find encoder!",k)
                        base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                        del base_ckpt[k]
                    

                incompatible = self.load_state_dict(base_ckpt, strict=False)
                if log:
                    if incompatible.missing_keys:
                        print_log('missing_keys', logger='SQUALL')
                        print_log(
                            get_missing_parameters_message(incompatible.missing_keys),
                            logger='SQUALL'
                        )
                    if incompatible.unexpected_keys:
                        print_log('unexpected_keys', logger='SQUALL')
                        print_log(
                            get_unexpected_parameters_message(incompatible.unexpected_keys),
                            logger='SQUALL'
                        )

                    print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='SQUALL')
            else:
                print_log('Training from scratch!!!', logger='SQUALL')

    def build_loss_func(self):
        loss_dict = {"CE":nn.CrossEntropyLoss(),
                    "MSE":nn.MSELoss(),
                    "NLL":nll_loss(),
                    "COX":CoxSurvLoss(),

                        }
        self.loss_abmil = loss_dict[self.config.loss]

    def get_loss_acc(self, ret, label):
        if self.config.loss == "COX":
            time,status = label
            # NLL 生存损失函数
            loss = self.loss_abmil(ret,time, status)
            print("loss",loss)
            batch_size = 1 if time.numel() == 1 else time.shape[0]
            if batch_size > 1:
                pred_risk = -ret.clone().float().detach().max(dim=2)[0].cpu().numpy()#.detach()
                pred_risk = pred_risk.flatten()
                status_bool = status.detach().float().cpu().numpy().astype(bool).flatten()
                time_array = time.detach().float().cpu().numpy().flatten()
                try:
                    cindex = concordance_index_censored(
                        status_bool,
                        time_array,
                        pred_risk 
                    )[0]
                except:
                    cindex = 0#some times no uncensored patients

                survival_data = np.core.records.fromarrays([status_bool, time_array], names='event, time')
                times = np.array([0,1, 2, 3])
                assert 'event' in survival_data.dtype.names, "Missing 'event' field in survival_data"
                assert 'time' in survival_data.dtype.names, "Missing 'time' field in survival_data"
                dynamic_auc = cindex
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
                pred_risk = -ret.clone().float().detach().max(dim=2)[0].cpu().numpy()#.detach()
                pred_risk = pred_risk.flatten()
                status_bool = status.detach().float().cpu().numpy().astype(bool).flatten()
                time_array = time.detach().float().cpu().numpy().flatten()
                try:
                    cindex= concordance_index_censored(
                        status_bool,
                        time_array,
                        pred_risk 
                    )[0]
                except:
                    cindex = 0 #some times no uncensored patients
                survival_data = np.core.records.fromarrays([status_bool, time_array], names='event, time')
                times = np.array([0,1, 2, 3])
                assert 'event' in survival_data.dtype.names, "Missing 'event' field in survival_data"
                assert 'time' in survival_data.dtype.names, "Missing 'time' field in survival_data"
                #dynamic_auc = cumulative_dynamic_auc(survival_data['time'], survival_data['event'], pred_risk, times)
                dynamic_auc = 0
            else:
                cindex  = 0 
                dynamic_auc = 0
            return loss, (cindex * 100,dynamic_auc,cindex * 100,)
        if self.config.loss == "CE":
            # ret :logits output，shape:[batch_size, num_classes]
            # label : shape: batch_size
            logits = ret.clone()  # logits 形状为 [batch_size, num_classes]
            labels = label.clone().view(-1).long()  # latten
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
            return loss, ( mean_pearson_corr,mean_r2, mean_cos_sim)


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

    def forward(self,  rgb, res):
        rgb = rgb.permute(0, 3, 1, 2)  # 将维度调整为 (B, C, H, W)
        cls = self.cls_token.expand(rgb.shape[0], -1, -1)# + self.cls_pos
        x = self.patch_embed_rgb(rgb)
        x = self.pos_embed(x, res)
        #x = torch.cat([x, cls], dim=1)# maybe for iRPB is work
        x = torch.cat([cls, x], dim=1)
        #z = self.blocks.forward(x, res)


        z = self.blocks.forward_rgb(x)



        features = z[:, 0]
        A, x = self.attention_net(features)
        ###from propoise
        A = torch.transpose(A, 1, 0)
        bag_feature = torch.mm(F.softmax(A, dim=1) , x)
        print("bag_feature.shape ",bag_feature.shape)
        attn_scores = A
        output = self.classifier(bag_feature)#.clone()
        
        return output,attn_scores

