import torch
import torch.nn as nn
from timm.models.layers import DropPath


# Transformers
class Mlp(nn.Module):
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


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, hidden_dim=64):
        super(RelativePositionBias, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Define an MLP to learn the position bias based on (x, y) distance
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input is (x, y) pair (2 values)
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)   # Output is num_heads biases (one for each attention head)
        )
    
    def forward(self, N, device):
        '''

        for N = 9
        relative_coords_x
         tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2],
         [1, 0, 1, 1, 0, 1, 1, 0, 1],
         [2, 1, 0, 2, 1, 0, 2, 1, 0],
         [0, 1, 2, 0, 1, 2, 0, 1, 2],
         [1, 0, 1, 1, 0, 1, 1, 0, 1],
         [2, 1, 0, 2, 1, 0, 2, 1, 0],
         [0, 1, 2, 0, 1, 2, 0, 1, 2],
         [1, 0, 1, 1, 0, 1, 1, 0, 1],
         [2, 1, 0, 2, 1, 0, 2, 1, 0]])

        relative_coords_y:
         tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                 [0, 0, 0, 1, 1, 1, 2, 2, 2],
                 [0, 0, 0, 1, 1, 1, 2, 2, 2],
                 [1, 1, 1, 0, 0, 0, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 1, 1, 1],
                 [2, 2, 2, 1, 1, 1, 0, 0, 0],
                 [2, 2, 2, 1, 1, 1, 0, 0, 0],
                 [2, 2, 2, 1, 1, 1, 0, 0, 0]])
        '''
        grid_size = int(N**0.5) 
        coords = torch.arange(grid_size, device=device)
        coords_x = coords.repeat(grid_size)  # 
        coords_y = coords.view(-1, 1).repeat(1, grid_size).view(-1)  

        coords = torch.stack([coords_x, coords_y], dim=-1)  # [N, 2]
        #print("N",N)
        #print("coords.shape ",coords.shape)
        #print("coords[:, 0].shape",coords[:, 0].shape)
        #print("coords[:, 1].shape",coords[:, 1].shape)
        if N != coords.shape[0]:
            k = N - coords.shape[0]
            pad_coords = torch.cat([torch.zeros(k, 2, device=device), coords], dim=0)##放到后面
            relative_coords_x = pad_coords[:, 0].view(N, 1) - pad_coords[:, 0].view(1, N)
            relative_coords_y = pad_coords[:, 1].view(N, 1) - pad_coords[:, 1].view(1, N)
        else:
            relative_coords_x = coords[:, 0].view(N, 1) - coords[:, 0].view(1, N)  #torch.abs( [N, N] x distance
            relative_coords_y = coords[:, 1].view(N, 1) - coords[:, 1].view(1, N)  # [N, N] y distance
        relative_coords = torch.stack([relative_coords_x, relative_coords_y], dim=-1).view(-1, 2)  # [N*N, 2]

        # MLP

        relative_position_bias = self.mlp(relative_coords.float())  # Shape [N*N, num_heads]
        # [N,N,num_heads]
        relative_position_bias = relative_position_bias.view(N, N, self.num_heads)
        
        return relative_position_bias


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,if_rpb = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.if_rpb = if_rpb
        #print("if_rpb ",if_rpb)
        if if_rpb:
            self.relative_position_bias = RelativePositionBias(num_heads)

    def forward(self, x):
        ''''''
        frozen = all(param.requires_grad is False for param in self.parameters())
        #print("attention frozen",frozen)
        if frozen:
            with torch.no_grad():
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                device = self.qkv.weight.device
                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

                attn = (q @ k.transpose(-2, -1)) * self.scale
                if self.if_rpb:
                    relative_position_bias = self.relative_position_bias(N, device=device)  # Shape [N, N, num_heads]
                    relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # Shape [1, num_heads, N, N]
                    attn = attn + relative_position_bias  # Broadcasting to [B, num_heads, N, N]
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            device = x.device
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.if_rpb:
                relative_position_bias = self.relative_position_bias(N, device=device)  # Shape [N, N, num_heads]
                relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # Shape [1, num_heads, N, N]
                attn = attn + relative_position_bias  # Broadcasting to [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        
        return x


class MultiWayMLP(nn.Module):
    def __init__(self, layer_index, merge_layer_depth, in_features=None, mlp_hidden_dim=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        self.layer_index = layer_index
        self.merge_layer_depth = merge_layer_depth
        if self.layer_index < self.merge_layer_depth:
            self.mlp_rgb = Mlp(in_features, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp_expr = Mlp(in_features, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.norm_rgb = norm_layer(in_features)
            self.norm_expr = norm_layer(in_features)
        else:
            self.mlp = Mlp(in_features, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.norm = norm_layer(in_features)

    def forward(self, x, mask):
        if self.layer_index < self.merge_layer_depth:
            z = torch.zeros_like(x, dtype=torch.float16)  # 将 z 的数据类型设置为 float16
            #z = torch.zeros_like(x)
            z[mask] = self.mlp_rgb(self.norm_rgb(x[mask]))
            z[~mask] = self.mlp_expr(self.norm_expr(x[~mask]))
        else:
            z = self.mlp(self.norm(x))
        return z

    def forward_rgb(self, x):
        frozen = all(param.requires_grad is False for param in self.parameters())
        #print("multiway frozen",frozen)
        
        if self.layer_index < self.merge_layer_depth:
            if frozen:
                with torch.no_grad():# if frozen else contextlib.suppress():  # Use no_grad if all params are frozen
                    z = self.mlp_rgb(self.norm_rgb(x))
            else:
                z = self.mlp_rgb(self.norm_rgb(x))
        else:
            if frozen:
                with torch.no_grad():# if frozen else contextlib.suppress():  # Use no_grad if all params are frozen
                    z = self.mlp(self.norm(x))
            else:
                z = self.mlp(self.norm(x))
        
        
        return z


    def forward_expr(self, x):
        if self.layer_index < self.merge_layer_depth:
            z = self.mlp_expr(self.norm_expr(x))
        else:
            z = self.mlp(self.norm(x))
        return z

    def forward_all(self, x):
        if self.layer_index < self.merge_layer_depth:
            N = x.shape[1]
            x_rgb = x[:, :N//2]
            x_expr = x[:, N//2:]
            z_rgb = self.mlp_rgb(self.norm_rgb(x_rgb))
            z_expr = self.mlp_expr(self.norm_expr(x_expr))
            z = torch.cat([z_rgb, z_expr], dim=1)
        else:
            z = self.mlp(self.norm(x))
        return z


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,if_rpb = False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,if_rpb = if_rpb)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,if_rpb = False):
        super().__init__()

        dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_list[i],if_rpb = if_rpb
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,if_rpb = False):
        super().__init__()
        dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_list[i],if_rpb = if_rpb
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x)
        x = self.norm(x)
        return x
