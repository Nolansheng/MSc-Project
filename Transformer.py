import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Embeddings(nn.Module):
    def __init__(self, img_size=(23, 30), in_channels=1024, patch_size=(1, 1), embed_dim=512, norm_layer=nn.LayerNorm, drop_ratio=0.1):
        super(Embeddings, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0]*self.grid_size[1]
        self.project = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.pos_embed = PositionalEncoding()
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size {H}*{W} doesn't match model ({self.img_size[0]}*{self.img_size[1]}."
        # project: [B,C,H,W] --> [B, Embed_dim, H, W]
        # flatten:[B,C,H,W] --> [B,C,HW]
        # transpose: [B, C, HW] --> [B, HW, C]
        x = self.project(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        # x shape is batch_size * num_patches * embed_dim

        # add position
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        # return [B, N, Embed_dim]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_patch=690, embed_dim=512):
        super(PositionalEncoding, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.position_table = nn.Parameter(self.get_position_table(num_patch=num_patch, embed_dim=embed_dim))

    def get_position_table(self, num_patch, embed_dim):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (i // 2) / embed_dim) for i in range(embed_dim)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_patch)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0) # (1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.position_table


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.qkv = nn.Linear(dim, dim*3)
        self.scale = head_dim ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # B = batch_size , N = num of patches, C = dim
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        # [3, batch_size, num_heads, num_patches, embed_dim_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        atten = (q @ k.transpose(-2, -1)) * self.scale
        # --> batch_size x num_heads x num_patches x num_patches
        atten = atten.softmax(dim=-1)
        # --> batch_size x num_heads x num_patches x num_patches

        x = (atten @ v).transpose(1, 2).reshape(B, N, C)
        # --> batch_size x num_heads x num_patches x embed_dim_per_head
        # --> batch_size x num_patches x num_heads x embed_dim_per_head
        # --> batch_size x num_patches x embed_dim
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop_ratio=0.1):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        # --> batch_size x num_patches x embed_dim
        x = self.fc1(x)
        # --> batch_size x num_patches x hidden_features
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # --> batch_size x num_patches x embed_dim
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, act_layer=nn.GELU(), drop_ratio=0.1, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.atten = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_ratio=drop_ratio)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = x + self.drop(self.atten(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, img_size=(23, 30), patch_size=(1, 1), in_c=1024, num_classes=32,
                 embed_dim=512, drop_ratio=0.1, depth=6, num_heads=8, mlp_ratio=4, norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU(), embed_layer=Embeddings):
        super(Transformer, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
        in_channels=in_c, embed_dim=embed_dim)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  norm_layer=norm_layer, act_layer=act_layer, drop_ratio=drop_ratio)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # --> (B,C,W,H)
        x = self.patch_embed(x)
        # -->(B, WH, embed_dim) = (1, 690, 729)

        x = self.blocks(x)
        x = self.norm(x)
        B, N, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, self.img_size[0], self.img_size[1])
        # --> batch_size x 512 x 23 x 30
        return x






