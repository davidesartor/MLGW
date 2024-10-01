import math
from torch import nn
import torch
from timm.models.vision_transformer import Attention


class MLP(nn.Sequential):
    def __init__(
        self,
        in_dim,
        hidden_dim=None,
        out_dim=None,
        hidden_layers=1,
        act_layer=nn.SiLU,
    ):
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        layers = [nn.Linear(in_dim, hidden_dim), act_layer()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_layer()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        super().__init__(*layers)


class Patchify1d(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = MLP(input_dim * patch_size, output_dim, output_dim)

    def forward(self, x):
        *B, L, C = x.shape
        assert L % self.patch_size == 0, "sequence length not divisible by patch size"
        x = x.view((*B, L // self.patch_size, C * self.patch_size))
        return self.patch_embed(x)


class UnPatchify1d(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.patch_unembed = MLP(input_dim, input_dim, output_dim * patch_size)

    def forward(self, x):
        x = self.patch_unembed(x)
        *B, L, C = x.shape
        return x.reshape((*B, L * self.patch_size, C // self.patch_size))


class PositionEmbedder(nn.Module):
    def __init__(self, embed_dim, max_period=10000):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.max_period = max_period

    def pos_embedding(self, k):
        freqs = torch.arange(0, self.embed_dim // 2, dtype=k.dtype, device=k.device)
        freqs = torch.exp(-freqs * math.log(self.max_period) / (self.embed_dim // 2))
        x = freqs[None, :] * k
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def forward(self, x):
        k = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
        return self.pos_embedding(k.unsqueeze(-1)).expand((*x.shape[:-1], -1))


class TimestepEmbedder(nn.Module):
    def __init__(self, embed_dim, max_period=100, pos_embed_dim=256):
        super().__init__()
        self.max_period = max_period
        self.pos_embedder = PositionEmbedder(pos_embed_dim, max_period)
        self.mlp = MLP(pos_embed_dim, embed_dim, embed_dim)

    def forward(self, t):
        return self.mlp(self.pos_embedder(t * self.max_period))


class ZeroInitMLP(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.zeros_(self[-1].weight)
        if self[-1].bias is not None:
            nn.init.constant_(self[-1].bias, 0)


class ConditionalLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, elementwise_affine=False, **kwargs)

    def forward(self, x, scale, shift):
        return super().forward(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.modulation = ZeroInitMLP(hidden_dim, hidden_dim, hidden_dim * 6)
        self.norm1 = ConditionalLayerNorm(hidden_dim)
        self.attention = Attention(hidden_dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = ConditionalLayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim)

    def forward(self, x, c):
        modulation = self.modulation(c).unsqueeze(1)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        x = x + gate1 * self.attention(self.norm1(x, scale1, shift1))
        x = x + gate2 * self.mlp(self.norm2(x, scale2, shift2))
        return x
