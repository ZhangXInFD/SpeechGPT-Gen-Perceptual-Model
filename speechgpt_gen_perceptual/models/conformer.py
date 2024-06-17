import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from .attend import Attend
from .utils import exists, default, divisible_by, AdaptiveRMSNorm, T5RelativePositionBias, RotaryEmbedding, apply_rotary_pos_emb, PreNorm, Swish, GLU


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


# conformer

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x, mask=None):
        if exists(mask):
            mask = rearrange(mask, "b n -> b 1 n")
            x = x.masked_fill(~mask, 0.)
        
        x = F.pad(x, self.padding)
        out = self.conv(x)
        if exists(mask):
            out = out.masked_fill(~mask, 0.)
        return out

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout) # need to remove

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None,
        attn_bias = None
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask = mask, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net1 = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1))
        self.ds_conv = DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding)
        self.net2 = nn.Sequential(Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = self.net1(x)
        x = self.ds_conv(x, mask=mask)
        return self.net2(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        attn_flash = True,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        
        if adaptive_rmsnorm:
            norm = AdaptiveRMSNorm
            norm_kwargs = dict(dim=dim, cond_dim=adaptive_rmsnorm_cond_dim_in)
        else:
            norm = nn.LayerNorm
            norm_kwargs = dict(normalized_shape=dim)
        self.attn = PreNorm(self.attn, norm=norm, **norm_kwargs)
        self.ff1 = PreNorm(self.ff1, norm=norm, **norm_kwargs)
        self.ff2 = PreNorm(self.ff2, norm=norm, **norm_kwargs)
        self.ff1 = Scale(0.5, self.ff1)
        self.ff2 = Scale(0.5, self.ff2)

        self.post_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None,
        attn_bias = None,
        adaptive_rmsnorm_cond = None
    ):
        cond = adaptive_rmsnorm_cond
        x = self.ff1(x, cond=cond) + x
        x = self.attn(x, cond=cond, mask = mask, rotary_emb = rotary_emb, attn_bias = attn_bias) + x
        x = self.conv(x, mask=mask) + x
        x = self.ff2(x, cond=cond) + x
        x = self.post_norm(x)
        return x

# Conformer

class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        attn_flash = True,
        t5_rel_pos_bias = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None
    ):
        super().__init__()

        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'

        self.dim = dim
        self.layers = nn.ModuleList([])
        assert divisible_by(dim, heads), 'dim must be divisible by heads'
        dim_head = dim // heads

        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads = heads) if t5_rel_pos_bias else None

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
                conv_causal = conv_causal,
                attn_flash = attn_flash,
                adaptive_rmsnorm=adaptive_rmsnorm,
                adaptive_rmsnorm_cond_dim_in=adaptive_rmsnorm_cond_dim_in
            ))

    def forward(self, x, mask = None, adaptive_rmsnorm_cond=None):
        seq_len = x.shape[-2]

        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None

        for block in self.layers:
            x = block(
                x,
                mask = mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias,
                adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
            )

        return x

class UConformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        attn_flash = True,
        t5_rel_pos_bias = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None
    ):
        super().__init__()

        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'

        self.dim = dim
        self.layers = nn.ModuleList([])
        assert divisible_by(dim, heads), 'dim must be divisible by heads'
        dim_head = dim // heads
        self.depth = depth

        contracting_depth = depth // 2 - (1 - depth % 2)
        self.contracting_layers = nn.ModuleList([])
        self.expanding_layers = nn.ModuleList([])
        self.bottleneck_layers = nn.ModuleList([])
        nets = [self.contracting_layers, self.expanding_layers, self.bottleneck_layers]

        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads = heads) if t5_rel_pos_bias else None

        for i in range(depth):
            if i < contracting_depth:
                net_idx = 0
            elif i >= depth - contracting_depth:
                net_idx = 1
            else:
                net_idx = 2
            nets[net_idx].append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
                conv_causal = conv_causal,
                attn_flash = attn_flash,
                adaptive_rmsnorm=adaptive_rmsnorm,
                adaptive_rmsnorm_cond_dim_in=adaptive_rmsnorm_cond_dim_in
            ))
        self.projs = nn.ModuleList([])
        for _ in range(contracting_depth):
            self.projs.append(nn.Linear(dim * 2, dim))

    def forward(self, x, mask = None, adaptive_rmsnorm_cond=None):
        seq_len = x.shape[-2]

        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None
        
        cache = []
        
        for block in self.contracting_layers:
            x = block(
                    x,
                    mask = mask,
                    rotary_emb = rotary_emb,
                    attn_bias = attn_bias,
                    adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
                )
            cache.append(x)
        
        for block in self.bottleneck_layers:
            x = block(
                    x,
                    mask = mask,
                    rotary_emb = rotary_emb,
                    attn_bias = attn_bias,
                    adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
                )
            
        for block, proj in zip(self.expanding_layers, self.projs):
            x = proj(torch.cat([x, cache.pop(-1)], dim=-1))
            x = block(
                    x,
                    mask = mask,
                    rotary_emb = rotary_emb,
                    attn_bias = attn_bias,
                    adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
                )
                

        return x
