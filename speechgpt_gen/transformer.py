import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from .attend import Attend
from .utils import exists, default, divisible_by, RotaryEmbedding, apply_rotary_pos_emb, PreNorm, Swish, GLU, PostNorm, AdaptiveRMSNorm, T5RelativePositionBias, RMSNorm

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
            if not exists(context):
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
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    
    def __init__(
        self, 
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        attn_flash = True,
        ff_dropout = 0.,
        cross_attention = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None):
        
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=attn_flash)
        self.ffn = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=attn_flash)
        
        if adaptive_rmsnorm:
            norm = AdaptiveRMSNorm
            norm_kwargs = dict(dim=dim, cond_dim=adaptive_rmsnorm_cond_dim_in)
        else:
            norm = RMSNorm
            norm_kwargs = dict(dim=dim)
        self.attn = PostNorm(self.attn, norm=norm, **norm_kwargs)
        self.ffn = PostNorm(self.ffn, norm=norm, **norm_kwargs)
        if cross_attention:
            self.cross_attn = PostNorm(self.cross_attn, norm=norm, **norm_kwargs)
            
    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None,
        rotary_emb = None,
        attn_bias = None,
        adaptive_rmsnorm_cond = None):
        
        cond = adaptive_rmsnorm_cond
        
        x = self.attn(x, cond=cond, mask = mask, rotary_emb = rotary_emb, attn_bias = attn_bias)
        if self.cross_attention:
            x = self.cross_attn(x, cond=cond, context=context, mask = context_mask, rotary_emb = rotary_emb, attn_bias = attn_bias)
        
        x = self.ffn(x, cond=cond)
        
        return x
    
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_flash = True,
        t5_rel_pos_bias = False,
        cross_attention = False,
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
            self.layers.append(TransformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                cross_attention=cross_attention,
                ff_mult = ff_mult,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                attn_flash = attn_flash,
                adaptive_rmsnorm=adaptive_rmsnorm,
                adaptive_rmsnorm_cond_dim_in=adaptive_rmsnorm_cond_dim_in
            ))
            
    def forward(self, 
                x, 
                context = None,
                context_mask = None,
                mask = None, 
                adaptive_rmsnorm_cond=None):
        seq_len = x.shape[-2]
        
        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None
        
        for block in self.layers:
            x = block(
                x,
                mask = mask,
                context = context,
                context_mask = context_mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias,
                adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
            )
        
        return x
    
class UformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_flash = True,
        t5_rel_pos_bias = False,
        cross_attention = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None
    ):
        super().__init__()

        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'
        self.dim = dim
        assert divisible_by(dim, heads), 'dim must be divisible by heads'
        dim_head = dim // heads

        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads = heads) if t5_rel_pos_bias else None
    
        self.depth = depth
        contracting_depth = depth // 2 - (1 - depth % 2)
        self.contracting_layers = nn.ModuleList([])
        self.expanding_layers = nn.ModuleList([])
        self.bottleneck_layers = nn.ModuleList([])
        nets = [self.contracting_layers, self.expanding_layers, self.bottleneck_layers]
        
        for i in range(depth):
            if i < contracting_depth:
                net_idx = 0
            elif i >= depth - contracting_depth:
                net_idx = 1
            else:
                net_idx = 2
            nets[net_idx].append(TransformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                cross_attention=cross_attention,
                ff_mult = ff_mult,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                attn_flash = attn_flash,
                adaptive_rmsnorm=adaptive_rmsnorm,
                adaptive_rmsnorm_cond_dim_in=adaptive_rmsnorm_cond_dim_in
            ))
        self.projs = nn.ModuleList([])
        for _ in range(contracting_depth):
            self.projs.append(nn.Sequential(nn.Linear(dim * 2, dim), RMSNorm(dim=dim)))
            
    def forward(self, 
                x, 
                context = None,
                context_mask = None,
                mask = None, 
                adaptive_rmsnorm_cond=None):
        seq_len = x.shape[-2]
        
        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None
        
        cache = []
        
        for block in self.contracting_layers:
            x = block(
                x,
                mask = mask,
                context = context,
                context_mask = context_mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias,
                adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
            )
            cache.append(x)
        
        for block in self.bottleneck_layers:    
            x = block(
                    x,
                    mask = mask,
                    context = context,
                    context_mask = context_mask,
                    rotary_emb = rotary_emb,
                    attn_bias = attn_bias,
                    adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
                    )
        
        for block, proj in zip(self.expanding_layers, self.projs):
            x = proj(torch.cat([x, cache.pop(-1)], dim=-1))
            x = block(
                x,
                mask = mask,
                context = context,
                context_mask = context_mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias,
                adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
            )
        
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        context_encoder_depth = None,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_flash = True,
        t5_rel_pos_bias = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None
    ):
        super().__init__()

        context_encoder_depth = default(context_encoder_depth, depth)
        
        self.context_encoder = TransformerEncoder(dim = dim,
                                                  depth = context_encoder_depth,
                                                  heads = heads,
                                                  ff_mult = ff_mult,
                                                  attn_dropout = attn_dropout,
                                                  ff_dropout = ff_dropout,
                                                  attn_flash = attn_flash,
                                                  t5_rel_pos_bias = t5_rel_pos_bias,
                                                  cross_attention = False,
                                                  adaptive_rmsnorm = False)
        
        self.encoder = TransformerEncoder(dim = dim,
                                            depth = depth,
                                            heads = heads,
                                            ff_mult = ff_mult,
                                            attn_dropout = attn_dropout,
                                            ff_dropout = ff_dropout,
                                            attn_flash = attn_flash,
                                            t5_rel_pos_bias = t5_rel_pos_bias,
                                            cross_attention = True,
                                            adaptive_rmsnorm = adaptive_rmsnorm,
                                            adaptive_rmsnorm_cond_dim_in = adaptive_rmsnorm_cond_dim_in)
            
    def forward(self, 
                x, 
                context,
                context_mask = None,
                mask = None, 
                adaptive_rmsnorm_cond=None):
        
        context = self.context_encoder(
                context,
                mask = context_mask,
            )
        
        x = self.encoder(
            x,
            mask = mask,
            context = context,
            context_mask = context_mask,
            adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
        )
        
        return x
    
class Uformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        context_encoder_depth = None,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_flash = True,
        t5_rel_pos_bias = False,
        adaptive_rmsnorm = False,
        adaptive_rmsnorm_cond_dim_in = None
    ):
        super().__init__()

        context_encoder_depth = default(context_encoder_depth, depth)
        
        self.context_encoder = TransformerEncoder(dim = dim,
                                                  depth = context_encoder_depth,
                                                  heads = heads,
                                                  ff_mult = ff_mult,
                                                  attn_dropout = attn_dropout,
                                                  ff_dropout = ff_dropout,
                                                  attn_flash = attn_flash,
                                                  t5_rel_pos_bias = t5_rel_pos_bias,
                                                  cross_attention = False,
                                                  adaptive_rmsnorm = False)
        
        self.encoder = UformerEncoder(dim = dim,
                                      depth = depth,
                                        heads = heads,
                                        ff_mult = ff_mult,
                                        attn_dropout = attn_dropout,
                                        ff_dropout = ff_dropout,
                                        attn_flash = attn_flash,
                                        t5_rel_pos_bias = t5_rel_pos_bias,
                                        cross_attention = True,
                                        adaptive_rmsnorm = adaptive_rmsnorm,
                                        adaptive_rmsnorm_cond_dim_in = adaptive_rmsnorm_cond_dim_in)
            
    def forward(self, 
                x, 
                context,
                context_mask = None,
                mask = None, 
                adaptive_rmsnorm_cond=None):
        
        context = self.context_encoder(
                context,
                mask = context_mask,
            )
        
        x = self.encoder(
            x,
            mask = mask,
            context = context,
            context_mask = context_mask,
            adaptive_rmsnorm_cond=adaptive_rmsnorm_cond
        )
        
        return x
        
        
        
    
        
        