import math
from random import randrange
import logging

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import torchode as to
from functools import partial

from torchdiffeq import odeint

from einops import rearrange, repeat, reduce, pack, unpack

from .conformer import UConformer, Conformer
from .transformer import Transformer, Uformer
from .utils import *
from pathlib import Path


LOGGER = logging.getLogger(__file__)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# sinusoidal positions

class LearnedSinusoidalPosEmb(Module):
    """ used by @crowsonkb """

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

class ConditionalFlowMatcher(Module):
    
    def __init__(self,
                 cfg,
                torchode_method_klass = to.Tsit5,
                 ):
        super().__init__()
        print(f'Initializing model from config:{cfg}')
        self.sigma = cfg.get("sigma", 0.)
        self.use_torchode = cfg.get("use_torchode", False)
        self.torchode_method_klass = torchode_method_klass
        self.odeint_kwargs = dict(
            atol = cfg.get("ode_atol", 1e-7),
            rtol = cfg.get("ode_rtol", 1e-9),
            method = cfg.get("torchdiffeq_ode_method", "midpoint"),
            options = dict(step_size = cfg.get("ode_step_size", 0.0625))
        )
        
        conformer_kwargs = cfg.get("conformer_kwargs")
        
        dim = conformer_kwargs['dim']
        dim_in = cfg.get("dim_in", dim)
        time_hidden_dim = conformer_kwargs['adaptive_rmsnorm_cond_dim_in']
        
        model_type = cfg.get("model_type", 'uconformer')
        if model_type == 'conformer':
            self.net = Conformer(**conformer_kwargs)
        elif model_type == 'uconformer':
            self.net = UConformer(**conformer_kwargs)
        else:
            raise NotImplementedError('Must be \'conformer\' or \'uconformer\'')
        self.adaptive_norm = conformer_kwargs["adaptive_rmsnorm"]
        self.start_from_cond = cfg.get("start_from_cond")
        self.concat_cond = cfg.get("concat_cond")
        self.explicit = cfg.get("explicit", False)
        if self.concat_cond or not self.start_from_cond:
            self.to_embed = nn.Linear(dim_in * 3 + 1, dim_in)
        else:
            self.to_embed = nn.Linear(dim_in * 2 + 1, dim_in)
        if self.adaptive_norm:
            self.sinu_pos_emb = nn.Sequential(
                LearnedSinusoidalPosEmb(dim),
                nn.Linear(dim, time_hidden_dim),
                nn.SiLU()
            )
        self.to_pred = nn.Linear(dim, dim_in, bias=False)
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        params = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(params, strict = strict)
    
    @torch.inference_mode()
    @eval_decorator
    def infer(self,
              *,
              x,
              times,
              semantic_emb = None,
              context = None,
              mask = None,
              ):
        batch, seq_len, dtype = *x.shape[:2], x.dtype
        # if not exists(mask):
        #     mask = torch.ones(batch, seq_len, dtype=torch.bool, device=self.device)
            
            
        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = batch)
            
        if self.adaptive_norm:
            time_emb = self.sinu_pos_emb(times)
            
        times = repeat(times, 'b -> b n 1', n=seq_len)
            
        if self.concat_cond or not self.start_from_cond:
            x = torch.cat([x, semantic_emb, context, times], axis=-1)
            x = self.to_embed(x)
        else:
            x = torch.cat([x, context, times], axis=-1)
            x = self.to_embed(x)
   
        
        if self.adaptive_norm:                
            out = self.net(x,
                        mask=mask,
                        adaptive_rmsnorm_cond=time_emb)
        else:
            out = self.net(x,
                        mask=mask)
            
        out = self.to_pred(out)
        
        return out
    
    @torch.inference_mode()
    @eval_decorator
    def generate(self,
                 *,
                 semantic_emb,
                 context = None,
                 context_semantic_emb = None,
                 mask = None,
                 steps = 3):
        batch, seq_len, dtype = *semantic_emb.shape[:2], semantic_emb.dtype
        
        if exists(context_semantic_emb):
            assert exists(context), 'Context and context\'s semantic embeddings must appear simultaneously'
            if exists(mask):
                context_mask = torch.ones(*context.shape[:2], dtype=torch.bool, device=self.device)
                mask = torch.cat([context_mask, mask], dim=-1)
            context = torch.cat([context, torch.zeros_like(semantic_emb, dtype=dtype, device=self.device)], dim=-2)
            semantic_emb = torch.cat([context_semantic_emb, semantic_emb], dim=-2)    
        else:
            context = torch.zeros_like(semantic_emb, dtype=dtype, device=self.device)
        
        if self.start_from_cond:
            y0 = torch.randn_like(semantic_emb) + semantic_emb
            if not self.concat_cond:
                semantic_emb = None
        else:
            y0 = torch.randn_like(semantic_emb)
        
        def fn(t, x, *, packed_shape = None):
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')
            
            out = self.infer(x = x,
                             times = t,
                             semantic_emb = semantic_emb,
                             context = context,
                             mask = mask)
            
            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')
            
            return out
        t = torch.linspace(0, 1, steps, device = self.device)
        
        if not self.use_torchode:
            # print('sampling with torchdiffeq')
            
            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
            generated = trajectory[-1]
        else:
            # print('sampling with torch ode')
            
            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')
            
            fn = partial(fn, packed_shape=packed_shape)
            
            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term=term)
            
            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )
            
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)

            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)

            sol = jit_solver.solve(init_value)

            generated = sol.ys[:, -1]
            generated = unpack_one(generated, packed_shape, 'b *')
            
        return generated[:, -seq_len:]
                        
        
    def forward(self,
                x1,
                x0,
                mask = None,
                max_cond_seq = 300
                ):
        batch, seq_len, dtype = *x1.shape[:2], x1.dtype
        
        if not exists(mask):
            mask = torch.ones(*x1.shape[:2], dtype=torch.bool, device=x1.device)
        
        seq_mask = mask.clone()
        min_seq_len = mask.sum(dim=-1).amin()
        
        # sample prompt delimiter
        if max_cond_seq != 0:
            t = randrange(0, min(min_seq_len, max_cond_seq) - 1)
            mask[:, :t] = False
            
        if self.concat_cond or not self.start_from_cond or self.explicit:
            cond = x0.clone()
        
        if self.start_from_cond:
            x0 = torch.randn_like(x0) +  x0
            
        else:
            x0 = torch.randn_like(x0)
        
        times = torch.rand((batch,), dtype=dtype, device=self.device)
        t = rearrange(times, 'b -> b 1 1')
        
        if self.explicit:
            w = (1 - (1 - self.sigma) * t) * x0 + t * (x1 - cond)
        else:
            w = (1 - (1 - self.sigma) * t) * x0 + t * x1
        
        if self.explicit:
            flow = (x1 - cond) - (1 - self.sigma) * x0
        else:
            flow = x1 - (1 - self.sigma) * x0
        
        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = batch)
        
        
        if self.adaptive_norm:
            time_emb = self.sinu_pos_emb(times)
        
        context = torch.where(mask.unsqueeze(-1), 0, x1)
        
        times = repeat(times, 'b -> b n 1', n=seq_len)
        
        if self.concat_cond or not self.start_from_cond:
            w = torch.cat([w, cond, context, times], axis=-1)
            w = self.to_embed(w)
        else:
            w = torch.cat([w, context, times], axis=-1)
            w = self.to_embed(w)
        
        if self.adaptive_norm:
            w = self.net(w, 
                        mask=seq_mask, 
                        adaptive_rmsnorm_cond=time_emb)
        else:
            w = self.net(w, 
                        mask=seq_mask)
        w = self.to_pred(w)
        
        loss = F.mse_loss(w[mask], flow[mask])
        
        return loss
        
class HierarchicalConditionalMatcher(Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        num_semantic_tokens = cfg.get('num_semantic_tokens')
        semantic_emb_dim = cfg.get('conformer_kwargs')['dim']
        self.semantic_embeddings = nn.Embedding(num_semantic_tokens, semantic_emb_dim)
        
        self.model = ConditionalFlowMatcher(cfg) # start_from_cond = False, concat_cond=True, explicit=False
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        params = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(params, strict = strict)
        
    def forward(self,
                x1,
                semantic_tokens,
                mask = None,
                max_cond_seq = 250
                ):
        
        x0 = self.semantic_embeddings(semantic_tokens)
        
        return self.model(x1=x1,
                          x0=x0,
                          mask=mask,
                          max_cond_seq=max_cond_seq)
        
    @torch.inference_mode()
    @eval_decorator
    def generate(self,
                 semantic_tokens,
                 **kwargs):
        semantic_emb = self.semantic_embeddings(semantic_tokens)
        return self.model.generate(semantic_emb=semantic_emb, **kwargs)
        
        
class TransformerGenerator(Module):
    
    def __init__(self,
                 cfg,
                torchode_method_klass = to.Tsit5,
                 ):
        super().__init__()
        print(f'Initializing model from config:{cfg}')
        self.sigma = cfg.get("sigma", 0.)
        self.use_torchode = cfg.get("use_torchode", False)
        self.torchode_method_klass = torchode_method_klass
        self.odeint_kwargs = dict(
            atol = cfg.get("ode_atol", 1e-7),
            rtol = cfg.get("ode_rtol", 1e-9),
            method = cfg.get("torchdiffeq_ode_method", "midpoint"),
            options = dict(step_size = cfg.get("ode_step_size", 0.0625))
        )
        
        transformer_kwargs = cfg.get("transformer_kwargs")
        
        dim = transformer_kwargs['dim']
        dim_in = cfg.get("dim_in", dim)
        time_hidden_dim = transformer_kwargs['adaptive_rmsnorm_cond_dim_in']
        self.model_type = cfg.get('model_type', 'uformer')
        if self.model_type == 'transformer':
            self.net = Transformer(**transformer_kwargs)
        elif self.model_type == 'uformer':
            self.net = Uformer(**transformer_kwargs)
        else:
            raise NotImplementedError
        self.start_from_cond = cfg.get("start_from_cond")
        self.concat_cond = cfg.get("concat_cond")
        self.explicit = cfg.get("explicit", False)
        if self.concat_cond or not self.start_from_cond:
            self.to_embed = nn.Linear(dim_in * 2, dim_in)
        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )
        self.to_pred = nn.Linear(dim, dim_in, bias=False)
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        params = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(params, strict = strict)
    
    @torch.inference_mode()
    @eval_decorator
    def infer(self,
              *,
              x,
              times,
              cond,
              semantic_emb = None,
              mask = None,
              ):
        batch, seq_len, dtype = *x.shape[:2], x.dtype
        if not exists(mask):
            mask = torch.ones(batch, seq_len, dtype=torch.bool, device=self.device)
            
        if self.concat_cond or not self.start_from_cond or self.explicit:
            x = torch.cat([x, semantic_emb], axis=-1)
            x = self.to_embed(x)
            
        
        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = batch)
            
        time_emb = self.sinu_pos_emb(times)
        
        out = self.net(x=x,
                       context=cond,
                       mask=mask,
                       adaptive_rmsnorm_cond=time_emb)
        out = self.to_pred(out)
        return out[:, -seq_len:]
    
    @torch.inference_mode()
    @eval_decorator
    def generate(self,
                 *,
                 semantic_emb,
                 cond = None,
                 mask = None,
                 steps = 3):
        batch, seq_len = semantic_emb.shape[:2]
        
        if self.start_from_cond:
            y0 = torch.randn_like(semantic_emb) + semantic_emb
            if not self.concat_cond:
                semantic_emb = None
        else:
            y0 = torch.randn_like(semantic_emb)
        
        def fn(t, x, *, packed_shape = None):
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')
            
            out = self.infer(x = x,
                             times = t,
                             semantic_emb = semantic_emb,
                             cond = cond,
                             mask = mask)
            
            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')
            
            return out
        t = torch.linspace(0, 1, steps, device = self.device)
        
        if not self.use_torchode:
            # print('sampling with torchdiffeq')
            
            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
            generated = trajectory[-1]
        else:
            # print('sampling with torch ode')
            
            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')
            
            fn = partial(fn, packed_shape=packed_shape)
            
            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term=term)
            
            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )
            
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)

            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)

            sol = jit_solver.solve(init_value)

            generated = sol.ys[:, -1]
            generated = unpack_one(generated, packed_shape, 'b *')
            
        return generated
                        
        
    def forward(self,
                x1,
                x0,
                mask = None,
                max_cond_seq = 500
                ):
        batch, seq_len, dtype = *x1.shape[:2], x1.dtype
        
        if not exists(mask):
            mask = torch.ones(*x1.shape[:2], dtype=torch.bool, device=x1.device)
        
        seq_mask = mask.clone()
        min_seq_len = mask.sum(dim=-1).amin()
        
        # sample prompt delimiter
        if max_cond_seq != 0:
            t = randrange(50, min(min_seq_len, max_cond_seq) - 1)
            context = x1[:, :t]
            x1 = x1[:, t:]
            x0 = x0[:, t:]
            seq_mask = seq_mask[:, t:]
            
        if self.concat_cond or not self.start_from_cond or self.explicit:
            cond = x0.clone()
        
        if self.start_from_cond:
            x0 = torch.randn_like(x0) +  x0
            
        else:
            x0 = torch.randn_like(x0)
        
        times = torch.rand((batch,), dtype=dtype, device=self.device)
        t = rearrange(times, 'b -> b 1 1')
        
        if self.explicit:
            w = (1 - (1 - self.sigma) * t) * x0 + t * (x1 - cond)
        else:
            w = (1 - (1 - self.sigma) * t) * x0 + t * x1
        
        if self.explicit:
            flow = (x1 - cond) - (1 - self.sigma) * x0
        else:
            flow = x1 - (1 - self.sigma) * x0
        
        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = batch)
        
        time_emb = self.sinu_pos_emb(times)
        
        if self.concat_cond or not self.start_from_cond or self.explicit:
            w = torch.cat([w, cond], axis=-1)
            w = self.to_embed(w)
        
        
        w = self.net(w, 
                     context=context,
                     mask=seq_mask, 
                     adaptive_rmsnorm_cond=time_emb)
        w = self.to_pred(w)
        
        loss = F.mse_loss(w, flow)
        
        return loss