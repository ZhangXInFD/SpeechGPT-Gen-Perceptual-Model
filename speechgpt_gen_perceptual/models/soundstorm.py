from random import randrange
from collections import namedtuple
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, EinMix

from beartype import beartype
from beartype.typing import Union, Dict
from .conformer import Conformer
from .utils import *
from .backbone import *



# tensor helpers

@beartype
def get_mask_subset_prob(
    mask: Tensor,
    prob: Union[float, Tensor],
    min_mask: int = 0
):
    batch, seq, device = *mask.shape, mask.device

    if isinstance(prob, Tensor):
        prob = rearrange(prob, 'b -> b 1')

    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).argsort(dim=-1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# conformer with sum reduction across quantized tokens at the beginning, along with heads

class BackboneWrapper(nn.Module):

    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_quantizers,
        backbone_type,
        backbone_kwargs,
        grouped_quantizers = 1
    ):
        super().__init__()
        self.net = NET_NAME_DICT[backbone_type](**backbone_kwargs)

        dim = self.net.dim

        self.embedding_proj = nn.Sequential(
            nn.Linear(dim * grouped_quantizers, dim),
            nn.LayerNorm(dim)
        ) if grouped_quantizers > 1 else nn.Identity()

        num_codes_with_mask = codebook_size + 1
        num_effective_quantizers = num_quantizers * grouped_quantizers

        self.code_embeds = nn.Embedding(num_codes_with_mask * num_effective_quantizers, dim)

        self.register_buffer('quantizer_offsets', torch.arange(num_effective_quantizers) * num_codes_with_mask, persistent = False)
        self.register_buffer('mask_tokens', self.quantizer_offsets + num_codes_with_mask, persistent = False)

        self.dim = dim
        self.codebook_size = codebook_size

        self.num_codes_with_mask = num_codes_with_mask
        self.num_quantizers = num_quantizers
        self.grouped_quantizers = grouped_quantizers

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * num_effective_quantizers),
            Rearrange('b n (h d) -> b (n h) d', h = num_effective_quantizers)
        )

        # each quantizer codebook would require its own logits weight and bias matrices
        # the amazing einops makes this easy with 'EinMix'

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (n gq) d -> b n gq d', gq = num_effective_quantizers),
            EinMix(
                'b n gq d -> b n gq l',
                weight_shape = 'gq d l',
                bias_shape = 'gq l',
                gq = num_effective_quantizers,
                l = codebook_size,
                d = dim
            ),
            Rearrange('b ... d -> b (...) d')
        )

    def forward(
        self,
        x,
        *,
        mask = None,
        cond = None,
        sum_embeds = None,
        return_embeddings = False,
        return_logits_and_embeddings = False
    ):
        """
        einops notation:
        b - batch
        n - sequence
        g - groups
        q - quantizers
        d - feature dimension
        """

        n, q, g = x.shape[-1], self.num_quantizers, self.grouped_quantizers
        assert divisible_by(n, g * q), 'sequence must be divisible by number of quantizers'

        x = rearrange(x, 'b (n gq) -> b n gq', gq = g * q)
        x = x + self.quantizer_offsets

        x = self.code_embeds(x)

        x = reduce(x, 'b n (g q) d -> b n (g d)', 'sum', g = g)

        x = self.embedding_proj(x)

        if exists(sum_embeds):
            x = x + sum_embeds

        if exists(cond):
            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

            x = x + cond

        x = self.net(x, mask = mask)
        embeds = self.heads(x)

        if return_embeddings or not exists(self.to_logits):
            return embeds

        logits = self.to_logits(embeds)

        if return_logits_and_embeddings:
            return logits, embeds

        return logits

# for main logits as well as self token critic

class LogitHead(nn.Module):
    def __init__(
        self,
        net: BackboneWrapper,
        logit_dim
    ):
        super().__init__()
        self.net = net
        dim = net.dim
        self.to_logits = nn.Linear(dim, logit_dim)

    def forward(self, x):
        embed = self.net(x, return_embeddings = True)
        return self.to_logits(embed)


class SoundStorm(nn.Module):
    
    def __init__(self,
                 cfg
                 ):
        super().__init__()
        self.net = BackboneWrapper(**cfg.get('BackboneWrapper'))
        self.dim = self.net.dim
        self.num_tokens = self.net.codebook_size
        self.pad_id = cfg.get('pad_id')
        self.num_semantic_token_ids = cfg.get('num_semantic_token_ids')
        self.semantic_token_emb = nn.Embedding(self.num_semantic_token_ids, self.dim)
        self.semantic_pad_id = cfg.get('semantic_pad_id', -1)
        schedule = cfg.get('schedule', 'linear')
        if callable(schedule):
            self.schedule_fn = schedule
        elif schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')
        self.num_quantizers = self.net.num_quantizers
        self.mask_id = self.net.codebook_size
    
    def get_condition(self, token_ids, length=None):
        mask = token_ids != self.semantic_pad_id
        token_ids = token_ids.masked_fill(~mask, 0)
        semantic_tokens = self.semantic_token_emb(token_ids)
        cond_tokens = semantic_tokens.masked_fill(~rearrange(mask, '... -> ... 1'), 0.)
        if exists(length):
            cond_length = cond_tokens.size(-2)
            if cond_length < length:
                cond_length = F.pad(cond_tokens, (0, 0, 0, length - cond_length), value=0.)
            elif cond_length > length:
                cond_tokens = cond_tokens[:, :length]
        return cond_tokens
    
    @property
    def device(self):
        return next(self.net.parameters()).device
    
    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        params = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(params, strict = strict)
        
    @classmethod    
    def from_pretrained(cls, path):
        import yaml
        with open(f'{path}/model_config.yml') as f:
            cfg = yaml.safe_load(f)
        model = cls(cfg)
        model.load(f'{path}/{model.__class__.__name__}_best_dev.pt')
        return model
        
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                  semantic_tokens,
                  prompt_tokens = None,
                  steps = 8,
                  num_full_sampling_levels = 1,
                  topk_pres = 0.7,
                  greedy=True,
                  once_all_tokens=False,
                  generator_sample_temperature = 1.0
                  ):
        device = self.device
        batch_size, seq_length = semantic_tokens.shape
        mask = torch.ones((batch_size, semantic_tokens.size(-1), self.num_quantizers), device=device)
        masked = mask * self.mask_id
        cond_tokens = semantic_tokens
        
        if exists(prompt_tokens):
            prompt_semantic_tokens = prompt_tokens[:, :, 0]
            prompt_acoustic_tokens = prompt_tokens[:, :, 1:]
            cond_tokens = torch.cat([prompt_semantic_tokens, cond_tokens], axis=-1)
            masked = torch.cat([prompt_acoustic_tokens, masked], axis=1)
            mask = torch.cat([torch.zeros_like(prompt_acoustic_tokens, dtype=torch.bool, device=device), mask.bool()], axis=1)
        else:
            mask = mask.bool()
            
            
        prompt_mask = mask.clone()
        seq_mask = torch.ones_like(cond_tokens, dtype=torch.bool, device=device)
        cond_tokens = self.semantic_token_emb(cond_tokens)
        seq_mask_with_quantizer = repeat(seq_mask, 'b n -> b (n q)', q = self.num_quantizers)
        times = torch.linspace(0., 1., steps + 1, device=device)
        rand_mask_probs = cosine_schedule(times)
        rand_mask_probs = rearrange(rand_mask_probs, 'n -> n 1')
        seq_len_from_mask = reduce(seq_mask, 'b n -> b', 'sum')
        all_mask_num_tokens = (rand_mask_probs * seq_len_from_mask).long()
        
        if once_all_tokens:
            masked = rearrange(masked, 'b n q -> b (n q)')
            logits = self.net(masked.long(), mask=seq_mask, cond=cond_tokens)
            if greedy:
                sampled_ids = logits.argmax(axis=-1)
            else:
                logits = top_k(logits, thres=topk_pres)
                sampled_ids = gumbel_sample(logits, temperature=generator_sample_temperature)
            masked = torch.where(rearrange(mask, 'b n q -> b (n q)'), sampled_ids, masked)
            masked = rearrange(masked, 'b (n q) -> b n q', q = self.num_quantizers)
            output = torch.cat([semantic_tokens.unsqueeze(-1), masked[:, -seq_length:]], axis=-1)
            return output.detach().long()
                
        
        for q in range(self.num_quantizers):
            mask_num_tokens = all_mask_num_tokens if q < num_full_sampling_levels else torch.zeros((1, batch_size), dtype = torch.long, device = device)
            for i, mask_num_tokens in enumerate(all_mask_num_tokens):
                masked_input = rearrange(masked, 'b n q -> b (n q)')
                logits = self.net(masked_input.long(), mask=seq_mask, cond=cond_tokens)
                if greedy and (q >= num_full_sampling_levels or (mask_num_tokens == 0).all()):
                    sampled_ids = logits.argmax(axis=-1)
                else:
                    logits = top_k(logits, thres=topk_pres)
                    temperature = generator_sample_temperature * i / steps
                    sampled_ids = gumbel_sample(logits, temperature=temperature)
                if q >= num_full_sampling_levels:
                    masked[:, :, q:] = rearrange(sampled_ids, 'b (n q) -> b n q', q=self.num_quantizers)[:, :, q:]
                    mask[:, :, q] = False
                    continue
                
                masked = torch.where(mask, rearrange(sampled_ids, 'b (n q) -> b n q', q=self.num_quantizers), masked)
                if (mask_num_tokens == 0).all():    
                    continue
                
                scores = 1 - logits.softmax(dim=-1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')
                mask_value = -torch.finfo(scores.dtype).max
                scores = scores.masked_fill(~seq_mask_with_quantizer, mask_value)
                scores_sorted = scores.argsort(dim=-1, descending=True).argsort(dim=-1)
                mask_num_tokens = rearrange(mask_num_tokens, 'b -> b 1')
                mask = scores_sorted < mask_num_tokens
                mask = rearrange(mask, 'b (n q) -> b n q', q=self.num_quantizers)
                mask[:, :, (q + 1):] = True
                mask = mask & prompt_mask
                
                masked = masked.masked_fill(mask, self.mask_id)
        output = torch.cat([semantic_tokens.unsqueeze(-1), masked[:, -seq_length:]], axis=-1)
        return output.detach().long()
        
        
    def forward(self,
                x,
                cond_ids,
                mask = None,
                generator_sample_temperature = 1,
                **kwargs):
        b, n, q = x.shape
        device = x.device
        seq_mask = mask
        
        if not exists(seq_mask):
            seq_mask = torch.ones((b, n), device=device, dtype=torch.bool)
        
        if exists(self.pad_id):
            pad_mask = (x == self.pad_id).any(dim = -1)
            seq_mask = seq_mask & ~pad_mask
            
        cond_tokens = self.get_condition(cond_ids)
        orig_seq = rearrange(x.clone(), 'b n q -> b (n q)')
        
        min_seq_len = seq_mask.sum(dim =-1).amin()
        
        # sample prompt delimiter
        t = randrange(0, min_seq_len - 1)
        
        mask = seq_mask[:, t:] 
        
        # sample time position mask
        
        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        rand_probs = self.schedule_fn(rand_times)
        
        mask = get_mask_subset_prob(mask, rand_probs)  
        
        # random quantizer position
        
        q = randrange(0, self.num_quantizers)
        
        masked = torch.where(mask, self.mask_id, x[:, t:, q])
        masked = rearrange(torch.cat((x[:, :t, q], masked), dim=1), 'b n -> b n 1')
        masked = torch.cat((x[:, :, :q], masked, x[:, :, q + 1:]), dim=2)
        masked[:, t:, q + 1:] = self.mask_id
        masked = rearrange(masked, 'b n q -> b (n q)')
        
        prompt_mask = torch.full((b, t), False, device=device)
        lower_quantizers_mask = torch.full((b, n, q), False, device=device)
        upper_quantizers_mask = torch.full((b, n, self.num_quantizers - q - 1), True, device=device)
        
        # upper_quantizers_mask in prompt also should be False

        upper_quantizers_mask[:, :t, :] = False
        mask = rearrange(torch.cat((prompt_mask, mask), dim=1), 'b n -> b n 1')
        mask = torch.cat((lower_quantizers_mask, mask, upper_quantizers_mask), dim = 2)
        
        # above is the right mask, but when computing loss, only consider level q
        
        mask[:, :, q + 1:] = False
        mask = rearrange(mask, 'b n q -> b (n q)')
        
        logits = self.net(masked,
                          mask=seq_mask,
                          cond = cond_tokens,
                          **kwargs)
        
        # CE loss
        loss = F.cross_entropy(logits[mask],
                               orig_seq[mask])
        # sampled_ids = logits.argmax(axis=-1)
        # # sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        # acc = (sampled_ids[mask] == orig_seq[mask]).sum() / mask.sum()
        # generated = torch.where(mask, sampled_ids, orig_seq)
        
        # return loss, acc, generated
        return loss
                 