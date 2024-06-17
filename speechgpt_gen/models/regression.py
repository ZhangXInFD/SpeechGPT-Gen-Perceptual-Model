from random import randrange

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

from beartype import beartype
from beartype.typing import Union
from .backbone import NET_NAME_DICT
from .utils import *
from pathlib import Path

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

    randperm = logits.argsort(dim = -1).float()

    # num_padding = (~mask).sum(dim = -1, keepdim = True)
    num_padding = (~mask).sum(dim = -1, keepdim = True) - 1
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

class Regression(nn.Module):
    
    def __init__(self,
                 cfg):
        super().__init__()
        schedule = cfg.get('schedule')
        backbone_kwargs = cfg.get("backbone_kwargs")
        backbone_type = cfg.get("backbone_type", 'uconformer')
        try:
            net_type = NET_NAME_DICT[backbone_type]
        except:
            raise NotImplementedError(f'No implement of {backbone_type}')
        
        self.net = net_type(**backbone_kwargs)
        if callable(schedule):
            self.schedule_fn = schedule
        elif schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')
        
    @property
    def device(self):
        return next(self.net.parameters()).device
    
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 x,
                 prompt_cond = None,
                 mask = None):
        b, n, d = x.shape
        if not exists(mask):
            mask = torch.ones((b, n), device=self.device, dtype=torch.bool)
        else:
            mask = mask

        if exists(prompt_cond):
            x = torch.cat([prompt_cond, x], axis=1)
            prompt_mask = torch.ones(prompt_cond.shape[:2], device=self.device, dtype=torch.bool)
            mask = torch.cat([prompt_mask, mask], axis=-1)
        
        output = self.net(x,
                          mask=mask)
        return output[:, -n:, :] + x[:, -n:, :]
    
    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        params = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(params, strict = strict)
    
    def forward(self,
                x,
                cond,
                mask = None):
        
        b, n, d = x.shape
        
        seq_mask = torch.ones((b, n), device=self.device, dtype=torch.bool)
            
        min_seq_len = mask.sum(dim=-1).amin()
            
        # sample prompt delimiter
        t = randrange(0, min_seq_len - 1)
        
        seq_mask[:, t:] = False
        masked = x * seq_mask.unsqueeze(-1) + cond
        output = self.net(masked,
                          mask = mask)
        target = x
        
        # regression loss
        seq_mask = (~seq_mask) & mask
        loss = (0.5 * F.mse_loss(output[seq_mask], target[seq_mask]) + F.l1_loss(output[seq_mask], target[seq_mask]))
        return loss
        