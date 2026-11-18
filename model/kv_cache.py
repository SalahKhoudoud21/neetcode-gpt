import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None and self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
            return self.cache_k, self.cache_v

        self.cache_k = torch.cat((self.cache_k, new_k), dim = 1)
        self.cache_v = torch.cat((self.cache_v, new_v), dim = 1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Apply a causal mask offset by the number of previously cached tokens
        # 6. Return (rounded output, kv_cache)
        
        # Q, K, V for new token
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if not kv_cache:
            kv_cache = KVCache()
            
        
        cached_k, cached_v = kv_cache.update(K, V)

        new_len = Q.size(-2)
        total_len = cached_k.size(-2) # this is the new len + prev cached len
        old_len = total_len - new_len
        
        scores = (Q @ cached_k.transpose(-1, -2))/(cached_k.size(-1) ** 0.5)

        old_cached_KVs = torch.ones((new_len, old_len))
        new_cached_KVs = torch.tril(torch.ones((new_len, new_len)))

        lower_triangular = torch.cat((old_cached_KVs, new_cached_KVs), dim = 1)
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        
        
        out = torch.softmax(scores, dim = -1) @ cached_v
        return out, kv_cache





        
        
