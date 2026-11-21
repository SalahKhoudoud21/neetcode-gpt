import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        # Since we implement multiple heads at the same time
        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        # The strength of GQA is below (fewer kv heads which are shared across groups instead of looking at all heads)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        Q = self.q_proj(x) # (B, T, num_heads * self.head_dim)
        K = self.k_proj(x) # (B, T, num_kv_heads * self.head_dim)
        V = self.v_proj(x) # (B, T, num_kv_heads * self.head_dim)
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        Q_new = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K_new = K.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        V_new = V.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        repeats = num_heads // num_kv_heads
        # Repeats each K/V head so their number of heads matches the query heads (num_heads).
        keys = torch.repeat_interleave(K_new, repeats=repeats, dim=1)
        values = torch.repeat_interleave(V_new, repeats=repeats, dim=1) 
        # 4. Compute scaled dot-product attention with causal mask
        scores = Q_new @ keys.transpose(-2, -1)/(keys.size(-1)**0.5)
        causal_mask = torch.tril(torch.ones((T, T))) == 0
        scores = scores.masked_fill(causal_mask, float('-inf'))
        out = torch.softmax(scores, dim=-1) @ values
        # 5. Concatenate heads and apply output projection
        out = out.transpose(1,2)
        out = out.contiguous().reshape(B, T, num_heads * self.head_dim)
        out = self.output_proj(out)
        # 6. Return rounded output (decimals=4)
        return torch.round(out, decimals=4)
