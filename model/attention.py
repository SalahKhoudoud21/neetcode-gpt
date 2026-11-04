import torch
import torch.nn as nn
from torchtyping import TensorType
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        K = self.Key(embedded)
        Q = self.Query(embedded)
        V = self.Value(embedded)
        
        attention_scores = (Q @ K.transpose(-2,-1))/math.sqrt(Q.shape[-1])

        causal_mask = torch.tril(torch.ones((Q.shape[-2]), Q.shape[-2], dtype=torch.bool))
        
        masked_scores = attention_scores.masked_fill(causal_mask==0, float('-inf')) # replace where causal mask if False by -inf so softmax goes to 0

        weights = F.softmax(masked_scores, dim=2)
        
        return torch.round(weights @ V, decimals=4)
        
