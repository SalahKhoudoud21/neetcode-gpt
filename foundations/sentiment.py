import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.encoder = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16)

        self.mlp = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        
        embeddings = self.encoder(x)
        inp = torch.mean(embeddings, dim = 1)
        out = self.mlp(inp)
        
        return torch.round(out, decimals=4)

        # Return a B, 1 tensor and round to 4 decimal places
        
