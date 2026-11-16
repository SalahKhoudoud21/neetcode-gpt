import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        final_loss = 0.0
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        
        # Here, each epoch samples one random batch (one training step).
        # Normally, an epoch processes the entire dataset across multiple batches.
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            start_indices = torch.randint(high=len(data) - context_length, size=(batch_size, 1))
            shifts = torch.arange(context_length)
            indices = start_indices + shifts
            X = data[indices]
            Y = data[indices+1]
            optimizer.zero_grad() # clear gradients from previous training step
            logits = model(X) # B, T, C (This GPT model returns raw logits)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = Y.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            final_loss = loss.item()
            loss.backward() # backprop recalculate new gradients
            optimizer.step() # update model parameters using those gradients
        
        return round(final_loss, 4)
