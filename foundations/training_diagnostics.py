import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        model.eval()
        stats = []
        with torch.no_grad():
            for _ , layer in model.named_children(): # only looks at top level layers not ones inside other layers like nn.linear(nn.Conv2d()) will only see nn.linear if want to visit all layers can use named_modules()
                x = layer(x)
                
                if isinstance(layer, nn.Linear):
                    
                    stats.append(
                        {
                            'mean': round(x.mean().item(), 4),
                            'std': round(x.std().item(), 4),
                            'dead_fraction': round((x <= 0).all(dim=0).float().mean().item(),4)
                        }
                    )

            return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        out = model(x)
        loss_func = nn.MSELoss()
        loss = loss_func(out, y)
        loss.backward()
        grad_stats = []
        for _ , layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    grad_stats.append(
                        {
                            # detach: Keeps the tensor values, but stop PyTorch from tracking how they were created.
                            'mean': round(layer.weight.grad.detach().mean().item(),4),
                             'std': round(layer.weight.grad.detach().std().item(),4),
                             'norm': round(torch.norm(layer.weight.grad.detach()).item(), 4) 
                            

                        }
                    )
        return grad_stats





    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        concensus = 'healthy'
        
        for layer in activation_stats:
            if layer['dead_fraction'] > 0.5:
                return 'dead_neurons'
            
            if layer['std'] < 0.1:
                concensus = 'vanishing_gradients'
            
            elif layer['std'] > 10.0:
                concensus = 'exploding_gradients'
        
        for layer in gradient_stats:
            if layer['norm'] > 1000:
                return 'exploding_gradients'
        
        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        
        return concensus
        
