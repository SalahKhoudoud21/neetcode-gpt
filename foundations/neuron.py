import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        def relu(z):
            return max(z, 0.0)
        
        def sigmoid(z):
            return 1/(1+np.exp(-z))
        
        z = x @ w + b

        y = 0
        if activation == 'sigmoid':
             y = sigmoid(z)
        else:
            y = relu(z)
        
        return np.round(y, 5)

        
