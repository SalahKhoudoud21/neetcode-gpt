import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        i = 0
        n = len(weights) - 1
        h = x
        for layer, bias in zip(weights, biases):
            if i == n:
                h = h @ layer + bias
                return np.round(h, 5)
            h = np.maximum(h @ layer + bias, 0)
            i += 1
        

        
        

        
