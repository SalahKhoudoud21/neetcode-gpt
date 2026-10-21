import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        def mse_loss(N, y_hat, y_true):
            return 1/N * (y_hat - y_true)**2
        
        z1 = W1 @ x + b1
        h1 = np.maximum(z1, 0)
        y_hat = W2 @ h1 + b2

        def deriv_l(N, y_hat, y): # dL/dy_hat
            return  2 * 1/N * (y_hat - y)
        
        deriv_loss_wrp_y_hat = deriv_l(1, y_hat, y_true)
        deriv_y_hat_wrp_w2 = h1
        deriv_y_hat_wrp_b2 = 1
        deriv_y_hat_wrp_h = W2
        deriv_h_wrp_z1 = (z1 > 0).astype(float)
        deriv_z1_wrp_w1 = x
        deriv_z1_wrp_b1 = 1

        new_W2 = np.round(np.outer(deriv_loss_wrp_y_hat,deriv_y_hat_wrp_w2), 4)
        
        new_b2 = np.round(deriv_loss_wrp_y_hat * deriv_y_hat_wrp_b2, 4)
        
        new_W1 = np.round(np.outer(deriv_y_hat_wrp_h.T @ deriv_loss_wrp_y_hat * deriv_h_wrp_z1, deriv_z1_wrp_w1), 4)
        
        new_b1 = np.round((deriv_y_hat_wrp_h.T @ deriv_loss_wrp_y_hat) * deriv_h_wrp_z1 * deriv_z1_wrp_b1, 4)

        loss = np.round(mse_loss(len(y_hat), y_hat, y_true), 4)

        return {
            'loss': loss[0],
            'dW1':   new_W1,
            'db1':   new_b1,
            'dW2':   new_W2,
            'db2':   new_b2,
        }




