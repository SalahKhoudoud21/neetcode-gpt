import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # forward propagation (make the prediction)
        def sigmoid(z):
            return 1/(1+np.exp(-z))
        
        def grad_l(y, y_hat): # dL/dy_hat
            return y_hat - y
        
        def grad_y_hat(y_hat): # dy_hat/dz
            return y_hat * (1 - y_hat) # deriv of sigmoid
        
        z = np.dot(x, w) + b 
        y_hat = sigmoid(z)

        grad_loss_wrp_y_hat = grad_l(y_true, y_hat)
        grad_y_hat_wrp_z = grad_y_hat(y_hat)
        grad_z_wrp_w = x # dz/dw
        grad_z_wrp_b = 1.0 # dz/db

        new_w = np.round(grad_loss_wrp_y_hat * grad_y_hat_wrp_z * grad_z_wrp_w, 5)
        new_b = np.round(grad_loss_wrp_y_hat * grad_y_hat_wrp_z * grad_z_wrp_b, 5)

        return (new_w, new_b)



