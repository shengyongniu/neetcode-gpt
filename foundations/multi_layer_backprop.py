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
        # x: input_size
        # w1: hidden_size, input_size
        # b1: hidden_size
        # w2: output_size, hidden_size
        # b2: output_size
        x = np.array(x, dtype=float)             # (input_size,)
        W1 = np.array(W1, dtype=float)           # (hidden_size, input_size)
        b1 = np.array(b1, dtype=float)           # (hidden_size,)
        W2 = np.array(W2, dtype=float)           # (output_size, hidden_size)
        b2 = np.array(b2, dtype=float)           # (output_size,)
        y_true = np.array(y_true, dtype=float) 

        z1 = W1 @ x + b1 # hidden_size
        a1 = np.maximum(z1, 0) 
        z2 = W2 @ a1 + b2 # output_size
        loss = round(np.mean((z2 - y_true) ** 2), 4)
        
        n = len(y_true)
        dz2 = (2/n)*(z2-y_true) 
        dw2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1) # (output_size, hidden_size)
        db2 = dz2
        da1 = W2.T @ dz2 # (hidden_dim,) = (hidden, output) * (output_size,)
        dz1 = da1 * (z1 > 0) 
        dw1 = dz1.reshape(-1, 1) @ x.reshape(1, -1) # (hidden, 1) * (1, input)
        db1 = dz1.squeeze()

        return {"loss": loss, "dW1": np.round(dw1, 4), "db1": np.round(db1, 4), 
        "dW2": np.round(dw2, 4), "db2": np.round(db2, 4)}

        pass
    
        # L = (1/n)*(z2 - y_true)**2
        # a2 = relu(z1)
        # pL/pz2 = (2/n)*(z2-y_true) = dz2
        # z2 = w2*a1+b2 
        # pL/pw2 = dw2 = pL/pz2 * pz2/pw2 = dz2 * a1
        # pL/pb2 = db2 = dz2 * 1 = dz2
        # da1 = pL/pa1 = pL/pz2 * pz2/pa1 = dz2 * w2
        # dz1 = pL/pz1 = pL/a1 * pa1/pz1 = da1 * 1 when z1 > 0
        # pL/pw1 = dz1 * x
        # pl/pb1 = dz1
