import numpy as np
from Sigmoid import *


def BackpropXOR_ad(W1, W2, W3, X, D):
    alpha = 0.9
    
    N = 5
    for k in range(N):
        x = X[k, :].T
        d = D[k]
        
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        v  = np.matmul(W3, y2)
        y  = Sigmoid(v)
        
        e     = d - y
        delta = y*(1-y) * e
        
        e2     = np.matmul(W3.T, delta)
        delta2 = y2*(1-y2) * e2

        e1     = np.matmul(W2.T, delta2)
        delta1 = y1*(1-y1) * e1
        
        dW1 = (alpha*delta1).reshape(5, 1) * x.reshape(1, 4)
        W1  = W1 + dW1

        dW2 = (alpha*delta2).reshape(5, 1) * y1.reshape(1, 5)
        W2  = W2 + dW2

        dW3 = alpha * delta * y2
        W3  = W3 + dW3
    
    return W1, W2, W3
