import numpy as np
from Sigmoid import *


def BackpropXOR_add(W1, W2, W3, W4, X, D):
    alpha = 0.2
    
    N = 5
    for k in range(N):
        x = X[k, :].T
        d = D[k]
        
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        v3 = np.matmul(W3, y2)
        y3 = Sigmoid(v3)
        v  = np.matmul(W4, y3)
        y  = Sigmoid(v)
        
        e     = d - y
        delta = y*(1-y) * e 
        # delta = e

        e3     = np.matmul(W4.T, delta)
        delta3 = y2*(1-y2) * e3

        e2     = np.matmul(W3.T, delta3)
        delta2 = y2*(1-y2) * e2

        e1     = np.matmul(W2.T, delta2)
        delta1 = y1*(1-y1) * e1
        
        dW1 = (alpha*delta1).reshape(10, 1) * x.reshape(1, 4)
        W1  = W1 + dW1

        dW2 = (alpha*delta2).reshape(10, 1) * y1.reshape(1, 10)
        W2  = W2 + dW2

        dW3 = (alpha*delta3).reshape(10, 1) * y2.reshape(1, 10)
        W3  = W3 + dW3

        dW4 = alpha * delta * y3
        W4  = W4 + dW4
    
    return W1, W2, W3, W4
