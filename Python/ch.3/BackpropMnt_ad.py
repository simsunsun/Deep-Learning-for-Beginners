import numpy as np
from Sigmoid import *


def BackPropMmt_ad(W1, W2, W3, X, D):
    alpha = 0.9
    beta  = 0.9
    
    mmt1 = np.zeros_like(W1)
    mmt2 = np.zeros_like(W2)
    mmt3 = np.zeros_like(W3)
    
    N = 4
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


        dW1  = (alpha*delta1).reshape(4, 1) * x.reshape(1, 3)
        mmt1 = dW1 + beta*mmt1
        W1   = W1 + mmt1
        
        dW2  = (alpha*delta2).reshape(4, 1) * y2.reshape(1, 4)
        mmt2 = dW2 + beta*mmt2
        W2   = W2 + mmt2

        dW3  = alpha * delta * y1
        mmt3 = dW3 + beta*mmt3
        W3   = W3 + mmt3
    
    return W1, W2, W3
