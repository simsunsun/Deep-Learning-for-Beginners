import numpy as np
from Sigmoid import *


def DeltaSGD(W, X, D):
    alpha = 0.3
    
    N = 5
    for k in range(N):
        x = X[k, :].T
        d = D[k]

        # print("x :", x)
    
        v = np.matmul(W, x)
        y = Sigmoid(v)
        
        e     = d - y
        delta = y*(1-y) * e
        
        dW = alpha*delta*x
        
        W[0][0] = W[0][0] + dW[0]
        W[0][1] = W[0][1] + dW[1]
        W[0][2] = W[0][2] + dW[2]
        
    return W
