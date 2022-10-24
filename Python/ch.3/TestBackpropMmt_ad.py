import numpy as np
from BackpropMnt_ad import *
from Sigmoid import *


def TestBackpropMmt():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    D = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    W1 = 2*np.random.random((4, 3)) - 1
    W2 = 2*np.random.random((4, 4)) - 1
    W3 = 2*np.random.random((1, 4)) - 1
    
    for _epoch in range(10000):
        W1, W2, W3 = BackPropMmt_ad(W1, W2, W3, X, D)
        
    N = 4
    for k in range(N):
        x  = X[k, :].T
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        v  = np.matmul(W3, y2)
        y  = Sigmoid(v)
        print(y)

if __name__ == '__main__':
    TestBackpropMmt()