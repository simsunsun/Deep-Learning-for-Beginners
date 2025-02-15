import numpy as np
from Sigmoid import *
from DeltaSGD import *


def TestDeltaSGD():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    D = np.array([[0],
                  [0],
                  [1],
                  [1]])
        
    W = 2*np.random.random((1, 3)) -1
        
    for _epoch in range(40000):
        W = DeltaSGD(W, X, D)
        # print(W)

    N = 4
    for k in range(N):
        x = X[k,:].T
        v = np.matmul(W, x)
        y = Sigmoid(v)
        print(y)

if __name__ == '__main__':
    TestDeltaSGD()