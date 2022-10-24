from Sigmoid import *
from BackpropXOR_ad import *
import numpy as np 


def TestBackpropXOR():
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 1, 0],
                  [1, 0, 1, 1],
                  [1, 1, 1, 100],
                  [0.5, 0.5, 1, 0]])
    
    D = np.array([[0],
                  [1],
                  [1],
                  [0],
                  [0]])
    
    W1 = 2*np.random.random((5, 4)) - 1
    W2 = 2*np.random.random((5, 5)) - 1
    W3 = 2*np.random.random((1, 5)) - 1
    
    for _epoch in range(10000):
        W1, W2, W3 = BackpropXOR_ad(W1, W2, W3, X, D)
        
    N = 5
    for k in range(N):
        x  = X[k ,:].T       
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        v  = np.matmul(W3, y2)
        y  = Sigmoid(v)
        print(y)

if __name__ == '__main__':
    TestBackpropXOR()
