from Sigmoid import *
from BackpropXOR_add import *
import numpy as np 


def TestBackpropXOR():
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 1, 0],
                  [1, 0, 1, 1],
                  [1, 1, 1, 1],
                  [0.5, 0.5, 1, 0]])
    
    D = np.array([[0],
                  [1],
                  [1],
                  [0],
                  [0]])
    
    W1 = 2*np.random.random((10, 4)) - 1
    W2 = 2*np.random.random((10, 10)) - 1
    W3 = 2*np.random.random((10, 10)) - 1
    W4 = 2*np.random.random((1, 10)) - 1
    
    for _epoch in range(20000):
        W1, W2, W3, W4 = BackpropXOR_add(W1, W2, W3, W4, X, D)
        
    N = 5
    for k in range(N):
        x  = X[k ,:].T       
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        v3 = np.matmul(W3, y2)
        y3 = Sigmoid(v3)
        v  = np.matmul(W4, y3)
        y  = Sigmoid(v)
        print(y)

if __name__ == '__main__':
    for i in range(500):
        TestBackpropXOR()
        print("---------")
