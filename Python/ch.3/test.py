import numpy as np 
from Sigmoid import *

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

W1 = 2*np.random.random((4, 3)) - 1
W2 = 2*np.random.random((1, 4)) - 1

# print(W1)
# print("---")
# print(W2)

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([[0],
              [1],
              [1],
              [0]])

k = 0
alpha = 0.9

x = X[k, :].T
# xx = X[k, :]
d = D[k]

# print("x :", x)
# print("d :", d)

v1 = np.matmul(W1, x)
# vv1 = np.matmul(W1, xx)
y1 = Sigmoid(v1)
v  = np.matmul(W2, y1)
y  = Sigmoid(v)


e     = d - y
delta = y*(1-y) * e

e1     = np.matmul(W2.T, delta)
delta1 = y1*(1-y1) * e1


dW1 = (alpha*delta1).reshape(4, 1) * x.reshape(1, 3)
W1  = W1 + dW1

dW2 = alpha * delta * y1
W2  = W2 + dW2


'''
은닉층 1개
[0.00723065]
[0.99199064]
[0.98706728]
[0.00159852]
[0.01263912]

은닉층 2개
[0.00450277]
[0.99367099]
[0.99387646]
[0.00149086]
[0.00627885]

은닉층 3개
[0.0017599]
[0.99267904]
[0.9926975]
[1.06561219e-06]
[0.00693218]

'''

''' 모멘텀
은닉층 1개
[0.00403209]
[0.99338112]
[0.99262116]
[0.01216342]

은닉층 2개
[0.00170241]
[0.99601228]
[0.9944777]
[0.00186239]

은닉층 3개
[0.00256654]
[0.99643643]
[0.99586614]
[0.00556071]

'''
