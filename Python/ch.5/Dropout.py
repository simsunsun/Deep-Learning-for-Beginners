from tkinter import Y
import numpy as np

'''
y = np.array([1, 2, 3, 4, 5])
y = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
y
ratio = 0.5
ym = np.zeros_like(y)
ym
num = round(y.size*(1-ratio))
num
y.size
idx = np.random.choice(y.size, num, replace=False)
idx
ym[idx] = 1.0 / (1.0 - ratio)
ym
'''

def Dropout(y, ratio):
    ym = np.zeros_like(y)  
    num = round(y.size*(1-ratio))
    idx = np.random.choice(y.size, num, replace=False)
    ym[idx] = 1.0 / (1.0 - ratio)
    
    return ym