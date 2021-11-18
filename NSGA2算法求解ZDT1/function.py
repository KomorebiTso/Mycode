"""
优化的目标函数，返回多个目标函数值 
"""
import numpy as np 

def function(X):
    m = X.shape[1]
    f1 = X[:,0]
    #print(m)
    #n = (np.sum(X, axis=1)-X[:, 0])*9
    #lie = X[:, 0]
    #print(n)
    #print(X.shape)
    #print(lie)
    #print(lie.shape)
    #print(n.shape)
    g = 1 + (np.sum(X, axis=1)-X[:, 0])*9/(m-1)
    #print(g)
    #print(f1/g)
    h = 1- np.sqrt(f1/g)
    #h = 1-(f1/g)**2
    #print(h)
    f2 = g*h
    #print(m)
    #print(f1)
    #print(g)
    #print(h)
    #print(f2)
    f = np.column_stack((f1,f2))
    #print(f.shape)
    #print (f)
    return f

if __name__ == "__main__":
    tX = np.array([[0.5,0.5, 0.5],[0.4,0.4, 0.4],[0.2,0.2,0.2],[0.1,0.1,0.1]])
    #print(tX)
    a=function(tX)
    #print(a)
    #print(function(tX))