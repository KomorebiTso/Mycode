"""
种群或个体的适应度 
"""
import numpy as np 
from function import * 

def fitness(pops,func):
    # 计算种群或者个体的适应度 
    # 如果是1维需要转为2维 
    if pops.ndim == 1:
        pops = pops.reshape(1,len(pops))
    fits = func(pops)
    return fits 

if __name__ == "__main__":
    pops = np.array([[0.5,0.5, 0.5],[0.4,0.4, 0.4],[0.2,0.2,0.2],[0.1,0.1,0.1]])
    fits = fitness(pops, function) 
    print(fits) 