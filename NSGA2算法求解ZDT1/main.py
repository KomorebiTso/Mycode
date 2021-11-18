"""
测试NSGA2算法 
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis 
from NSGA2 import * 
from function import * 
from fitness import *

def main(): 
    nIter = 100
    nChr = 30
    nPop = 200
    pc = 0.6  
    pm = 0.1 
    etaC = 1 
    etaM = 1 
    func = function 
    lb = 0
    rb = 1
    paretoPops, paretoFits = NSGA2(nIter, nChr, nPop, pc, pm, etaC, etaM, func, lb, rb) 
    print(paretoFits) 
    print(f"paretoFront: {paretoFits.shape}") 

    # 理论最优解集合 
    pops = np.random.rand(nPop, nChr) * (rb - lb) + lb
    f1 = pops[:,0]
    f2 = 1 - np.sqrt(f1)
    #f2 = 1 - (f1)**2
    thFits = np.column_stack((f1, f2))

    plt.rcParams['font.sans-serif'] = 'KaiTi'  # 设置显示中文
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.scatter(thFits[:,0], thFits[:,1], color='green', label='理论帕累托前沿')
    ax.scatter(paretoFits[:,0], paretoFits[:,1], color='red', label='实际解集')
    ax.legend()
    fig.show()
    fig.savefig('解集.png', dpi=400)

    print(paretoPops) 

if __name__ == "__main__": 
    main() 