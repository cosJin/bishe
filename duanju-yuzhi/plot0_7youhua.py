import numpy as np
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
x=[0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7]
y=[0.9364,0.937,0.936,0.941,0.9419,0.943,0.949,0.957,0.955,0.952,0.952]
plt.bar(x,y,width=0.005)
plt.plot(x,y,'--')
plt.ylabel("正确率",size=20)
plt.xlabel("阈值",size=20)
plt.ylim(0.92, 0.97)
plt.xlim(0.59, 0.71)
plt.legend()
plt.show()