import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.bar(["$T_1$","$T_2$","$T_3$","$T_4$","$T_5$"],[1732,894,629,132,119],width=0.4)
plt.xlabel("标签",size = 20)
plt.ylabel("句子数量",size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 17)
plt.xticks(fontproperties = 'Times New Roman', size = 17)
plt.show()


