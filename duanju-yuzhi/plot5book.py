import numpy as np
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
N = 5


y1 = [1508,194,185,123 ,181 ]
y2 = [663 ,521,188,258 ,202 ]
y3 = [555 ,187,688,321 ,426 ]
y4 = [369 ,137,173,1102,344 ]
y5 = [126 ,19 ,32 ,247 ,1309]

index = np.array([1,7,13,19,25])
 
bar_width = 1
plt.bar(index , y1, width=bar_width , color='grey',hatch="\\\\\\", alpha = 0.7, label = "T1")
plt.bar(index +  bar_width, y2, width=bar_width ,hatch="++", color='grey', alpha = 0.7, label = "T2")
plt.bar(index +  bar_width*2, y3, width=bar_width ,hatch="///", color='grey', alpha = 0.7, label = "T3")
plt.bar(index +  bar_width*3, y4, width=bar_width ,hatch="---", color='grey', alpha = 0.7, label = "T4")
plt.bar(index +  bar_width*4, y5, width=bar_width ,hatch="//\\\\", color='grey', alpha = 0.7, label = "T5")
plt.bar(np.array([1,8,15,22,29]), [1508,521,688,1102,1309], width=bar_width , color='grey', alpha = 0.2)
plt.ylabel("句子条数",size=20)
plt.xlabel("书籍",size=20)
plt.xticks([3,9,15,21,27],["《左传》","《素问》","《世说新语》","《全相平话》","《西游记》"],size = 15)
plt.legend()
plt.show()