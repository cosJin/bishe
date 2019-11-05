import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

filename = ["32-16","32-32","32-64","64-16","64-32","64-64"]
markers = ["--","-0","--","-.","-","-."]
rate = []
for i,name in enumerate(filename):
    file = open(name+"/log/sanguozhi.txt")
    for line in file.readlines():
        if line.strip()[1] == 'r':
            rate.append(round(float(line.strip().split("=")[-1]),4))
    x = range(len(rate)+1)[1:]
    print(rate)
    plt.plot(x,rate,markers[i],label="hidden:"+name.split('-')[0]+",embedding:"+name.split('-')[1])
    rate = []
plt.yticks(fontproperties = 'Times New Roman', size = 17)
plt.xticks(fontproperties = 'Times New Roman', size = 17)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
# 设置主次y轴的title
plt.ylabel('正确率', size = 20)
# 设置x轴title
plt.xlabel('代数', size = 20)
plt.show()
    
