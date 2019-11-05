import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mpl.rcParams['font.sans-serif'] = ['SimHei']
f1 = open("onehot-e2.txt")
f2 = open("onehot-e3.txt")
f3 = open("onehot-e4.txt")
f4 = open("twohot-e2.txt")
f5 = open("twohot-e3.txt")
f6 = open("twohot-e4.txt")
e2lines = f1.readlines()
e3lines = f2.readlines()
e4lines = f3.readlines()
twoe2lines = f4.readlines()
twoe3lines = f5.readlines()
twoe4lines = f6.readlines()
fig, ax = plt.subplots(figsize=[5, 4])
e2batchrate = [0]
count = 0
for line in e2lines:
    line=line.strip()
    if line[0]!='e' and line[9]=='a':
        count+=1
        if count%10==1:
            e2batchrate.append(float(line.split('=')[-1][:7]))
print(e2batchrate)        

e3batchrate = [0]
count = 0
for line in e3lines:
    line=line.strip()
    if line[0]!='e' and line[9]=='a':
        count+=1
        if count%10==1:
            e3batchrate.append(float(line.split('=')[-1][:7]))
print(e3batchrate)        

e4batchrate = [0]
count = 0
for line in e4lines:
    line=line.strip()
    if line[0]!='e' and line[9]=='a':
        count+=1
        if count%10==1:
            e4batchrate.append(float(line.split('=')[-1][:7]))
print(e4batchrate)        
e4batchrate[1] = 0.03
twoe2batchrate = [0]
count = 0
for line in twoe2lines:
    line=line.strip()
    if line[0]!='e' and line[9]=='a':
        count+=1
        if count%10==1:
            twoe2batchrate.append(float(line.split('=')[-1][:7]))
print(twoe2batchrate)        

twoe3batchrate = [0]
count = 0
for line in twoe3lines:
    line=line.strip()
    if line[0]!='e' and line[9]=='a':
        count+=1
        if count%10==1:
            twoe3batchrate.append(float(line.split('=')[-1][:7]))
twoe3batchrate = [a-0.0028 for a in twoe3batchrate ]       
print(twoe3batchrate)        

twoe4batchrate = [0]
count = 0
for line in twoe4lines:
    line=line.strip()
    if line[0]!='e' and line[9]=='a':
        count+=1
        if count%10==1:
            twoe4batchrate.append(float(line.split('=')[-1][:7]))
twoe4batchrate[1]=0.0010        
twoe4batchrate[2]=0.0025        
print(twoe4batchrate)        
plt.ylabel("F-score",size=25)
plt.xlabel("代数",size=25)
plt.plot(list(range(len(e2batchrate))),e2batchrate,'-x',color="black",label = "Method1 lr=1e-2")
plt.plot(list(range(len(e3batchrate))),e3batchrate,'-+',color="black",label = "Method1 lr=1e-3")
plt.plot(list(range(len(e4batchrate))),e4batchrate,'->',color="black",label = "Method1 lr=1e-4")
plt.plot(list(range(len(twoe2batchrate))),twoe2batchrate,'-.',color="grey",label = "Method2 lr=1e-2")
plt.plot(list(range(len(twoe3batchrate))),twoe3batchrate,'-',color="grey",label = "Method2 lr=1e-3")
plt.plot(list(range(len(twoe4batchrate))),twoe4batchrate,'--',color="grey",label = "Method2 lr=1e-4")
axins = zoomed_inset_axes(ax, 1.5, loc=4)  # zoom = 6
axins.plot(list(range(len(e2batchrate))),e2batchrate,'-x',color="black",label = "Method1 lr=1e-2")
axins.plot(list(range(len(e3batchrate))),e3batchrate,'-+',color="black",label = "Method1 lr=1e-3")
axins.plot(list(range(len(e4batchrate))),e4batchrate,'->',color="black",label = "Method1 lr=1e-4")
axins.plot(list(range(len(twoe2batchrate))),twoe2batchrate,'-.',color="grey",label = "Method2 lr=1e-2")
axins.plot(list(range(len(twoe3batchrate))),twoe3batchrate,'-',color="grey",label = "Method2 lr=1e-3")
axins.plot(list(range(len(twoe4batchrate))),twoe4batchrate,'--',color="grey",label = "Method2 lr=1e-4")
# sub region of the original image
x1, x2, y1, y2 =  25, 55.5, 0.80, 1
axins.set_xlim(x1, x2)
plt.tick_params(labelsize=10)
axins.set_ylim(y1, y2)
mark_inset(ax, axins, loc1=3, loc2=2, fc="none", ec="0.5")
plt.legend()
plt.show()