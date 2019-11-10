import matplotlib.pyplot as plt
f=open("dataOfwithDic.txt","r")
strData = f.read()
strData = strData.split('\n')
# print(len(strData))
# for i in range(len(strData)-1):
for i in range(20000):
    if(i%300==0):
        data = strData[i][1:-1].split(',')
        plt.scatter(int(data[1]),round(float(data[2]),4),c="b")
        plt.title("LSTM+Dic")
        plt.xlabel("Number of word")
        plt.ylabel("Correct Rate")
        print(data)
plt.show()