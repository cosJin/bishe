f = open("log.log","r")
lenth_right={}
for line in f.readlines():
    info = line.strip()[1:-1].split(",")
    rightnum = float(info[0])
    sentencelen= len(info[1:-1])
#if(rightnum/sentencelen < 0.5):continue
    if sentencelen in lenth_right.keys():
        lenth_right[sentencelen][0]+=rightnum
        lenth_right[sentencelen][1]+=sentencelen
    else:
        lenth_right[sentencelen]=[rightnum,sentencelen]
print(lenth_right)
len=1
for k_v in lenth_right.values():
    print(k_v[0]/k_v[1])
    len+=1

                                               

    

