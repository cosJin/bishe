#coding:utf-8
import re
import codecs
from bs4 import BeautifulSoup
f1 = open("shanggu/text/cleaned_shanggu_train1600-1971.txt", 'w', encoding="utf-16")  #每次打开都会覆盖
# for fileIndex in range(1160)[0:800]:
f=codecs.open('shanggu/text/raw_shanggu_train1600-1971.txt','rb',encoding='utf-16',errors="replace")
txt=f.read()
# soup = BeautifulSoup(htmlhandle,'lxml')
# a=soup.select("body > div:nth-of-type(1) > center:nth-of-type(1) > div:nth-of-type(1)")
# b=txt.decode()
b=re.sub(r'\(+[a-zA-Z0-9\_]+\)'," ",txt)
b=re.sub(r'\[+[/+a-zA-Z0-9/]+\]'," ",b)  #将标签去掉
f1.write(b)
f.close()
f1.close()

