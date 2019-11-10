#coding:utf-8
import codecs
from bs4 import BeautifulSoup
f1 = open("shanggu/text/raw_shanggu_train1600-1971.txt", 'w', encoding="utf-16")  #每次打开都会覆盖
for fileIndex in range(1972)[1600:]:
    f=codecs.open('shanggu\\'+str(fileIndex)+'.html','rb')
    htmlhandle=f.read()
    soup = BeautifulSoup(htmlhandle,'lxml')
    a=soup.select("html body div center div")
    b=a[0].decode().split("</div>")[1]
    b=b.replace("。","。\n").replace("？","？\n").replace("！","！\n").replace("；","；\n").replace("<b>"," ").replace("</b>"," ").replace("<br/>","").replace("<span class=\"k\">","").replace("</span>","")
    print(b)
    f1.write(b)
    f.close()
f1.close()
# print(b)