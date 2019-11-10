# -*- coding: UTF-8 -*-
# from urllib import request
# response = request.urlopen("http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh?ukey=-1615052793&qtype=11&tid=5&swTag=3")  #tid是文章号，swTag是显示标记转换的，swTag设置为1，则每次刷新都会改变显示标记。
# html = response.read()                                                                                                                #ukey是一个时间戳，隔一段时间一刷新。
# html = html.decode("big5",errors="replace")
# print(html)

import urllib.request as urllib2
from time import sleep
ua_headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0',
}

for pageNum in range(1972)[1035:]:

    f = open("shanggu/"+str(pageNum)+".html", 'w', encoding="utf-16")  # 每次打开都会覆盖
    # 通过Request()方法构造一个请求对象
    url = 'http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh?ukey=-49807357&qtype=11&tid='+str(pageNum)   #对于上古汉语，tid=1971则为上限。
    # request = urllib2.Request(url, headers = ua_headers)
    # response = urllib2.urlopen(request)
    # url = 'http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh?ukey=-49807357&qtype=11&tid='+str(pageNum)+'&swTag=1'   #对于上古汉语，tid=1971则为上限。
    request = urllib2.Request(url, headers = ua_headers)
    # 向指定的url地址发送请求，并返回服务器响应的类文件对象
    response = urllib2.urlopen(request)

    # 服务器返回的类文件对象支持python文件对象的操作方法
    html = response.read()
    htmlDown = html.decode('big5', errors="replace")
    sleep(2)
    f.write(htmlDown)
    print(pageNum)
    f.close()
