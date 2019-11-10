import pandas as pd
import pickle
import re
# word2dictionary = open("word2dictionary.txt", 'w', encoding="utf-16")
with open('word2dictionary.txt', 'rb') as ind:
    dic = ind.read().decode('utf-16')
    dic=dic.split()

# sentense="你 看 我 盡 節 存忠 立 功勛，單 注 著 楚 霸 王 大軍 盡 霸 王 的"  #你  看  我  盡  節  存  忠  立  功勛  ，單  注  著  楚霸王  大軍  盡  。
sentences = [
"那 知 州 聽 得 這 話 ， 從 頂 門 上 不 見 了 三 魂 ， 腳 底 下 疏 失 了 七 魄 ， 便 投 後 殿 走 了 。",              #那  知州  聽   得   這  話  ，從  頂門  上  不  見  了  三魂  ，腳  底下  疏失  了  七魄  ，便  投  後 殿  走  了  。
"失 卻 龍 駒 怎 戰 爭 ， 了 虞 姬 那 痛 增 。 ",                                                    #失卻  龍駒  怎  戰爭  ，了  虞姬  那  痛  增
"你 看 我 盡 節 存 忠 立 功 勛 ， 單 注 著 楚 霸 王 大 軍 盡 。",                                        #你  看  我  盡  節  存  忠  立  功勛  ，單  注  著  楚霸王  大軍  盡  。
"今 朝 希 遇 大 乘 經 ， 見 優 曇 花 一 種 ；",                                                    #今朝  希  遇  大乘經  ，見  優曇花  一  種  ；
"全 不 見 鴻 門 會 那 氣 性 ， 今 日 向 烏 江 岸 滅 盡 形 。",                                          #全  不  見  鴻門會  那  氣性  ，今日  向  烏江  岸  滅   盡   形  。
"昔 者 齊 晏 子 使 於 梁 國 為 使 ",                                                            #昔  者  齊  晏子  使  於  梁國  為  使
"劍 雖 三 尺 ， 定 四 方 ， 麒 麟 雖 小 ， 聖 君 瑞 應 ；",                                            #劍  雖  三  尺  ，定  四方  ，麒麟  雖  小  ，聖君  瑞應  ；
"學 而 時 習 之 ， 不 亦 說 乎 ？ 有 朋 自 遠 方 來 ， 不 亦 樂 乎 ？ 人 不 知 而 不 慍 ， 不 亦 君 子 乎 ？",          #學 而 時 習 之 ， 不 亦 說 乎 ？有 朋 自 遠方 來 ，不 亦 樂 乎 ？人 不 知 而 不 慍 ，不 亦 君子 乎？
"弟 子 入 則 孝 ， 出 則 弟 ， 謹 而 信 ， 汎 愛 眾 ， 而 親 仁 。"                                       #弟子 入 則 孝 ，出 則 弟 ，謹 而 信 ，汎 愛 眾 ，而 親 仁 。
]
for sentense in sentences:
    letter=sentense.split()
    sig=-2
    result=[]

    for i in range(len(letter)-1):
        if(sig+1==i):
            continue
        else:
            word=letter[i]+letter[i+1]
            if word in dic:
                result.append(word)
                sig=i
            else:
                result.append(letter[i])
    if(sig!=len(letter)-2):
        result.append(letter[i+1])
    print(result)
# for wor in dic:
#     word2dictionary.write(wor+' ')