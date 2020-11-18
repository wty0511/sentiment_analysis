import re
import jieba
import math
import numpy as np
import copy
import jieba.posseg
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
path1 = r"data2utf8.csv"  # 语料


def splitSentence(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


def yuchuli(para, stopwordsdic):
    # para为包含该篇文章中各条句子的list[]

    paracixing = []
    for i in range(0, len(para)):
        splited = jieba.posseg.cut(para[i])

        # 去除停止词
        senten = []
        cixing = []
        for w in splited:
            if w.flag[0] not in stopwordsdic:
                if w.word != '\n':
                    senten.append(w.word)
                    cixing.append(w.flag)
                if w.word == '时':
                    pass
                    #print('dsdasaaada' + str(w))
            else:
                pass
                # print(w)
        para[i] = senten
        paracixing.append(cixing)
    return paracixing


'''
def yuchulititle(title, stopwordsdic):
    # para为包含该篇文章标题的str
    title=jieba.lcut(title, cut_all=False)

    # 去除停止词
    senten = []
    for i in range(0, len(title)):
        if title[i] not in stopwordsdic:
            if title[i] != '\n':
                senten.append(title[i])
        if title[i] in stopwordsdic:
            print(title[i])
    return senten
'''


def yuchulititle(title, stopwordsdic):
    # para为包含该篇文章标题的str
    splited = jieba.posseg.cut(title)

    # 去除停止词
    senten = []
    cixing = []
    for w in splited:
        if w.flag[0] not in stopwordsdic:
            if w.word != '\n':
                senten.append(w.word)
                cixing.append(w.flag)
            if w.word == '时':
                print('dsdasaaada' + str(w))
        else:
            pass
            # print(w)
    return [senten, cixing]


def highfrequencyfeatures(title, para):
    para2 = copy.deepcopy(para)
    para.append(title)
    allparawords = para[0]
    for i in range(1, len(para)):
        allparawords = allparawords + para[i]

    allparawordssigle = list(set(allparawords))  # 文章中出现的所有词list
    yuzhi = 0
    # 阈值
    numsenten = len(para)
    if numsenten < 10:
        yuzhi = 2
    if 10 <= numsenten < 30:
        yuzhi = 3
    if 30 <= numsenten < 70:
        yuzhi = 4
    if 70 <= numsenten:
        yuzhi = 5

    # 统计词频
    dicParaWordFreq = {}

    for i in range(len(allparawordssigle)):
        nowword = allparawordssigle[i]
        dicParaWordFreq[nowword] = 0  # 词频初始为0
        for j in range(len(allparawords)):
            if nowword == allparawords[j]:
                dicParaWordFreq[nowword] += 1

    hfwl1 = {}
    for (key, value) in dicParaWordFreq.items():
        if value >= 2:
            hfwl1[key] = value

    hfwl2 = {}
    allparawordslianxu = ''.join(allparawords)
    dicdaichaw2 = {}
    for key in hfwl1:
        for sentence in para:
            if key in sentence:
                lwindex = []
                for i in range(len(sentence)):
                    if sentence[i] == key:
                        lwindex.append(i)
                for windex in lwindex:
                    zuheword = key
                    slen = len(sentence)
                    for i in range(windex + 1, windex + 5):  # 后续1-4个词
                        if i < slen:
                            zuheword = zuheword + sentence[i]
                            dicdaichaw2[zuheword] = 0

    for key in dicdaichaw2:
        dicdaichaw2[key] = allparawordslianxu.count(key)

    for (key, value) in dicdaichaw2.items():
        if value >= yuzhi:
            hfwl2[key] = value

    # hfwl2去重
    dellist = []
    for (key1, value1) in hfwl2.items():
        maxnum = value1
        dqkey = key1
        for (key2, value2) in hfwl2.items():
            if key1 == key2:
                continue
            if key1 in key2:
                if value2 >= maxnum:
                    dellist.append(dqkey)
                    maxnum = value2
                    dqkey = key2
                else:
                    dellist.append(key2)
    dellist = list(set(dellist))
    for i in dellist:
        del hfwl2[i]

    # hfwl1更新
    dellist = []
    for (k2, v2) in hfwl2.items():
        for (k1, v1) in hfwl1.items():
            if k1 in k2:
                if v2 < v1:
                    hfwl1[k1] = v1 - v2
                else:
                    dellist.append(k1)
    dellist = list(set(dellist))
    for i in dellist:
        del hfwl1[i]

    lshfwl1 = sorted(hfwl1.items(), key=lambda item: item[1], reverse=True)
    lshfwl2 = sorted(hfwl2.items(), key=lambda item: item[1], reverse=True)

    lshfwlfinal = lshfwl2[0:21]
    if len(lshfwlfinal) < 20:
        lshfwlfinal = lshfwlfinal + lshfwl1[0:20 - len(lshfwlfinal)]

    # print(lshfwlfinal)

    para = para2
    firstlastSsentens = para[:yuzhi + 1] + para[-yuzhi - 1:]
    for i in range(len(firstlastSsentens)):
        firstlastSsentens[i] = ''.join(firstlastSsentens[i])
    flSsstr = ''.join(firstlastSsentens)
    titlestr = ''.join(title)

    # 根据词所处不同位置分配权
    dicwordweight = {}
    hfwlfinal = dict(lshfwlfinal)
    for key in hfwlfinal:
        if key in titlestr:
            dicwordweight[key] = 5
            continue
        if key in flSsstr:
            dicwordweight[key] = 3
            continue
        if key in allparawordslianxu:
            dicwordweight[key] = 1
            continue

    desentensubw = {}  # 统计每一句的权
    for i in range(len(para)):
        weight = 0
        parastr = ''.join(para[i])
        for key in hfwlfinal:
            if key in parastr:
                weight += hfwlfinal[key] * dicwordweight[key]
        desentensubw[i] = weight

    weightsum = sum(desentensubw.values())

    for i in range(len(para)):
        desentensubw[i] = desentensubw[i] / weightsum

    weightsed = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)
    # print(weightsed)
    return [weightsed, hfwl2]  # [[文章中每一句的权tuple(句子index：权)]，该文章的HFWL2词典dic{词：词频}]


def titlefeatures(title, titlecx, para, paracixing, foudingdic, hfwl2):
    titlen = []
    titleo = []
    for i in range(len(title)):
        if titlecx[i][0] == 'n':
            titlen.append(title[i])
        else:
            titleo.append(title[i])
    cnounst = len(titlen)
    titlestr = ''.join(title)
    titleospastr = ' '.join(titleo)

    desentensubw = {}  # 统计每一句的权
    for i in range(len(para)):
        sentenn = []
        senteno = []
        for k in range(len(para[i])):
            if paracixing[i][k][0] == 'n':
                sentenn.append(para[i][k])
            else:
                senteno.append(para[i][k])
        cnounssi = len(sentenn)
        alln = list(set(sentenn + titlen))
        counsame = 0
        for k in alln:
            if k in sentenn and k in titlen:
                counsame += 1
        sentenstr = ''.join(para[i])
        for (key, value2) in hfwl2.items():
            if key in sentenstr and key in titlestr:
                counsame += 1
        simn = counsame / (cnounssi + cnounst - counsame)

        sentenospastr = ' '.join(senteno)

        try:
            # 待改进
            count_vec = CountVectorizer(stop_words=None)
            sentences = [sentenospastr, titleospastr]

            #print('sentences')
            #print(sentences)
            # 求相应的词袋向量
            wordvec = count_vec.fit_transform(sentences).toarray()

            # 求余弦相似度
            X = np.vstack(wordvec)
            ndsimo = 1 - pdist(X, 'cosine')
            simo = ndsimo[0]
            if simo == np.nan:
                simo=0
        except ValueError:
            simo = 0

        # 否定词未实现

        res = simn * 0.5 + simo * 0.5

        desentensubw[i] = res

    fres = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)

    # print(fres)
    return fres  # [[文章中每一句的权tuple(句子index：权)]


def sentencepositionfeature(para):
    # 阈值
    numsenten = len(para)
    yuzhi=0
    if numsenten < 10:
        yuzhi = 1
    if 10 <= numsenten < 30:
        yuzhi = 2
    if 30 <= numsenten < 70:
        yuzhi = 3
    if 70 <= numsenten < 110:
        yuzhi = 4

    desentensubw = {}  # 统计每一句的权
    snum = len(para)
    for i in range(snum):
        nowsenten = para[i]
        if i <= yuzhi:
            desentensubw[i] = 1
        else:
            desentensubw[i] = 1 - (math.log2(i) / math.log2(snum))

    fres = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)

    # print(fres)
    return fres  # [[文章中每一句的权tuple(句子index：权)]


def tendfeature(para,trendworddic):
    desentensubw = {}  # 统计每一句的权
    snum = len(para)
    for i in range(snum):
        nowsenten = para[i]
        for tw in trendworddic:
            if tw in nowsenten:
                desentensubw[i] = 1
            else:
                desentensubw[i] = 0

    fres = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)

    # print(fres)
    return fres  # [[文章中每一句的权tuple(句子index：权)]


def emotionalwordfeatures(para,emotionworddic):
    desentensubw = {}  # 统计每一句的权
    for i in range(len(para)):
        weight = 0
        parastr = ''.join(para[i])
        for key in emotionworddic:
            if key in parastr:
                weight += 1
        desentensubw[i] = weight

    weightsum = sum(desentensubw.values())

    for i in range(len(para)):
        desentensubw[i] = desentensubw[i] / weightsum

    weightsed = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)

    return weightsed  # [文章中每一句的权tuple(句子index：权)]


def txt_spliter(pathf, splitstr):
    f2 = open(pathf, encoding='utf-8')
    data = []
    i = 0
    for line in f2:
        if (i > 0):  # 跳过第一行
            if (line == '\n'):
                continue
            line = line.rstrip()
            if len(line.split(splitstr)) != 1:
                line = line.split(splitstr)
            data.append(line)
        i = i + 1
    return data


def get_key_sentences(t,c,contextlength):
    
    stopworddic = txt_spliter("stopwords.txt", '\n')
    trendworddic = txt_spliter("倾向词.txt", '\n')
    emoworddic = txt_spliter("总情感词典.txt", '\n')



    c = splitSentence(c)

    rawpara=copy.deepcopy(c)

    paracixing = yuchuli(c, stopworddic)
    pas = c
    lsa = yuchulititle(t, stopworddic)
    t = lsa[0]
    titlecixing = lsa[1]

    para = copy.deepcopy(c)
    title = copy.deepcopy(t)

    lget = highfrequencyfeatures(title, para)
    hffweight = dict(lget[0])
    hfwl2 = lget[1]

    para = copy.deepcopy(c)
    title = copy.deepcopy(t)

    tfweight = dict(titlefeatures(title, titlecixing, para, paracixing, [], hfwl2))

    spfweight = dict(sentencepositionfeature(para))

    tdfweight = dict(tendfeature(para,trendworddic))

    ewfweight= dict(emotionalwordfeatures(para,emoworddic))

    dicweightzong ={}
    for k in range(len(para)):# 对上述各评价指标加权
        dicweightzong[k] = 0.3*hffweight[k]+0.25*tfweight[k]+0.05*spfweight[k]+0.1*tdfweight[k]+0.3*ewfweight[k]

    weightsed = sorted(dicweightzong.items(), key=lambda item: item[1], reverse=True)

    improtantsentence = []
    for i in range(len(weightsed)):
        if i>=contextlength:break
        improtantsentence.append(rawpara[weightsed[i][0]])
    return improtantsentence


