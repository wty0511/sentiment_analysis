import re
import jieba
import math
import numpy as np
import copy
import jieba.posseg
import jieba.analyse
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Doc2Vec,KeyedVectors

class ImportantSentence:
    """
    抽取重点句构造方法：
    objname=ImportantSentence(loadw2vmodle=False)
          loadw2vmodle=是否加载word2vec预训练模型，默认False不加载，不加载时仅能进行分词与提取词性，加载模型时间较长
    分析，抽取重点句，分词：
    objname.analyze(rawdatalist=None, pathdata='',impsentnum1=5, save=False,outputrawpara=False,delwordcixing=['c','e','f','p','r','u','w','x','q','n']):
                    (rawdatalist=语料，语料格式见下,pathdata=语料文件路径（可选）, 抽取前几句，默认5, 是否保存结果到文件，默认False不保存
                    outputrawpara=输出原语句还是删除特定词性后的词表，delwordcixing=要删除的词性list，按jieba的词性名称)
    语料格式：[['标题','内容句子。内容句子。内容句子......'],['标题','内容句子。内容句子。内容句子......']......]
    获取第i篇文章的重点句，返回为句子组成的list：
        objname.improtantsentences[i]
    获取第i篇文章的标题的词性list，返回为词性的list：
    objname.cixing[i]['title']
    获取第i篇文章的标题的分词结果list，返回为分词后的list：
    objname.fenci[i]['title']
    获取第i篇文章的正文第j句的词性list，返回为每句词性的list：
    objname.cixing[i]['para'][j]
    获取第i篇文章的正文第j句的分词结果list，返回为每句分词后的list：
    imsen.fenci[i]['para'][j]
    """

    data = [] # 语料
    pathsave = r"data抽取完重点句.csv"  # 保存文件名
    impsentnum =7 #取前几句重点句
    improtantsentences=[]
    cixing=[]
    fenci=[]
    splitedtap=[[],[]]



    def splitSentence(self,para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        return para.split("\n")


    def yuchuli(self,para, stopwordsdic):
        # para为包含该篇文章中各条句子的list[]

        paracixing = []
        for i in range(0, len(para)):
            splited = jieba.posseg.cut(para[i])


            # 去除停止词
            sentenpair=[]
            senten = []
            cixing = []
            for w in splited:
                sentenpair.append(w)
                if w.flag[0] not in stopwordsdic:
                    if w.word != '\n':
                        senten.append(w.word)
                        cixing.append(w.flag)
                    if w.word == '时':
                        pass
                        # print('dsdasaaada' + str(w))
                else:
                    pass
                    # print(w)
            para[i] = senten
            paracixing.append(cixing)
            self.splitedtap[1].append(sentenpair)
        return paracixing


    def yuchulititle(self,title, stopwordsdic):
        # para为包含该篇文章标题的str
        splited = jieba.posseg.cut(title)
        self.splitedtap[0] = splited

        # 去除停止词
        senten = []
        cixing = []
        for w in splited:
            if w.flag[0] not in stopwordsdic:
                if w.word != '\n':
                    senten.append(w.word)
                    cixing.append(w.flag)
                if w.word == '时':
                    #print('dsdasaaada' + str(w))
                    pass
            else:
                pass
                # print(w)
        return [senten, cixing]


    def highfrequencyfeatures(self,title, para,rawpara,rawtitle,dicParaWordtfidf):
        para2 = copy.deepcopy(para)
        para.append(title)
        allparawords = para[0]
        for i in range(1, len(para)):
            allparawords = allparawords + para[i]

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

        # 统计tfidf

        totalparawt = '。'.join([rawtitle,rawpara])
        dicParaWordtfidf = [w for w in dicParaWordtfidf if w[1]>=0.04]

        hfwl1 = {}
        for (word, value) in dicParaWordtfidf:
                hfwl1[word] = totalparawt.count(word)

        hfwl2 = {}
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
            dicdaichaw2[key] = totalparawt.count(key)

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
            if key in totalparawt:
                dicwordweight[key] = 1
                continue
            print('掉出')
            print(key)

        #print('fsfsfs')

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
            try:
                desentensubw[i] = desentensubw[i] / weightsum
            except ZeroDivisionError:
                desentensubw[i] = 0

        #weightsed = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)
        # print(weightsed)
        return [desentensubw, hfwl2]  # [[文章中每一句的权tuple(句子index：权)]，该文章的HFWL2词典dic{词：词频}]


    def getsentencevecbytfidf(self,dicParaWordtfidf,sentence):
        dicParaWordtfidf=dict(dicParaWordtfidf)
        res = np.zeros(128)
        wtfidfsum =0
        for word in sentence:
            try:
                wtfidf=dicParaWordtfidf[word]
            except KeyError:
                wtfidf=0.005
                #print(word)
            wtfidfsum+=wtfidf

            try:
                wvec = self.w2vmodel[word]
            except KeyError:
                wvec = np.zeros(128)
                #print(word)
            res=res+(wvec*wtfidf)
        res=res/wtfidfsum
        return res


    def newtitlefeatures(self,title, para,dicParaWordtfidf,titlecx, paracixing,hfwl2):
        titlestr = ''.join(title)
        titlen = []
        titleo = []

        for i in range(len(title)):
            if titlecx[i][0] == 'n':
                titlen.append(title[i])
            else:
                titleo.append(title[i])
        for (key, value2) in hfwl2.items():
            if key in titlestr:
                titlen.append(key)

        cnounst = len(titlen)
        titleospastr = ' '.join(titleo)

        titlvec = self.getsentencevecbytfidf(dicParaWordtfidf, title)

        desentensubw = {}  # 统计每一句的权
        for i in range(len(para)):
            sentenstr = ''.join(para[i])
            sentenn = []
            senteno = []
            for k in range(len(para[i])):
                if paracixing[i][k][0] == 'n':
                    sentenn.append(para[i][k])
                else:
                    senteno.append(para[i][k])
            for (key, value2) in hfwl2.items():
                if key in sentenstr:
                    sentenn.append(key)
            cnounssi = len(sentenn)
            alln = list(set(sentenn + titlen))

            counsame = 0
            for k in alln:
                if k in sentenn and k in titlen:
                    counsame += 1

            for (key, value2) in hfwl2.items():
                if key in sentenstr and key in titlestr:
                    counsame += 1

            try:
                simn = counsame / (cnounssi + cnounst - counsame)
            except ZeroDivisionError:
                simn = 0



            sentvec = self.getsentencevecbytfidf(dicParaWordtfidf, para[i])
            try:
                # 求余弦相似度
                X = np.vstack([sentvec, titlvec])
                ndsimo = 1 - pdist(X, 'cosine')
                simo = ndsimo[0]
                if simo == np.nan:
                    simo = 0
            except ValueError:
                simo = 0

            desentensubw[i] = simn * 0.3 + simo * 0.7

        fres = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)


        pass
        return desentensubw


    def sentencepositionfeature(self,para):
        # 阈值
        numsenten = len(para)
        if numsenten < 10:
            yuzhi = 1
        if 10 <= numsenten < 30:
            yuzhi = 2
        if 30 <= numsenten < 70:
            yuzhi = 3
        if 70 <= numsenten < 110:
            yuzhi = 4
        else:
            yuzhi = 5

        desentensubw = {}  # 统计每一句的权
        snum = len(para)
        for i in range(snum):
            nowsenten = para[i]
            if i <= yuzhi or snum-1 <= yuzhi:
                desentensubw[i] = 1
            else:
                desentensubw[i] = 1 - (math.log2(i) / math.log2(snum))

        #fres = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)

        # print(fres)
        return desentensubw  # [[文章中每一句的权tuple(句子index：权)]


    def tendfeature(self,para, trendworddic):
        desentensubw = {}  # 统计每一句的权
        snum = len(para)
        for i in range(snum):
            nowsenten = para[i]
            for tw in trendworddic:
                if tw in nowsenten:
                    desentensubw[i] = 1
                else:
                    desentensubw[i] = 0

        #fres = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)

        # print(fres)
        return desentensubw  # [[文章中每一句的权tuple(句子index：权)]try:try:


    def emotionalwordfeatures(self,para, emotionworddic):
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
            try:
                desentensubw[i] = desentensubw[i] / weightsum
            except ZeroDivisionError:
                desentensubw[i] = 0

        #weightsed = sorted(desentensubw.items(), key=lambda item: item[1], reverse=True)
        # print(weightsed)
        return desentensubw  # [文章中每一句的权tuple(句子index：权)]


    def txt_spliter(self,pathf, splitstr):
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


    def __init__(self, rawdatalist=None, pathdata='', pathsave1="data抽取完重点句.csv",loadw2vmodle=False):
        self.pathsave=pathsave1
        self.path1=pathdata
        self.justsplit=not loadw2vmodle
        if loadw2vmodle:
            self.w2vmodel = KeyedVectors.load_word2vec_format('baike_26g_news_13g_novel_229g.bin', binary=True)
        else:
            self.w2vmodel =None


    def analyze(self, rawdatalist=None, pathdata='', impsentnum1=5, save=False, outputrawpara=True, delwordcixing=None):
        if delwordcixing is None:
            delwordcixing = []
        self.path1 = pathdata
        self.improtantsentences=[]
        self.impsentnum = impsentnum1
        if rawdatalist is None:
            data = self.txt_spliter(self.path1, ',')
            it = 1
            ip = 2
        else:
            data = rawdatalist
            it = 0
            ip = 1
        datasave = open(self.pathsave, 'w')
        stopworddic = self.txt_spliter("stopwords2.txt", '\n')
        trendworddic = self.txt_spliter("倾向词.txt", '\n')
        emoworddic = self.txt_spliter("总情感词典.txt", '\n')

        rawrawdatacopy = copy.deepcopy(data)

        for i in range(len(data)):
            '''
            print('标题：')
            print(data[i][1])
            print('文章：')
            print(data[i][ip])
            '''
            self.splitedtap=[[],[]]


            totalparawt = '。'.join([data[i][it], data[i][ip]])
            dicParaWordtfidf = jieba.analyse.extract_tags(totalparawt, topK=None, withWeight=True)

            data[i][ip] = self.splitSentence(data[i][ip])

            rawpara = copy.deepcopy(data[i][ip])  # 仅分句之后的
            rawtitle = copy.deepcopy(data[i][it])

            paracixing = self.yuchuli(data[i][ip], stopworddic)

            lsa = self.yuchulititle(data[i][it], stopworddic)
            data[i][it] = lsa[0]
            titlecixing = lsa[1]

            self.fenci.append({'title': data[i][it], 'para': data[i][ip]})
            self.cixing.append({'title': titlecixing, 'para': paracixing})

            if self.justsplit:
                continue

            para = copy.deepcopy(data[i][ip])
            title = copy.deepcopy(data[i][it])

            lget = self.highfrequencyfeatures(title, para, rawrawdatacopy[i][ip], rawrawdatacopy[i][it],
                                              dicParaWordtfidf)
            hffweight = lget[0]
            hfwl2 = lget[1]

            para = copy.deepcopy(data[i][ip])
            title = copy.deepcopy(data[i][it])

            # tfweight = titlefeatures(title, titlecixing, para, paracixing, [], hfwl2)

            tfweight = self.newtitlefeatures(title, para, dicParaWordtfidf, titlecixing, paracixing, hfwl2)

            para = copy.deepcopy(data[i][ip])
            title = copy.deepcopy(data[i][it])

            spfweight = self.sentencepositionfeature(para)

            tdfweight = self.tendfeature(para, trendworddic)

            ewfweight = self.emotionalwordfeatures(para, emoworddic)

            dicweightzong = {}
            for k in range(len(para)):  # 对上述各评价指标加权
                dicweightzong[k] = 0.3 * hffweight[k] + 0.20 * tfweight[k] + 0.05 * spfweight[k] + 0.1 * tdfweight[
                    k] + 0.325 * ewfweight[k]

            weightsed = sorted(dicweightzong.items(), key=lambda item: item[1], reverse=True)
            weightsed = weightsed[:self.impsentnum]  # 取前几句
            weightsed = sorted(weightsed, key=lambda item: item[0], reverse=False)

            improtantsentence = []
            for k in range(len(weightsed)):
                if outputrawpara:
                    improtantsentence.append(rawpara[weightsed[k][0]])
                else:
                    sentence = self.splitedtap[1][weightsed[k][0]]
                    sentenceproce=[]
                    for wo in sentence:
                        #print('fsafdsaf')
                        if wo.flag[0] not in delwordcixing:
                            if wo.word != '\n':
                                sentenceproce.append(wo.word)
                    improtantsentence.append(''.join(sentenceproce))


            '''
            print('重点句：')
            print(improtantsentence)
            '''
            self.improtantsentences.append(improtantsentence)

            if outputrawpara:
                rawrawdatacopy[i].append(''.join(improtantsentence))
            if save:
                datasave.write(','.join(rawrawdatacopy[i]) + '\n')
            #print('now processing :', end='')
            #print(i)

        datasave.close()

