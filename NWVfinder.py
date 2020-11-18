import pandas as pd
import json
from langconv import *
from template import *
import jieba.posseg
import jiagu


class NWVfinder:

    def __init__(self):
        print("NWVfinder: 正在载入知识图谱")
        self.data = pd.read_csv("DATA.csv", delimiter='\t')
        self.data.columns = ['uri', 'relation', 'start', 'end', 'json']
        self.data = self.data[
            self.data['start'].apply(lambda row: row.find('zh') > 0) &
            self.data['end'].apply(lambda row: row.find('zh') > 0)
            ]
        self.data.index = range(self.data.shape[0])
        weights = self.data['json'].apply(lambda row: json.loads(row)['weight'])
        self.data.pop('json')
        self.data.insert(4, 'weights', weights)
        print("NWVfinder: 知识图谱载入完成")

    def __cht_to_chs(self, line):
        line = Converter('zh-hans').convert(line)
        line.encode('utf-8')
        return line

    def __chs_to_cht(self, line):
        line = Converter('zh-hant').convert(line)
        line.encode('utf-8')
        return line

    def __search(self, words, n=50):
        result = self.data[self.data['start'].str.contains(self.__chs_to_cht(words), regex=True)]
        topK_result = result.sort_values("weights", ascending=False).head(n)
        return topK_result

    def searchNWVector(self, words):
        if not isinstance(words,list):
            word = [words]

        ret = ''
        for word in words:
            topK_result = self.__search(word)
            sentence = ''
            c = 0
            for i in topK_result.index[:3]:
                i = topK_result.loc[i]

                if len(template[i['relation']]) > 0 and self.__cht_to_chs((i['start']).split('/')[3]) == word:
                    if(c==0):
                        c+=1
                        #print(c)
                        #print(self.__cht_to_chs((i['start']).split('/')[3]))
                        sentence+=self.__cht_to_chs((i['start']).split('/')[3])


                    sentence+=self.__cht_to_chs(template[i['relation']].format((i['end']).split('/')[3]))
            ret+=sentence
        return ret
    '''
    def searchSentence(self, sentence):
        words = jiagu.seg(sentence)  # 分词
        #print(words)
        word_list=[]
        ner = jiagu.ner(words)  # 命名实体识别
        #print(ner)
        i = 0
        while i < len(ner):
            if ner[i][0] == 'B':
                word_list.append(words[i])
                i += 1
            else:
                i += 1
        word_list=list(set(word_list))
        return [word_list, self.searchNWVector(word_list)]
    '''
    def searchSentence(self, sentence):
        #w5 = jieba.posseg.cut(sentence)
        #words = [item.word for item in w5 if item.flag in ['n', 'nt']]
        words=jieba.analyse.extract_tags(sentence,allowPOS=('n','nt'),topK=3,withWeight=False)
        return [words, self.searchNWVector(words)]
