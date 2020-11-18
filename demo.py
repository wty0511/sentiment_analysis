from NWVfinder import *

finder = NWVfinder()

# 通过实体列表构造addSentence
# @param: 实体列表
# @return: 实体列表对应的addSentence
addSentence = finder.searchNWVector(["火车", "飞机", "轮船"])
print(addSentence)

# 通过句子构造addSentence
# @param: 句子
# @return: [句子中的实体列表, 实体列表对应的addSentence]
[wordLIst, addSentence] = finder.searchSentence('据悉，上半年江苏共查处违反八项规定问题起，处理人，其中党纪政纪处分人。对于备受关注的公车改革，通报透露，公车改革总体方案、省级机关实施方案已获中央公车改革领导小组批复同意。')
print(wordLIst)
print(addSentence)
