template = {
    '/r/RelatedTo': '和{}相关',
    '/r/FormOf': '的形式为{}',
    '/r/IsA': '是{}',
    '/r/PartOf': '是{}的一部分',
    '/r/HasA': '具有{}',
    '/r/UsedFor': '用来{}',
    '/r/CapableOf': '可以{}',
    '/r/AtLocation': '在{}',
    '/r/Causes': '导致{}',
    '/r/HasSubevent': ',接下来,{}',
    '/r/HasFirstSubevent': '，紧接着，{}',
    '/r/HasLastSubevent': '的最后一步是{}',
    '/r/HasPrerequisite': '的前提为{}',
    '/r/HasProperty': '具有{}的属性',
    '/r/MotivatedByGoal': '受到{}的驱动',
    '/r/ObstructedBy': '受到{}的影响',
    '/r/Desires': '想要{}',
    '/r/CreatedBy': '被{}创造',
    '/r/Synonym': '和{}同义',
    '/r/Antonym': '和{}反义',
    '/r/DistinctFrom': '和{}相区别',
    '/r/DerivedFrom': '由{}导致',
    '/r/SymbolOf': '象征着{}',
    '/r/DefinedAs': '定义为{}',
    '/r/MannerOf': '',
    '/r/LocatedNear': '和{}相邻',
    '/r/HasContext': '的背景是{}',
    '/r/SimilarTo': '和{}相似',
    '/r/EtymologicallyRelatedTo': '',
    '/r/EtymologicallyDerivedFrom': '',
    '/r/CausesDesire': '',
    '/r/MadeOf': '由{}制成',
    '/r/ReceivesAction': '',
    '/r/ExternalURL': ''
}


def strip(str):
    return str.split('/')[3]
