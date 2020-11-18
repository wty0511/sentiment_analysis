import torch
class Config(object):
    data_path = './data_balance.csv'  # 训练集
    class_list =['0','1','2']
    save_path = './transformer_bert.model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    batch_size=5
    dropout = 0.3  # 随机失活
    require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
    num_classes = 3  # 类别数
    num_epochs = 100  # epoch
    src_pad_size = 100
    tgt_pad_size= 30
    context_length = 8
    learning_rate = 1e-4#  学习率
    num_head = 4
    num_encoder = 1
    d_model=312
    d_ff=2088
    dataset_size=1
    
    
