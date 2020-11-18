import pandas as pd
from sklearn.utils import shuffle
data=pd.read_csv('./data_balance.csv')
data=shuffle(data)
tran=data[:int(len(data)*0.7)]
test=data[int(len(data)*0.7):len(data)]
tran.to_csv('./train.csv',index=None,header=None)
test.to_csv('./test.csv',index=None,header=None)
