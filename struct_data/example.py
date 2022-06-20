from cProfile import label
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv('/Users/bytedance/Downloads/Rexam-main/heart.csv')
dataframe.fillna(0)
print(dataframe.head())
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#使用tf.data包装dataframe，使它成为可以直接使用的模型训练数据
# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  dataframe = dataframe.astype('float64')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
batch_size = 5
print(train.shape)


train_ds = df_to_dataset(train, batch_size=batch_size)
#val_ds = df_to_dataset(val,batch_size=batch_size)
#test_ds = df_to_dataset(test,batch_size=batch_size)

for feature_batch,label_batch in train_ds.take(1):
    print("feature:", list(feature_batch.keys()))
    print("a batch of ages:",feature_batch['age'])
    print("a batch of targets:", label_batch)


