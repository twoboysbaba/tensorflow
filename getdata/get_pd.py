import tensorflow as tf
import pandas as pd

#csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
csv_file = '/Users/bytedance/Downloads/Datasets-master/Heart.csv'
df = pd.read_csv(csv_file)

print(df.head)
print(df.dtypes)

df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
print(df.head)
print(df.dtypes)