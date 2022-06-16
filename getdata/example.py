import tensorflow as tf
import matplotlib.pyplot as plt
import os

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32
#返回一个（features，label）对构建的tf.data.Dataset,其中features是一个词典{'feature_name':value}
train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp, batch_size,column_names=column_names,label_name=label_name,num_epochs=1)

features,labels = next(iter(train_dataset))
print(features)

plt.scatter(features['petal_length'],features['sepal_length'],c=labels)
plt.xlabel('petal_length')
plt.ylabel('sepal_length')
#plt.show()

#把特征词典打包为（batch_size,num_features）的单个数组
def pack_features_vector(features,labels):
    features = tf.stack(list(features.values()),axis=1)
    return features,labels
train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
print(features[:5])

#开始训练
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])
predictions = model(features)
print(predictions[:5])
tf.nn.softmax(predictions[:5])
#对每个类别执行tf.argmax运算可以得出预测的类别索引。
#不过，该模型尚未接受训练，因此这些预测并不理想
print("prediction:{}".format(tf.argmax(predictions,axis=1)))
print("labels:{}".format(labels))