from distutils.command.build import build
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values="?",comment='\t',sep=" ",skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

#数据预处理
dataset = dataset.dropna()
#onehot处理
origin = dataset.pop('Origin')
dataset['USA'] = (origin==1)*1.0
dataset['Europe'] = (origin==2)*1.0
dataset['Japan'] = (origin==3)*1.0

print(dataset.tail())

#拆分训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#提出数据中的label
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
#归一化
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
def norm(x):
    return (x-train_stats['mean'])/train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.RMSprop(0.001),
                    metrics=['mae','mse'])
    return model
model = build_model()

history = model.fit(normed_train_data, train_labels, epochs=1000, validation_split=0.2, verbose=0)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

#plot_history(history)
#添加earlystop
model = build_model()
history = model.fit(normed_train_data, train_labels, epochs=1000, validation_split=0.2, verbose=0,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
#plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("test set mean abs error:{:5.2f}".format(mae))