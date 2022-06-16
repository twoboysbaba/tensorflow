from pickletools import optimize
from tabnanny import verbose
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)
print(test_images.shape)

train_images = train_images/255
test_images = test_images/255

def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(10)]
    )
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

model = build_model()
#查看模型的基本架构
model.summary()
#通过checkpoint回调的方式保存训练期间的权重
import os
checkpoint_patch = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_patch)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,save_weights_only=True,verbose=1
)
model.fit(train_images, train_labels, epochs=10,callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_loss)
print(test_acc)

#从checkpoint加载后的model，验证是不是和直接训练的效果一致
model = build_model()
model.load_weights(checkpoint_dir)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("load weight from pointcheck!!!")
print(test_loss)
print(test_acc)


predictions = model.predict(test_images)
#print(predictions[0])

probability_model = tf.keras.Sequential(
    [model,
    tf.keras.layers.Softmax()]
)
predictions = probability_model.predict(test_images)
#print(predictions[0])

