import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images,test_images = train_images/255,test_images/255

print(train_images.shape)

def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=[28,28]),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    return model
model = build_model()

model.fit(train_images,train_labels,epochs=10)
test_loss, test_acc = model.evaluate(test_images,test_labels)
print(test_loss)
print(test_acc)