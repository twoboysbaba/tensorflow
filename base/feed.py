import tensorflow as tf
tf.compat.v1.disable_eager_execution()

input1 = tf.compat.v1.placeholder(tf.float32)
input2 = tf.compat.v1.placeholder(tf.float32)
output = tf.compat.v1.multiply(input1, input2)

with tf.compat.v1.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# 输出:
# [array([ 14.], dtype=float32)]