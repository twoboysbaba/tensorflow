import tensorflow as tf
tf.compat.v1.disable_eager_execution()
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.compat.v1.multiply(input1, intermed)

with tf.compat.v1.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

#    "python.autoComplete.extraPaths": [
##        "/usr/bin/python3/site-packages"
#   ]