import tensorflow as tf
tf.compat.v1.disable_eager_execution()

x1 = tf.Variable(tf.compat.v1.random_normal(shape=[1],mean=-0.1,stddev=0.01))
x2 = -x1
loss = 10 - x1*x1 - x2*x2
train_op = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(10000):
        xs,loss_show = sess.run([x1,loss])
        if xs*xs+xs <= 0 :
            _= sess.run(train_op)
        print('step =%d,loss=%.8f,x1=%.8f'%(i,loss_show,xs))