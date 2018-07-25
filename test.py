import tensorflow as tf

hello = tf.constant('hello World')
sess = tf.Session()
print(sess.run(hello))

