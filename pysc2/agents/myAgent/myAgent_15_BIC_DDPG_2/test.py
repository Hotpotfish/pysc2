# import tensorflow as tf
#
# a = tf.constant([[1, 2, 3, 4, 5], [9, 2, 1, 1, 2]])
# b = tf.contrib.layers.embed_sequence(ids=a, vocab_size=100, embed_dim=1)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(b))
