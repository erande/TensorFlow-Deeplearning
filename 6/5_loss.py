import tensorflow as tf

# # MSE: 平方差累计和 / N
# # l2-norm: 平方差累计和开根
# y = tf.constant([1, 2, 3, 0, 2])
# y = tf.one_hot(y, depth=4)
# y = tf.cast(y, dtype=tf.float32)
# out = tf.random.normal([5, 4])
# loss1 = tf.reduce_mean(tf.square(y - out))
# loss2 = tf.square(tf.norm(y - out)) / (5 * 4)
# loss3 = tf.reduce_mean(tf.losses.MSE(y, out))
# # loss1 = loss2 = loss3

sess = tf.Session()

# Entropy: 衡量某一个分布的平稳性，entropy越小越certainty
a = tf.fill([4], 0.25)
entropy = -tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.))
print(sess.run(entropy))  # entropy = 2.0

a = tf.constant([0.01, 0.01, 0.01, 0.97])
entropy = -tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.))
print(sess.run(entropy))  # entropy = 0.24194068

# Cross Entropy: 两个分布之间，例如预测label和实际label两个分布


