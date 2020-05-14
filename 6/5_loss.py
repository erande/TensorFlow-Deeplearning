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
# 二分类交叉熵损失函数 = -(y * log(y') + (1 - y) * log(1 - y'))
# 多分类eg: y = [1 0 0 0 0], y' = [0.4 0.3 0.05 0.05 0.2]
# Cross Entropy Loss = -(1log0.4 + 0log0.3 + 0log0.05 + 0log0.05 + 0log0.2) = -log0.4

# tf.losses.categorical_crossentropy
loss = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25])  # 1.3862944
loss = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.1, 0.8, 0])  # 2.3978953
loss = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.7, 0.1, 0.1])  # 0.35667497
loss = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.01, 0.97, 0.01, 0.01])  # 0.030459179

# tf.losses.binary_crossentropy
loss = tf.losses.binary_crossentropy([1], [0.1])  # 2.3025842

# why not use MSE?
# when sigmoid + MSE => gradient vanish => converge slower


# numerical stability
x = tf.random.normal([1, 784])
w = tf.random.normal([784, 2])
b = tf.zeros([2])

# loss1 = loss2
# 但是loss1的计算方式使用了from_logits=True，这样使得不会发生数值不稳定现象
# 而loss2的计算方式，由于人为softmax再categorical_crossentropy，可能会导致数值不稳定
logits = x @ w + b
loss1 = tf.losses.categorical_crossentropy([0, 1], logits, from_logits=True)

prob = tf.math.softmax(logits, axis=1)
loss2 = tf.losses.categorical_crossentropy([0, 1], prob)
