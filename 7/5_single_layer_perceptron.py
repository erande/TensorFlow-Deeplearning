import tensorflow as tf


# y = x @ w + b
# loss = E = (y - p)^2 / 2
# dE_dwji = (p_i - y_i) * p_i * (1 - p_i) * x_j

# one-output node
x = tf.random.normal([1, 3])  # input node = 3
y = tf.constant([1])
w = tf.ones([3, 1])
b = tf.ones([1])
with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.sigmoid(x @ w + b)
    loss = tf.reduce_mean(tf.losses.MSE(y, prob))
grads = tape.gradient(loss, [w, b])
print(grads[0])
print(grads[1])

# multi-output nodes
x = tf.random.normal([2, 4])  # input node = 4
y = tf.one_hot(tf.constant([2, 0]), depth=3)
w = tf.ones([4, 3])
b = tf.ones([3])
with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x @ w + b, axis=1)
    loss = tf.reduce_mean(tf.losses.MSE(y, prob))
grads = tape.gradient(loss, [w, b])
