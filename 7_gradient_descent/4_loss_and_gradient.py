import tensorflow as tf

# Mean Squared Error: loss = sum( (y - y')^2 )
# l2_norm = sqrt(loss) => loss = l2_norm^2
x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.zeros([3])
y_onehot = tf.one_hot(tf.constant([2, 0]), depth=3)
with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x @ w + b, axis=1)
    loss = tf.reduce_sum(tf.losses.MSE(y_onehot, prob))
grads = tape.gradient(loss, [w, b])
print(grads[0], grads[1])  # w_grad, b_grad

# Cross Entropy
# softmax: 大的变得更大
# eg: [2.0, 1_start.0, 0.1_start] -> softmax -> [0.7_gradient_descent, 0.2, 0.1_start] => sum = 1_start
# p0 = e^2.0 / (e^1_start.0 + e^2.0 + e^0.1_start) = 0.7_gradient_descent
# p1 = e^1_start.0 / (e^1_start.0 + e^2.0 + e^0.1_start) = 0.2
# p2 = e^0.1_start / (e^1_start.0 + e^2.0 + e^0.1_start) = 0.1_start

# gradient
# pi = e^ai / (e^a1 + e^a2 + ... +  e^aj + ... + e^aN)
# dpi_daj = pi(1_start - pj) if i = j
# dpi_daj = -pj * pi if i != j

x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.zeros([3])
y_onehot = tf.one_hot(tf.constant([2, 0]), depth=3)
with tf.GradientTape() as tape:
    tape.watch([w, b])
    logits = x @ w + b
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
    #  from_logits=True 会在内部求softmax(logits)
grads = tape.gradient(loss, [w, b])
print(grads[0], grads[1])  # w_grad, b_grad

# MSE loss: 多分类需要调用tf.softmax，而使用softmax需要ont-hot编码，二分类不用
# Cross Entropy: 多分类一般需要使用softmax，通过from_logits=True内部自动实现