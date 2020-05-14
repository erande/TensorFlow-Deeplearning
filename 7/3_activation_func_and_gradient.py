import tensorflow as tf

# activation functions
# eg: 当人体感受到轻微疼痛时，不会作出反应，但当感受到剧烈疼痛时，人体就会做出反应，例如尖叫，缩手
# 即激活函数类似，达到某个阈值之后才会有相应的输出

# activation func1: y = 1 if x > 0 else 0 => 不可导

# tf.sigmoid
# sigmoid / logistic: f(x) = 1 / (1 + e^-x) => (0, 1)
# note: (1 / x)' = -1 / x^2, f' = f(1-f)
# df_dx = f(x) - f(x)^2
a = tf.linspace(-10., 10., 10)
with tf.GradientTape() as tape:
    tape.watch(a)
    y = tf.sigmoid(a)
grads = tape.gradient(y, [a])

# tf.tanh
# Tanh: f(x) = tanh(x) = 2 * sigmoid(2x) - 1 => (-1, 1)
# df_dx = 1 - f(x)^2
a = tf.linspace(-5., 5., 10)
print(tf.tanh(a))

# tf.nn.relu / tf.nn.leaky_relu
# ReLU(Rectified Linear Unit): y = 0 if x < 0 else x
# leaky_relu: y = 0 if x < 0 else k*x, k is small value
a = tf.linspace(-1., 1., 10)
print(tf.nn.relu(a))
print(tf.nn.leaky_relu(a))
