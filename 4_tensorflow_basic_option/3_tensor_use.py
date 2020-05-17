import tensorflow as tf
import numpy as np

# scalar:[], 常用于loss，accuracy
out = tf.random.uniform([4, 10])
y = tf.range(4)  # label: [0, 1_start, 2, 3_regression]
y = tf.one_hot(y, depth=10)
loss = tf.keras.losses.mse(y, out)  # mse:误差平方和
loss = tf.reduce_mean(loss)

# vector:[m], 常用于bias
# example：8维变成10维的网络，f = x@w + b
net = tf.layers.Dense(10)  # 全连接层
net.build((4, 8))  # 4代表输入的数据有4个？
print(net.kernel)  # weight.shape=(8_keras_high_level_api, 10)，随机初始化值
print(net.bias)  # b.shape=(10,), 默认初始化数值全为0

# matrix:[m, n], 常用于input，weight
# example：四张照片，每张照片28x28打平成784，再进行数字识别，输出维度为10
x = tf.random.normal([4, 784])
net = tf.layers.Dense(10)
net.build((4, 784))
print(net(x).shape)  # TensorShape=([4_tensorflow_basic_option, 10])
print(net.kernel.shape)  # TensorShape=([784, 10])
print(net.bias.shape)  # TensorShape=([10])
