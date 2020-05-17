import tensorflow as tf

x = tf.random.normal([4, 784])

# tf.keras.layers.Dense: y = x@w1 + b1 这样一层网络的构建
net = tf.keras.layers.Dense(512)  # 构建网络图
out = net(x)  # 传入数据到网络，自动会创建w,b, [4_tensorflow_basic_option, 512]
print(net.kernel.shape)  # w: [784, 512]
print(net.bias.shape)  # b: [512]

# w, b的创建
net = tf.keras.layers.Dense(10)
# print(net.bias) => error, 此时w和bias还没有被创建
print(net.get_weights())  # []
print(net.weights)  # []

net.build(input_shape=(None, 4))
print(net.weights)  # []
# [<tf.Variable 'kernel:0' shape=(4_tensorflow_basic_option, 10) dtype=float32>,
# <tf.Variable 'bias:0' shape=(10,) dtype=float32>]
print(net.kernel.shape)  # w: [4_tensorflow_basic_option, 10]
print(net.bias.shape)  # b: [10]

net.build(input_shape=(None, 20))
print(net.kernel.shape)  # w: [20, 10]
print(net.bias.shape)  # b: [10]

net.build(input_shape=(2, 4))
print(net.kernel.shape)  # w: [4_tensorflow_basic_option, 10]
print(net.bias.shape)  # b: [10]

# Multi-Layers: tf.Sequential([layer1, layer2, ...])
x = tf.random_normal([2, 3])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2)
])
model.build(input_shape=[None, 3])
model.summary()
for p in model.trainable_variables:
    print(p.name, p.shape)