import tensorflow as tf
import numpy as np

# 从numpy创建tensor
a = tf.convert_to_tensor(np.ones([2, 3]))
print(tf.cast(a, dtype=tf.float32))

b = tf.convert_to_tensor(np.zeros([2, 3]))
print(tf.cast(b, dtype=tf.float32))

print(tf.convert_to_tensor([1, 2]))
print(tf.convert_to_tensor([1, 2.]))
print(tf.convert_to_tensor([[1], [2.]]))

# 直接新建tensor
print(tf.zeros([]))  # scalar 0
print(tf.zeros([1]))  # vector [0.]
print(tf.zeros([2, 2]))  # 2dim matrix
print(tf.zeros([2, 3, 3]))  # 3dim matrix

a = tf.zeros([2, 3, 3])
b = tf.zeros_like(a)
c = tf.zeros(a.shape)
print(a, b, c)  # a = b = c

# ones: 元素全为1
print(tf.ones(1))  # shape=(1_start,), numpy=array([1_start.])
print(tf.ones([]))  # shape=(), numpy=1_start.0
print(tf.ones([2]))  # shape=(2,), numpy=array([1_start., 1_start.])
print(tf.ones([2, 3]))  # shape=(2, 3_regression),
print(tf.ones_like(a))  # shape=(2, 3_regression, 3_regression)

# fill: 填充任意类型的值，且每次填充值全部相同
print(tf.fill([2, 2]), 0)
print(tf.fill([2, 2]), 0.)
print(tf.fill([2, 2]), 1)
print(tf.fill([2, 2]), 9)

# random.normal
print(tf.random.normal([2, 2]), mean=1, stddev=1)  # normal正态分布，mean均值，stddev方差
print(tf.random.normal([2, 2]))  # mean=0, stddev=1_start

print(tf.random.truncated_normal([2, 2], mean=0, stddev=1))
# 截断正态分布，在原正态分布上截取去一段
# 主要是由于梯度消失，例如sigmoid曲线两端，梯度非常平缓，称为Gradient Vanish

# random.uniform
print(tf.random.uniform([2, 2], minval=0, maxval=1))  # dtype=float32
print(tf.random.uniform([2, 2], minval=0, maxval=100))  # dtype=float32
print(tf.random.uniform([2, 2], minval=0, maxval=100, dtype=tf.int32))

# example: 随机打散数据
idx = tf.range(10)
idx = tf.random.shuffle(idx)
print(idx)

image = tf.random.normal([10, 784])
label = tf.random.uniform([10], maxval=10, dtype=tf.int32)
print(a, b)

image = tf.gather(image, idx)
label = tf.gather(label, idx)
print(a, b)

# tf.constant: 与tf.convert_to_tensor几乎完全类似
print(tf.constant(1))  # shape=(), numpy=1_start
print(tf.constant([1]))  # shape=(1_start,), numpy=array([1_start], dtype=int32)
print(tf.constant([1, 2.]))  # shape=(2,), numpy=array([1_start., 2.], dtype=float32)

