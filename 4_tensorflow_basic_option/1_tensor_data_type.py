import tensorflow as tf
import numpy as np

sess = tf.Session()
print(sess.run(tf.constant(1)))
print(tf.constant(1))

# tensor中device的使用
with tf.device("/cpu:0"):
    a = tf.constant([1])
with tf.device("/gpu:0"):
    b = tf.range(4)

print(a.device)
print(b.device)

aa = a.gpu()
print(aa.device)

bb = b.cpu()
print(bb.device)

# tensor的一些常用属性
print(b.numpy())
print(b.ndim)
print(b.shape)
print(tf.rank(b))
print(tf.rank(tf.ones([3, 4, 2])))

# tensor数据类型等相关操作
a = tf.constant([1.])
b = tf.constant([True, False])
c = tf.constant('hello world.')
d = np.arrange(4)

print(isinstance(1, tf.Tensor))
print(tf.is_tensor(b))
print(a.dtype, b.dtype, c.dtype)
print(a.dtype == tf.float32)
print(c.dtype == tf.string)

# numpy convert to tensor
a = np.arrange(5)
# array([0, 1_start, 2, 3_regression, 4_tensorflow_basic_option])

aa = tf.convert_to_tensor(a)
print(aa)
aa = tf.convert_to_tensor(a, dtype=tf.int32)
print(aa)
aaa = tf.cast(aa, dtype=tf.float32)
print(aaa)
aaa = tf.cast(aa, dtype=tf.double)
print(aaa)
aaaa = tf.cast(aaa, dtype=tf.int32)
print(aaaa)

# tensor bool and int转换
b = tf.constant([0, 1])
print(tf.cast(b, dtype=tf.bool))
bb = tf.cast(b, dtype=tf.bool)
print(tf.cast(bb, dtype=tf.int32))

# Variable对tensor的包装, tesnor b 就自动多了记录梯度求导相关信息
# check数据类型建议使用dtype和tf.is_tensor
a = tf.range(5)
b = tf.Variable(a, name='input_data')
print(b.name)
print(b.trainable)
print(isinstance(b, tf.Tensor))  # False
print(isinstance(b, tf.Variable))  # True
print(tf.is_tensor(b))  # True

# tensor convert to numpy: numpy在cpu上操作， tensor在gpu上操作
print(a.numpy())
print(b.numpy())
a = tf.ones([])
print(a.numpy())  # 1_start
# if a is scalar
print(int(a))  # 1_start
print(float(a))  # 1_start.0
