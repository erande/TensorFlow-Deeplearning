import tensorflow as tf

# tf.reshape
a = tf.random.normal([4, 28, 28, 3])
print(tf.reshape(a, [4, 28 * 28, 3]).shape)  # TensorShape=([4, 728, 3])
print(tf.reshape(a, [4, -1, 3]).shape)  # TensorShape=([4, 728, 3])
print(tf.reshape(a, [4, 784 * 3]).shape)  # TensorShape=([4, 2352])
print(tf.reshape(a, [4, -1]).shape)  # TensorShape=([4, 2352])

# tf.transpose, perm可指定转置哪些维度的数据
a = tf.random.normal([4, 3, 2, 1])  # TensorShape=([4, 3, 2, 1])
print(tf.transpose(a).shape)  # TensorShape=([1, 2, 3, 4])
print(tf.transpose(a, perm=[0, 1, 3, 2]).shape)  # TensorShape=([4, 3, 1, 2])

# expand dim
# example, a:[classes, students, classes] -> [4, 35, 8]
# add school dim
a = tf.random.normal([4, 25, 8])
print(tf.expand_dims(a, axis=0).shape)  # TensorShape=([1, 4, 25, 8])
print(tf.expand_dims(a, axis=1).shape)  # TensorShape=([4, 1, 25, 8])
print(tf.expand_dims(a, axis=3).shape)  # TensorShape=([4, 25, 8, 1])
print(tf.expand_dims(a, axis=-1).shape)  # TensorShape=([4, 25, 8, 1])
print(tf.expand_dims(a, axis=-4).shape)  # TensorShape=([1, 4, 25, 8])

# squeeze dim
a = tf.random.normal([1, 4, 25, 8])
print(tf.squeeze(a).shape)  # TensorShape=([4, 25, 8])
print(tf.squeeze(tf.zeros([1, 2, 1, 1, 3])).shape)  # TensorShape=([2, 3])
a = tf.zeros([1, 2, 1, 3])
print(tf.squeeze(a, axis=0))  # TensorShape=([2, 1, 3])
print(tf.squeeze(a, axis=2))  # TensorShape=([1, 2, 3])
print(tf.squeeze(a, axis=-2))  # TensorShape=([1, 2, 3])
print(tf.squeeze(a, axis=-4))  # TensorShape=([2, 1, 3])
