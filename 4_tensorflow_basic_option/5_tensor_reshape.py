import tensorflow as tf

# tf.reshape
a = tf.random.normal([4, 28, 28, 3])
print(tf.reshape(a, [4, 28 * 28, 3]).shape)  # TensorShape=([4_tensorflow_basic_option, 728, 3_regression])
print(tf.reshape(a, [4, -1, 3]).shape)  # TensorShape=([4_tensorflow_basic_option, 728, 3_regression])
print(tf.reshape(a, [4, 784 * 3]).shape)  # TensorShape=([4_tensorflow_basic_option, 2352])
print(tf.reshape(a, [4, -1]).shape)  # TensorShape=([4_tensorflow_basic_option, 2352])

# tf.transpose, perm可指定转置哪些维度的数据
a = tf.random.normal([4, 3, 2, 1])  # TensorShape=([4_tensorflow_basic_option, 3_regression, 2, 1_start])
print(tf.transpose(a).shape)  # TensorShape=([1_start, 2, 3_regression, 4_tensorflow_basic_option])
print(tf.transpose(a, perm=[0, 1, 3, 2]).shape)  # TensorShape=([4_tensorflow_basic_option, 3_regression, 1_start, 2])

# expand dim
# example, a:[classes, students, classes] -> [4_tensorflow_basic_option, 35, 8_keras_high_level_api]
# add school dim
a = tf.random.normal([4, 25, 8])
print(tf.expand_dims(a, axis=0).shape)  # TensorShape=([1_start, 4_tensorflow_basic_option, 25, 8_keras_high_level_api])
print(tf.expand_dims(a, axis=1).shape)  # TensorShape=([4_tensorflow_basic_option, 1_start, 25, 8_keras_high_level_api])
print(tf.expand_dims(a, axis=3).shape)  # TensorShape=([4_tensorflow_basic_option, 25, 8_keras_high_level_api, 1_start])
print(tf.expand_dims(a, axis=-1).shape)  # TensorShape=([4_tensorflow_basic_option, 25, 8_keras_high_level_api, 1_start])
print(tf.expand_dims(a, axis=-4).shape)  # TensorShape=([1_start, 4_tensorflow_basic_option, 25, 8_keras_high_level_api])

# squeeze dim
a = tf.random.normal([1, 4, 25, 8])
print(tf.squeeze(a).shape)  # TensorShape=([4_tensorflow_basic_option, 25, 8_keras_high_level_api])
print(tf.squeeze(tf.zeros([1, 2, 1, 1, 3])).shape)  # TensorShape=([2, 3_regression])
a = tf.zeros([1, 2, 1, 3])
print(tf.squeeze(a, axis=0))  # TensorShape=([2, 1_start, 3_regression])
print(tf.squeeze(a, axis=2))  # TensorShape=([1_start, 2, 3_regression])
print(tf.squeeze(a, axis=-2))  # TensorShape=([1_start, 2, 3_regression])
print(tf.squeeze(a, axis=-4))  # TensorShape=([2, 1_start, 3_regression])
