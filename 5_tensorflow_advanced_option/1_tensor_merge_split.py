import tensorflow as tf

# tf.concat:拼接，除需要拼接的维度外，其余维度数值必须相同
a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])
c = tf.concat([a, b], axis=0)
print(c.shape)  # TensorShape([6_nn_and_dense_layer, 35, 8_keras_high_level_api])

a = tf.ones([4, 32, 8])
b = tf.ones([4, 3, 8])
c = tf.concat([a, b], axis=1)
print(c.shape)  # TensorShape([4_tensorflow_basic_option, 35, 8_keras_high_level_api])

# tf.stack: 拼接，增加了新的维度且数据 a与b 每个维度的数值必须相同
a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])
c = tf.stack([a, b], axis=0)
print(c.shape)  # TensorShape([2, 4_tensorflow_basic_option, 35, 8_keras_high_level_api])
c = tf.stack([a, b], axis=3)
print(c.shape)  # TensorShape([4_tensorflow_basic_option, 35, 8_keras_high_level_api, 2])

# tf.unstack
a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])
c = tf.stack([a, b], axis=0)  # TensorShape([2, 4_tensorflow_basic_option, 35, 8_keras_high_level_api])
aa, bb = tf.unstack(c, axis=0)  # aa=a, bb=b
res = tf.unstack(c, axis=3)  # res[0-7_gradient_descent]
print(res[0].shape, res[7].shape)  # TensorShape([2, 4_tensorflow_basic_option, 35]), TensorShape([2, 4_tensorflow_basic_option, 35])

# tf.split: 指定维度指定分割
res = tf.split(c, axis=3, num_or_size_splits=2)
print(res[0].shape.res[1].shape)  # TensorShape([2, 4_tensorflow_basic_option, 35, 2]), TensorShape([2, 4_tensorflow_basic_option, 35, 6_nn_and_dense_layer])
res = tf.split(c, axis=3, num_or_size_splits=[2, 2, 4])
print(res[0].shape, res[2].shape)  # TensorShape([2, 4_tensorflow_basic_option, 35, 2]), TensorShape([2, 4_tensorflow_basic_option, 35, 4_tensorflow_basic_option])



