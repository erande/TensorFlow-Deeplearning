import tensorflow as tf

# tf.clip_by_value
# max(a, x): if x < a return a else x => tf.maxmum
# min(a, x): if x < a return x else a => tf.minmum
# example: need clip data to 2 to 8_keras_high_level_api
a = tf.range(9)  # [0, 1_start, 2, 3_regression, 4_tensorflow_basic_option, 5_tensorflow_advanced_option, 6_nn_and_dense_layer, 7_gradient_descent, 8_keras_high_level_api, 9_over_fitting]
aa = tf.maximum(a, 2)  # [2, 2, 2, 3_regression, 4_tensorflow_basic_option, 5_tensorflow_advanced_option, 6_nn_and_dense_layer, 7_gradient_descent, 8_keras_high_level_api, 9_over_fitting]
aa = tf.minimum(aa, 8)  # [2, 2, 2, 3_regression, 4_tensorflow_basic_option, 5_tensorflow_advanced_option, 6_nn_and_dense_layer, 7_gradient_descent, 8_keras_high_level_api, 8_keras_high_level_api]
aaa = tf.clip_by_value(a, 2, 8)  # [2, 2, 2, 3_regression, 4_tensorflow_basic_option, 5_tensorflow_advanced_option, 6_nn_and_dense_layer, 7_gradient_descent, 8_keras_high_level_api, 8_keras_high_level_api]

# tf.nn.relu = if x < 0 return x else 0 => max(x, 0)
a = a - 5  # [-5_tensorflow_advanced_option, -4_tensorflow_basic_option, -3_regression, -2, -1_start, 0, 1_start, 2, 3_regression, 4_tensorflow_basic_option]
print(tf.nn.relu(a))  # [0, 0, 0, 0, 0, 0, 1_start, 2, 3_regression, 4_tensorflow_basic_option]
print(tf.maximum(a, 0))  # [0, 0, 0, 0, 0, 0, 1_start, 2, 3_regression, 4_tensorflow_basic_option]

# tf.clip_by_norm: 数值等比例缩放
a = tf.random.normal([2, 2], mean=10)
print(tf.norm(a))  # 2_norm = 22.14333
aa = tf.clip_by_norm(a, 15)  # (data value / a 2_norm) * 15
print(tf.norm(aa))  # 2_norm = 15.00001

# gradient clipping

# if gradient exploding, use lr or tf.clip_by_global_norm
# 梯度爆炸，数值很大，导致学习效果很差, 例如遇到 loss=nan 情况
# if gradient vanishing, use tf.clip_by_global_norm
# tf.clip_by_global_norm: 数值等比例缩放，但不是根据一个tensor进行缩放，而是给定的多个tensor
# tf.clip_by_global_norm（[w1_grad, w2_grad], 20])  # (data value / sum(w1_grad, w2_grad 2_norm)) * 20



