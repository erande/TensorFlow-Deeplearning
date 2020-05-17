import tensorflow as tf

# element-wise: +-*/
# 1_start,2维数据的加减乘除，//相除取整， %相除取余
# tf.math.log = td.exp, 自然底数为e
# log 2底 8_keras_high_level_api
print(tf.math.log(8.) / tf.math.log(2.))
b = tf.fill([2, 2], 2.)
print(tf.pow(b, 3))  # b中每个元素求立方
print(b ** 2)  # b中每个元素求平方 = tf.pow(b, 2)
print(tf.sqrt(b))  # b中每个元素开方

# matrix-wise: @ = matmul 矩阵相乘
# [2, 3_regression, 4_tensorflow_basic_option] @ [2, 4_tensorflow_basic_option, 5_tensorflow_advanced_option] -> [2, 3_regression, 5_tensorflow_advanced_option]

# dim-wise: reduce_mean/reduce_max/reduce_min/reduce_sum
# 对某一个维度进行的计算
