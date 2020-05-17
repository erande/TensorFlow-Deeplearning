import tensorflow as tf

# tf.broadcast_to
# if a dim != b dim
# we need to expand a and b to same dim
# broadcast_to function auto expand to same dim
# and original data a and b not change
# example: [students, scores]
# 需要给所有的学生加上5分
# [students=4_tensorflow_basic_option, scores] + 5_tensorflow_advanced_option
# 以上数据维度不同不能直接相加，因此需要broadcast为[5_tensorflow_advanced_option, 5_tensorflow_advanced_option，5_tensorflow_advanced_option，5_tensorflow_advanced_option].T
# tile也可以进行这种扩张，但tile方式扩张的数据会存储在内容，占用空间
# 而broadcast方式是在计算时自动扩张，计算完成后释放扩张的数据空间
# broadcast：b数据维度不够增加维度，数值都为1，再将数值1扩张为需要的数值
# 例如矩阵相乘，就要扩张为矩阵相乘需要的维度，矩阵加碱则需要扩张为a dim = b dim
# a = [4_tensorflow_basic_option, 32, 14, 14]
# b1 = [14, 14] -> [1_start, 1_start, 14, 14] -> [4_tensorflow_basic_option, 32, 14, 14] = a -> True
# b1 = [1_start, 32, 1_start, 1_start] -> [4_tensorflow_basic_option, 32, 14, 14] = a -> True
# b3 = [2, 32, 1_start, 1_start] -> [2, 32, 14, 14] != a -> False
a = tf.random.normal([4, 32, 32, 3])
b = tf.random.normal([3])
y = a + b  # 隐式broadcast
b = tf.broadcast_to(tf.random.normal([4, 1, 1, 1]), [4, 32, 32, 3])  # 显式broadcast
y = a + b

# tf.tile,此种方式进行的数据扩张会使得数据存储在内存中不会释放
a = tf.ones([3, 4])  # TensorShape=([3_regression, 4_tensorflow_basic_option])
a1 = tf.broadcast_to(a, [2, 3, 4])  # TensorShape=([2, 3_regression, 4_tensorflow_basic_option])
a2 = tf.expand_dims(a, axis=0)  # TensorShape=([1_start, 3_regression, 4_tensorflow_basic_option])
a2 = tf.tile(a2, [2, 1, 1])  # TensorShape=([2, 3_regression, 4_tensorflow_basic_option])

