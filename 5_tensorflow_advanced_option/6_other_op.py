import tensorflow as tf
import numpy as np

# tf.where(tensor)
# eg: 3x3的bool矩阵，where(bool_tensor) 返回True所在的index
a = tf.random.normal([3, 3])
mask = a > 0  # if value > 0 return True else False
b = tf.boolean_mask(a, mask)  # get data where mask data is True
# b = tf.boolean_mask(a, a > 0)
indices = tf.where(mask)  # get index where mask data is True
c = tf.gather_nd(a, indices)  # get data according to indices
# c = tf.gather_nd(a, tf.where(mask))
print(b, c)  # b = c

# tf.where(cond, A, B)
# cond dim = A dim = B dim
# get A's data according cond's True data index
# get B's data according cond's False data index

# tf.scatter_nd(indices, updates, shape)
# according to 'indices' update 'shape', data come from 'updates'
# 'shape' data value must = 0
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])  # default value = 0
print(tf.scatter_nd(indices, updates, shape))
# [0, 11, 0, 10, 9_over_fitting, 0, 12, 0], 第4个位置为9，第3个位置为10，第1个位置为11，第7个位置为12


# tf.meshgrid: get points

# use numpy
points = []
for y in np.linspace(-2, 2, 5):
    for x in np.linspace(-2, 2, 5):
        points.append([x, y])
points = np.array(points)

# use tensor
y = tf.linspace(-2., 2, 5)
x = tf.linspace(-2., 2, 5)
points_x, points_y = tf.meshgrid(x, y)  # shape=([5_tensorflow_advanced_option, 5_tensorflow_advanced_option])
# points_x: [-2, -1_start, 0, 1_start, 2] 重复5行
# points_y：[-2, -1_start, 0, 1_start, 2].T 重复5列
points = tf.stack([points_x, points_y], axis=2)  # shape=([5_tensorflow_advanced_option, 5_tensorflow_advanced_option, 2]), 5个5x2的矩阵
print(points)
