import numpy as np
import tensorflow as tf

# numpy indexing: 四张rgb图片
a = tf.random.normal([4, 28, 28, 3])
print(a[1].shape)  # 取第二张图片，TensorShape([28, 28, 3])
print(a[1, 2].shape)  # 取第二张照片的第三行，TensorShape([28, 3])
print(a[1, 2, 3].shape)  # rgb: TensorShape([3])
print(a[1, 2, 3, 2].shape)  # rgb中b通道的一个具体值: TensorShape([])

# 切片':'
# [start:end] -> default value: [0:-1]，不包含end所在元素
# 0代表第一个元素，-1代表最后一个元素，-2代表到数第二个元素
a = tf.range(10)
print(a[-1:])  # a[-1:-1]，取最后一个元素
print(a[-2:])  # a[-2:-1]，取到数两个元素
print(a[:2])  # a[0:2]，取前2个元素
print(a[:-1])  # a[0:-1], 除最后一个元素都取
print(a[:])  # a[0:-1], 除最后一个元素都取

# 规则但不连续的索引方式
# [start:end:step]，step表示间隔，step=2代表隔一个索引一个
# 如果step<0，表示从start元素反向索引到end所在元素
# 如果step<-1，可直接理解为从start到end的元素进行逆序

# ...表示省略多个 ':,:'索引方式
# 前提是可以从写法推断出...代表哪些维度
print(a[0, :, :])
print(a[0, ...])

# Selective Indexing : 无规律索引，先给出索引号，根据索引号取元素
# tf.gather, tf.gather_nd, tf.boolean_mask

# tf.gather
# example data:[classes, students, subjects]
# 假设有四个班级， 每个班35人，每个学生有8门课程 [4, 35, 8]
print(tf.gather(a, axis=0, indices=[2, 3]).shape)  # TensorShape=([2, 35, 8])
# axis代表取哪个维度的元素，indices代表具体的索引号
# 以上表示在数据a中从班级维度进行索引，选出2，3号班级进行查看
print(tf.gather(a, axis=0, indices=[2, 1, 3, 0]).shape)  # TensorShape=([4, 35, 8])
print(tf.gather(a, axis=1, indices=[2, 3, 5, 7, 9, 16]))  # TensorShape=([4, 6, 8]), 表示抽取学号为indices的学生的相关数据
print(tf.gather(a, axis=2, indices=[2, 3, 7]).shape)  # TensorShape=([4, 35, 3]), 表示抽取课程为indices的相关信息

# tf.gather_nd:多个维度指定index
print(tf.gather_nd(a, [0]))  # a[0], shape=([35, 8])
print(tf.gather_nd(a, [0, 1]))  # a[0,1], shape=([8])
print(tf.gather_nd(a, [0, 1, 2]))  # a[0,1,2], shape=([])

print(tf.gather_nd(a, [[0]]))  # [a[0]], shape=([1])
print(tf.gather_nd(a, [[0, 1]]))  # [a[0,1]], shape=([1])
print(tf.gather_nd(a, [[0, 1, 2]]))  # [a[0,1,2]], shape=([1])

print(tf.gather_nd(a, [[0, 0], [1, 1]]))  # [a[0,1], s[1,2]], shape=([2, 8])
print(tf.gather_nd(a, [[0, 0, 0], [1, 1, 1], [2, 2, 2]]))  # shape=([3])
print(tf.gather_nd(a, [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]]))  # shape=([1, 3])

# tf.boolean_mask:根据True/False进行索引，选取对应位置为True的元素
print(tf.boolean_mask(a, mask=[True, True, False, False]))

