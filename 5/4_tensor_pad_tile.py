import tensorflow as tf

# tf.pad: 数据填充，default value = 0
a = tf.reshape(tf.range(9), [3, 3])
print(tf.pad(a, [[0, 0], [0, 0]]))  # 无任何填充
print(tf.pad(a, [[1, 0], [0, 0]]))  # [1, 0]增加首行
print(tf.pad(a, [[1, 1], [0, 0]]))  # [1, 1]首尾各增加一行
print(tf.pad(a, [[1, 1], [1, 0]]))  # 首尾各增加一行，左边增加一列
print(tf.pad(a, [[1, 1], [1, 1]]))  # 首尾各增加一行，左右增加一列
# 多维数据，对应维度两端各padding [A, B] 行或列数据

# tf.tile: 和broadcast_to相同，tile扩充后的数据真是存在，broadcast_to虚拟存在
a = tf.reshape(tf.range(9), [3, 3])
print(tf.tile(a, [1, 2]))  # 第一个维度不变，第二个维度复制一遍，shape=(3, 6)
print(tf.tile(a, [2, 1]))  # shape=(6, 3)
print(tf.tile(a, [2, 2]))  # shape=(6, 6), 先复制小维度，即后面的维度
# 很多操作符自动支持broadcast优化，对这些操作符不需要人为tile

