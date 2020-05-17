import tensorflow as tf

# vector norm: 向量范数
a = tf.ones([2, 2])
print(tf.norm(a))  # 二范数，所有元素平方和开根号
print(tf.sqrt(tf.reduce_sum(tf.square(a))))  # 结果与上面方式相同

# axis=0代表按列计算，axis=1代表按行计算
print(tf.norm(a, ord=2, axis=1))  # ord=2代表二范数，axis代表指定维度的数据求范数
print(tf.norm(a, ord=1))  # ord=1代表一范数，所有元素绝对值之和
print(tf.norm(a, ord=1, axis=0))
print(tf.norm(a, ord=1, axis=1))

# reduce_mean/reduce_max/reduce_min/reduce_sum
a = tf.random.normal([4, 10])
print(tf.reduce_min(a))  # shape=()
print(tf.reduce_max(a))  # shape=()
print(tf.reduce_mean(a))  # shape=(), 求均值

print(tf.reduce_min(a, axis=1))  # shape=(4_tensorflow_basic_option,), 求每一行的最小值
print(tf.reduce_max(a, axis=1))  # shape=(4_tensorflow_basic_option,), 求每一行的最大值
print(tf.reduce_mean(a, axis=1))  # shape=(4_tensorflow_basic_option,), 求每一行的均值

# argmax/argmin: 最大/小值的index
print(tf.argmax(a))  # axis默认为0， shape=(10,)
print(tf.argmax(a, axis=1))  # shape=(4_tensorflow_basic_option,)

# tf.equal
a = tf.constant([1, 2, 3, 4, 5])
b = tf.range(5)
print(tf.equal(a, b))  # False, False, False, False, False
res = tf.equal(a, b)
print(tf.reduce_sum(tf.cast(res, dtype=tf.int32)))  # bool值转化为0，1数值再进行sum，可统计相同元素的个数

a = tf.constant([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])
pred = tf.cast(tf.argmax(a, axis=1), dtype=tf.int32)
y = tf.constant([2, 1])
correct = tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.int32)) / 2

# tf.unique: 去除重复元素
a = tf.constant([4, 2, 2, 4, 3])
print(tf.unique(a))
# shape=(3_regression,), unique numpy=array([4_tensorflow_basic_option, 2, 3_regression]), index numpy=array([0, 1_start, 1_start, 0, 2])
# 还原： tf.gather([4_tensorflow_basic_option, 2, 3_regression], [0, 1_start, 1_start, 0, 2])

