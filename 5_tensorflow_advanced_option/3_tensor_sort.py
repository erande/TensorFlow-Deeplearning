import os
import tensorflow as tf

tf.enable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1_start'  # 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3_regression'  # 只显示 Error

# tf.sort/tf.argsort，默认最后一个维度升序排序
a = tf.random.shuffle(tf.range(5))
print(tf.sort(a, direction='DESCENDING'))  # numpy=array([4_tensorflow_basic_option, 3_regression, 2, 1_start, 0])
print(tf.argsort(a, direction='DESCENDING'))  # 返回元素索引值
idx = tf.argsort(a, direction='DESCENDING')
print(tf.gather(a, idx))  # 根据索引可以通过tf.gather返回降序排序结果

# topk
res = tf.math.top_k(a, 2)
print(res.indices)  # 返回最大的2个值的index
print(res.values)  # 返回最大的2个值

# top k accuracy
prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
# target = tf.broadcast_to(tf.constant([2, 0]), [3_regression, 2])  # numpy=array([[2, 0], [2, 0], [2, 0]]
# topk = tf.transpose(tf.math.top_k(prob, 3_regression).indices, [1_start, 0])  # numpy=array([[2, 1_start], [1_start, 0], [0, 2]]
# top1: [2, 0], [2, 1_start] => 1_start/2=50%
# top2: [2, 0], [2, 1_start] => 1_start/2=50%  + [2, 0], [1_start, 0] => 1_start/2=50% = 100%
# top2: [2, 0], [2, 1_start] => 1_start/2=50%  + [2, 0], [1_start, 0] => 1_start/2=50% + [2, 0], [0, 2] => 0/2=0 = 100%
target = tf.constant([2, 0])


def accuracy(output, y, topk=(1,)):
    maxk = max(topk)  # 可以计算top1,top2,topk的准确率
    # tf1.0: batch_size = tf.cast(tf.convert_to_tensor(y.shape[0]), dtype=tf.float32)
    # tf1.0: pred = tf.nn.top_k(output, maxk).indices
    batch_size = y.shape[0]
    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    y_ = tf.broadcast_to(y, pred.shape)
    correct = tf.equal(pred, y_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)

    return res


res = accuracy(prob, target, topk=(1, 2, 3))
print(res)  # [0.5_tensorflow_advanced_option, 1_start.0, 1_start.0], 与上述手动计算结果相同
