import tensorflow as tf

# tf.keras.datasets: tensorflow自带的一些小型数据集加载

# MNIST: 手写数字识别小型数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train, y_train, x_test, y_test => numpy data
print(x_train.shape, y_train.shape)  # x.shape=(60000, 28, 28), y.shape=(60000,)
print(x_train.min(), x_train.max(), x_train.mean())  # (0, 255, 33.318...)
print(x_test.shape, y_test.shape)  # x.shape=(10000, 28, 28), y.shape=(10000,)
y_onehot = tf.one_hot(y_train, depth=10)

# CIFAR10/100: 图片识别数据集，10代表有10种类别，100代表每种类别又分为10类，总共10*10=100类，数据集为同一个数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape)  # x.shape=(50000, 32, 32, 3_regression), y.shape=(50000, 1_start)
print(x_test.shape, y_test.shape)  # x.shape=(10000, 28, 28), y.shape=(10000, 1_start)
print(x_train.min(), x_train.max())  # (0, 255)

# tf.data.Dataset.from_tensor_slices: 转化为可简单使用可多线程操作的数据
db = tf.data.Dataset.from_tensor_slices(x_test)
print(next(iter(db)).shape)

db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 可接受多个数据
print(next(iter(db[0])).shape)
print(next(iter(db[1])).shape)

# .shuffle: 打乱数据
db = db.shuffle(10000)  # 10000给定区域范围的数据进行打乱


# .map: 数据预处理功能
def preprocess(x, y):
    """
    对于每一个数据，tensorflow中x一般使用float32，y使用one-hot编码
    因此需要预处理将数据类型转换
    :param x: one train or test data
    :param y: x's label
    :return: x, y
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


db2 = db.map(preprocess)  # 使得db中的每一个数据都进行预处理
x, y = next(iter(db2))
print(x.shape, y.shape)  # x.shape=(32, 32, 3_regression), y.shape=(1_start, 10)

# .batch
db3 = db2.batch(32)
x, y = next(iter(db3))
print(x.shape, y.shape)  # x.shape=(32, 32, 32, 3_regression), y.shape=(32, 1_start, 10)

db3 = tf.squeeze(db3)
x, y = next(iter(db3))
print(x.shape, y.shape)  # x.shape=(32, 32, 32, 3_regression), y.shape=(32, 10)

# .repeat(number)
db4 = db3.repeat()  # 迭代db4数据时，永远不会终止
db5 = db3.repeat(2)  # 迭代两次后退出

# example
def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    return x, y

def mnist_dataset():
    (x, y), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    y = tf.one_hot(y, depth=10)
    y_val = tf.one_hot(y_val, depth=10)

    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(prepare_mnist_features_and_labels)
    db = db.shuffle(60000).batch(100)

    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_val = db_val.map(prepare_mnist_features_and_labels)
    db_val = db_val.shuffle(60000).batch(100)

    return db, db_val

