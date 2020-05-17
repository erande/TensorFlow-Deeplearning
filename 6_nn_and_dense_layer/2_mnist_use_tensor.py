import tensorflow as tf
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只打印error信息

(x, y), (x_test, y_test) = datasets.mnist.load_data()  # x: [60k, 28, 28], y: [60k]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # x:[0~255] -> [0~1_start]
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.  # x:[0~255] -> [0~1_start]
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)  # 一次取128张照片 => batch=128
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

# [b, 784] -> [b, 256] -> [b, 128] -> [b, 10]
# w shape: [dim_in, dim_out], b shape: [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 0.001  # lr = 1e-3_regression

for epoch in range(100):  # for every batch
    for step, (x, y) in enumerate(train_db):
        # x shape:[128, 28, 28], y shape:[128], need x shape:[b, 28*28]
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:  # tf.GradientTape默认只会跟踪tf.Variable类型
            # [b, 784] @ [784, 256] + [256]
            # h1 = x@w1 + b1
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            out = h2 @ w3 + b3

            # loss mse = mean(sum((y-out)^2))
            # out shape:[b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.reduce_mean(tf.square(y_onehot - out))

        # w = w - lr * w_grad
        # 更新w1, assign_sub在原w1的基础上进行更新,sub减法，数据类型不会变
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print('epoch:', epoch, ' step:', step, ' loss:', loss)

    # test / evaluation
    total_correct, total_number = 0, 0
    for step, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 28 * 28])

        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3  # out: [b, 10] is R

        prob = tf.nn.softmax(out, axis=1)  # prob: [b, 10] is [0, 1_start]
        pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)  # [b, 10] => [b] is predict data
        correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))

        total_correct += int(correct)
        total_number += x.shape[0]

    acc = total_correct / total_number
    print('test acc:', acc)
