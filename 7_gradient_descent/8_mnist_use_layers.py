import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
lr = 0.001

(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

db_iter = iter(train_db)
sample = next(db_iter)
print('batch: ', sample[0].shape, sample[1].shape)

model = tf.keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)])
model.build(input_shape=[None, 28 * 28])
model.summary()  # print net structure and parameters
optimizer = optimizers.Adam(lr=lr)


def main():
    for epoch in range(30):

        # train
        for step, (x, y) in enumerate(train_db):
            x = tf.reshape(x, (-1, 28 * 28))
            with tf.GradientTape() as tape:
                logits = model(x)  # forward
                y = tf.one_hot(y, depth=10)
                # loss_mse = tf.reduce_mean(tf.losses.MSE(y, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))
            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  # backward
            # if step % 100 == 0:
            #     print('epoch:', epoch, ' step:', step, ' loss_mse:', loss_mse.numpy(), ' loss_ce:', loss_ce.numpy())

        # test / evaluation
        total_correct, total_number = 0, 0
        for step, (x, y) in enumerate(test_db):
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
            total_correct += int(correct)
            total_number += x.shape[0]
        acc = total_correct / total_number
        print('test acc:', acc)


if __name__ == '__main__':
    main()
