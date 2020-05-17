import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_val, y_val) = datasets.mnist.load_data()
xs = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)

model = tf.keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
        # get w1_grad, w2_grad, b1_grad, b2_grad
        grads = tape.gradient(loss, model.trainable_variables)
        # update: w1, w2, b1, b2, w3, b3
        # w1 = w1 - lr * w1_grad, b1 = b1 - lr * b1_grad
        # w2 = w2 - lr * w2_grad, b2 = b2 - lr * b2_grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print('epoch:', epoch, ' step:', step, ' loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
