import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import io
import datetime
from matplotlib import pyplot as plt

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def image_grid(images):
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    return figure

batchsz = 128
(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz).repeat(10)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz, drop_remainder=True)

model = tf.keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)])
model.build(input_shape=[None, 28 * 28])
model.summary()  # print net structure and parameters
optimizer = optimizers.Adam(lr=1e-3)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'E:/PythonProjects/tensorflow_study/logs/' + current_time
summary_writer = tf.summary.create_file_writer(logdir)

sample_img = next(iter(train_db))[0]
sample_img = sample_img[0]
sample_img = tf.reshape(sample_img, [1, 28, 28, 1])
with summary_writer.as_default():
    tf.summary.image('training sample', sample_img, step=0)

for step, (x, y) in enumerate(train_db):
    # train
    x = tf.reshape(x, (-1, 28 * 28))
    with tf.GradientTape() as tape:
        logits = model(x)  # forward
        y = tf.one_hot(y, depth=10)
        # loss_mse = tf.reduce_mean(tf.losses.MSE(y, logits))
        loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))
    grads = tape.gradient(loss_ce, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # backward
    if step % 100 == 0:
        print(' step:', step, 'train-loss', float(loss_ce))
        with summary_writer.as_default():
            tf.summary.scalar('loss', float(loss_ce), step=step)

    # test / evaluation
    if step % 500 == 0:
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

        val_images = x[:25]
        val_images = tf.reshape(val_images, [-1, 29, 28, 1])
        with summary_writer.as_default():
            tf.summary.scalar('test-acc', float(acc), step=step)
            tf.summary.image('val-onebyone-images:', val_images, max_outputs=25, step=step)

            val_images = tf.reshape(val_images, [-1, 28, 28])
            figure = image_grid(val_images)
            tf.summary.image('val-images:', plot_to_image(figure), step=step)