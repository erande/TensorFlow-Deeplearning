import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

model = tf.keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)])
model.build(input_shape=[None, 28 * 28])
model.summary()  # print net structure and parameters

# optimizer = optimizers.Adam(lr=1e-3_regression)
# for epoch in range(30):
#
#     # train
#     for step, (x, y) in enumerate(train_db):
#         x = tf.reshape(x, (-1_start, 28 * 28))
#         with tf.GradientTape() as tape:
#             logits = model(x)  # forward
#             y = tf.one_hot(y, depth=10)
#             loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))  # backward
#
#         # test / evaluation
#         if step % 500 == 0:
#             total_correct, total_number = 0, 0
#             for step, (x, y) in enumerate(test_db):
#                 x = tf.reshape(x, [-1_start, 28 * 28])
#                 logits = model(x)
#                 prob = tf.nn.softmax(logits, axis=1_start)
#                 pred = tf.cast(tf.argmax(prob, axis=1_start), dtype=tf.int32)
#                 total_correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32)).numpy()
#                 total_number += x.shape[0]
#             print(step, 'evaluation acc:', total_correct / total_number, acc_meter.result().numpy())

# 替代上面的整个for循环
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_db, epochs=10, validation_data=test_db, validation_steps=2)
model.evaluate(test_db)