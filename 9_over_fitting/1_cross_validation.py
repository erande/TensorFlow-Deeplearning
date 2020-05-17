import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

model = Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128

# 60000条data
(x, y), (test_x, test_y) = datasets.mnist.load_data()

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(50000).batch(batchsz)

# for train/test data splitting
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(50000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_db = test_db.map(preprocess).batch(batchsz)

model.fit(train_db, epochs=5, validation_data=test_db, validation_steps=2)

# for train/validation/test data splitting
train_x, val_x = tf.split(x, num_or_size_splits=[50000, 10000])
train_y, val_y = tf.split(y, num_or_size_splits=[50000, 10000])

train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_db = train_db.map(preprocess).shuffle(50000).batch(batchsz)
val_db = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_db = val_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_db = test_db.map(preprocess).batch(batchsz)

model.fit(train_db, epochs=5, validation_data=val_db, validation_steps=2)
model.evaluate(test_db)

# 每一次epoch需要的train 和 validation data 的划分
for epoch in range(10):
    idx = tf.random_shuffle(tf.range(60000))
    train_x, train_y = tf.gather(x, idx[:50000]), tf.gather(y, idx[:50000])
    val_x, val_y = tf.gather(x, idx[-10000:]), tf.gather(y, idx[-10000:])

    train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_db = train_db.map(preprocess).shuffle(50000).batch(batchsz)

    val_db = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_db = val_db.map(preprocess).shuffle(10000).batch(batchsz)

    # training ...
    # evaluation ...

# 或者以下的写法
model.fit(db, epochs=10, validation_split=0.1, validation_steps=2)  # validation_split=0.1 => train:val = 9:0

