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
model.summary()
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_db, epochs=3, validation_data=test_db, validation_freq=2)
model.evaluate(test_db)

# model.save_weights('filename'): 此种方法之后重新加载model时，会导致运行结果有偏差
# model.save('filename'): 此种方法会保存整个模型，重新加载后无任何偏差
model.save_weights('weights.ckpt')  # save model's weights: [w1, b1, w2, b2, ... ]
model.save('model.h5')
del model

# load model.save_weights('filename'): network structure must = model structure
network = tf.keras.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.categorical_crossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('weights.ckpt')
network.evaluate(test_db)  # 判定模型的加载是否成功

# load model.save('filename')
network2 = tf.keras.models.load_model('model.h5')
network2.evaluate(test_db)


# tf.saved_model.save(m, 'filepath'): 保存为标准的可以给其它语言使用的模型
tf.saved_model.save(model, '/tmp/saved_model/')
imported = tf.saved_model.load('/tmp/saved_model/')
f = imported.signatures['serving_default']
print(f(x=tf.noes([1, 28, 28, 3])))  # forward for loaded model