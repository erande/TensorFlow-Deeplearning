import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1  # [0, 255] = [-1_start, 1_start]
    y = tf.squeeze(y)  # [b, 1_start, 10] => [b, 10]
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
# ([50000, 32, 32, 3_regression], [50000, 1_start, 10]), ([10000, 32, 32, 3_regression], [10000, 1_start, 10])
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)


class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [input_dim, output_dim])
        self.bias = self.add_variable('b', [output_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(32 * 32 * 3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        y = tf.nn.relu(self.fc1(inputs))
        y = tf.nn.relu(self.fc2(y))
        y = tf.nn.relu(self.fc3(y))
        y = tf.nn.relu(self.fc4(y))
        y = self.fc5(y)
        return y

model = MyModel()
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)
model.evaluate(test_db)
model.save_weights('weights.ckpt')
print('saved to weights.ckpt')
del model

model = MyModel()
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])
model.load_weights('weights.ckpt')
print('loaded weights from file')
model.evaluate(test_db)