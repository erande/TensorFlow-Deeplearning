import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


# 自定义的layer必须要继承 keras.layers.Layer 或 keras.Model

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
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        y = tf.nn.relu(self.fc1(inputs))
        y = tf.nn.relu(self.fc2(y))
        y = tf.nn.relu(self.fc3(y))
        y = tf.nn.relu(self.fc4(y))
        y = self.fc5(y)
        return y

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

network = MyModel()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.categorical_crossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_db, epochs=10, validation_data=test_db, validation_freq=2)
network.evaluate(test_db)