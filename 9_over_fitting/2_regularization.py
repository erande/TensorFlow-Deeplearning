import tensorflow as tf
from tensorflow.keras import datasets, layers, regularizers, Sequential, optimizers, metrics

# reduce over-fitting
# 1. more data
# 2. reduce the complexity of the model: shallow, regularization
# 3. dropout
# 4. data argumentation
# 5. early stop training

# regularization(other name 'weight decay')
# loss = mse_loss/... + 正则项（lambda * w_norm）
# 当最小化损失函数时，正则项也会被带入一起最小化
# 这使得正则项的范数会接近于0，如果w范数接近于0，那么w的每一个元素都接近于0
# 这样一来，模型的高次方系数约等于0，近似等于降低了模型复杂度，避免过拟合

# method 1
l2_model = Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation=tf.nn.relu),
    layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.sigmoid)])
l2_model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.categorical_crossentropy(from_logits=True),
              metrics=['accuracy'])

# method 2
model = Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.sigmoid)])
optimizer = optimizers.Adam(1e-3)

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

batchsz = 128
(x, y), (test_x, test_y) = datasets.mnist.load_data()
y_onehot = tf.one_hot(tf.cast(y, dtype=tf.int32), depth=10)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(50000).batch(batchsz)

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, from_logits=True))

        loss_regular = []
        for w in l2_model.trainable_variables:  # w, b
            loss_regular.append(tf.nn.l2_loss(w))
        loss_regular = tf.reduce_sum(tf.stack(loss_regular))

        loss = loss + 0.001 * loss_regular
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
