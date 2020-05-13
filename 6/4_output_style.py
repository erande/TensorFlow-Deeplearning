import tensorflow as tf

# output R
# linear regression
# naive classification with MSE

# image generation: rgb: [0, 255] => [0, 1] use sigmoid(out)

# tf.sigmoid(a): binary classification， if y > 0.5 => 1, else => 0
a = tf.linspace(-6, 6, 10)
print(tf.sigmoid(a))

# tf.nn.softmax(a): multi-classification
# 使每一类的数值在0到1（概率），且所有类别概率累计和=1
print(tf.nn.softmax(a))

# example
logits = tf.random.normal([1, 10], minval=-2, maxval=2)
prob = tf.nn.softmax(logits)
print(tf.reduce_sum(prob, axis=1))  # 约等于1

# tf.tanh(a): 使每一类的数值在-1到1

