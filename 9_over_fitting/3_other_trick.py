import tensorflow as tf
from tensorflow.keras import datasets, layers, regularizers, Sequential, optimizers, metrics

# momentum: 动量or惯性
# 指的是每次迭代梯度更新不仅需要考虑当前梯度方向，还要考虑上一次的梯度方向
# w_t+1 = w_t - alpha * (beta * z_k + w_t_grad)
# z_k 可以等于 w_t-1, beta表示考虑过去的梯度方向的程度

optimizer = optimizers.SGD(lr=0.02, momentum=0.9)  # beta = 0.9
optimizer = optimizers.RMSprop(lr=0.02, momentum=0.9)
optimizer = optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999)  # Adam本身内置momentum

# learning rate decay
# 一般设置略大的lr，然后施行learning rate decay
# 使得一开始梯度快速下降，为了避免出现振荡，所以进行学习率衰减
optimizer = optimizers.SGD(lr=0.2)
for epoch in range(100):
    optimizer.lr = 0.2 * (100 - epoch) / 100  # when epoch=100, lr=0

# early stop
# training acc continue slowly increase, but test acc start decrease => over-fitting
# 什么时候stop？
# 当test acc保持不变或下降一段时间后
# 但是不能确保此后test acc还会不会上升
# this is deep learning's trick

# dropout
# 原本每一层的输出会直接作为下一层的输入
# dropout代表会删除一部分输出，剩下的才作为下一层的输入

# method 1
model = Sequential([layers.Dense(256, activation=tf.nn.relu),
                    layers.Dropout(0.5),  # 0.5 rate to drop
                    layers.Dense(128, activation=tf.nn.relu),
                    layers.Dropout(0.5),  # 0.5 rate to drop
                    layers.Dense(64, activation=tf.nn.relu),
                    layers.Dense(32, activation=tf.nn.relu),
                    layers.Dense(10)])
# for step, (x, y) in enumerate(db):
#     # training
#     with tf.GradientTape() as tape:
#         out = model(x, training=True)
#     # test
#     out = model(x, training=False)

# stochastic gradient descent: 随机梯度下降
# not random! 是符合某一个分布的
# 因为当数据量很大时，每次使用全部的数据来计算梯度时，比较消耗内存
# 32GB的显存需要80999元。。。。。。太贵了！
# 因此考虑使用部分数据来计算loss(loss用于梯度更新），减少内存消耗
# 每次取batch的数量来计算loss