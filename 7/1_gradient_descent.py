import tensorflow as tf

# gradient: 函数在每个方向上的导数（偏微分）组合成的一个向量或矩阵
# gradient方向代表函数值增大的方向, 而要求的是最小化loss
# 因此每次朝着gradient的反方向进行迭代更新 => w = w - lr * w_grad

# autograd
x = tf.constant(1.)
w = tf.constant(2.)
y = x * w

with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w
grad1 = tape.gradient(y, [w])  # [None]

with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w
grad2 = tape.gradient(y2, [w])  # 2.0
grad2 = tape.gradient(y2, [w])  # error

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w])
    y2 = x * w
grad2 = tape.gradient(y2, [w])  # 2.0
grad2 = tape.gradient(y2, [w])  # 2.0

# 二阶求导
b = tf.constant(2.)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * w + b
    dy_dw, dy_db = t2.gradient(y, [w, b])
d2y_d2w = t1.gradient(dy_dw, w)
