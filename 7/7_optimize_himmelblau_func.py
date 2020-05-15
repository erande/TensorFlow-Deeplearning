import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

sess = tf.Session()

# gradient descent
# [-4., 0.], [4., 0.], [1., 0.]
x = tf.constant([-4., 0.])  # input data
for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grads = tape.gradient(y, [x])[0]
    x = x - 0.01 * grads

    if step % 20 == 0:
        print('step{}: x = {}. f(x) = {}'.format(step, sess.run(x), sess.run(y)))
    # x = [-4., 0.]: step40: x = [-3.7793102 - 3.283186].f(x) = 0.0, 找到了一个最小值
