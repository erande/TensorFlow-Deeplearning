import tensorflow as tf
import matplotlib.pyplot as plt


def func(x):
    """
    z = sin(x) + sin(y)
    :param x: [b, 2]
    :return:
    """
    z = tf.math.sin(x[..., 0]) + tf.math.sin(x[..., 1])
    return z


x = tf.linspace(0., 2 * 3.14, 500)
y = tf.linspace(0., 2 * 3.14, 500)
point_x, point_y = tf.meshgrid(x, y)
points = tf.stack([point_x, point_y], axis=2)  # shape=(500, 500, 2)
z = func(points)  # shape=(500, 500)

# tf1.0
sess = tf.Session()
point_x = sess.run(point_x)
point_y = sess.run(point_y)
z = sess.run(z)

plt.figure('plot 2d func contour')
plt.contour(point_x, point_y, z)
plt.colorbar()
plt.show()
