import numpy as np


def loss(w, b, points):
    error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (w * x + b)) ** 2
    return error / float(len(points))


def step_gradient(w_current, b_current, points, lr):
    w_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
    w_new = w_current - (lr * w_gradient)
    b_new = b_current - (lr * b_gradient)
    return [w_new, b_new]


def gradient_descent(points, w, b, lr, iterations):
    for i in range(iterations):
        w, b = step_gradient(w, b, np.array(points), lr)
    return [w, b]


def run():
    points = np.genfromtxt('data.csv', delimiter=",")
    lr = 0.0001
    w = 0
    b = 0
    iterations = 1000
    print("Staring gradient descent at w = {0}, b = {1}, error = {2}".format(w, b, loss(w, b, points)))
    print('Running...')
    [w, b] = gradient_descent(points, w, b, lr, iterations)
    print("After {0} iterations w = {1}, b = {2}, error = {3}".format(iterations, w, b, loss(w, b, points)))


if __name__ == '__main__':
    run()
