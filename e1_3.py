import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = pathlib.Path(__file__).parent.resolve()
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
EXERCISE_DIR = os.path.join(IMAGES_DIR, 'e1.3')
if not os.path.exists(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)
if not os.path.exists(EXERCISE_DIR):
    os.mkdir(EXERCISE_DIR)

# plot f(x)
figure = plt.figure()
axis = plt.axes(projection='3d')

domain = np.linspace(-5, 5, 100)
x, y = np.meshgrid(domain, domain)
z = (x**2 - y - 1)**2 + (x - 2*y - 1)**2

axis.plot_surface(x, y, z)
plt.savefig(os.path.join(EXERCISE_DIR, 'fx.png'))

# plot gradient descent
rates = [0.01, 0.05]

for rate in rates:
    figure.clear()
    axis = plt.axes()

    points = [(1, 1)]

    for _ in range(10):
        x = points[-1][0] - rate * (4 * (points[-1][0] ** 3) - 4 *
                                    points[-1][0] * points[-1][1] - 2 * points[-1][0] - 4 *
                                    points[-1][1] - 2)
        y = points[-1][1] - rate * (-2 * (points[-1][0] **
                                    2) + 10 * points[-1][1] + 4 * points[-1][0] + 6)
        points.append((x, y))

    axis.plot(*zip(*points), marker='o')
    plt.text(
        *points[-1],
        f'({round(points[-1][0], 3)}, {round(points[-1][1], 3)})')
    plt.savefig(os.path.join(EXERCISE_DIR, f'r{rate}.png'))
