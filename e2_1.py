import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = pathlib.Path(__file__).parent.resolve()
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
EXERCISE_DIR = os.path.join(IMAGES_DIR, 'e2.1')
if not os.path.exists(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)
if not os.path.exists(EXERCISE_DIR):
    os.mkdir(EXERCISE_DIR)


def add_noise(x, y, total: int = 100, mean: float = 0.0, scale: float = 0.2):
    x_noise = []
    y_noise = []

    noise = np.random.normal(mean, scale, size=(total, 2))

    for i in range(total):
        k = np.random.randint(len(x))

        x_noise.append((x[k][0] + noise[i][0], x[k][1] + noise[i][1]))
        y_noise.append(y[k])

    return np.array(x_noise), np.array(y_noise)


def plot_clusters(x, y, labels):
    for label in labels:
        points = y == label
        plt.scatter(x[points, 0], x[points, 1], marker='o')
    plt.title('Noisy XOR dataset')
    plt.grid(True)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def intialize_parameters(w, w0):
    w = np.array(w)
    w0 = w0

    return w, w0


def get_activation_loss(x, y, w, w0):
    z = np.dot(w.T, x) + w0
    a = sigmoid(z)
    m = x.shape[0]

    cost = (1/m) * np.sum(-1 * (y * np.log(a) + (1 - y) * (np.log(1 - a))))
    return a, cost, z


def update_parameters(x, w, w0, a, y, lr):
    dw = np.sum(np.dot(x, (a-y).T))
    dw0 = np.sum(a - y)

    w = w - (lr*dw)
    w0 = w0 - (lr*dw0)

    return (w, w0)


def optimize_parameters(x, y, w, w0, lr, epoch):
    loss = []

    for _ in range(epoch):
        a, cost, _ = get_activation_loss(x, y, w, w0)
        w, w0 = update_parameters(x, w, w0, a, y, lr)
        loss.append(cost)

    return (w, w0, loss)


def predict(x, w, w0):
    z = np.dot(w.T, x) + w0
    a = sigmoid(z)
    return a


# Create XOR dataset
x = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y = np.array([0, 1, 1, 0])

# Train
sizes = [10, 50, 100]
configs = [
    {'lr': 0.01, 'epoch': 10},
    {'lr': 0.01, 'epoch': 100},
    {'lr': 0.001, 'epoch': 10},
    {'lr': 0.001, 'epoch': 100},
]

w, w0 = intialize_parameters([1.0, 1.0], 0.5)

for size in sizes:
    figure, axis = plt.subplots(2, 2, constrained_layout=True)
    positions = [[0, 0], [0, 1], [1, 0], [1, 1]]
    index = 0

    x_noise, y_noise = add_noise(x, y, size)

    x_noise = x_noise.T
    y_noise = y_noise.T

    for config in configs:

        _, _, loss = optimize_parameters(
            x_noise, y_noise, w, w0, config['lr'],
            config['epoch'])

        axis[positions[index][0], positions[index][1]].plot(loss)
        axis[positions[index][0], positions[index][1]].set_title(
            f'size: {size}, lr: {config["lr"]}, epoch: {config["epoch"]}')
        index += 1

    plt.savefig(os.path.join(EXERCISE_DIR, f's{size}.png'))
