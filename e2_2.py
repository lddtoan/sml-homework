import os
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


BASE_DIR = pathlib.Path(__file__).parent.resolve()
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
EXERCISE_DIR = os.path.join(IMAGES_DIR, 'e2.2')
if not os.path.exists(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)
if not os.path.exists(EXERCISE_DIR):
    os.mkdir(EXERCISE_DIR)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Hidden layer
        self.hl = nn.Linear(2, 2, bias=False)
        self.hlaf = nn.Sigmoid()

        # Output layer
        self.ol = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        out = self.hl(x)
        out = self.hlaf(out)
        out = self.ol(out)

        return out


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


def optimize_parameters(x, y, lr, epoch):
    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    loss = []

    for _ in range(epoch):
        model.train()
        out = model(torch.from_numpy(x).float().view(len(x), -1))

        cost = criterion(out, torch.from_numpy(y).long())

        loss.append(cost.item())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return (model, loss)


# Create XOR dataset
x = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y = np.array([0, 1, 1, 0])

sizes = [10, 50, 100]
configs = [
    {'lr': 0.01, 'epoch': 10},
    {'lr': 0.01, 'epoch': 100},
    {'lr': 0.001, 'epoch': 10},
    {'lr': 0.001, 'epoch': 100},
]

for size in sizes:
    figure, axis = plt.subplots(2, 2, constrained_layout=True)
    positions = [[0, 0], [0, 1], [1, 0], [1, 1]]
    index = 0

    x_noise, y_noise = add_noise(x, y, size)

    for config in configs:
        model, loss = optimize_parameters(
            x_noise, y_noise, config['lr'],
            config['epoch'])

        axis[positions[index][0], positions[index][1]].plot(loss)
        axis[positions[index][0], positions[index][1]].set_title(
            f'size: {size}, lr: {config["lr"]}, epoch: {config["epoch"]}')
        index += 1

    plt.savefig(os.path.join(EXERCISE_DIR, f's{size}.png'))
