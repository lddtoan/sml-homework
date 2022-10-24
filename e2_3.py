import e2_2
import e2_1
import os
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = pathlib.Path(__file__).parent.resolve()
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
EXERCISE_DIR = os.path.join(IMAGES_DIR, 'e2.3')
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


# Create XOR dataset
x = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y = np.array([0, 1, 1, 0])

# Train
w, w0 = e2_1.intialize_parameters([1.0, 1.0], 0.5)

x_train, y_train = add_noise(x, y, 100)

w, w0, _ = e2_1.optimize_parameters(x_train.T, y_train.T, w, w0, 0.001, 100)

model, _ = e2_2.optimize_parameters(x_train, y_train, 0.001, 100)

# Test accuracy
lf_corrects = []
mlp_corrects = []

for _ in range(20):
    x_test, y_test = add_noise(x, y, 20)

    # Test accuracy logistic function
    lf_correct = 0
    for pred, actual in zip(e2_1.predict(x_test.T, w, w0), y_test):
        if pred <= 0.5 and actual == 0 or pred > 0.5 and actual == 1:
            lf_correct += 1
    lf_corrects.append(lf_correct / 20)

    # Test accuracy MLP
    mlp_correct = 0
    with torch.no_grad():
        out = model(torch.from_numpy(x_test).float())
        _, pred = torch.max(out, 1)
        mlp_correct = (pred == torch.from_numpy(y_test).int()).sum().item()
    mlp_corrects.append(mlp_correct / 20)

figure = plt.figure()
axis = plt.axes()
axis.plot(lf_corrects, marker='o', label='Logistic function')
axis.plot(mlp_corrects, marker='o', label='MLP')
plt.xticks(range(0, 20, 2))
plt.legend()
plt.title('Model\'s accuracy')
plt.savefig(os.path.join(EXERCISE_DIR, 'accuracy.png'))
