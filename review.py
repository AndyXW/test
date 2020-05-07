from pathlib import Path
# import request

DATA_PATH = Path("MNIST_DATA")
PATH = DATA_PATH / "mnist"

# PATH.mkdir(parents=True, exist_ok=True)
#
# URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
#
# if not(PATH / FILENAME).exists():
#     content = requests.get(URL + FILENAME).content
#     (PATH / FILENAME).open("wb").write(content)


import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(),"rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

import matplotlib.pyplot as plt
import numpy as np

# plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# plt.show()

# PyTorch uses torch.tensor, rather than numpy arrays, so we need to convert our data
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.shape)
# print(y_train.min(), y_train.max())

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

batch_size = 64
xb = x_train[0: batch_size]
preds = model(xb)

# print(preds[0], preds.shape)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:batch_size]
# print(loss_func(preds, yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))

from IPython.core.debugger import set_trace

lr = 0.5
epochs = 2
for epoch in range(epochs):
    for i in range((n - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = x_train[start_i: end_i]
        yb = y_train[start_i: end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        loss.backward()

        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

# print(loss_func(model(xb), yb), accuracy(model(xb), yb))

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic()

def fit():
    for epoch in range(epochs):
        for i in range((n-1) // batch_size + 1):
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb = x_train[start_i: end_i]
            yb = y_train[start_i: end_i]
            pred = model(xb)

            loss = loss_func(pred, yb)

            with torch.no_grad():
                for param in model.parameters():
                    param -= param.grad * lr
                model.zero_grad()

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, xb):
        return self.linear(xb)


# Refactor using Dataset
# A Dataset can be anything that has a __len__ function and a __getitem__ functions as a way
# of indexing into it
# Pytorch's TensorDataset is a Dataset wrapping tensors. By defining a length and way of indexing
# this also gives us a way to iterate, index, and slice along the first dimension of a tensor.
# This will make it easier to access both the independent and dependent variables in the same line as we train

from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

# xb, yb = train_ds[i*batch_size, i*batch_size + batch_size]

"""
Refactor using DataLoader
Pytorch's DataLoader is responsible for managing batches. You can create a DataLoader from
andy Dataset. DataLoader makes it easier to iterate over batches. Rather than having to us
train_ds[i*batch_size, i*batch_size + batch_size], the DataLoader gives us each minibatch automatically
"""

from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=batch_size)

from torch import optim
opt = optim.SGD(model.parameters(), lr=lr)

for xb, yb in train_dl:
    pred = model(xb)

# train

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size*2)

# Note that we always call model.train() before training and model.eval() before inference,
# because these are used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate
# behaviour for these different phase

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.zero_grad()
        opt.step()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
    print(epoch, valid_loss / len(valid_dl))

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in epochs:
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs*2))

train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# switch to CNN

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

class Lambda(nn.Module):
    def __init__(self, func):
        super.__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
def preprocess(x):
    return x.view(-1, 1, 28, 28)
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv3d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
print("This is a new change in local")
