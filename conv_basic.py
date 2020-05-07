import torch
import torch.nn as nn


def corr2d(X, K):
    H, W = X.shape
    h, w = K.shape
    y = torch.zeros(H - h + 1, W - w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
y = corr2d(X, K)
print(y)


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones(6, 8)
Y = torch.zeros(6, 7)
X[:, 2:6] = 0
Y[:, 1] = 1
Y[:, 5] = -1

conv2d = Conv2D(kernel_size=(1, 2))
step = 30
lr = 0.01

for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()
# gradient descent
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
# gradient clear
    conv2d.weight.grad.zero_()
    conv2d.bias.grad.zero_()
    
    if(i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

print(conv2d.weight.data)
print(conv2d.bias.data)

# we use nn.Conv2d class to realize 2d convolution
# in_channels number of channels of input image (means colors)
# out_channels number of channels produced by the convolution
# kernel_size (int or tuple) size of the convolving kernel
# stride = step (int or tuple, optional) stride of the convolution default:1
# padding (int, tuple, optional) zero-padding added to both side of the input default: 0
# bias(bool, optional) -If true, adds a learnable bias to the output. Default: True

# forward function has four dimensions parameters(N(batch_size), Cin, Hin, Win)
# and the return value is also a 4 dimensions (N, Cout, Hout, Wout)

X = torch.rand(4, 2, 3, 5)

conv2d = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 5), stride=1, padding=(1, 2))
Y = conv2d(X)
print('Y.shape: ', Y.shape)
print('weight.shape: ', conv2d.weight.shape)
print('bias.shape: ', conv2d.bias.shape)


# pooling in order to mitigate over sensitivity of the convolution layer
# usually we have max pooling or average pooling

X = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
pool2d = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))
Y = pool2d(X)
print(X)
print(Y)
print(Y.shape)
