import time
import torch
from torch import nn, optim
import torch.nn.functional as F


def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # judge whether train model or test
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var, eps)

    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # condition 1 fully connected
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # condition 2 convolution layer, calculate mean and var on channels
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)

        # update moving mean and moving var
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # stretch and translation

    return Y, moving_mean, moving_var


# for protect the batch normalization
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # engage to calculate gradient
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameters(torch.zeros(shape))
        # do not calculate gradient
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
                                                          X, self.gamma, self.beta, self.moving_mean, self.moving_var,
                                                          eps=1e-5, momentum=0.9)
        return Y

# using built in function nn.BatchNorm2d and nn.BatchNorm1d


# ResNet 残差网络
class Residual(nn.Module):
    # 可以设定输出通道数， 是否使用额外的1*1卷积层来修改通道数以及卷积层的步骤
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


