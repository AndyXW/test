import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.ReLU(),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.ReLU(),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
# 这里的全连接层的输出个数比LeNet中的大数倍，使用丢弃层来缓解过拟合

        self.fc = nn.Sequential(
            nn.Linear(265 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),   # mitigate overfitting neuron has probability shot down
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)

        )

    def forward(self, img):

        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


# VGG using repetitive simple block for constructing neural network

# define a vgg template
def vgg_block(num_convs, in_channels, out_channels):

    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))

fc_features = 512 * 7 * 7
fc_hidden_units = 4096   # the number you can choose as you want


def vgg(conv_arch, fc_feature, fc_hidden_units=4096):

    net = nn.Sequential()

    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))

    net.add_module("fc", nn.Sequential(
        # flatten layer TODO
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(fc_hidden_units, 10)
    ))

    return net

# NiN Net IN NET


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk


class GlobalAvgPool2d(nn.Module):
    #  全局平均池化层可通过将池化窗口形状设置成输入高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self,x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


net = nn.Sequential(nin_block(1, 96, kernel_size=11, stride=4, padding=0),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nin_block(96, 256, kernel_size=5, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nin_block(256, 384, kernel_size=3, stide=1, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Dropout(0.5),
                    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
                    GlobalAvgPool2d())

# GoogLeNet
# 由Inception基础块组成


class Inception(nn.Module):
    # c1 - c4 are the out_channels for each circuit
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # circuit 1  one layer 1*1 convolution
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # circuit 2 first 1*1 convolution then 3*3
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_c, c2[1], kernel_size=3, padding=1)  # in order to make the feature map the same size
        # circuit 3 first 1*1 convolution then 5*5
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_c, c3[1], kernel_size=5)
        # circuit 4 first 3*3 max pooling then 1*1 convolution
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # in the dimension of channels, combine all




