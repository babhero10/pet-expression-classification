import torch.nn as nn
import torch

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchNormalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.batchNormalization(out)
        out = self.activation(out)
        return out

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()

        self.conv1 = ConvolutionBlock(3, 32, 3, 2, 0)
        self.conv2 = ConvolutionBlock(32, 32, 3, 1, 0)
        self.conv3 = ConvolutionBlock(32, 64, 3, 1, 1)
        self.conv4 = ConvolutionBlock(64, 80, 3, 1, 0)
        self.conv5 = ConvolutionBlock(80, 192, 3, 1, 1)
        self.maxPool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.maxPool(out)

        out = self.conv4(out)
        out = self.conv5(out)

        out = self.maxPool(out)

        return out


class InceptionBlock_A(nn.Module):
    def __init__(self, in_channels, nbr_kernels):
        super(InceptionBlock_A, self).__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels, 64, 1, 1, 0),
            ConvolutionBlock(64, 96, 3, 1, 1),
            ConvolutionBlock(96, 96, 3, 1, 1)
        )

        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels, 48, 1, 1, 0),
            ConvolutionBlock(48, 64, 3, 1, 1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            ConvolutionBlock(in_channels, 64, 1, 1, 0)
        )

        self.branch4 = ConvolutionBlock(in_channels, 64, 1, 1, 0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out

class InceptionBlock_B(nn.Module):
    def __init__(self, in_channels, nbr_kernels):
        super(InceptionBlock_B, self).__init__()

        self.branch1 = ConvolutionBlock(in_channels, 192, 1, 1, 0)

        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels, nbr_kernels, 1, 1, 0),
            ConvolutionBlock(nbr_kernels, nbr_kernels, (1, 7), 1, (0, 3)),
            ConvolutionBlock(nbr_kernels, 192, (7, 1), 1, (3, 0))
        )

        self.branch3 = nn.Sequential(
            ConvolutionBlock(in_channels, nbr_kernels, 1, 1, 0),
            ConvolutionBlock(nbr_kernels, nbr_kernels, (7, 1), 1, (0, 3)),
            ConvolutionBlock(nbr_kernels, nbr_kernels, (1, 7), 1, (3, 0)),
            ConvolutionBlock(nbr_kernels, nbr_kernels, (7, 1), 1, (0, 3)),
            ConvolutionBlock(nbr_kernels, 192, (1, 7), 1, (3, 0)),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            ConvolutionBlock(in_channels, 192, 1, 1, 0)
        )

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out

class InceptionBlock_C(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_C, self).__init__()

        self.branch1 = ConvolutionBlock(in_channels, 320, 1, 1, 0)

        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            ConvolutionBlock(in_channels, 192, 1, 1, 0)
        )

        self.branch3 = ConvolutionBlock(in_channels, 384, 1, 1, 0)

        self.branch3_1 = ConvolutionBlock(384, 384, (1, 3), 1, (0, 1))

        self.branch3_2 = ConvolutionBlock(384, 384, (3, 1), 1, (1, 0))

        self.branch4 = nn.Sequential(
            ConvolutionBlock(in_channels, 448, 1, 1, 0),
            ConvolutionBlock(448, 384, 3, 1, 1)
        )

        self.branch4_1 = ConvolutionBlock(384, 384, (1, 3), 1, (0, 1))
        self.branch4_2 = ConvolutionBlock(384, 384, (3, 1), 1, (1, 0))

    def forward(self, x):

        branch1 = self.branch1(x)

        branch2 = self.branch2(x)

        branch3 = self.branch3(x)

        branch3 = torch.cat([self.branch3_1(branch3), self.branch3_2(branch3)], 1)

        branch4 = self.branch4(x)

        branch4_1 = self.branch4_1(branch4)
        branch4_2 = self.branch4_2(branch4)

        branch4 = torch.cat([branch4_1, branch4_2], 1)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out


class ReductionBlock_A(nn.Module):
    def __init__(self, in_channels):

        super(ReductionBlock_A, self).__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels, 64, 1, 1, 0),
            ConvolutionBlock(64, 96, 3, 1, 1),
            ConvolutionBlock(96, 96, 3, 2, 0)
        )

        self.branch2 = ConvolutionBlock(in_channels, 384, 3, 2, 0)

        self.branch3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat([branch1, branch2, branch3], 1)

        return out

class ReductionBlock_B(nn.Module):

    def __init__(self, in_channels):
        super(ReductionBlock_B, self).__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels, 192, 1, 1, 0),
            ConvolutionBlock(192, 192, (1, 7), 1, (0, 3)),
            ConvolutionBlock(192, 192, (7, 1), 1, (3, 0)),
            ConvolutionBlock(192, 192, 3, 2, 0)
        )

        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels, 192, 1, 1, 0),
            ConvolutionBlock(192, 320, 3, 2, 0)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat([branch1, branch2, branch3], 1)

        return out

class Aux_Block(nn.Module):

    def __init__(self, in_channels, num_classes=1000):
        super(Aux_Block, self).__init__()

        self.avgPool = nn.AvgPool2d(kernel_size=(5, 5), stride=3, padding=0)
        self.conv1 = ConvolutionBlock(in_channels, 128, 1, 1, 0)
        self.conv2 = ConvolutionBlock(128, 768, 5, 1, 0)
        self.fc1 = nn.Linear(in_features=768, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):

        out = self.avgPool(x)

        out = self.conv1(out)

        out = self.conv2(out)

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = nn.ReLU()(out)

        out = self.fc2(out)

        return out


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()
        self.stem = StemBlock()

        self.inceptionA_1 = InceptionBlock_A(192, 32)
        self.inceptionA_2 = InceptionBlock_A(288, 64)
        self.inceptionA_3 = InceptionBlock_A(288, 64)

        self.reductionA = ReductionBlock_A(288)

        self.inceptionB_1 = InceptionBlock_B(768, 128)
        self.inceptionB_2 = InceptionBlock_B(768, 160)
        self.inceptionB_3 = InceptionBlock_B(768, 160)
        self.inceptionB_4 = InceptionBlock_B(768, 192)

        self.aux = Aux_Block(768)

        self.reductionB = ReductionBlock_B(768)

        self.inceptionC_1 = InceptionBlock_C(1280)
        self.inceptionC_2 = InceptionBlock_C(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=2048, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):

        out = self.stem(x)

        out = self.inceptionA_1(out)
        out = self.inceptionA_2(out)
        out = self.inceptionA_3(out)

        out = self.reductionA(out)

        out = self.inceptionB_1(out)
        out = self.inceptionB_2(out)
        out = self.inceptionB_3(out)
        out = self.inceptionB_4(out)

        aux = self.aux(out)

        out = self.reductionB(out)

        out = self.inceptionC_1(out)
        out = self.inceptionC_2(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)
        out = nn.ReLU()(out)

        out = self.fc2(out)

        return out, aux

def create_model(num_classes=1000):
    return InceptionV3(num_classes=num_classes)
