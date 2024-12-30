import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DenseLayer(nn.Module):
    """
    A single dense layer within a Dense Block.
    """

    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], 1)  # Concatenate along channel dim


class DenseBlock(nn.Module):
    """
    A Dense Block, a sequence of DenseLayers.
    """

    def __init__(self, in_channels, num_layers, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TransitionLayer(nn.Module):
    """
    Transition Layer - performs batch norm, 1x1 convolution, and pooling.
    """

    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out


class DenseNet121(nn.Module):
    """
    Implementation of DenseNet-121 architecture.
    """

    def __init__(self, num_classes=1000, growth_rate=32, bn_size=4, drop_rate=0.0):
        super(DenseNet121, self).__init__()
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

        # Initial Convolution Layer (same as in torchvision's implementation)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Dense Blocks and Transition Layers
        num_features = 64
        num_layers_block1 = 6
        num_layers_block2 = 12
        num_layers_block3 = 24
        num_layers_block4 = 16
        self.features.add_module('denseblock1', DenseBlock(num_features, num_layers_block1, growth_rate, bn_size, drop_rate))
        num_features = num_features + num_layers_block1 * growth_rate
        self.features.add_module('transition1', TransitionLayer(num_features, num_features // 2, drop_rate))
        num_features = num_features // 2

        self.features.add_module('denseblock2', DenseBlock(num_features, num_layers_block2, growth_rate, bn_size, drop_rate))
        num_features = num_features + num_layers_block2 * growth_rate
        self.features.add_module('transition2', TransitionLayer(num_features, num_features // 2, drop_rate))
        num_features = num_features // 2

        self.features.add_module('denseblock3', DenseBlock(num_features, num_layers_block3, growth_rate, bn_size, drop_rate))
        num_features = num_features + num_layers_block3 * growth_rate
        self.features.add_module('transition3', TransitionLayer(num_features, num_features // 2, drop_rate))
        num_features = num_features // 2

        self.features.add_module('denseblock4', DenseBlock(num_features, num_layers_block4, growth_rate, bn_size, drop_rate))
        num_features = num_features + num_layers_block4 * growth_rate

        # Final Batch Norm and Classifier
        self.features.add_module('bn5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def create_model(num_classes=1000):
    return DenseNet121(num_classes=num_classes)
