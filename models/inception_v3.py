import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, branch1x1, branch3x3_reduce, branch3x3,
                 branch5x5_reduce, branch5x5, branch_pool):
        super(InceptionBlock, self).__init__()

        # 1x1 Convolution Branch
        self.branch1x1 = nn.Conv2d(in_channels, branch1x1, kernel_size=1)

        # 1x1 -> 3x3 Convolution Branch
        self.branch3x3_1 = nn.Conv2d(in_channels, branch3x3_reduce, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(branch3x3_reduce, branch3x3, kernel_size=3, padding=1)

        # 1x1 -> 5x5 Convolution Branch
        self.branch5x5_1 = nn.Conv2d(in_channels, branch5x5_reduce, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(branch5x5_reduce, branch5x5, kernel_size=5, padding=2)

        # Pooling Branch
        self.branch_pool = nn.Conv2d(in_channels, branch_pool, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()

        # Initial Convolution and Pooling Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv4 = nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Inception Blocks
        self.inception3a = InceptionBlock(192, 64, 48, 64, 64, 96, 32)
        self.inception3b = InceptionBlock(256, 64, 48, 64, 64, 96, 64)
        self.inception3c = InceptionBlock(288, 64, 48, 64, 64, 96, 64)

        # Average Pooling and Fully Connected Layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(288, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_model(num_classes=1000):
    return InceptionV3(num_classes=num_classes)
