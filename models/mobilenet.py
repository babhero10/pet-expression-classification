import torch
import torch.nn as nn

class MobileNet(nn.Module):
    def _init_(self, num_classes=1000):
        super(MobileNet, self)._init_()

        def depthwise_separable_conv(in_channels, out_channels, stride):
            """Defines a depthwise separable convolution block."""
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Define the layers for MobileNet
        self.layers = nn.Sequential(
            depthwise_separable_conv(32, 64, stride=1),
            depthwise_separable_conv(64, 128, stride=2),
            depthwise_separable_conv(128, 128, stride=1),
            depthwise_separable_conv(128, 256, stride=2),
            depthwise_separable_conv(256, 256, stride=1),
            depthwise_separable_conv(256, 512, stride=2),
            *[depthwise_separable_conv(512, 512, stride=1) for _ in range(5)],
            depthwise_separable_conv(512, 1024, stride=2),
            depthwise_separable_conv(1024, 1024, stride=1)
        )

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_model(num_classes=1000):
    return MobileNet(num_classes=num_classes)
