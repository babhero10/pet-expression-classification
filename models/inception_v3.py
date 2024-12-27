class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()
        # Initial layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2)

        # Simplified Inception blocks
        self.inception_block = nn.Sequential(
            self._inception_module(64, 32, 32, 64),
            self._inception_module(64, 64, 64, 128)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _inception_module(self, in_channels, branch1, branch3, branch5):
        return nn.ModuleList([
            nn.Conv2d(in_channels, branch1, kernel_size=1),
            nn.Conv2d(in_channels, branch3, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, branch5, kernel_size=5, padding=2)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        for module in self.inception_block:
            branch1 = F.relu(module[0](x))
            branch3 = F.relu(module[1](x))
            branch5 = F.relu(module[2](x))
            x = torch.cat([branch1, branch3, branch5], dim=1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_model(num_classes=1000):
    return InceptionV3(num_classes=num_classes)

