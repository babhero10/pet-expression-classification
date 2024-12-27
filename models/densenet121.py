class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000):
        super(DenseNet121, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense layers
        self.dense_block1 = self._dense_block(64, 32, 6)
        self.transition1 = self._transition(256, 128)

        self.dense_block2 = self._dense_block(128, 32, 12)
        self.transition2 = self._transition(512, 256)

        self.dense_block3 = self._dense_block(256, 32, 24)
        self.transition3 = self._transition(1024, 512)

        self.dense_block4 = self._dense_block(512, 32, 16)

        # Classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(self._bottleneck(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def _bottleneck(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def _transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.transition2(x)
        x = self.dense_block3(x)
        x = self.transition3(x)
        x = self.dense_block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_model(num_classes=1000):
    return DenseNet121(num_classes=num_classes)

