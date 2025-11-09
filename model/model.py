import torch.nn as nn
import torch


class model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(in_features=3, out_features=8, kenel_size=3, padding=1, stride=1)
        self.conv2 = self.make_block(in_features=8, out_features=16, kenel_size=3, padding=1, stride=1)
        self.conv3 = self.make_block(in_features=16, out_features=32, kenel_size=3, padding=1, stride=1)
        self.conv4 = self.make_block(in_features=32, out_features=64, kenel_size=3, padding=1, stride=1)
        self.conv5 = self.make_block(in_features=64, out_features=128, kenel_size=2, padding=1, stride=1)

        self.Linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def make_block(self, in_features, out_features, kenel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kenel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm2d(num_features=out_features),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=kenel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm2d(num_features=out_features),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.Linear(x)
        return x

if __name__ == '__main__':
    data = torch.rand(8,3,224,224)
    model = model(num_classes=10)
    output = model(data)
    print(output.shape)