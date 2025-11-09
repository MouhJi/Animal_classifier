import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from pprint import pprint
class Transfer_ResNet(nn.Module):
    def __init__(self, pretrained=True, num_class=10):
        super().__init__()

        # Load backbone
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)

        # Lấy số feature của fc gốc (đặt sau khi load backbone)
        in_feature = self.backbone.fc.in_features

        for name, params in self.backbone.named_parameters():
            if name.startswith("layer4"):
                params.requires_grad = True
            else:
                params.requires_grad = False

        # Bỏ layer FC gốc
        self.backbone.fc = nn.Identity()

        # Thêm phần head (classifier mới)
        self.fc1 = nn.Linear(in_feature, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    model = Transfer_ResNet(num_class=10)
    for name, params in model.backbone.named_parameters():
        print(name, params.requires_grad, sep="\t")