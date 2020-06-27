import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.features = nn.Sequential(*modules)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        # convert to (batch_size, 8, 8, 2048)
        return x.permute(0, 2, 3, 1)


if __name__ == '__main__':
    x = ResNet()
    arr = torch.ones((2, 3, 255, 256), dtype=torch.float32)
    m = x(arr)
    print(m.shape)
    # print(x)
