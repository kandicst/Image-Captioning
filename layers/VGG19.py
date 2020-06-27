import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        original = models.vgg19_bn(pretrained=True)
        conv_layers = list(original.children())
        # take only first 3 conv blocks before maxpool
        # print(conv_layers)
        self.features = nn.Sequential(*list(conv_layers[0].children())[:52])
        # self.features = nn.Sequential(*list(conv_layers[0].children())[:39])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

if __name__ == '__main__':
    x = VGG19()
    arr = torch.ones((2, 3, 224, 224), dtype=torch.float32)
    m = x(arr)
    print(m.shape)
    # print(x)
