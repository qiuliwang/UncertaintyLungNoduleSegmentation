from resnet_custom import *
import torch.nn as nn
# torch.nn.Linear(in_features, out_features, bias=True)

class ResNet_Binary(nn.Module):
    def __init__(self):
        self.res_model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.res_model(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x