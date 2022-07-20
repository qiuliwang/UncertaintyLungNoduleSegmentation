from byol_pytorch import BYOL
from torchvision import models, transforms
from torch import nn
import torch.nn.functional as F
import torch

class byol(nn.Module):
    """
    Double Conv for U-Net
    """
    def __init__(self, config, load = False):
        super(byol, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.model = BYOL(
            resnet,
            image_size = 512,
            hidden_layer = 'avgpool',
            projection_size = 512,
            projection_hidden_size = 4096,
            moving_average_decay = 0.99
        )

        if load:
            self.model.load_state_dict(torch.load('/home1/qiuliwang/Code/byol-pytorch-master/examples/trad_torch/checkpoint/80.pth.tar'), strict = False)

        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        # print('x size:', x.size())
        projection, representation = self.model(x, return_embedding = True)
        # print('projection size:', projection.size())
        # print('representation size:', representation.size())
        # projection size: torch.Size([32, 512])
        # representation size: torch.Size([32, 2048])

        x = representation
        x = self.fc1(x)
        x = self.fc2(x)
        return x
