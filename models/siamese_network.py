import torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models.mobilenetv3 import mobilenet_v3_large

class Backbone(nn.Module):
    def __init__(self, *args, **kwargs ,):
        super().__init__(*args, **kwargs)
        self.embeding_model = mobilenet_v3_large(pretrained=True)
        self.embeding_model = nn.Sequential(*list(self.embeding_model.children())[:-1])
        ## Apply Pooling

    def forward(self, x):
        return self.embeding_model(x)

class SiameseNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Backbone()

    def forward(self, x1 , x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        return x1 , x2
    

