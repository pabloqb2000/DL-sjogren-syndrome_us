import torch
import torch.nn as nn
import torchvision.models as models

class ResNetOwn(nn.Module):
    def __init__(self, hidden_dim=256, last_dim=4, freeze=True, discard_layers=2, dropout=0.1, pretrained='', weights_path='', device='cpu', **kwargs):
        super(ResNetOwn, self).__init__()

        # self.resnet = models.resnet18(pretrained=pretrained)
        # self.resnet = models.resnet50(pretrained = pretrained)
        self.resnet = models.vit_b_16(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, last_dim)   

        self.device = torch.device(device)
        self.resnet.to(self.device)


    def forward(self, x):
        self.resnet(x)
        return x