import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet18_Weights
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ResNetOwn(nn.Module):
    def __init__(self, last_dim=4, resnet_type='ResNeXt', device='cpu', **kwargs):
        super(ResNetOwn, self).__init__()
        
        if resnet_type == 'ResNet':
            # So far best results but there is no clear convergence (quite unstable loss and accs)
            self.weights = ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=self.weights)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, last_dim)

        elif resnet_type == 'ResNeXt':
            # This architecture has a very nice convergence of train and test losses unlike any other model tried. However, performance might not be as great
            self.weights = ResNeXt50_32X4D_Weights.DEFAULT
            self.resnet = resnext50_32x4d(weights = self.weights)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, last_dim)
        
        elif resnet_type == 'ViT_b_16':
            self.weights = ViT_B_16_Weights.DEFAULT
            self.resnet = models.vit_b_16(weigths = self.weights, num_classes = 4)

        elif resnet_type == 'ConvNeXt':
            # Similar conclusions as with ResNeXt. Very slow.
            self.weights = ConvNeXt_Base_Weights.DEFAULT
            self.resnet = convnext_base(weights = self.weights)
            self.resnet.classifier[2] = torch.nn.Linear(
                in_features= self.resnet.classifier[2].in_features,
                out_features=4
            )

        self.device = torch.device(device)
        self.resnet.to(self.device)


    def forward(self, x):
        x = self.resnet(x)
        return x