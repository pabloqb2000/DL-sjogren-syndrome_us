import torch
import torch.nn as nn
import torchvision.models as models
import pdb
from torchvision.models import resnet50, ResNet18_Weights
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ResNetOwn(nn.Module):
    def __init__(self, hidden_dim=224, last_dim=4, freeze=True, discard_layers=2, dropout=0.1, pretrained='', weights_path='', device='cpu', **kwargs):
        super(ResNetOwn, self).__init__()
        
        # --- ResNet ------
        # So far best results but there is no clear convergence (quite unstable loss and accs)
        '''self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, last_dim) '''

        # --- ResNeXt -----
        # This architecture has a very nice convergence of train and test losses unlike any other model tried. However, performance might not be as great
        self.resnet = resnext50_32x4d(weights = ResNeXt50_32X4D_Weights.DEFAULT ) 
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, last_dim)
        
        # --- ViT_b_16 ----
        '''self.resnet = models.vit_b_16(weigths = ViT_B_16_Weights, num_classes = 4)'''
        

        # --- ConvNeXt ----
        # Similar conclusions as with ResNeXt. Very slow.
        '''self.resnet = convnext_base(weights = ConvNeXt_Base_Weights.DEFAULT)
        self.resnet.classifier[2] = torch.nn.Linear(
                    in_features= self.resnet.classifier[2].in_features,
                    out_features=4
                )'''
        

        self.device = torch.device(device)
        self.resnet.to(self.device)


    def forward(self, x):
        x = self.resnet(x)
        return x