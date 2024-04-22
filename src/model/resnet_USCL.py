import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUSCL(nn.Module):
    def __init__(self, hidden_dim=256, last_dim=4, pretrained='', weights_path='', device='cpu', **kwargs):
        super(ResNetUSCL, self).__init__()

        resnet = models.resnet18(pretrained=pretrained)
        num_ftrs = resnet.fc.in_features        

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Discard the last fc layer

        self.linear = nn.Linear(num_ftrs, hidden_dim)
        self.fc = nn.Linear(hidden_dim, last_dim)

        self.device = torch.device(device)
        self.to(self.device)

        if weights_path:
            self.load_weights(weights_path)
        
    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        for k in state_dict:
            print(k)
        new_dict = {
            k: state_dict[k] 
                for k in state_dict
                if k.startswith('features') # Keep only first features layers
        }  
        model_dict = self.state_dict()

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        h = self.linear(h)
        x = self.fc(h)

        return x
