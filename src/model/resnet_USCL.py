import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUSCL(nn.Module):
    def __init__(self, hidden_dim=256, last_dim=4, freeze=True, last_cnn_layers=2, last_cnn_in_channels=512, cnn_features_out=1568, discard_layers=2, dropout=0.1, pretrained='', weights_path='', device='cpu', **kwargs):
        super(ResNetUSCL, self).__init__()

        resnet = models.resnet18(pretrained=pretrained)
        num_ftrs = resnet.fc.in_features        

        self.discard_layers = discard_layers
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.features = nn.Sequential(*list(resnet.children())[:-2-discard_layers])  # Discard the last layers
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        in_channels = last_cnn_in_channels
        self.cnns = nn.ModuleList()
        for i in range(last_cnn_layers):
            out_channels = in_channels // 2
            self.cnns.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                nn.BatchNorm2d(out_channels),
                self.activation,
                self.dropout
            ])
            in_channels = out_channels

        self.linear = nn.Linear(cnn_features_out, hidden_dim)
        self.fc = nn.Linear(hidden_dim, last_dim)

        self.device = torch.device(device)
        self.to(self.device)

        if weights_path:
            self.load_weights(weights_path)
        
    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        # for k in state_dict:
        #     print(k)
        new_dict = { # Keep only first features layers
            k: state_dict[k] 
                for k in state_dict
                if k.startswith('features.') and k[9] in list(range(8-self.discard_layers))
        }  
        model_dict = self.state_dict()

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        n = x.shape[0]
        h = self.features(x)
        h = self.activation(h)
        h = self.dropout(h)

        for layer in self.cnns:
            h = layer(h)
        h = h.view((n, -1))
        
        h = self.linear(h)
        h = self.activation(h)
        h = self.dropout(h)

        x = self.fc(h)

        return x
