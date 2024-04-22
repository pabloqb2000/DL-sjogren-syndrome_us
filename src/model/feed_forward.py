import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, layer_dims, dropout_rate=0.0, device='cpu'):
        super(FeedForwardNN, self).__init__()
        self.layer_dims = layer_dims
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList()

        for i in range(0, len(layer_dims) - 2):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

        self.output_layer = nn.Linear(layer_dims[-2], layer_dims[-1])
        
        self.device = torch.device(device)
        self.to(self.device)
        
    def forward(self, x):
        x = x.view((x.shape[0], -1))
        
        for layer in self.layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x
