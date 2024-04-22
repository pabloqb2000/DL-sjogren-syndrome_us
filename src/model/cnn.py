import torch
import torch.nn as nn
from src.model.feed_forward import FeedForwardNN
from src.model.functions import get_nonlinearity, get_pooling_layer

class CNN(nn.Module):
    def __init__(self, 
                 n_blocks=3, convolutions_per_block=2,
                 ch_first=64, ch_factor=2, ch_in=1,
                 first_conv_config={'kernel_size': 3},
                 normal_conv_config={'kernel_size': 3},
                 dropout_rate=0.1, 
                 pooling_layer="MaxPool2d", 
                 pooling_layer_config={'kernel_size': 2},
                 non_linearity="ReLU", 
                 non_linearity_config={'inplace': True},
                 last_layer_dims=[64, 4],
                 device='cpu',
                 **kwargs,
            ):
        super(CNN, self).__init__()
        self.blocks = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pooling_layer = get_pooling_layer(pooling_layer, pooling_layer_config) \
            if pooling_layer else None
        self.non_linearity = get_nonlinearity(non_linearity, non_linearity_config)

        self.n_channels = [ch_in, *(ch_first * (ch_factor**(i//2)) for i in range(n_blocks))]
        
        for i, (ch_in, ch_out) in enumerate(zip(self.n_channels[:-1], self.n_channels[1:])):
            block = nn.ModuleList()

            if i != 0:
                if pooling_layer:
                    block.append(self.pooling_layer)
                else:
                    block.extend([
                        nn.Conv2d(ch_in, ch_out, **first_conv_config),
                        nn.BatchNorm2d(ch_out),
                        self.non_linearity,
                        self.dropout,
                    ])
            else:
                block.extend([
                    nn.Conv2d(ch_in, ch_out, **normal_conv_config),
                    nn.BatchNorm2d(ch_out),
                    self.non_linearity,
                    self.dropout,
                ])
                
            for _ in range(convolutions_per_block-1): 
                block.extend([
                    nn.Conv2d(ch_out, ch_out, **normal_conv_config),
                    nn.BatchNorm2d(ch_out),
                    self.non_linearity,
                    self.dropout,
                ])

            self.blocks.append(block)
        
        self.ffnn = FeedForwardNN(last_layer_dims, dropout_rate)

        self.device = torch.device(device)
        self.to(self.device)
        
    def forward(self, x):
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        x = self.ffnn(x)

        return x

