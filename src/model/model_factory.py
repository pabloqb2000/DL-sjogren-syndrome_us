from src.model.feed_forward import FeedForwardNN
from src.model.cnn import CNN
from src.model.resnet_USCL import ResNetUSCL
from src.model.resnet_own import ResNetOwn
import torch

def build_model(config):
    if config.type == "FeedForwardNN":
        model = FeedForwardNN(**config.model_config)
    elif config.type == "CNN":
        model = CNN(**config.model_config)
    elif config.type == "ResNetUSCL":
        model = ResNetUSCL(**config.model_config)
    elif config.type == "ResNetOwn":
        model = ResNetOwn(**config.model_config)

    return model