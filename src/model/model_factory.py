from src.model.feed_forward import FeedForwardNN
from src.model.cnn import CNN
import torch

def build_model(config):
    if config.type == "FeedForwardNN":
        model = FeedForwardNN(**config.model_config)
    elif config.type == "CNN":
        model = CNN(**config.model_config)

    return model