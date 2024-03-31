from src.model.feed_forward import FeedForwardNN
import torch

def build_model(config):
    if config.type == "FeedForwardNN":
        model = FeedForwardNN(**config.model_config)

    return model