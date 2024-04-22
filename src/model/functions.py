import torch.nn as nn

def get_nonlinearity(name, config={}):
    try:
        f = getattr(nn, name)
    except AttributeError:
        raise ValueError("Unrecognized non-linearity [" + name + "]")
    return f(**config)

def get_pooling_layer(name, config={}):
    try:
        f = getattr(nn, name)
    except AttributeError:
        raise ValueError("Unrecognized pooling layer [" + name + "]")
    return f(**config)
