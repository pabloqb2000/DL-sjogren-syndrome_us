from torch.optim import lr_scheduler

def build_lr_scheduler(optimizer, config):
    if config.lr_scheduler == "LambdaLR":
        return lr_scheduler.LambdaLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "MultiplicativeLR":
        return lr_scheduler.MultiplicativeLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "StepLR":
        return lr_scheduler.StepLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "MultiStepLR":
        return lr_scheduler.MultiStepLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "ConstantLR":
        return lr_scheduler.ConstantLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "LinearLR":
        return lr_scheduler.LinearLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "PolynomialLR":
        return lr_scheduler.PolynomialLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "ChainedScheduler":
        return lr_scheduler.ChainedScheduler(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "SequentialLR":
        return lr_scheduler.SequentialLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "CyclicLR":
        return lr_scheduler.CyclicLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "OneCycleLR":
        return lr_scheduler.OneCycleLR(optimizer, **config.lr_scheduler_config)
    elif config.lr_scheduler == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config.lr_scheduler_config)
    else:
        raise ValueError("Unrecognized loss [" + config.loss + "]")