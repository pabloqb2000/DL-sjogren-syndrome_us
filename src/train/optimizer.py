from torch import optim

def build_optimizer(model_parameters, config):
    if config.optimizer == "Adadelta":
        return optim.Adadelta(model_parameters, **config.optimizer_config)
    elif config.optimizer == "Adagrad":
        return optim.Adagrad(model_parameters, **config.optimizer_config)
    elif config.optimizer == "Adam":
        return optim.Adam(model_parameters, **config.optimizer_config)
    elif config.optimizer == "AdamW":
        return optim.AdamW(model_parameters, **config.optimizer_config)
    elif config.optimizer == "SparseAdam":
        return optim.SparseAdam(model_parameters, **config.optimizer_config)
    elif config.optimizer == "Adamax":
        return optim.Adamax(model_parameters, **config.optimizer_config)
    elif config.optimizer == "ASGD":
        return optim.ASGD(model_parameters, **config.optimizer_config)
    elif config.optimizer == "LBFGS":
        return optim.LBFGS(model_parameters, **config.optimizer_config)
    elif config.optimizer == "NAdam":
        return optim.NAdam(model_parameters, **config.optimizer_config)
    elif config.optimizer == "RAdam":
        return optim.RAdam(model_parameters, **config.optimizer_config)
    elif config.optimizer == "RMSprop":
        return optim.RMSprop(model_parameters, **config.optimizer_config)
    elif config.optimizer == "Rprop":
        return optim.Rprop(model_parameters, **config.optimizer_config)
    elif config.optimizer == "SGD":
        return optim.SGD(model_parameters, **config.optimizer_config)
    else:
        raise ValueError("Unrecognized optimizer [" + config.optimizer + "]")