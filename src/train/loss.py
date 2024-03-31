from torch import nn

def build_loss(config):
    if config.loss == "L1Loss":
        return nn.L1Loss(**config.loss_config)
    elif config.loss == "MSELoss":
        return nn.MSELoss(**config.loss_config)
    elif config.loss == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**config.loss_config)
    elif config.loss == "CTCLoss":
        return nn.CTCLoss(**config.loss_config)
    elif config.loss == "NLLLoss":
        return nn.NLLLoss(**config.loss_config)
    elif config.loss == "PoissonNLLLoss":
        return nn.PoissonNLLLoss(**config.loss_config)
    elif config.loss == "GaussianNLLLoss":
        return nn.GaussianNLLLoss(**config.loss_config)
    elif config.loss == "KLDivLoss":
        return nn.KLDivLoss(**config.loss_config)
    elif config.loss == "BCELoss":
        return nn.BCELoss(**config.loss_config)
    elif config.loss == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**config.loss_config)
    elif config.loss == "MarginRankingLoss":
        return nn.MarginRankingLoss(**config.loss_config)
    elif config.loss == "HingeEmbeddingLoss":
        return nn.HingeEmbeddingLoss(**config.loss_config)
    elif config.loss == "MultiLabelMarginLoss":
        return nn.MultiLabelMarginLoss(**config.loss_config)
    elif config.loss == "HuberLoss":
        return nn.HuberLoss(**config.loss_config)
    elif config.loss == "SmoothL1Loss":
        return nn.SmoothL1Loss(**config.loss_config)
    elif config.loss == "SoftMarginLoss":
        return nn.SoftMarginLoss(**config.loss_config)
    elif config.loss == "MultiLabelSoftMarginLoss":
        return nn.MultiLabelSoftMarginLoss(**config.loss_config)
    elif config.loss == "CosineEmbeddingLoss":
        return nn.CosineEmbeddingLoss(**config.loss_config)
    elif config.loss == "MultiMarginLoss":
        return nn.MultiMarginLoss(**config.loss_config)
    elif config.loss == "TripletMarginLoss":
        return nn.TripletMarginLoss(**config.loss_config)
    elif config.loss == "TripletMarginWithDistanceLoss":
        return nn.TripletMarginWithDistanceLoss(**config.loss_config)
    else:
        raise ValueError("Unrecognized loss [" + config.loss + "]")