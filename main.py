import sys
from src.train.trainer import Trainer
from src.model.model_factory import build_model
from src.logger.loggers import build_logger
from src.utils.load_config import load_config 
from src.utils.dataset_type import DatasetType 
from src.evalutation.writers import build_writers
from src.evalutation.evaluators import build_evaluator

config_file = sys.argv[1]
config = load_config(config_file)

model = build_model(config.model)
logger = build_logger(config)
writers = build_writers(config, config.train.out_path, logger)
train_evaluator = build_evaluator(config.evaluation.train_metrics, writers, DatasetType.Train)
valid_evaluator = build_evaluator(config.evaluation.valid_metrics, writers, DatasetType.Valid)

trainer = Trainer(model, logger, train_evaluator, valid_evaluator, config)



import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

d_in = 15
d_out = 10
N = 10000

x = np.random.normal(0, 1, (N, d_in))
A = np.random.normal(0, 1, (d_out, d_in))
y = (A @ x.T).T + np.random.normal(0, .2, (N, d_out))
logger.log("Y shape", y.shape)

x = torch.Tensor(x)
y = torch.Tensor(y)

dataset = TensorDataset(x, y)
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2])
trainloader = DataLoader(
    train_dataset, 
    batch_size=config.data.batch_size, 
    shuffle=config.data.shuffle
)
validloader = DataLoader(
    validation_dataset, 
    batch_size=config.data.batch_size, 
    shuffle=config.data.shuffle
)
testloader = DataLoader(
    test_dataset, 
    batch_size=config.data.batch_size, 
    shuffle=config.data.shuffle
)

trainer.train(trainloader, validloader)
