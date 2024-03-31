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


import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(0, 1)
])

train_dataset = datasets.ImageFolder(
    root=config.data.path,
    transform=train_transform
)
train_loader = DataLoader(
    train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle
)


trainer.train(train_loader, train_loader)
