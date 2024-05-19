import sys
import torch
from src.data.preprocessing import *
from src.train.trainer import Trainer
from src.test.tester import Tester
from src.model.model_factory import build_model
from src.logger.loggers import build_logger
from src.utils.load_config import load_config 
from src.utils.dataset_type import DatasetType 
from src.evalutation.writers import build_writers
from src.evalutation.evaluators import build_evaluator
from src.utils.misc import set_seed
from src.data.split import data_split
from sklearn.model_selection import train_test_split, StratifiedKFold
import pdb
from torchvision.models import ViT_B_16_Weights

from src.data.datasets import CachedImageDataset, CustomImageDataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.transforms import v2

# Config
config_file = sys.argv[1]
config = load_config(config_file)
set_seed(config.random_seed)


# Transforms
train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(p=0.5),

    RandomCropHorizontal(),
    # AutoContrast(),
    # v2.RandomRotation(
    #           (-config.data.rotation_angle, config.data.rotation_angle),
    #          v2.InterpolationMode.BILINEAR),
    v2.Resize(config.data.crop_size),
    # v2.GaussianBlur(kernel_size = 3),
    # v2.ColorJitter(),
    v2.Normalize([0.25176433, 0.25176433, 0.25176433], [0.1612002, 0.1612002, 0.1612002])
    # v2.RandomEqualize(p = 0.5),
])

valid_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    CropCenterHorizontal(),
    # AutoContrast(),
    v2.Resize(config.data.crop_size),
    v2.Normalize([0.25176433, 0.25176433, 0.25176433], [0.1612002, 0.1612002, 0.1612002])
])

train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(300),
    #v2.RandomRotation(

    #           (-config.data.rotation_angle, config.data.rotation_angle),
    #          v2.InterpolationMode.BILINEAR),
    v2.RandomResizedCrop(size=config.data.crop_size, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.GaussianBlur(kernel_size = 3),
                # v2.ColorJitter(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # v2.RandomEqualize(p = 0.5),
])

valid_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(300),
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=config.data.crop_size),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Datasets and data loaders
train_data, val_data, test_data = data_split(random_state=42)
im_train, y_train = train_data
im_val, y_val = val_data
im_test, y_test = test_data

train_dataset = CustomImageDataset(im_train, y_train, train_transform)
valid_dataset = CustomImageDataset(im_val, y_val, valid_transform)
test_dataset = CustomImageDataset(im_test, y_test, valid_transform)

train_loader = DataLoader(
    train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle
)
valid_loader = DataLoader(
    valid_dataset, batch_size=config.data.batch_size, shuffle=False,
)
test_loader = DataLoader(
    test_dataset, batch_size=config.data.batch_size, shuffle=False,
)


# Model and evaluators
model = build_model(config.model)
logger = build_logger(config)
writers = build_writers(config, config.train.out_path, logger)
train_evaluator = build_evaluator(config.evaluation.train_metrics, writers, DatasetType.Train)
valid_evaluator = build_evaluator(config.evaluation.valid_metrics, writers, DatasetType.Valid)

if config.model.type == 'ResNetOwn':
    train_transform.transforms.append(model.weights.transforms())
    valid_transform.transforms.append(model.weights.transforms())


# Train model
trainer = Trainer(model, logger, train_evaluator, valid_evaluator, config)
trainer.train(train_loader, valid_loader)


# Test on valid set
tester = Tester(model, logger, config)
tester.test(valid_loader)

# Last step would be to check results in test (once training is done correctly)
'''tester.test(test_loader)'''
