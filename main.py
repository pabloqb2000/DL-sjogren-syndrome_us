import sys
import torch
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

from src.data.datasets import CachedImageDataset, CustomImageDataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.transforms import v2


config_file = sys.argv[1]
config = load_config(config_file)
set_seed(config.random_seed)


transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomResizedCrop(size=config.data.crop_size, antialias=True),
    # v2.Grayscale()
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(
        (-config.data.rotation_angle, config.data.rotation_angle),
        v2.InterpolationMode.BILINEAR),
    # v2.GaussianBlur(kernel_size = 5),
    # v2.ColorJitter(),
    v2.Normalize([60.21704499799203]*3, [40.62192559049014]*3),
])


if config.train.full_train:

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_dataset = CustomImageDataset(im_train, y_train)
    test_dataset = CustomImageDataset(im_test, y_test)

    for fold, (train_idx, val_idx) in enumerate(skfold.split(im_train, y_train)):

        model = build_model(config.model)
        logger = build_logger(config)
        writers = build_writers(config, config.train.out_path, logger)
        train_evaluator = build_evaluator(config.evaluation.train_metrics, writers, DatasetType.Train)
        valid_evaluator = build_evaluator(config.evaluation.valid_metrics, writers, DatasetType.Valid)

        trainer = Trainer(model, logger, train_evaluator, valid_evaluator, config)
        tester = Tester(model, logger, train_evaluator, valid_evaluator, config)

        logger.log(f"Fold {fold + 1}")
        logger.log("-------")

        train_loader = DataLoader(
            train_dataset, batch_size=config.data.batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx)
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=config.data.batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        )

        trainer.train(train_loader, valid_loader)


        test_loader = DataLoader(
            test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle,
        )

        tester.test(test_loader)

else:
    model = build_model(config.model)
    logger = build_logger(config)
    writers = build_writers(config, config.train.out_path, logger)
    train_evaluator = build_evaluator(config.evaluation.train_metrics, writers, DatasetType.Train)
    valid_evaluator = build_evaluator(config.evaluation.valid_metrics, writers, DatasetType.Valid)

    trainer = Trainer(model, logger, train_evaluator, valid_evaluator, config)
    tester = Tester(model, logger, train_evaluator, valid_evaluator, config)

    train_data, val_data, test_data = data_split(random_state=42)
    im_train, y_train = train_data
    im_val, y_val = val_data
    im_test, y_test = test_data

    train_dataset = CustomImageDataset(im_train, y_train, transform)
    valid_dataset = CustomImageDataset(im_val, y_val, transform)
    test_dataset = CustomImageDataset(im_test, y_test, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle,
    )

    trainer.train(train_loader, valid_loader)

    tester.test(test_loader)
