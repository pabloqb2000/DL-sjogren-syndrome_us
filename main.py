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
from sklearn.model_selection import train_test_split
import pdb


config_file = sys.argv[1]
config = load_config(config_file)
set_seed(config.random_seed)

model = build_model(config.model)
logger = build_logger(config)
writers = build_writers(config, config.train.out_path, logger)
train_evaluator = build_evaluator(config.evaluation.train_metrics, writers, DatasetType.Train)
valid_evaluator = build_evaluator(config.evaluation.valid_metrics, writers, DatasetType.Valid)

trainer = Trainer(model, logger, train_evaluator, valid_evaluator, config)
tester = Tester(model, logger, train_evaluator, valid_evaluator, config)


from src.data.datasets import CachedImageDataset, CustomImageDataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.transforms import v2

cached_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.Grayscale()
])

online_transform = v2.Compose([
    v2.RandomResizedCrop(size=config.data.crop_size, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(
        (-config.data.rotation_angle, config.data.rotation_angle),
        v2.InterpolationMode.BILINEAR),
    v2.Normalize([0]*3, [1]*3),
])

dataset = CachedImageDataset(
    root_dir=config.data.path,
    cached_transform=cached_transform,
    online_transform=online_transform
)


images = []
labels = []
for i, tuple in enumerate(dataset):
    images.append(tuple[0])
    labels.append(tuple[1])

im_train, im_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=dataset.labels, random_state=42)
im_train, im_val, y_train, y_val = train_test_split(im_train, y_train, test_size=0.1, stratify= y_train, random_state=42)
# train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])

train_dataset = CustomImageDataset(im_train, y_train)
valid_dataset = CustomImageDataset(im_val, y_val)
test_dataset = CustomImageDataset(im_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle
)
valid_loader = DataLoader(
    valid_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle
)
test_loader = DataLoader(
    test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle
)


trainer.train(train_loader, valid_loader)

tester.test(test_loader)