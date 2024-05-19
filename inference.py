from sklearn.metrics import confusion_matrix
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from src.data.preprocessing import preprocessing
from src.model.model_factory import build_model
from src.utils.load_config import load_config 
from src.utils.misc import set_seed
from torchvision.transforms import v2

# Parse command line arguments
parser = argparse.ArgumentParser(
    prog='Sjogren-Syndrome',
    description='Classify US images into their corresponding OMERACT score')

parser.add_argument('-c', '--config', help=".yaml config file", required=True)
parser.add_argument('-w', '--weights', help=".pth file with model weights", required=True)
parser.add_argument('-l', '--labels', help=".csv file with data info", required=True)
args = parser.parse_args()

# Load model and config
config = load_config(args.config)
set_seed(config.random_seed)
model = build_model(config.model)
state_dict = torch.load(args.weights)
model.load_state_dict(state_dict)
model.eval()

# Transforms
valid_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(300),
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=config.data.crop_size),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# if config.model.type == 'ResNetOwn':
#     valid_transform.transforms.append(model.weights.transforms())

# Process images
preprocessing(args.labels, config)
data = pd.read_csv(args.labels, sep=',')
softmax = torch.nn.Softmax(dim=1)

n = 0
correct = 0
labels = []
predictions = []
print("   ID    | Real label | Predicted | Probabilities")
for id, row in data.iterrows():
    img = Image.open(f'./data/imgs/preprocessed_images/{row['Anonymized ID']:03}.jpg')
    img = np.repeat(np.array(img)[:, :, np.newaxis], 3, -1)
    img = valid_transform(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        output = softmax(model(img))
        output = output.detach().cpu().numpy()
    output_label = np.argmax(output)
    label = row["OMERACT score"]

    n += 1
    if output_label == label:
        correct += 1
    labels.append(label)
    predictions.append(output_label)

    print(str(row["Anonymized ID"]).center(9) + '|' + \
          str(label).center(12) + '|' + \
          str(output_label).center(11) + '| ' + \
          f'[{', '.join([str(round(v, 3)) for v in output[0]])}]')

print("Final accuracy:", correct/n)
print("Confusion matrix")
print(confusion_matrix(labels, predictions))
