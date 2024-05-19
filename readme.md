
# Sjögren Syndrome Deep Learning Project

## Contents
- [Sjögren Syndrome Deep Learning Project](#sj-gren-syndrome-deep-learning-project)
  * [Overview](#overview)
  * [Objectives](#objectives)
  * [Dataset](#dataset)
  * [Architecture](#architecture)
  * [Workflow](#workflow)
  * [Impact](#impact)
  * [Project Structure](#project-structure)
- [Usage](#usage)
    + [Inference](#inference)
      - [Prerequisites](#prerequisites)
      - [Command Line Arguments](#command-line-arguments)
      - [Usage](#usage-1)
      - [Example](#example)
      - [Options](#options)
- [Results](#results)
  * [Validation set](#validation-set)
  * [Test set](#test-set)
- [References](#references)
- [Authors](#authors)

## Overview

Sjögren syndrome is an autoimmune disorder characterized by dryness of the eyes and mouth due to inflammation of the lacrimal and salivary glands. Accurate diagnosis and monitoring of disease progression are crucial for effective management.

This deep learning project focuses on classifying ultrasound (US) images of the parotid glands (PGs) and submandibular glands (SMGs) using the OMERACT scale. The OMERACT (Outcome Measures in Rheumatology) scale is a standardized metric developed by experts to assess the morphological and structural appearance of these glands.

## Objectives
   - The primary goal is to classify US images into their OMERACT scale value.
   - The OMERACT scale provides a quantitative assessment of glandular features.
   - Features include echogenicity, homogeneity, and glandular boundaries.
   - The model will predict OMERACT scores based on these features.

## Dataset

- The dataset consists of labeled US images of PGs and SMGs.
- Annotations include OMERACT scores for each image.
- Data augmentation techniques are applied to enhance model robustness.

## Architecture

1. **Model Selection**:
   - We explore various architectures (e.g., CNNs, ResNets) to find the best fit.
   - Transfer learning may be employed using pre-trained models.

2. **Training and Evaluation**:
   - The model is trained using labeled data.
   - Evaluation metrics include accuracy, precision, recall, and F1-score.

## Workflow

1. **Data Preprocessing**:
   - Image normalization, resizing, and augmentation.
   - Splitting data into training, validation, and test sets.

2. **Model Training**:
   - Hyperparameter tuning.
   - Regularization techniques (dropout, lr scheduling, early stopping,...).

## Impact

- Accurate classification and OMERACT score prediction can aid clinicians in early diagnosis and monitoring of Sjögren syndrome.
- This project contributes to the field of medical imaging and autoimmune disease research.

## Project Structure

This deep learning project follows a well-organized structure to manage code, data, and other resources. Below, we describe the purpose of each directory and file:

1. **Root Directory:**
   - `.gitignore`: Specifies files and directories to be ignored by Git.
   - `data_overview.ipynb`: Jupyter Notebook providing an overview of the dataset.
   - `inference.py`: Script for model inference.
   - `main.py`: Main entry point for training and evaluation.
   - `model_info.ipynb`: Jupyter Notebook with model architecture details.
   - `readme.md`: This README file.
   - `requirements.txt`: Lists project dependencies.
   - `tests.ipynb`: Jupyter Notebook for testing.

2. **`configs` Directory:**
   - Contains YAML configuration files for different model variants (`base_config.yaml`, `cnn_config.yaml`, etc.).

3. **`data` Directory:**
   - `.gitkeep`: Placeholder file to ensure Git tracks the directory.
   - `labels.csv`: CSV file containing class labels.
   - `preprocess.ipynb`: Jupyter Notebook for data preprocessing.
   - `imgs/`: Subdirectory for storing image data.

4. **`logs` Directory:**
   - Contains logs generated during training and evaluation.

5. **`mlruns` Directory:**
   - Stores MLflow experiment runs.

6. **`output_images` Directory:**
   - Stores output images generated during inference.

7. **`scripts` Directory:**
   - Contains utility scripts (`activate.ps1`, `mlflow.ps1`, etc.).

8. **`src` Directory:**
   - Organized into subdirectories:
     - `data`: Data-related modules (`datasets.py`, `preprocessing.py`, etc.).
     - `evaluation`: Evaluation-related modules (`evaluators.py`, `writers.py`, etc.).
     - `logger`: Logging utilities (`loggers.py`).
     - `model`: Model architecture modules (`cnn.py`, `resnet_own.py`, etc.).
     - `test`: Testing utilities (`tester.py`).
     - `train`: Training components (`loss.py`, `lr_scheduler.py`, etc.).
     - `utils`: General utility functions (`dataset_type.py`, `load_config.py`, etc.).

9. **`weights` Directory:**
   - Contains model weights (`best_model.pth`) and configuration (`config.yaml`).


# Usage
### Inference

To perform inference using the `inference.py` script, follow the steps below:

#### Command Line Arguments

The `inference.py` script requires three main arguments:
- `--config` (`-c`): Path to the `.yaml` configuration file.
- `--weights` (`-w`): Path to the `.pth` file with the model weights.
- `--labels` (`-l`): Path to the `.csv` file containing the data information.

#### Usage

To run the inference script, use the following command:

```bash
python .\inference.py -c <path_to_config_file> -w <path_to_weights_file> -l <path_to_labels_file>
```

Replace `<path_to_config_file>`, `<path_to_weights_file>`, and `<path_to_labels_file>` with the actual paths to your configuration file, model weights file, and data information file, respectively.

> The images should be under the path specified in the config file, which by default is `./data/imgs/`.

#### Options

- `-h, --help`: Show the help message and exit.
- `-c CONFIG, --config CONFIG`: Specify the path to the `.yaml` configuration file.
- `-w WEIGHTS, --weights WEIGHTS`: Specify the path to the `.pth` file with model weights.
- `-l LABELS, --labels LABELS`: Specify the path to the `.csv` file with data information.

This command will classify the ultrasound images into their corresponding OMERACT scores based on the provided configuration, weights, and label information.

# Results 
## Validation set
Final **loss** in validation: 0.95

**Accuracy**:  0.57

**Confusion matrix**
 . | **0** | **1** | **2** | **3** 
--- | --- | --- | --- | --- |
**0** | 7 | 4 | 1 | 0
**1** | 1 | 4 | 3 | 1
**2** | 1 | 1 | 3 | 2
**3** | 0 | 0 | 1 | 6

Classification report:
 OMERACT score | precision | recall | f1-score | support
--- | --- | --- | --- | --- |
 0 | 0.78 | 0.58 | 0.67 | 12
 1 | 0.44 | 0.44 | 0.44 |  9
 2 | 0.38 | 0.43 | 0.40 |  7
 3 | 0.67 | 0.86 | 0.75 |  7 
|
accuracy |      |      | 0.57 | 35
macro avg | 0.57 | 0.58 | 0.57 | 35
weighted avg | 0.59 | 0.57 | 0.57 | 35

## Test set
Final **loss** in test: 1.06

**Accuracy**:  0.57

**Confusion Matrix**:
 . | **0** | **1** | **2** | **3** 
--- | --- | --- | --- | --- |
**0** | 8 | 2 | 1 | 0
**1** | 1 | 0 | 1 | 1
**2** | 1 | 2 | 1 | 0
**3** | 1 | 1 | 1 | 7

Classification report:

 OMERACT score | precision | recall | f1-score | support
--- | --- | --- | --- | --- |
 0 | 0.73 | 0.73 | 0.73 | 11
 1 | 0.00 | 0.00 | 0.00 |  3
 2 | 0.25 | 0.25 | 0.25 |  4
 3 | 0.88 | 0.70 | 0.78 | 10
|
accuracy |      |      | 0.57 | 28
macro avg | 0.46 | 0.42 | 0.44 | 28
weighted avg | 0.63 | 0.57 | 0.60 | 28

# References
- Kise, Y., Shimizu, M., Ikeda, H., Fujii, T., Kuwada, C., Nishiyama, M., ... & Ariji, E. (2020). Usefulness of a deep learning system for diagnosing Sjögren’s syndrome using ultrasonography images. _Dentomaxillofacial Radiology_, 49(3), 20190348.

- Arso, V. M., Alen, Z., Vera, M., Alojzija, H., Georgios, F., Tzioufas, A., & Nenad, F. (2021, October). Scoring Primary Sjögren's syndrome affected salivary glands ultrasonography images by using deep learning algorithms. In 2021 IEEE 21st International Conference on Bioinformatics and Bioengineering (BIBE) (pp. 1-4). _IEEE_.

- Vukicevic, A. M., Radovic, M., Zabotti, A., Milic, V., Hocevar, A., Callegher, S. Z., ... & Filipovic, N. (2021). Deep learning segmentation of primary Sjögren's syndrome affected salivary glands from ultrasonography images. _Computers in Biology and Medicine_, 129, 104154.

- Chen, Y., Zhang, C., Liu, L., Feng, C., Dong, C., Luo, Y., & Wan, X. (2021). USCL: pretraining deep ultrasound image diagnosis model through video contrastive representation learning. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part VIII 24 (pp. 627-637). _Springer International Publishing_.

# Authors
- Daniel Corrales Alonso
- Pablo Quintanilla Berriochoa