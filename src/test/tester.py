import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from src.train.loss import build_loss
from src.logger.loggers import AbstractLogger
from src.train.optimizer import build_optimizer
from src.utils.dict_as_member import DictAsMember
from src.evalutation.evaluators import AbstractEvaluator
from src.train.lr_scheduler import build_lr_scheduler
from src.utils.dict_as_member import DictAsMember

from sklearn.metrics import confusion_matrix, classification_report
import pdb
import cv2
from torchvision.transforms import v2

# 3. Iterate over batches in your test loader

class Tester:
    def __init__(self, 
                 model: torch.nn.Module, 
                 logger: AbstractLogger, 
                 train_evaluator: AbstractEvaluator, 
                 valid_evaluator: AbstractEvaluator, 
                 config: DictAsMember) -> None:
        
        self.model = model
        self.logger = logger
        self.config = config

    def test(self, testloader):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        true_labels_list = []
        predicted_labels_list = []

        self.model.eval()
        self.loss = build_loss(self.config.train)

        correct_indices = {0: None, 1: None, 2: None, 3: None}
        incorrect_indices = {0: None, 1: None, 2: None, 3: None}

        inverse_transform = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            ])

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(self.model.device)  #
                labels = labels.to(self.model.device)  # Move labels to device
                
                # 4. Forward pass each batch through the model
                outputs = self.model(inputs)
                
                # 5. Compute metrics
                loss = self.loss(outputs, labels)  # Calculate loss
                total_loss += loss.item()  # Accumulate loss
                
                _, predicted = torch.max(outputs, 1)  # Get predicted labels
                correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
                total_samples += labels.size(0)  # Count total samples

                true_labels_list.extend(labels.cpu().numpy())
                predicted_labels_list.extend(predicted.cpu().numpy())

                for class_label in range(4):
                    class_mask = (labels == class_label)
                    correct_mask = (predicted == labels) & class_mask
                    incorrect_mask = (predicted != labels) & class_mask
                    
                    '''if correct_indices[class_label] is None:
                        correct_indices[class_label] = np.where(correct_mask)[0]
                    if incorrect_indices[class_label] is None:
                        incorrect_indices[class_label] = np.where(incorrect_mask)[0]'''

                    correct_indices[class_label] = np.where(correct_mask)[0] if correct_indices[class_label] is None else np.concatenate((correct_indices[class_label], np.where(correct_mask)[0]))
                    incorrect_indices[class_label] = np.where(incorrect_mask)[0] if incorrect_indices[class_label] is None else np.concatenate((incorrect_indices[class_label], np.where(incorrect_mask)[0]))


        conf_matrix = confusion_matrix(true_labels_list, predicted_labels_list)

        # Calculate average loss and accuracy
        average_loss = total_loss / len(testloader)
        accuracy = correct_predictions / total_samples

        self.logger.log("Final loss in test:", average_loss)
        self.logger.log("Accuracy: ", accuracy)

        self.logger.log("Confusion Matrix:\n", conf_matrix)
        self.logger.log("Classification report:")
        self.logger.log("\n", classification_report(true_labels_list, predicted_labels_list, labels=[0,1,2,3], zero_division=0.0))


        # Save some of the images
        output_dir = "output_images/"
        os.makedirs(output_dir, exist_ok=True)

        for class_label in range(4):

            # Get one correctly predicted image index for the class
            try:
                correct_idx = correct_indices[class_label][0]
                correct_image, true_label = testloader.dataset[correct_idx]

                correct_image_original = inverse_transform(correct_image)
                correct_image_original = correct_image_original.permute(1, 2, 0).numpy()

                cv2.imwrite(os.path.join(output_dir, f"true_class_{true_label}_predicted_class_{class_label}.jpg"), correct_image_original[:,:,0]*255)
            except:
                self.logger.log(f"No correctly classified images for class {class_label}")
            
            # Get one incorrectly predicted image index for the class
            try:
                incorrect_idx = incorrect_indices[class_label][0]
                incorrect_image, true_label  = testloader.dataset[incorrect_idx]
                prediction = predicted_labels_list[incorrect_idx]

                incorrect_image_original = inverse_transform(incorrect_image)
                incorrect_image_original = incorrect_image_original.permute(1, 2, 0).numpy()

                cv2.imwrite(os.path.join(output_dir, f"true_class_{true_label}_predicted_class_{prediction}.jpg"), incorrect_image_original[:,:,0]*255)
            except:
                self.logger.log(f"No incorrectly classified image as class {class_label}")
            