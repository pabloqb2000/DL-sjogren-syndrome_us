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

from sklearn.metrics import confusion_matrix

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

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(self.model.device)  # Assuming you're using GPU, move inputs to device
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


        conf_matrix = confusion_matrix(true_labels_list, predicted_labels_list)

        # Calculate average loss and accuracy
        average_loss = total_loss / len(testloader)
        accuracy = correct_predictions / total_samples

        self.logger.log("Final loss in test:", average_loss)
        self.logger.log("Accuracy: ", accuracy)

        self.logger.log("Confusion Matrix:\n", conf_matrix)