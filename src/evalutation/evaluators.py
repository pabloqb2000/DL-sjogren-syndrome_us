from src.train.loss import build_loss
from src.utils.dataset_type import DatasetType
from torch import nn
import torch

class AbstractEvaluator:
    def __init__(self, name, writers, dataset_type, perc = 1.0):
        self.name = name
        self.writers = writers
        self.dataset_type = dataset_type
        self.perc = perc
        self.value = self.evaluations = 0

    def evaluate(self, prediction, target, perc):
        if perc >= self.perc:
            return
        
        self.value += self.__eval__(prediction, target)
        self.evaluations += 1

    def write(self, n_batch):
        self.value /= self.evaluations
        for writer in self.writers:
            writer.write(
                self.value, n_batch, 
                self.name + '/' + DatasetType.Names[self.dataset_type]
            )
        
        self.value = self.evaluations = 0
    
    def flush(self):
        for writer in self.writers:
            writer.flush()


class CallableEvaluator(AbstractEvaluator):
    def __init__(self, name, evaluator, writers, dataset_type, perc=1):
        super().__init__(name, writers, dataset_type, perc)
        self.evaluator = evaluator

    def __eval__(self, prediction, target):
        return self.evaluator(prediction, target).item()

class MultipleEvaluator:
    def __init__(self, evaluators):
        self.evaluators = evaluators
    
    def evaluate(self, prediction, target, perc):
        for evaluator in self.evaluators:
            evaluator.evaluate(prediction, target, perc)    
    
    def write(self, n_batch):
        for evaluator in self.evaluators:
            evaluator.write(n_batch)    
    
    def flush(self):
        for evaluator in self.evaluators:
            evaluator.flush()

def build_evaluator(metrics, writers, dataset_type):
    evaluators = []
    for metric in metrics:
        evaluators.append(
            CallableEvaluator(
                metric.name, 
                __get_metric(metric.name), 
                writers,
                dataset_type,
                metric.get('perc', 1)
            )
        )

    return MultipleEvaluator(evaluators)

def __get_metric(metric):
    if metric == "L1Loss":
        return nn.L1Loss()
    elif metric == "MSELoss":
        return nn.MSELoss()
    elif metric == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif metric == "CTCLoss":
        return nn.CTCLoss()
    elif metric == "NLLLoss":
        return nn.NLLLoss()
    elif metric == "PoissonNLLLoss":
        return nn.PoissonNLLLoss()
    elif metric == "GaussianNLLLoss":
        return nn.GaussianNLLLoss()
    elif metric == "KLDivLoss":
        return nn.KLDivLoss()
    elif metric == "BCELoss":
        return nn.BCELoss()
    elif metric == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif metric == "MarginRankingLoss":
        return nn.MarginRankingLoss()
    elif metric == "HingeEmbeddingLoss":
        return nn.HingeEmbeddingLoss()
    elif metric == "MultiLabelMarginLoss":
        return nn.MultiLabelMarginLoss()
    elif metric == "HuberLoss":
        return nn.HuberLoss()
    elif metric == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    elif metric == "SoftMarginLoss":
        return nn.SoftMarginLoss()
    elif metric == "MultiLabelSoftMarginLoss":
        return nn.MultiLabelSoftMarginLoss()
    elif metric == "CosineEmbeddingLoss":
        return nn.CosineEmbeddingLoss()
    elif metric == "MultiMarginLoss":
        return nn.MultiMarginLoss()
    elif metric == "TripletMarginLoss":
        return nn.TripletMarginLoss()
    elif metric == "TripletMarginWithDistanceLoss":
        return nn.TripletMarginWithDistanceLoss()
    elif metric == 'Accuracy':
        return accuracy
    else:
        raise ValueError(f"Unrecognized metric [{metric}]")

def accuracy(x, y):
    return (torch.argmax(x, dim=1) == y).float().sum() / y.shape[0]
