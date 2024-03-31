from src.utils.dataset_type import DatasetType
from torch import nn

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
    if metric == 'MSE':
        return nn.MSELoss()
    elif metric == 'L1':
        return nn.L1Loss()
