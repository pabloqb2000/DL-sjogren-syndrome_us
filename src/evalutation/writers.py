import os
import mlflow
import numpy as np
from os.path import join
from torch.utils.tensorboard import SummaryWriter

class LogWriter:
    def __init__(self, logger, color=None):
        self.logger = logger
        self.color = color
    
    def write(self, value, n_batch, name):
        if self.color is not None:
            self.logger.log_color(self.color, name + ':', value)
        else:
            self.logger.log(name + ':', value)

    def flush(self):
        pass

class NumpyWriter:
    def __init__(self, out_path):
        self.out_path = join(out_path, 'numpy')
        self.values = {}

    def write(self, value, n_batch, name):
        if name not in self.values:
            self.values[name] = {'x': [], 'y': [] }
        self.values[name]['x'].append(n_batch)
        self.values[name]['y'].append(value)
    
    def flush(self):
        for name in self.values:
            path = '/'.join(join(self.out_path, name).split('/')[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(join(self.out_path, name + '_x.npy'), np.array(self.values[name]['x']))
            np.save(join(self.out_path, name + '.npy'), np.array(self.values[name]['y']))
            

class TensorboardWriter:
    def __init__(self, out_path):
        self.summary_writer = SummaryWriter(join(out_path, 'tensorboard'))

    def write(self, value, n_batch, name):
        self.summary_writer.add_scalar(name, value, n_batch)

    def flush(self):
        self.summary_writer.close()


class MLflowWriter:
    def __init__(self, params, experiment, host='127.0.0.1', port='8080'):
        mlflow.set_tracking_uri(uri=f"http://{host}:{port}")
        mlflow.set_experiment(experiment)
        self.mlflow_writer = mlflow.start_run()
        mlflow.log_params(params)
    
    def write(self, value, n_batch, name):
        mlflow.log_metric(name, value, n_batch)
    
    def flush(self):
        mlflow.end_run()

def build_writers(config, out_path=None, logger=None):
    writers = []
    for writer in config.evaluation.writers:
        if writer == 'LogWritter':
            writers.append(LogWriter(logger))
        elif writer == 'NumpyWriter':
            writers.append(NumpyWriter(out_path))
        elif writer == 'TensorboardWriter':
            writers.append(TensorboardWriter(out_path))
        elif writer == 'MLflowWriter':
            params = config.model | config.train
            params.pop('out_path')
            params = get_recursive_params(params)
            writers.append(MLflowWriter(params, config.name))

    return writers

def get_recursive_params(params):
    result = {}
    for k, v in params.items():
        if isinstance(v, dict):
            result |= get_recursive_params(v)
        else:
            result[k] = v
    return result
