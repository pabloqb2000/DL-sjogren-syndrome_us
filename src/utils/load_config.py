import os
import yaml
import shutil
from datetime import datetime

from src.utils.dict_as_member import DictAsMember

def load_config(config_file: str) -> DictAsMember:
    with open(config_file, 'r') as file:
        config = DictAsMember(yaml.safe_load(file))

        now = datetime.now()
        config.train.out_path = config.train.out_path.format(
            date = str(now.strftime('%Y-%m-%d')), 
            now = str(now.strftime('%Y-%m-%d__%H-%M-%S')), 
            name = config.name
        )    
        
        if config.train.save_model:
            if not os.path.exists(config.train.out_path):
                os.makedirs(config.train.out_path)
            shutil.copy(config_file, os.path.join(config.train.out_path, 'config.yaml'))

    return config