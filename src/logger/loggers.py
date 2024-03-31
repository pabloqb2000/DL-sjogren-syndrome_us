import os
import yaml
from datetime import datetime
from src.utils.dict_as_member import DictAsMember


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def format_timedelta_to_HHMM(td: datetime) -> str:
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, _ = divmod(remainder, 60)
    return "{:02d}:{:02d}".format(int(hours), int(minutes))


class AbstractLogger:
    def __init__(self, verbose: bool = True, log_start: bool = True) -> None:
        self.init_time = datetime.now()
        self.verbose = verbose
        if log_start:
            self.log('Log started', self.init_time)

    def get_header(self) -> str:
        return f'[{format_timedelta_to_HHMM(datetime.now() - self.init_time)}]'
    
    def log_header(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        self.log_color(Colors.HEADER + Colors.BOLD, *args, sep=sep, end=end)

    def log_success(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        self.log_color(Colors.OKGREEN, *args, sep=sep, end=end)
        
    def log_warning(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        self.log_color(Colors.WARNING, *args, sep=sep, end=end)
        
    def log_error(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        self.log_color(Colors.FAIL, *args, sep=sep, end=end)

    def log_bold(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        self.log_color(Colors.BOLD, *args, sep=sep, end=end)

    def log_underline(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        self.log_color(Colors.UNDERLINE, *args, sep=sep, end=end)
    
    def log_color(self, color, *args, sep: str = ' ', end: str = '\n') -> None:
        self.print(color + self.get_header(), *args, Colors.ENDC, sep=sep, end=end)

    def log(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        if self.verbose:
            self.print(self.get_header(), *args, sep=sep, end=end)
    
    def print(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        pass

class StdoutLogger(AbstractLogger):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    def print(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        print(*args, sep=sep, end=end)

class FileLogger(AbstractLogger):
    def __init__(self, path, **kwargs) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        self.file_name = os.path.join(path, 'log.txt')
        super().__init__(**kwargs)

    def print(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        with open(self.file_name, 'a+', encoding='utf-8') as f:
            f.write(sep.join(str(arg) for arg in args) + end)

class MultipleLogger(AbstractLogger):
    def __init__(self, *loggers, **kwargs) -> None:
        self.loggers = loggers
        super().__init__(log_start=False, **kwargs)
    
    def print(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        for logger in self.loggers:
            logger.print(*args, sep=sep, end=end)


def build_logger(config: DictAsMember) -> MultipleLogger:
    loggers = []
    if config.log.stdout:
        loggers.append(StdoutLogger(**config.log.logger_config))
    if config.log.file:
        loggers.append(FileLogger(config.train.out_path, **config.log.logger_config))
    logger = MultipleLogger(*loggers, **config.log.logger_config)
    
    if config.log.log_config:
        config_str = yaml.dump(config) \
            .replace('!!python/object/new:src.utils.dict_as_member.DictAsMember', '') \
            .replace('!!python/object:src.utils.dict_as_member.DictAsMember', '')
        config_str = '\n'.join(
            line for line in config_str.split('\n')
            if 'dictitems:' not in line
        )
        logger.log(' Configuration:\n', config_str, sep='')
    
    return logger
