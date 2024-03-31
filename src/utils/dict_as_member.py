from typing import Any

class DictAsMember(dict):
    def __init__(self, __dict: dict) -> None:
        for key in __dict:
            value = __dict[key]
            if isinstance(value, dict):
                self[key] = DictAsMember(value)
            elif isinstance(value, list):
                self[key] = parse_list(value)
            else:
                self[key] = value

    def __getattr__(self, __name: str) -> object:
        return self[__name]
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        self[__name] = __value

def parse_list(__list):
    return [
        DictAsMember(item) if isinstance(item, dict) else 
        parse_list(item) if isinstance(item, list) else
        item for item in __list
    ]