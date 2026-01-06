import random
import torch
import numpy as np
import datetime
import os


def initialize_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def get_local_dict(locals_: dict) -> dict:
    local_dict: dict = locals_
    local_dict.pop('self')
    if 'kwargs' in locals_.keys():
        kwargs: dict = locals_['kwargs']
        local_dict.pop('kwargs')
        return local_dict | kwargs
    return local_dict


def get_current_time(for_file_name: bool = False) -> str:
    if for_file_name:
        return datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S')
    else:
        return datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')


def logging(log: str, log_name: str = 'logs.log') -> None:
    print(f'[{get_current_time()}] {log}')
    if log_name is not None:
        with open(log_name, 'a', encoding='utf-8_sig') as txt:
            txt.write(f'[{get_current_time()}] {log}\n')
            txt.close()


_device_alart = True
def get_device(device_name: str) -> torch.device:
    global _device_alart
    if 'cuda' in device_name:
        if torch.cuda.is_available():
            device = torch.device(device_name)
            if _device_alart:
                logging('CUDA was found!')
                _device_alart = False
        else:
            device = torch.device('cpu')
            if _device_alart:
                logging('***WARNING*** CUDA was NOT found! CPU will be used.')
                _device_alart = False
    else:
        device = torch.device('cpu')
        if _device_alart:
            logging('CPU will be used.')
            _device_alart = False
    return device


def save_txt(txt_name: str, **kwargs):
    with open(f'{txt_name}.txt', 'w', encoding='utf-8_sig') as txt:
        for key, value in kwargs.items():
            if value is None:
                continue
            if callable(value):
                if hasattr(value, '__name__'):
                    txt.write(f'{key}\t{type(value).__name__}\t{value.__name__}\n')
                elif hasattr(value, '__class__'):
                    txt.write(f'{key}\t{type(value).__name__}\t{value.__class__.__name__}\n')
                else:
                    txt.write(f'{key}\t{type(value).__name__}\t{value}\n')
            else:
                if '\t' in str(value) or '\n' in str(value):
                    txt.write(f'{key}\tarray_like\t{type(value).__name__}\n')
                else:
                    txt.write(f'{key}\t{type(value).__name__}\t{value}\n')
        txt.close()

def load_txt(txt_name: str, default_value: dict = None):
    kwargs = {}
    if os.path.exists(f'{txt_name}.txt'):
        with open(f'{txt_name}.txt', 'r', encoding='utf-8_sig') as txt:
            lines = txt.readlines()
            for line in lines:
                key, type_str, value = line.strip().split('\t')
                if type_str == 'int':
                    kwargs[key] = int(value)
                elif type_str == 'float':
                    kwargs[key] = float(value)
                elif type_str == 'str':
                    kwargs[key] = value
                elif type_str == 'bool':
                    kwargs[key] = value == 'True'
                elif type_str == 'array_like':
                    kwargs[key] = 'array_like'
                else:
                    kwargs[key] = value
        return kwargs
    else:
        if default_value is None:
            raise ValueError
        return default_value