import os
import torch
import torch.nn as nn
import cifar10.model_loader
from simple_cnn import SimpleCNN, SimpleXORNet

def load(dataset, model_name, model_file=None, data_parallel=False):
    if dataset == 'cifar10':
        if model_name == 'custom_cnn':
            net = SimpleCNN()
            if model_file and os.path.isfile(model_file):
                net.load_state_dict(torch.load(model_file, map_location='cpu'))
            return net
        else:
            return cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'xor':
        net = SimpleXORNet()
        if model_file and os.path.isfile(model_file):
            net.load_state_dict(torch.load(model_file, map_location='cpu'))
        return net
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")