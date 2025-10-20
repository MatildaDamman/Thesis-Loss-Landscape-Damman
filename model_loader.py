import os
import torch
import torch.nn as nn
from simple_cnn import SimpleCNN, SimpleXORNet, SimpleXORNet_332

def load(dataset, model_name, model_file=None, data_parallel=False):
    if dataset == 'cifar10':
        if model_name == 'custom_cnn':
            net = SimpleCNN()
            if model_file and os.path.isfile(model_file):
                net.load_state_dict(torch.load(model_file, map_location='cpu'))
            return net
        else:
            raise ValueError(f"CIFAR-10 model '{model_name}' not supported")
    elif dataset == 'xor':
        if model_name == 'xor_332':
            net = SimpleXORNet_332()
        else:
            net = SimpleXORNet()
        if model_file and os.path.isfile(model_file):
            net.load_state_dict(torch.load(model_file, map_location='cpu'))
        return net
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")