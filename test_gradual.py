#!/usr/bin/env python3
"""
Gradual import test to find the problematic combination
"""

print("Testing gradual imports...")

# Basic imports
import torch
import torch.nn as nn
import torch.optim as optim
print("✅ Basic torch imports OK")

import torchvision
import torchvision.transforms as transforms
print("✅ torchvision imports OK")

import os
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
print("✅ Standard library imports OK")

from simple_cnn import SimpleCNN, SimpleXORNet
print("✅ Custom CNN imports OK")

# Now try the potentially problematic ones
print("Testing sklearn import...")
from sklearn.decomposition import PCA
print("✅ sklearn PCA import OK")

print("Testing h5py import...")
import h5py
print("✅ h5py import OK")

print("Testing h5_util import...")
import h5_util
print("✅ h5_util import OK")

print("🎉 All imports successful together!")

# Test basic functionality
def test_basic():
    print("Testing basic functionality...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='hello')
    
    # Don't parse actual args, just test the setup
    print("✅ argparse setup OK")
    print("✅ All tests passed!")

if __name__ == '__main__':
    test_basic()