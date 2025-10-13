#!/usr/bin/env python3
"""
Step-by-step version of train_model_new.py with detailed logging
"""

import sys
print("Python version:", sys.version)
print("Step 1: Starting imports...")

import torch
print("✅ Step 1a: torch imported")

import torch.nn as nn
import torch.optim as optim
print("✅ Step 1b: torch.nn and torch.optim imported")

import torchvision
import torchvision.transforms as transforms
print("✅ Step 1c: torchvision imported")

import os
print("✅ Step 1d: os imported")

from simple_cnn import SimpleCNN, SimpleXORNet
print("✅ Step 1e: simple_cnn imported")

import pandas as pd
print("✅ Step 1f: pandas imported")

import numpy as np
print("✅ Step 1g: numpy imported")

from torch.utils.data import TensorDataset, DataLoader
print("✅ Step 1h: torch data utils imported")

import argparse
print("✅ Step 1i: argparse imported")

from sklearn.decomposition import PCA
print("✅ Step 1j: sklearn PCA imported")

import h5py
print("✅ Step 1k: h5py imported")

import h5_util
print("✅ Step 1l: h5_util imported")

print("🎉 All imports completed successfully!")

def main():
    print("🚀 Step 2: Starting main function...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datapath', default='cifar10/data')
    parser.add_argument('--model', default='custom_cnn')
    args = parser.parse_args()
    
    print(f"✅ Step 2a: Arguments parsed: {args.dataset}, {args.datapath}, {args.model}")
    
    # Create directories
    print("📁 Step 2b: Creating directories...")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('gradients', exist_ok=True)
    print("✅ Step 2b: Directories created")
    
    # Load dataset
    if args.dataset == 'xor' or args.model == 'xor':
        print("📊 Step 2c: Loading XOR dataset...")
        try:
            df = pd.read_csv(args.datapath)
            print(f"✅ Step 2c: Loaded {len(df)} samples")
            
            X = df.iloc[:, :-1].values.astype('float32')
            y = df.iloc[:, -1].values.astype('int64')
            tensor_x = torch.tensor(X)
            tensor_y = torch.tensor(y)
            trainset = TensorDataset(tensor_x, tensor_y)
            trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=3)
            model = SimpleXORNet()
            print("✅ Step 2c: XOR dataset and model ready")
            
        except Exception as e:
            print(f"❌ Step 2c failed: {e}")
            return
    else:
        print("📊 Step 2c: Loading CIFAR-10...")
        # CIFAR-10 setup would go here
        print("✅ Step 2c: CIFAR-10 setup (placeholder)")
    
    print("🎉 Basic setup completed! Ready for training...")
    
    # For now, just return without actual training to test the setup
    print("ℹ️  Training step skipped for testing")

if __name__ == '__main__':
    try:
        print("🏃 Starting main...")
        main()
        print("✅ Script completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()