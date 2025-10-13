#!/usr/bin/env python3
"""
Minimal test version of train_model_new.py
"""

print("🚀 Starting minimal training script...")

import torch
print("✅ torch imported")

import torch.nn as nn
import torch.optim as optim
print("✅ torch modules imported")

import pandas as pd
import numpy as np
print("✅ pandas and numpy imported")

from simple_cnn import SimpleCNN, SimpleXORNet
print("✅ CNN models imported")

import argparse
print("✅ argparse imported")

def main():
    print("🏃 In main function...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datapath', default='cifar10/data')
    parser.add_argument('--model', default='custom_cnn')
    args = parser.parse_args()
    
    print(f"📝 Arguments: dataset={args.dataset}, datapath={args.datapath}, model={args.model}")
    
    if args.dataset == 'xor' or args.model == 'xor':
        print("📊 Loading XOR dataset...")
        try:
            df = pd.read_csv(args.datapath)
            print(f"✅ Loaded {len(df)} samples from {args.datapath}")
            print(f"📋 Dataset shape: {df.shape}")
            print(f"🎯 First few rows:\n{df.head()}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
    
    print("✅ Script completed successfully!")

if __name__ == '__main__':
    try:
        print("🏃 Starting main function...")
        main()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()