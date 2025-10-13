#!/usr/bin/env python3
"""
Simplified version of train_model_new.py to test basic functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from simple_cnn import SimpleCNN, SimpleXORNet
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse

print("✅ All imports successful!")

def main():
    print("🚀 Starting training script...")
    print("📦 Parsing arguments...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datapath', default='cifar10/data')
    parser.add_argument('--model', default='custom_cnn')
    args = parser.parse_args()
    
    print(f"📝 Arguments: dataset={args.dataset}, datapath={args.datapath}, model={args.model}")
    
    # Create necessary directories
    print("📁 Creating directories...")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('gradients', exist_ok=True)

    if args.dataset == 'xor' or args.model == 'xor':
        print("📊 Loading XOR dataset...")
        # Load XOR dataset from CSV
        try:
            df = pd.read_csv(args.datapath)
            print(f"✅ Loaded {len(df)} samples from {args.datapath}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
        
        X = df.iloc[:, :-1].values.astype('float32')
        y = df.iloc[:, -1].values.astype('int64')
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        trainset = TensorDataset(tensor_x, tensor_y)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=3)
        model = SimpleXORNet()
        print("✅ XOR dataset and model ready")
    else:
        # CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        model = SimpleCNN()
        print("✅ CIFAR-10 dataset and model ready")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🎯 Using device: {device}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("✅ Loss function and optimizer ready")

    # Training loop with checkpoints
    total_epochs = 5  # Reduced for testing
    print(f"🚀 Starting training for {total_epochs} epochs...")

    for epoch in range(total_epochs):
        running_loss = 0.0
        print(f"📈 Epoch {epoch + 1}/{total_epochs}")
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            
            # Break early for testing
            if i >= 2:
                break
        
        # Save checkpoint after each epoch
        checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'✅ Saved checkpoint: {checkpoint_path}')

    print('🎉 Training completed successfully!')

if __name__ == '__main__':
    try:
        print("🏃 Starting main function...")
        main()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()