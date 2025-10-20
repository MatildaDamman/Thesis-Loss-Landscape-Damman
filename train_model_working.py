#!/usr/bin/env python3
"""
Working version of train_model_new.py with PCA functionality
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

print(" Basic imports successful!")

# Only import PCA when needed to avoid conflicts
def create_pca_directions_from_gradients(all_gradients, save_path='pca_gradient_directions.h5'):
    """Create PCA directions from gradients"""
    print(" Creating PCA directions from gradients...")
    
    # Import PCA only when needed
    from sklearn.decomposition import PCA
    import h5py
    import h5_util
    
    if not all_gradients:
        print(" No gradients provided!")
        return None
    
    # Convert to matrix
    gradient_matrix = np.array(all_gradients)
    print(f" Gradient matrix shape: {gradient_matrix.shape}")
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(gradient_matrix)
    
    print(f" PCA explained variance: {pca.explained_variance_ratio_}")
    print(f" Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # For simplicity, just save the PCA info for now
    # Full implementation can be added later
    np.save('pca_components.npy', pca.components_)
    np.save('pca_variance_ratio.npy', pca.explained_variance_ratio_)
    
    print(f"PCA directions saved to numpy files")
    return save_path

def main():
    print(" Starting training script...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datapath', default='cifar10/data')
    parser.add_argument('--model', default='custom_cnn')
    args = parser.parse_args()
    
    print(f"Arguments: dataset={args.dataset}, datapath={args.datapath}, model={args.model}")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('gradients', exist_ok=True)

    if args.dataset == 'xor' or args.model == 'xor':
        print(" Loading XOR dataset...")
        df = pd.read_csv(args.datapath)
        print(f" Loaded {len(df)} samples")
        
        X = df.iloc[:, :-1].values.astype('float32')
        y = df.iloc[:, -1].values.astype('int64')
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        trainset = TensorDataset(tensor_x, tensor_y)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=3)
        model = SimpleXORNet()
    else:
        # CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        model = SimpleCNN()
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f" Using device: {device}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop with gradient collection
    total_epochs = 10  # Reasonable number for testing
    all_gradients = []
    
    print(f" Starting training for {total_epochs} epochs...")

    for epoch in range(total_epochs):
        running_loss = 0.0
        epoch_gradients = []
        
        print(f" Epoch {epoch + 1}/{total_epochs}")
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Collect gradients
            grad_vector = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vector.append(param.grad.detach().cpu().numpy().flatten())
            
            if grad_vector:
                grad_vector = np.concatenate(grad_vector)
                epoch_gradients.append(grad_vector)
                all_gradients.append(grad_vector)
            
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Save gradients for this epoch
        if epoch_gradients:
            np.save(f'gradients/gradients_epoch_{epoch}.npy', np.stack(epoch_gradients))
        
        # Save checkpoint
        checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

    print(' Training completed!')
    
    # Create PCA directions
    if all_gradients:
        print(f" Collected {len(all_gradients)} gradient vectors")
        create_pca_directions_from_gradients(all_gradients)
    else:
        print(" No gradients collected")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()