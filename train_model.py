import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from simple_cnn import SimpleCNN, SimpleXORNet
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import argparse

# Model (using a simple CNN for quick training)
#NEW

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datapath', default='cifar10/data')
    parser.add_argument('--model', default='custom_cnn')
    args = parser.parse_args()
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    if args.dataset == 'xor' or args.model == 'xor':
        # Load XOR dataset from CSV
        df = pd.read_csv(args.datapath)
        X = df.iloc[:, :-1].values.astype('float32')
        y = df.iloc[:, -1].values.astype('int64')
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        trainset = TensorDataset(tensor_x, tensor_y)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
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

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop with checkpoints
    total_epochs = 20  # We'll save 10 checkpoints

    for epoch in range(total_epochs):
        running_loss = 0.0
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
        
        # Save checkpoint after each epoch
        checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint at {checkpoint_path}')

    print('Finished Training')

if __name__ == '__main__':
    main()
