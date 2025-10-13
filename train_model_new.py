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
from sklearn.decomposition import PCA
import h5py
import h5_util

print("✅ All imports successful!")

# Model (using a simple CNN for quick training)
#NEW

def create_pca_directions_from_gradients(all_gradients, model, n_components=2, save_path='pca_directions.h5'):
    """
    Create PCA directions from collected gradients and save them in h5 format.
    
    Args:
        all_gradients: List of gradient arrays collected during training
        model: The trained model to get parameter structure
        n_components: Number of PCA components (default: 2 for 2D visualization)
        save_path: Path to save the PCA directions h5 file
    
    Returns:
        Path to the saved direction file
    """
    print("Creating PCA directions from gradients...")
    
    # Flatten all gradients into a matrix where each row is one gradient vector
    gradient_matrix = np.array(all_gradients)
    print(f"Gradient matrix shape: {gradient_matrix.shape}")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(gradient_matrix)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Get the principal components
    pc1 = pca.components_[0]
    pc2 = pca.components_[1] if n_components > 1 else None
    
    # Convert PCA components back to model parameter structure
    def gradient_to_param_structure(grad_vector, model):
        """Convert a flattened gradient vector back to model parameter structure"""
        directions = []
        start_idx = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_size = param.numel()
                param_grad = grad_vector[start_idx:start_idx + param_size]
                directions.append(torch.tensor(param_grad.reshape(param.shape), dtype=torch.float32))
                start_idx += param_size
            else:
                # For parameters without gradients, add zero tensors
                directions.append(torch.zeros_like(param, dtype=torch.float32))
        
        return directions
    
    # Convert principal components to parameter structure
    xdirection = gradient_to_param_structure(pc1, model)
    ydirection = gradient_to_param_structure(pc2, model) if pc2 is not None else None
    
    # Save directions to h5 file
    print(f"Saving PCA directions to: {save_path}")
    f = h5py.File(save_path, 'w')
    
    # Save directions
    h5_util.write_list(f, 'xdirection', xdirection)
    if ydirection is not None:
        h5_util.write_list(f, 'ydirection', ydirection)
    
    # Save PCA information
    f['explained_variance_ratio_'] = pca.explained_variance_ratio_
    f['singular_values_'] = pca.singular_values_
    f['explained_variance_'] = pca.explained_variance_
    f['n_components'] = n_components
    
    f.close()
    
    print(f"PCA directions saved successfully to: {save_path}")
    return save_path

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
    total_epochs = 50  # We'll save 50 checkpoints

    all_gradients = []
    
    for epoch in range(total_epochs):
        running_loss = 0.0
        epoch_gradients = []
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Collect gradient vector for this step
            grad_vector = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vector.append(param.grad.detach().cpu().numpy().flatten())
            
            if grad_vector:  # Only add if we have gradients
                grad_vector = np.concatenate(grad_vector)
                epoch_gradients.append(grad_vector)
                all_gradients.append(grad_vector)
            
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Save per-epoch gradients
        if epoch_gradients:
            np.save(f'gradients/gradients_epoch_{epoch}.npy', np.stack(epoch_gradients))
        
        # Save checkpoint after each epoch
        checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint at {checkpoint_path}')

    print('Finished Training')
    
    # Create PCA directions from collected gradients
    if all_gradients:
        print(f"\nCollected {len(all_gradients)} gradient vectors during training")
        pca_file = create_pca_directions_from_gradients(
            all_gradients, 
            model, 
            n_components=2, 
            save_path='pca_gradient_directions.h5'
        )
        print(f"PCA directions saved to: {pca_file}")
    else:
        print("No gradients collected during training")

if __name__ == '__main__':
    try:
        print("🏃 Starting main function...")
        main()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
