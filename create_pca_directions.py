#!/usr/bin/env python3
"""
Create PCA directions from either gradients or model checkpoints.
This script provides flexibility to generate PCA directions for loss landscape visualization.
"""

import torch
import numpy as np
import argparse
import os
import glob
from sklearn.decomposition import PCA
import h5py
import h5_util
from simple_cnn import SimpleCNN, SimpleXORNet
import model_loader
import net_plotter
from projection import tensorlist_to_tensor, npvec_to_tensorlist


def create_pca_from_gradients(gradient_files, model, n_components=2, save_path='pca_gradient_directions.h5'):
    """
    Create PCA directions from gradient files.
    
    Args:
        gradient_files: List of paths to gradient .npy files
        model: Model instance for parameter structure
        n_components: Number of PCA components
        save_path: Path to save directions
    """
    print("Loading gradients from files...")
    all_gradients = []
    
    for grad_file in gradient_files:
        if os.path.exists(grad_file):
            gradients = np.load(grad_file)
            if gradients.ndim == 2:  # Multiple gradients in file
                all_gradients.extend(gradients)
            else:  # Single gradient
                all_gradients.append(gradients)
            print(f"Loaded {len(gradients)} gradients from {grad_file}")
    
    if not all_gradients:
        raise ValueError("No gradients loaded from files")
    
    # Convert to matrix
    gradient_matrix = np.array(all_gradients)
    print(f"Total gradient matrix shape: {gradient_matrix.shape}")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(gradient_matrix)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Convert PCA components to model parameter structure
    pc1 = pca.components_[0]
    pc2 = pca.components_[1] if n_components > 1 else None
    
    xdirection = gradient_to_param_structure(pc1, model)
    ydirection = gradient_to_param_structure(pc2, model) if pc2 is not None else None
    
    # Save to h5 file
    save_directions_to_h5(save_path, xdirection, ydirection, pca)
    return save_path


def create_pca_from_checkpoints(checkpoint_files, args, n_components=2, save_path='pca_model_directions.h5'):
    """
    Create PCA directions from model checkpoints (similar to existing setup_PCA_directions).
    
    Args:
        checkpoint_files: List of paths to model checkpoint files
        args: Arguments containing dataset, model info
        n_components: Number of PCA components
        save_path: Path to save directions
    """
    print("Loading models from checkpoints...")
    
    # Load reference model (first checkpoint)
    net = model_loader.load(args.dataset, args.model, checkpoint_files[0])
    if args.dir_type == 'weights':
        w = net_plotter.get_weights(net)
    elif args.dir_type == 'states':
        s = net.state_dict()
    
    # Load all models and create difference matrix
    matrix = []
    for model_file in checkpoint_files:
        print(f"Processing {model_file}")
        net2 = model_loader.load(args.dataset, args.model, model_file)
        
        if args.dir_type == 'weights':
            w2 = net_plotter.get_weights(net2)
            d = net_plotter.get_diff_weights(w, w2)
        elif args.dir_type == 'states':
            s2 = net2.state_dict()
            d = net_plotter.get_diff_states(s, s2)
        
        if hasattr(args, 'ignore') and args.ignore == 'biasbn':
            net_plotter.ignore_biasbn(d)
        
        d = tensorlist_to_tensor(d)
        matrix.append(d.numpy())
    
    # Perform PCA
    print("Performing PCA on model differences...")
    pca = PCA(n_components=n_components)
    pca.fit(np.array(matrix))
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Convert back to parameter structure
    pc1 = pca.components_[0]
    pc2 = pca.components_[1] if n_components > 1 else None
    
    if args.dir_type == 'weights':
        xdirection = npvec_to_tensorlist(pc1, w)
        ydirection = npvec_to_tensorlist(pc2, w) if pc2 is not None else None
    elif args.dir_type == 'states':
        xdirection = npvec_to_tensorlist(pc1, s)
        ydirection = npvec_to_tensorlist(pc2, s) if pc2 is not None else None
    
    if hasattr(args, 'ignore') and args.ignore == 'biasbn':
        net_plotter.ignore_biasbn(xdirection)
        if ydirection is not None:
            net_plotter.ignore_biasbn(ydirection)
    
    # Save to h5 file
    save_directions_to_h5(save_path, xdirection, ydirection, pca)
    return save_path


def gradient_to_param_structure(grad_vector, model):
    """Convert a flattened gradient vector back to model parameter structure"""
    directions = []
    start_idx = 0
    
    for param in model.parameters():
        param_size = param.numel()
        param_grad = grad_vector[start_idx:start_idx + param_size]
        directions.append(torch.tensor(param_grad.reshape(param.shape), dtype=torch.float32))
        start_idx += param_size
    
    return directions


def save_directions_to_h5(save_path, xdirection, ydirection, pca):
    """Save directions and PCA info to h5 file"""
    print(f"Saving directions to: {save_path}")
    
    f = h5py.File(save_path, 'w')
    h5_util.write_list(f, 'xdirection', xdirection)
    if ydirection is not None:
        h5_util.write_list(f, 'ydirection', ydirection)
    
    f['explained_variance_ratio_'] = pca.explained_variance_ratio_
    f['singular_values_'] = pca.singular_values_
    f['explained_variance_'] = pca.explained_variance_
    f['n_components'] = len(pca.components_)
    
    f.close()
    print(f"Directions saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Create PCA directions from gradients or model checkpoints')
    parser.add_argument('--mode', choices=['gradients', 'checkpoints'], required=True,
                       help='Create PCA from gradients or model checkpoints')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--model', default='custom_cnn', help='Model architecture')
    parser.add_argument('--dir_type', default='weights', choices=['weights', 'states'],
                       help='Direction type for checkpoint mode')
    parser.add_argument('--ignore', default='', help='Ignore bias and batchnorm')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components')
    parser.add_argument('--output', default='', help='Output file path')
    parser.add_argument('--gradient_dir', default='gradients', help='Directory containing gradient files')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory containing model checkpoints')
    
    args = parser.parse_args()
    
    if args.mode == 'gradients':
        # Find all gradient files
        gradient_pattern = os.path.join(args.gradient_dir, 'gradients_epoch_*.npy')
        gradient_files = sorted(glob.glob(gradient_pattern))
        
        if not gradient_files:
            print(f"No gradient files found in {args.gradient_dir}")
            return
        
        print(f"Found {len(gradient_files)} gradient files")
        
        # Create a model instance for parameter structure
        if args.dataset == 'xor' or args.model == 'xor':
            model = SimpleXORNet()
        else:
            model = SimpleCNN()
        
        output_path = args.output or 'pca_gradient_directions.h5'
        create_pca_from_gradients(gradient_files, model, args.n_components, output_path)
        
    elif args.mode == 'checkpoints':
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(args.checkpoint_dir, 'model_epoch_*.pth')
        checkpoint_files = sorted(glob.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            print(f"No checkpoint files found in {args.checkpoint_dir}")
            return
        
        print(f"Found {len(checkpoint_files)} checkpoint files")
        
        output_path = args.output or 'pca_model_directions.h5'
        create_pca_from_checkpoints(checkpoint_files, args, args.n_components, output_path)


if __name__ == '__main__':
    main()