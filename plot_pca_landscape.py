#!/usr/bin/env python3
"""
Plot loss landscape using PCA directions instead of random directions.
This script demonstrates how to use PCA directions for 2D loss landscape visualization.
"""

import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

import net_plotter
import plot_2D
import plot_1D
from simple_cnn import SimpleCNN, SimpleXORNet
import model_loader
import scheduler
import projection


def plot_loss_landscape_with_pca(args):
    """
    Plot 2D loss landscape using PCA directions instead of random directions.
    """
    
    # Load the trained model
    net = model_loader.load(args.dataset, args.model, args.model_file)
    
    # Load PCA direction file
    if not os.path.exists(args.dir_file):
        print(f"PCA direction file {args.dir_file} does not exist!")
        print("Please create PCA directions first using create_pca_directions.py")
        return
    
    print(f"Using PCA directions from: {args.dir_file}")
    
    # Verify that the direction file contains PCA directions
    with h5py.File(args.dir_file, 'r') as f:
        if 'explained_variance_ratio_' in f.keys():
            variance_ratio = f['explained_variance_ratio_'][:]
            print(f"PCA explained variance ratio: {variance_ratio}")
            print(f"Total explained variance: {np.sum(variance_ratio):.4f}")
        else:
            print("Warning: Direction file doesn't appear to contain PCA information")
    
    # Setup loss function and dataloader
    criterion = nn.CrossEntropyLoss()
    
    if args.dataset == 'xor' or args.model == 'xor':
        # Load XOR dataset
        df = pd.read_csv(args.datapath)
        X = df.iloc[:, :-1].values.astype('float32')
        y = df.iloc[:, -1].values.astype('int64')
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        trainset = TensorDataset(tensor_x, tensor_y)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=False)
        testloader = trainloader  # Use same for simplicity
    else:
        # CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    # Create surface file name
    surf_file = f"pca_surface_{args.dataset}_{args.model}"
    if args.model_file:
        surf_file += f"_{os.path.basename(args.model_file).replace('.pth', '')}"
    surf_file += f"_{args.x}_{args.y}_{args.xnum}.h5"
    
    # Calculate the loss surface
    plot_2D.crunch(surf_file, net, args.x, args.y, args.xnum, args.ynum, 
                   trainloader, criterion, args.dir_file, args.xnorm, args.ynorm,
                   args.xmin, args.xmax, args.ymin, args.ymax, args.cuda,
                   args.raw_data, args.print_freq, args.log, args.verbose)
    
    # Plot the surface
    plot_2D.plot_2d_contour(surf_file, args.vmin, args.vmax, args.vlevel, args.show)


def main():
    parser = argparse.ArgumentParser(description='Plot loss landscape using PCA directions')
    
    # Model and data parameters
    parser.add_argument('--model_file', required=True, help='Path to the trained model file')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--model', default='custom_cnn', help='Model architecture')
    parser.add_argument('--datapath', default='./data', help='Path to dataset')
    
    # PCA direction file
    parser.add_argument('--dir_file', required=True, help='Path to PCA directions h5 file')
    
    # Surface parameters
    parser.add_argument('--x', default='-1:1:51', help='x range and number of points')
    parser.add_argument('--y', default='-1:1:51', help='y range and number of points') 
    parser.add_argument('--xnum', default=51, type=int, help='Number of points along x-axis')
    parser.add_argument('--ynum', default=51, type=int, help='Number of points along y-axis')
    parser.add_argument('--xmin', default=-1.0, type=float, help='x minimum')
    parser.add_argument('--xmax', default=1.0, type=float, help='x maximum')
    parser.add_argument('--ymin', default=-1.0, type=float, help='y minimum')
    parser.add_argument('--ymax', default=1.0, type=float, help='y maximum')
    
    # Normalization
    parser.add_argument('--xnorm', default='filter', help='x direction normalization')
    parser.add_argument('--ynorm', default='filter', help='y direction normalization')
    
    # Plotting parameters
    parser.add_argument('--vmin', default=0.1, type=float, help='Minimum value for contour plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value for contour plot')
    parser.add_argument('--vlevel', default=0.5, type=float, help='Level interval for contour plot')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    
    # Technical parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--threads', default=2, type=int, help='Number of threads')
    parser.add_argument('--raw_data', action='store_true', help='Store raw data')
    parser.add_argument('--print_freq', default=10, type=int, help='Print frequency')
    parser.add_argument('--log', action='store_true', help='Use log scale')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    plot_loss_landscape_with_pca(args)


if __name__ == '__main__':
    main()