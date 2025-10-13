#!/usr/bin/env python3
"""
Complete example: Train model, create PCA directions, and plot loss landscape.
This script demonstrates the full workflow from training to visualization using PCA.
"""

import os
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and print its output"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Complete PCA loss landscape workflow')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'xor'],
                       help='Dataset to use')
    parser.add_argument('--model', default='custom_cnn', help='Model architecture')
    parser.add_argument('--datapath', default='./data', help='Dataset path')
    parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--skip_training', action='store_true', help='Skip training step')
    parser.add_argument('--mode', default='gradients', choices=['gradients', 'checkpoints'],
                       help='PCA mode: use gradients or model checkpoints')
    
    args = parser.parse_args()
    
    # Adjust parameters for XOR dataset
    if args.dataset == 'xor':
        args.datapath = 'Xor_Dataset.csv'
        args.model = 'xor'
    
    print("PCA Loss Landscape Visualization Workflow")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"PCA Mode: {args.mode}")
    
    # Step 1: Train the model (if not skipping)
    if not args.skip_training:
        train_cmd = f"python train_model_new.py --dataset {args.dataset} --datapath {args.datapath}"
        if not run_command(train_cmd, "Training model and collecting gradients"):
            print("Training failed!")
            return
    else:
        print("Skipping training step...")
    
    # Step 2: Create PCA directions
    if args.mode == 'gradients':
        pca_cmd = f"python create_pca_directions.py --mode gradients --dataset {args.dataset} --model {args.model} --output pca_gradient_directions.h5"
        dir_file = "pca_gradient_directions.h5"
    else:
        pca_cmd = f"python create_pca_directions.py --mode checkpoints --dataset {args.dataset} --model {args.model} --output pca_model_directions.h5"
        dir_file = "pca_model_directions.h5"
    
    if not run_command(pca_cmd, f"Creating PCA directions from {args.mode}"):
        print("PCA direction creation failed!")
        return
    
    # Step 3: Find the latest model checkpoint
    import glob
    checkpoint_files = glob.glob("checkpoints/model_epoch_*.pth")
    if not checkpoint_files:
        print("No model checkpoints found!")
        return
    
    latest_checkpoint = sorted(checkpoint_files)[-1]
    print(f"Using latest checkpoint: {latest_checkpoint}")
    
    # Step 4: Plot the loss landscape
    plot_cmd = (f"python plot_pca_landscape.py "
                f"--model_file {latest_checkpoint} "
                f"--dataset {args.dataset} "
                f"--model {args.model} "
                f"--datapath {args.datapath} "
                f"--dir_file {dir_file} "
                f"--show")
    
    if not run_command(plot_cmd, "Plotting loss landscape with PCA directions"):
        print("Plotting failed!")
        return
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"PCA directions saved to: {dir_file}")
    print(f"Loss landscape plots generated using PCA directions")
    print("\nOutput files:")
    print(f"- PCA directions: {dir_file}")
    print(f"- Surface data: pca_surface_*.h5")
    print(f"- Plot images: *.png")
    
    # Step 5: Show PCA information
    if os.path.exists(dir_file):
        try:
            import h5py
            import numpy as np
            
            print(f"\nPCA Direction Information:")
            print("-" * 40)
            
            with h5py.File(dir_file, 'r') as f:
                if 'explained_variance_ratio_' in f.keys():
                    variance_ratio = f['explained_variance_ratio_'][:]
                    print(f"Explained variance ratio: {variance_ratio}")
                    print(f"Total explained variance: {np.sum(variance_ratio):.4f}")
                    print(f"First component explains: {variance_ratio[0]:.4f} of variance")
                    print(f"Second component explains: {variance_ratio[1]:.4f} of variance")
                
                print(f"Available directions: {list(f.keys())}")
                
        except Exception as e:
            print(f"Could not read PCA information: {e}")


if __name__ == '__main__':
    main()