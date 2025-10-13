# PCA-based Loss Landscape Visualization

This project has been enhanced to use Principal Component Analysis (PCA) instead of random directions for loss landscape visualization. This provides more meaningful and interpretable visualizations of the loss landscape.

## Overview

The PCA approach captures the directions of maximum variance in either:
1. **Gradient space**: Principal components of gradients collected during training
2. **Parameter space**: Principal components of model parameter differences across training epochs

## Files Added/Modified

### Core Files
- `train_model_new.py` - Modified to collect gradients during training and create PCA directions
- `create_pca_directions.py` - Standalone script to create PCA directions from gradients or checkpoints
- `plot_pca_landscape.py` - Plot loss landscapes using PCA directions
- `run_pca_workflow.py` - Complete workflow from training to visualization

## Quick Start

### Option 1: Complete Workflow (Recommended)
```bash
# Run the complete workflow for CIFAR-10
python run_pca_workflow.py --dataset cifar10

# Run the complete workflow for XOR dataset
python run_pca_workflow.py --dataset xor
```

### Option 2: Step-by-Step

1. **Train model and collect gradients:**
```bash
python train_model_new.py --dataset cifar10
```

2. **Create PCA directions from gradients:**
```bash
python create_pca_directions.py --mode gradients --dataset cifar10 --output pca_directions.h5
```

3. **Plot loss landscape:**
```bash
python plot_pca_landscape.py --model_file checkpoints/model_epoch_49.pth --dir_file pca_directions.h5 --show
```

## PCA Modes

### Gradient-based PCA
- Uses gradients collected during training
- Captures directions of maximum gradient variance
- More sensitive to optimization dynamics
```bash
python create_pca_directions.py --mode gradients
```

### Checkpoint-based PCA  
- Uses model parameter differences across epochs
- Captures directions of maximum parameter change
- Similar to the original implementation but with PCA
```bash
python create_pca_directions.py --mode checkpoints
```

## Command Line Options

### create_pca_directions.py
- `--mode`: Choose 'gradients' or 'checkpoints'
- `--dataset`: Dataset name (cifar10, xor)
- `--model`: Model architecture
- `--n_components`: Number of PCA components (default: 2)
- `--output`: Output file path for directions
- `--gradient_dir`: Directory containing gradient files
- `--checkpoint_dir`: Directory containing model checkpoints

### plot_pca_landscape.py
- `--model_file`: Path to trained model
- `--dir_file`: Path to PCA directions file
- `--x`, `--y`: Range and resolution for visualization
- `--show`: Display the plot
- `--vmin`, `--vmax`: Contour plot value range

## Understanding PCA Output

When you run the scripts, you'll see PCA information like:
```
PCA explained variance ratio: [0.65432109 0.23456789]
Total explained variance: 0.8889
```

This means:
- First principal component explains 65.4% of the variance
- Second principal component explains 23.5% of the variance  
- Together they explain 88.9% of the total variance

Higher explained variance indicates that the PCA directions capture more of the meaningful variation in your data.

## Comparison: Random vs PCA Directions

| Aspect | Random Directions | PCA Directions |
|--------|------------------|----------------|
| **Interpretability** | Low - arbitrary directions | High - directions of maximum variance |
| **Reproducibility** | Low - different each run | High - deterministic given data |
| **Relevance** | Low - may miss important features | High - captures dominant patterns |
| **Computational Cost** | Low | Medium - requires PCA computation |

## Examples

### CIFAR-10 with Gradient PCA
```bash
python run_pca_workflow.py --dataset cifar10 --mode gradients
```

### XOR with Checkpoint PCA  
```bash
python run_pca_workflow.py --dataset xor --mode checkpoints
```

### Custom Training then PCA
```bash
# Train for 20 epochs
python train_model_new.py --dataset cifar10

# Create PCA from gradients with 3 components
python create_pca_directions.py --mode gradients --n_components 3

# Plot with custom range
python plot_pca_landscape.py --model_file checkpoints/model_epoch_19.pth \
    --dir_file pca_gradient_directions.h5 --x -2:2:101 --y -2:2:101
```

## Tips

1. **Choose the right mode**: 
   - Use gradient PCA for studying optimization dynamics
   - Use checkpoint PCA for studying parameter evolution

2. **Check explained variance**: 
   - Aim for >80% total explained variance for meaningful visualizations
   - If low, consider collecting more diverse data or increasing components

3. **Experiment with resolution**: 
   - Start with 51x51 grids for quick previews
   - Use 101x101 or higher for publication-quality plots

4. **Compare with random**: 
   - Run both PCA and random directions to see the difference
   - PCA should show more structured, meaningful landscapes