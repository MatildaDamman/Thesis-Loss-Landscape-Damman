import os
import h5py
import torch
import numpy as np
import sys
from simple_cnn import SimpleXORNet
import net_plotter
import dataloader
import model_loader
import copy
import plot_2D
import plot_1D
import evaluation
import scheduler
from multiprocessing import Process, current_process
import argparse

EPOCHS = 50
EPOCH_PAD = len(str(EPOCHS))

# Global variables that will be set by command line arguments
DATASET = 'xor'
DATAPATH = 'Xor_Dataset.csv'
MODEL = 'xor'
BATCH_SIZE = 16
CHECKPOINT_DIR = 'checkpoints'
SURFACE_DIR = 'surfaces'
X_RANGE = (-1, 1, 101)
Y_RANGE = (-1, 1, 101)
PCA_DIRECTIONS_FILE = 'pca_gradient_directions.h5'

def compute_surface_for_epoch(epoch):
    epochs = EPOCHS  # Use the global value
    trainloader, _ = dataloader.load_dataset(DATASET, DATAPATH, BATCH_SIZE, 0, model_type=MODEL)
    model_file = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")  # No zero-padding
    if not os.path.exists(model_file):
        print(f"Checkpoint {model_file} not found, skipping.")
        return

    net = model_loader.load(DATASET, MODEL, model_file)
    w = net_plotter.get_weights(net)
    s = copy.deepcopy(net.state_dict())

    args = type('Args', (), {})()
    args.epoch = epoch
    args.xmin, args.xmax, args.xnum = X_RANGE
    args.ymin, args.ymax, args.ynum = Y_RANGE
    args.y = True
    args.dir_type = 'weights'
    args.ngpu = 1
    args.xnorm = ''
    args.ynorm = ''
    args.xignore = ''
    args.yignore = ''
    args.same_dir = False
    args.idx = 0
    args.surf_file = ''
    args.raw_data = False
    args.data_split = 1
    args.split_idx = 0
    args.trainloader = ''
    args.testloader = ''
    args.model = MODEL
    args.dataset = DATASET
    args.cuda = False
    args.loss_name = 'crossentropy'
    args.dir_file = PCA_DIRECTIONS_FILE if PCA_DIRECTIONS_FILE else 'pca_gradient_directions.h5'  # Use PCA directions instead of random
    args.model_file = model_file
    args.model_file2 = ''
    args.model_file3 = ''

    # Check if PCA directions file exists, otherwise fall back to random
    if not os.path.exists(args.dir_file):
        print(f"Warning: PCA directions file {args.dir_file} not found!")
        print("Falling back to random directions. Run train_model_new.py first to generate PCA directions.")
        args.dir_file = ''  # This will create random directions
    else:
        print(f"Using PCA directions from: {args.dir_file}")

    dir_file = net_plotter.name_direction_file(args)
    if args.dir_file:  # If we have PCA directions, use them
        dir_file = args.dir_file
    else:  # Otherwise create random directions
        net_plotter.setup_direction(args, dir_file, net)

    def name_surface_file(args, dir_file, epoch):
        surf_file = dir_file
        surf_file += f'_epoch_{epoch}'  # Make unique per epoch
        surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
        if args.y:
            surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))
        return surf_file + ".h5"

    surf_file = name_surface_file(args, dir_file, epoch)
    if not os.path.exists(surf_file):
        f = h5py.File(surf_file, 'a')
        f['dir_file'] = dir_file
        xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
        f['xcoordinates'] = xcoordinates
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
        f.close()

    d = net_plotter.load_directions(dir_file)
    scheduler_comm = None
    scheduler_rank = 0
    
    # Add some delay to avoid file conflicts
    import time
    import random
    time.sleep(random.uniform(0.1, 0.5))
    
    crunch = __import__('plot_surface').crunch
    try:
        crunch(surf_file, net, w, s, d, trainloader,
               'train_loss', 'train_acc', scheduler_comm, scheduler_rank, args, epoch=epoch)
        print(f"Completed surface for epoch {epoch}: {surf_file}")
    except Exception as e:
        print(f"Error processing epoch {epoch}: {e}")
        return

    output_file = os.path.join(SURFACE_DIR, f'surface_epoch_{epoch}.h5')  # No zero-padding
    with h5py.File(surf_file, 'r') as src, h5py.File(output_file, 'w') as dst:
        for key in src.keys():
            src.copy(key, dst)
    print(f"Copied surface to {output_file}")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch plot loss landscapes using PCA directions')
    parser.add_argument('--dataset', default='xor', help='Dataset name')
    parser.add_argument('--datapath', default='Xor_Dataset.csv', help='Path to dataset')
    parser.add_argument('--model', default='xor', help='Model architecture')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory containing model checkpoints')
    parser.add_argument('--surface_dir', default='surfaces', help='Directory to save surface files')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to process')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch for resume (default: 0)')
    parser.add_argument('--x_range', default='(-1,1,101)', help='X range as (min,max,num)')
    parser.add_argument('--y_range', default='(-1,1,101)', help='Y range as (min,max,num)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of worker processes')
    parser.add_argument('--pca_directions_file', default='pca_gradient_directions.h5', 
                       help='Path to PCA directions file')
    parser.add_argument('--max_epochs', default=None, type=int, 
                       help='Maximum number of epochs to process (for testing)')
    
    args = parser.parse_args()
    
    # Set global variables
    DATASET = args.dataset
    DATAPATH = args.datapath
    MODEL = args.model
    BATCH_SIZE = args.batch_size
    CHECKPOINT_DIR = args.checkpoint_dir
    SURFACE_DIR = args.surface_dir
    PCA_DIRECTIONS_FILE = args.pca_directions_file
    
    # Parse range strings
    def parse_range(range_str):
        # Remove parentheses and split by comma
        range_str = range_str.strip('()')
        parts = [float(x) if '.' in x else int(x) for x in range_str.split(',')]
        return tuple(parts)
    
    X_RANGE = parse_range(args.x_range)
    Y_RANGE = parse_range(args.y_range)
    
    # Create output directory
    os.makedirs(SURFACE_DIR, exist_ok=True)
    
    # Auto-detect which checkpoints exist and need processing
    print(" Auto-detecting available checkpoints...")
    checkpoint_files = []
    available_epochs = []
    
    for epoch in range(args.start_epoch, args.epochs):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        surface_path = os.path.join(SURFACE_DIR, f"surface_epoch_{epoch}.h5")
        
        if os.path.exists(checkpoint_path):
            # Only process if surface file doesn't exist (resume capability)
            if not os.path.exists(surface_path):
                checkpoint_files.append(checkpoint_path)
                available_epochs.append(epoch)
            else:
                print(f"⏭️  Skipping epoch {epoch} - surface file already exists")
    
    print(f" Found {len(available_epochs)} checkpoints: {available_epochs}")
    
    # Limit epochs if max_epochs is specified
    if args.max_epochs and len(available_epochs) > args.max_epochs:
        available_epochs = available_epochs[:args.max_epochs]
        print(f"Limited to first {args.max_epochs} epochs: {available_epochs}")
    
    if not available_epochs:
        print(" No checkpoints found! Please run training first.")
        sys.exit(1)
    
    print(f"Using PCA directions from: {PCA_DIRECTIONS_FILE}")
    print(f"Processing {len(available_epochs)} available epochs sequentially to avoid file conflicts")
    print(f"Surface range: X{X_RANGE}, Y{Y_RANGE}")
    
    # Process only available epochs sequentially to avoid file conflicts
    for i, epoch in enumerate(available_epochs):
        print(f"Processing epoch {epoch} ({i+1}/{len(available_epochs)})...")
        compute_surface_for_epoch(epoch)
    
    print(f"Completed processing {len(available_epochs)} epochs!")
