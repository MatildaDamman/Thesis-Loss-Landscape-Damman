#!/usr/bin/env python3
"""
Example script showing how to run the PCA-based loss landscape workflow.
This demonstrates the exact sequence you mentioned in your question.
"""

import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and show its output"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ Command completed successfully")
        return True

def main():
    print("🔬 PCA-based Loss Landscape Workflow")
    print("This script runs your exact workflow with PCA directions instead of random directions.")
    
    # Step 1: Train model and collect gradients for PCA
    print("\n📍 STEP 1: Training model and collecting gradients...")
    cmd1 = "python train_model_new.py --dataset xor --datapath Xor_Dataset.csv --model xor"
    
    if not run_command(cmd1, "Training model and creating PCA directions"):
        print("❌ Training failed!")
        return False
    
    # Check if PCA directions were created
    if os.path.exists('pca_gradient_directions.h5'):
        print("✅ PCA directions file created successfully!")
    else:
        print("❌ PCA directions file not found!")
        return False
    
    # Step 2: Generate loss surfaces using PCA directions
    print("\n📍 STEP 2: Generating loss surfaces with PCA directions...")
    cmd2 = "python batch_plot_new_new.py --num_workers 3"
    
    if not run_command(cmd2, "Generating loss surfaces with PCA directions"):
        print("❌ Surface generation failed!")
        return False
    
    # Check if surfaces were created
    os.makedirs('surfaces', exist_ok=True)  # Ensure surfaces directory exists
    
    try:
        surfaces_exist = any(f.startswith('surface_epoch_') and f.endswith('.h5') 
                            for f in os.listdir('surfaces') if os.path.isfile(os.path.join('surfaces', f)))
    except FileNotFoundError:
        print("❌ Surfaces directory not found!")
        return False
    
    if surfaces_exist:
        print("✅ Surface files created successfully!")
        surface_count = len([f for f in os.listdir('surfaces') if f.startswith('surface_epoch_') and f.endswith('.h5')])
        print(f"📊 Created {surface_count} surface files")
    else:
        print("❌ No surface files found!")
        print("💡 Check if batch_plot_new_new.py completed successfully")
        return False
    
    # Step 3: Launch dashboard
    print("\n📍 STEP 3: Launching interactive dashboard...")
    cmd3 = "python dashboard2.py"
    
    print(f"Running: {cmd3}")
    print("🌐 Dashboard will open in your web browser.")
    print("📊 You'll see PCA-based loss landscapes instead of random directions!")
    print("🔄 The dashboard will show the evolution of loss landscapes across training epochs.")
    print("\n⚠️  Note: Press Ctrl+C to stop the dashboard when you're done.")
    
    # Run dashboard (this will block until user stops it)
    try:
        subprocess.run(cmd3, shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user.")
    
    print("\n🎉 Workflow completed!")
    print("\n📈 Summary of what you saw:")
    print("- Training collected gradients during optimization")
    print("- PCA found the principal directions of gradient variation")
    print("- Loss surfaces were computed along these meaningful PCA directions")
    print("- Dashboard visualized the PCA-based loss landscape evolution")
    print("\n🔬 This gives you much more interpretable results than random directions!")

if __name__ == '__main__':
    main()