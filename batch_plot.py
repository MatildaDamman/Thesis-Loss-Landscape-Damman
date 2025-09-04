import subprocess
import os

os.makedirs('surfaces', exist_ok=True)

for epoch in range(10):
    output_file = f'surfaces/surface_epoch_{epoch}.h5'
    intermediate_file = f"checkpoints/model_epoch_{epoch}.pth_weights_xnorm=filter_ynorm=filter.h5"
    
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
        print(f"Removed cache file: {intermediate_file}")

    try:
        result = subprocess.run([
            "python", "plot_surface.py",
            "--model", "custom_cnn",
            "--x=-1:1:4",
            "--y=-1:1:4",
            "--model_file", f"checkpoints/model_epoch_{epoch}.pth",
            "--dir_type", "weights",
            "--xnorm", "filter",
            "--ynorm", "filter",
            "--surf_file", output_file,
        ], timeout=300)  # 5-minute timeout
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for epoch {epoch}")
        continue

    if result.returncode != 0:
        print(f"Error occurred at epoch {epoch}")
    else:
        print(f"Completed surface for epoch {epoch}")


