import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import numpy as np

# Function to load surface from h5 file
def load_h5_surface(filename):
    with h5py.File(filename, 'r') as f:
        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        z = np.array(f['train_loss'][:])
    return x, y, z

# Preload data for all epochs
epochs = 10
surfaces = []
for epoch in range(epochs):
    try:
        x, y, z = load_h5_surface(f'surfaces/surface_epoch_{epoch}.h5')
        X, Y = np.meshgrid(x, y)
        surfaces.append((X, Y, z))
        print(f"Loaded surface for epoch {epoch}")
    except Exception as e:
        print(f"Failed to load epoch {epoch}: {e}")
        surfaces.append(None)

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(epoch):
    ax.clear()
    data = surfaces[epoch]
    if data is None:
        return
    X, Y, Z = data
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Direction X')
    ax.set_ylabel('Direction Y')
    ax.set_zlabel('Loss')
    ax.set_title(f'Loss Landscape at Epoch {epoch}')

ani = animation.FuncAnimation(fig, update, frames=epochs, interval=500)

ani.save('loss_landscape_evolution.gif', writer='pillow', fps=2)
print("Animation saved as 'loss_landscape done")
