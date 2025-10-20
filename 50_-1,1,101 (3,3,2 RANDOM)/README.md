# 3-3-2 Architecture Results

## Architecture Details
- **Model**: SimpleXORNet_332 (inferred)
- **Structure**: Linear(3,3) → ReLU → Linear(3,2) 
- **Parameters**: ~17 parameters total
- **Training Epochs**: 10-49 (40 epochs)

## Contents

### Checkpoints
- `model_epoch_10.pth` to `model_epoch_49.pth` - Model weights for each epoch
- `model_epoch_X.pth_weights_epoch_X.h5` - Direction files for loss landscape computation

### Surfaces
- `surface_epoch_10.h5` to `surface_epoch_49.h5` - Loss landscape surfaces for each epoch
- Grid resolution: 101 x 101 points  
- Range: [-1, 1] x [-1, 1]

## Model Architecture
```python
class SimpleXORNet_332(nn.Module):
    def __init__(self):
        super(SimpleXORNet_332, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 3),    # Input: 3 → Hidden: 3
            nn.ReLU(),          # Activation  
            nn.Linear(3, 2)     # Hidden: 3 → Output: 2
        )
```

## File Sizes
- Model checkpoints: 2312 bytes each
- This corresponds to the larger architecture with different parameter configuration

## Notes
- Despite having fewer neurons (3-3-2 vs 2-8-2), the checkpoint files are slightly larger
- This suggests different training configurations, optimizer states, or additional metadata
- The 3-input architecture may have been designed for a different variant of the XOR problem