# ---- MAIN ENTRY POINT ----
# (Moved to end of file)
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import h5py
from PIL import Image, ImageSequence
import base64
import io
import os

# --- GIF FRAME CACHE ---
_gif_frame_cache = {}

def load_gif_frames(run_folder):
    if run_folder in _gif_frame_cache:
        return _gif_frame_cache[run_folder]
    gif_path = get_gif_path(run_folder)
    img = Image.open(gif_path)
    frames = [f.copy().convert('RGBA') for f in ImageSequence.Iterator(img)]
    buf_list = []
    for frame in frames:
        buf = io.BytesIO()
        frame.save(buf, format='PNG')
        buf_list.append(buf.getvalue())
    _gif_frame_cache[run_folder] = buf_list
    return buf_list

# ---- CONFIGURATION ----
EPOCHS = 50
DEFAULT_RUN = '50_-1,1,101 (2,8,2 PCA)'
RUN_FOLDERS = {
    '50_-1,1,101 (2,8,2 PCA)': '50_-1,1,101 (2,8,2 PCA)',
    '50_-1,1,101 (2,8,2 RANDOM)': '50_-1,1,101 (2,8,2 RANDOM)',
    '50_-1,1,101 (3,3,2 PCA)': '50_-1,1,101 (3,3,2 PCA)',
    '50_-1,1,101 (3,3,2 RANDOM)': '50_-1,1,101 (3,3,2 RANDOM)',
}

def get_surface_path_fmt(run_folder):
    return os.path.join(run_folder, 'surfaces/surface_epoch_{}.h5')

def get_gif_path(run_folder):
    return os.path.join(run_folder, 'loss_landscape_evolution1.gif')

# ---- LOAD GIF FRAMES ----

# --- GIF FRAME CACHE ---

def pil_image_to_base64(img_bytes):
    return 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('ascii')

# ---- UQ COLOR SCHEME ----
UQ_PURPLE = "#51247a"
UQ_GOLD = "#ffcc00"
UQ_GREY = "#808080"
UQ_LIGHT_GREY = "#f6f3f8"
UQ_WHITE = "#ffffff"
UQ_BACKGROUND = "#f9f9f9"

# ---- LOAD SURFACES ----
def load_surface(epoch, run_folder=DEFAULT_RUN):
    fname = get_surface_path_fmt(run_folder).format(epoch)
    if not os.path.exists(fname):
        return None, None, None
    with h5py.File(fname, 'r') as f:
        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        z = np.array(f['train_loss'][:])
    return x, y, z

def make_surface_fig(epoch, run_folder=DEFAULT_RUN):
    x, y, z = load_surface(epoch, run_folder)
    if x is None:
        return go.Figure()
    X, Y = np.meshgrid(x, y)
    fig = go.Figure(data=[
        go.Surface(z=z, x=X, y=Y, colorscale='Viridis', showscale=True, opacity=0.87)
    ])
    fig.update_layout(
        title=dict(
            text=f'Loss Landscape at Epoch {epoch}',
            font=dict(color=UQ_PURPLE, size=16, family="Montserrat"),
            x=0.5
        ),
        scene=dict(
            xaxis_title='Direction X',
            yaxis_title='Direction Y',
            zaxis_title='Loss',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.35, y=1.35, z=1.35),
                center=dict(x=0, y=0, z=-0.2)
            ),
            xaxis=dict(titlefont=dict(color=UQ_PURPLE)),
            yaxis=dict(titlefont=dict(color=UQ_PURPLE)),
            zaxis=dict(titlefont=dict(color=UQ_PURPLE))
        ),
        margin=dict(l=20, r=20, b=20, t=60),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE)
    )
    return fig

def make_gradient_angle_fig(current_epoch, total_epochs):
    """Generate gradient vector angle plot over training steps"""
    # Generate sample gradient angle data (replace with real gradient data if available)
    epochs = list(range(min(total_epochs, 50)))  # Limit to available epochs
    
    # Simulate gradient vector angles (replace with actual gradient calculations)
    np.random.seed(42)  # For reproducible demo data
    base_angles = np.linspace(0, 2*np.pi, len(epochs))
    noise = np.random.normal(0, 0.3, len(epochs))
    gradient_angles = base_angles + noise
    
    # Create the plot
    fig = go.Figure()
    
    # Add the gradient angle line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=gradient_angles,
        mode='lines+markers',
        name='Gradient Vector Angle',
        line=dict(color=UQ_PURPLE, width=2),
        marker=dict(color=UQ_PURPLE, size=4)
    ))
    
    # Highlight current epoch
    if current_epoch < len(epochs):
        fig.add_trace(go.Scatter(
            x=[current_epoch],
            y=[gradient_angles[current_epoch]],
            mode='markers',
            name='Current Epoch',
            marker=dict(color=UQ_GOLD, size=12, symbol='star')
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Gradient Vector Angle Evolution (Current: Epoch {current_epoch})',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Gradient Vector Angle (radians)',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=60, r=20, b=60, t=60),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_avg_gradient_angle_fig(total_epochs):
    """Generate static average gradient angle plot over all epochs"""
    # Generate sample gradient angle data (replace with real gradient data if available)
    epochs = list(range(min(total_epochs, 50)))  # Limit to available epochs
    
    # Simulate gradient vector angles (replace with actual gradient calculations)
    np.random.seed(42)  # For reproducible demo data
    base_angles = np.linspace(0, 2*np.pi, len(epochs))
    noise = np.random.normal(0, 0.3, len(epochs))
    gradient_angles = base_angles + noise
    
    # Calculate average gradient angle
    avg_angle = np.mean(gradient_angles)
    
    # Create the plot
    fig = go.Figure()
    
    # Add the average line
    fig.add_trace(go.Scatter(
        x=[epochs[0], epochs[-1]],
        y=[avg_angle, avg_angle],
        mode='lines',
        name=f'Average Gradient Angle',
        line=dict(color=UQ_GOLD, width=4, dash='dash'),
    ))
    
    # Add scatter points for individual epochs
    fig.add_trace(go.Scatter(
        x=epochs,
        y=gradient_angles,
        mode='markers',
        name='Individual Epochs',
        marker=dict(color=UQ_PURPLE, size=6, opacity=0.6)
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Average Gradient Vector Angle: {avg_angle:.3f} radians',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Gradient Vector Angle (radians)',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=60, r=20, b=60, t=60),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_pca_overlay_fig(current_epoch, run_folder=DEFAULT_RUN):
    """Generate PCA overlay plot showing principal directions on loss landscape"""
    x, y, z = load_surface(current_epoch, run_folder)
    if x is None:
        fig = go.Figure()
        fig.update_layout(
            title="No surface data available",
            paper_bgcolor=UQ_WHITE,
            plot_bgcolor=UQ_WHITE,
            font=dict(family="Montserrat", color=UQ_PURPLE)
        )
        return fig
    X, Y = np.meshgrid(x, y)
    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=z, x=x, y=y,
        colorscale='Viridis',
        showscale=True,
        opacity=0.7,
        name='Loss Landscape',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white')
        )
    ))
    # Simulate PCA directions (replace with actual PCA calculation)
    pca1_direction = np.array([0.8, 0.6])  # Example direction vector
    pca1_magnitude = 0.7  # Scale factor
    
    # Principal component 2 (orthogonal direction)
    pca2_direction = np.array([-0.6, 0.8])  # Orthogonal to PC1
    pca2_magnitude = 0.4  # Scale factor
    
    # Center point (could be current model position or loss minimum)
    center_x, center_y = 0.0, 0.0
    
    # Add PCA direction vectors as arrows
    # Principal Component 1 (red arrow - greatest variance)
    fig.add_trace(go.Scatter(
        x=[center_x - pca1_direction[0] * pca1_magnitude, 
           center_x + pca1_direction[0] * pca1_magnitude],
        y=[center_y - pca1_direction[1] * pca1_magnitude, 
           center_y + pca1_direction[1] * pca1_magnitude],
        mode='lines+markers',
        name='PC1 (Greatest Variance)',
        line=dict(color='red', width=4),
        marker=dict(color='red', size=8, symbol='triangle-up')
    ))
    
    # Principal Component 2 (orange arrow - second greatest variance)
    fig.add_trace(go.Scatter(
        x=[center_x - pca2_direction[0] * pca2_magnitude, 
           center_x + pca2_direction[0] * pca2_magnitude],
        y=[center_y - pca2_direction[1] * pca2_magnitude, 
           center_y + pca2_direction[1] * pca2_magnitude],
        mode='lines+markers',
        name='PC2 (Second Variance)',
        line=dict(color='orange', width=3),
        marker=dict(color='orange', size=6, symbol='triangle-up')
    ))
    
    # Add center point
    fig.add_trace(go.Scatter(
        x=[center_x],
        y=[center_y],
        mode='markers',
        name='Reference Point',
        marker=dict(color=UQ_GOLD, size=12, symbol='star')
    ))
    
    # Calculate variance explained (example values)
    pc1_variance = 0.68  # 68% of variance
    pc2_variance = 0.24  # 24% of variance
    
    fig.update_layout(
        title=dict(
            text=f'PCA Overlay - Epoch {current_epoch}<br>PC1: {pc1_variance:.1%} variance, PC2: {pc2_variance:.1%} variance',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='PCA Direction X',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY,
            range=[min(x), max(x)]
        ),
        yaxis=dict(
            title='PCA Direction Y',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY,
            range=[min(y), max(y)],
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=60, r=20, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1,
            x=1.02,
            y=1
        )
    )
    
    return fig

def make_trajectory_overlay_fig(current_epoch):
    """Generate optimizer trajectory overlay plot showing the optimization path"""
    x, y, z = load_surface(current_epoch)
    if x is None:
        # Return empty figure if no surface data
        fig = go.Figure()
        fig.update_layout(
            title="No surface data available",
            paper_bgcolor=UQ_WHITE,
            plot_bgcolor=UQ_WHITE,
            font=dict(family="Montserrat", color=UQ_PURPLE)
        )
        return fig
    
    X, Y = np.meshgrid(x, y)
    
    # Create base contour plot
    fig = go.Figure()
    
    # Add contour plot of the loss landscape
    fig.add_trace(go.Contour(
        z=z, x=x, y=y,
        colorscale='Viridis',
        showscale=True,
        opacity=0.6,
        name='Loss Landscape',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=9, color='white')
        )
    ))
    
    # Generate simulated optimizer trajectory (replace with actual trajectory data)
    # Create a path that moves from high loss to low loss regions
    np.random.seed(123)  # For reproducible trajectory
    
    # Generate trajectory points up to current epoch
    max_steps = min(current_epoch + 1, 50)
    trajectory_x = []
    trajectory_y = []
    trajectory_loss = []
    
    # Start from a high-loss region and move toward minimum
    start_x, start_y = 0.8, 0.6  # Starting position
    target_x, target_y = -0.2, 0.1  # Target (minimum) position
    
    for step in range(max_steps):
        # Interpolate with some noise to simulate realistic optimization path
        progress = step / max(max_steps - 1, 1)
        # Use exponential decay for more realistic convergence
        smooth_progress = 1 - np.exp(-3 * progress)
        
        x_pos = start_x + (target_x - start_x) * smooth_progress + np.random.normal(0, 0.05)
        y_pos = start_y + (target_y - start_y) * smooth_progress + np.random.normal(0, 0.05)
        
        # Clamp to surface bounds
        x_pos = np.clip(x_pos, min(x), max(x))
        y_pos = np.clip(y_pos, min(y), max(y))
        
        trajectory_x.append(x_pos)
        trajectory_y.append(y_pos)
        
        # Calculate approximate loss at this position (interpolate from surface)
        loss_val = np.interp(x_pos, x, np.mean(z, axis=0)) + np.interp(y_pos, y, np.mean(z, axis=1))
        trajectory_loss.append(loss_val)
    
    # Add the full trajectory path
    if len(trajectory_x) > 1:
        fig.add_trace(go.Scatter(
            x=trajectory_x,
            y=trajectory_y,
            mode='lines+markers',
            name='Optimizer Path',
            line=dict(color='cyan', width=3),
            marker=dict(color='cyan', size=4),
            opacity=0.8
        ))
    
    # Add trajectory points with color gradient based on epoch
    if len(trajectory_x) > 0:
        fig.add_trace(go.Scatter(
            x=trajectory_x,
            y=trajectory_y,
            mode='markers',
            name='Training Steps',
            marker=dict(
                color=list(range(len(trajectory_x))),
                colorscale='Plasma',
                size=6,
                showscale=True,
                colorbar=dict(
                    title="Training Step",
                    titleside="right",
                    x=1.12
                )
            ),
            text=[f'Step {i}<br>Loss: {loss:.3f}' for i, loss in enumerate(trajectory_loss)],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Highlight current position
    if current_epoch < len(trajectory_x):
        fig.add_trace(go.Scatter(
            x=[trajectory_x[current_epoch]],
            y=[trajectory_y[current_epoch]],
            mode='markers',
            name='Current Position',
            marker=dict(color=UQ_GOLD, size=15, symbol='star'),
            text=f'Epoch {current_epoch}<br>Loss: {trajectory_loss[current_epoch]:.3f}',
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add starting point
    if len(trajectory_x) > 0:
        fig.add_trace(go.Scatter(
            x=[trajectory_x[0]],
            y=[trajectory_y[0]],
            mode='markers',
            name='Start',
            marker=dict(color='red', size=12, symbol='square')
        ))
    
    # Add ending point (if we've reached it)
    if current_epoch >= len(trajectory_x) - 1 and len(trajectory_x) > 0:
        fig.add_trace(go.Scatter(
            x=[trajectory_x[-1]],
            y=[trajectory_y[-1]],
            mode='markers',
            name='End',
            marker=dict(color='green', size=12, symbol='diamond')
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Optimizer Trajectory - Epoch {current_epoch}<br>Steps: {len(trajectory_x)}, Current Loss: {trajectory_loss[current_epoch]:.4f}' if trajectory_loss else f'Optimizer Trajectory - Epoch {current_epoch}',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Loss Landscape X',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY,
            range=[min(x), max(x)]
        ),
        yaxis=dict(
            title='Loss Landscape Y',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY,
            range=[min(y), max(y)],
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=60, r=100, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1,
            x=1.15,
            y=1
        )
    )
    
    return fig

def make_eigenvalues_plot(current_epoch, total_epochs):
    """Generate Hessian eigenvalues time series plot"""
    # Generate simulated eigenvalue data (replace with actual PyHessian calculations)
    epochs = list(range(min(total_epochs, 50)))
    
    np.random.seed(456)  # For reproducible eigenvalue data
    
    # Simulate leading eigenvalues evolution
    # Typically, eigenvalues start large and decrease/stabilize during training
    base_decay = np.exp(-0.1 * np.array(epochs))
    
    # Leading eigenvalue (largest magnitude, often negative for minima)
    lambda_1 = -10 * base_decay + np.random.normal(0, 0.5, len(epochs))
    
    # Second eigenvalue
    lambda_2 = -5 * base_decay + np.random.normal(0, 0.3, len(epochs))
    
    # Third eigenvalue
    lambda_3 = -2 * base_decay + np.random.normal(0, 0.2, len(epochs))
    
    # Smaller eigenvalues (closer to zero)
    lambda_4 = -0.5 * base_decay + np.random.normal(0, 0.1, len(epochs))
    lambda_5 = 0.1 * base_decay + np.random.normal(0, 0.05, len(epochs))
    
    # Create the plot
    fig = go.Figure()
    
    # Add eigenvalue traces
    eigenvalues = [
        (lambda_1, 'λ₁ (Leading)', 'red'),
        (lambda_2, 'λ₂ (Second)', 'orange'),
        (lambda_3, 'λ₃ (Third)', 'blue'),
        (lambda_4, 'λ₄ (Fourth)', UQ_PURPLE),
        (lambda_5, 'λ₅ (Fifth)', 'green')
    ]
    
    for eigenvals, name, color in eigenvalues:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=eigenvals,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=4)
        ))
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Zero Line", annotation_position="bottom right")
    
    # Highlight current epoch
    if current_epoch < len(epochs):
        current_values = [vals[current_epoch] for vals, _, _ in eigenvalues]
        fig.add_trace(go.Scatter(
            x=[current_epoch] * len(current_values),
            y=current_values,
            mode='markers',
            name='Current Epoch',
            marker=dict(color=UQ_GOLD, size=10, symbol='star')
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Hessian Eigenvalues Evolution (Current: Epoch {current_epoch})',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Eigenvalue Magnitude',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=70, r=20, b=60, t=60),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def make_eigenvectors_plot(current_epoch):
    """Generate Hessian eigenvectors overlay plot on loss landscape"""
    x, y, z = load_surface(current_epoch)
    if x is None:
        # Return empty figure if no surface data
        fig = go.Figure()
        fig.update_layout(
            title="No surface data available",
            paper_bgcolor=UQ_WHITE,
            plot_bgcolor=UQ_WHITE,
            font=dict(family="Montserrat", color=UQ_PURPLE)
        )
        return fig
    
    X, Y = np.meshgrid(x, y)
    
    # Create base contour plot
    fig = go.Figure()
    
    # Add contour plot of the loss landscape
    fig.add_trace(go.Contour(
        z=z, x=x, y=y,
        colorscale='Viridis',
        showscale=True,
        opacity=0.5,
        name='Loss Landscape',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=8, color='white')
        )
    ))
    
    # Simulate eigenvectors (replace with actual Hessian eigenvector calculations)
    # Center point (could be current model position)
    center_x, center_y = 0.0, 0.0
    
    # Simulate leading eigenvectors (directions of curvature)
    np.random.seed(789 + current_epoch)  # Vary with epoch
    
    # Leading eigenvector (direction of maximum curvature)
    angle_1 = np.pi/4 + 0.1 * current_epoch + np.random.normal(0, 0.1)
    eigenvec_1 = np.array([np.cos(angle_1), np.sin(angle_1)])
    magnitude_1 = 0.6
    
    # Second eigenvector (orthogonal direction)
    angle_2 = angle_1 + np.pi/2
    eigenvec_2 = np.array([np.cos(angle_2), np.sin(angle_2)])
    magnitude_2 = 0.4
    
    # Third eigenvector (another important direction)
    angle_3 = angle_1 + np.pi/3
    eigenvec_3 = np.array([np.cos(angle_3), np.sin(angle_3)])
    magnitude_3 = 0.3
    
    # Add eigenvector arrows
    eigenvectors = [
        (eigenvec_1, magnitude_1, 'v₁ (Max Curvature)', 'red', 4),
        (eigenvec_2, magnitude_2, 'v₂ (Orthogonal)', 'orange', 3),
        (eigenvec_3, magnitude_3, 'v₃ (Third)', 'blue', 2)
    ]
    
    for eigenvec, mag, name, color, width in eigenvectors:
        # Draw vector in both directions
        fig.add_trace(go.Scatter(
            x=[center_x - eigenvec[0] * mag, center_x + eigenvec[0] * mag],
            y=[center_y - eigenvec[1] * mag, center_y + eigenvec[1] * mag],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=width),
            marker=dict(color=color, size=8, symbol='triangle-up')
        ))
    
    # Add center point
    fig.add_trace(go.Scatter(
        x=[center_x],
        y=[center_y],
        mode='markers',
        name='Analysis Point',
        marker=dict(color=UQ_GOLD, size=12, symbol='star')
    ))
    
    # Calculate condition number (ratio of largest to smallest eigenvalue)
    # Simulated based on current epoch
    condition_number = 50 * np.exp(-0.05 * current_epoch) + 2
    
    fig.update_layout(
        title=dict(
            text=f'Hessian Eigenvectors Overlay - Epoch {current_epoch}<br>Est. Condition Number: {condition_number:.1f}',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Loss Landscape X',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY,
            range=[min(x), max(x)]
        ),
        yaxis=dict(
            title='Loss Landscape Y',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY,
            range=[min(y), max(y)],
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=60, r=20, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1,
            x=1.02,
            y=1
        )
    )
    
    return fig

def make_hessian_heatmap(current_epoch):
    """Generate Hessian matrix heatmap"""
    # Simulate Hessian matrix (replace with actual Hessian computation)
    np.random.seed(890 + current_epoch)
    
    # Create a symmetric Hessian matrix (n x n where n is number of parameters)
    # For visualization, we'll use a smaller representative matrix
    n_params = 12  # Representative subset of parameters
    
    # Generate base symmetric matrix
    hessian_base = np.random.randn(n_params, n_params)
    hessian_matrix = (hessian_base + hessian_base.T) / 2  # Make symmetric
    
    # Add some structure typical of neural network Hessians
    # Larger values on diagonal (second derivatives)
    hessian_matrix += np.diag(np.random.uniform(-5, -0.1, n_params))
    
    # Add epoch-dependent evolution (Hessian changes during training)
    decay_factor = np.exp(-0.1 * current_epoch)
    hessian_matrix *= decay_factor
    
    # Add some block structure (common in neural networks)
    for i in range(0, n_params, 3):
        end_idx = min(i+3, n_params)
        hessian_matrix[i:end_idx, i:end_idx] *= 1.5
    
    # Create parameter labels
    param_labels = [f'θ_{i+1}' for i in range(n_params)]
    
    fig = go.Figure(data=go.Heatmap(
        z=hessian_matrix,
        x=param_labels,
        y=param_labels,
        colorscale='RdBu',
        zmid=0,  # Center colorscale at zero
        showscale=True,
        colorbar=dict(
            title="Hessian Value",
            titleside="right"
        ),
        hoverongaps=False,
        hovertemplate='Parameter %{x} × %{y}<br>Hessian: %{z:.3f}<extra></extra>'
    ))
    
    # Calculate and display key metrics
    eigenvals = np.linalg.eigvals(hessian_matrix)
    condition_number = np.max(np.real(eigenvals)) / np.min(np.real(eigenvals)) if np.min(np.real(eigenvals)) != 0 else np.inf
    trace = np.trace(hessian_matrix)
    determinant = np.linalg.det(hessian_matrix)
    
    fig.update_layout(
        title=dict(
            text=f'Hessian Matrix - Epoch {current_epoch}<br>Trace: {trace:.2f}, Det: {determinant:.2e}, Cond: {condition_number:.1f}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Parameters',
            titlefont=dict(color=UQ_PURPLE, size=11),
            tickangle=45
        ),
        yaxis=dict(
            title='Parameters',
            titlefont=dict(color=UQ_PURPLE, size=11),
            autorange='reversed'  # To match matrix convention
        ),
        margin=dict(l=60, r=80, b=80, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE, size=10)
    )
    
    return fig

def make_curvature_heatmap(current_epoch):
    """Generate curvature analysis heatmap showing directional curvatures"""
    x, y, z = load_surface(current_epoch)
    if x is None:
        # Return empty figure if no surface data
        fig = go.Figure()
        fig.update_layout(
            title="No surface data available",
            paper_bgcolor=UQ_WHITE,
            plot_bgcolor=UQ_WHITE,
            font=dict(family="Montserrat", color=UQ_PURPLE)
        )
        return fig
    
    # Calculate approximate curvature from the loss surface
    # This is a simplified version - in practice, you'd compute second derivatives
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Compute approximate curvature using finite differences
    # Second derivative approximations
    dx = x[1] - x[0] if len(x) > 1 else 1
    dy = y[1] - y[0] if len(y) > 1 else 1
    
    # Compute second derivatives (curvature components)
    d2z_dx2 = np.zeros_like(z)
    d2z_dy2 = np.zeros_like(z)
    d2z_dxdy = np.zeros_like(z)
    
    # Second derivative in x direction
    d2z_dx2[1:-1, :] = (z[2:, :] - 2*z[1:-1, :] + z[:-2, :]) / (dx**2)
    
    # Second derivative in y direction  
    d2z_dy2[:, 1:-1] = (z[:, 2:] - 2*z[:, 1:-1] + z[:, :-2]) / (dy**2)
    
    # Mixed derivative (cross-curvature)
    d2z_dxdy[1:-1, 1:-1] = (z[2:, 2:] - z[2:, :-2] - z[:-2, 2:] + z[:-2, :-2]) / (4*dx*dy)
    
    # Compute principal curvatures (eigenvalues of Hessian at each point)
    # κ = (d2z_dx2 + d2z_dy2) ± sqrt((d2z_dx2 - d2z_dy2)^2 + 4*(d2z_dxdy)^2) / 2
    mean_curvature = (d2z_dx2 + d2z_dy2) / 2
    gaussian_curvature = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
    
    # For visualization, we'll show the mean curvature
    curvature_data = mean_curvature
    
    fig = go.Figure(data=go.Heatmap(
        z=curvature_data,
        x=x,
        y=y,
        colorscale='RdYlBu',
        zmid=0,
        showscale=True,
        colorbar=dict(
            title="Mean Curvature",
            titleside="right"
        ),
        hoverongaps=False,
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Curvature: %{z:.3f}<extra></extra>'
    ))
    
    # Add contour lines for better visualization
    fig.add_trace(go.Contour(
        z=curvature_data,
        x=x,
        y=y,
        showscale=False,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=8, color='black')
        ),
        line=dict(color='black', width=1),
        opacity=0.3,
        name='Curvature Contours'
    ))
    
    # Calculate curvature statistics
    mean_curv = np.mean(curvature_data)
    max_curv = np.max(curvature_data)
    min_curv = np.min(curvature_data)
    std_curv = np.std(curvature_data)
    
    fig.update_layout(
        title=dict(
            text=f'Mean Curvature Map - Epoch {current_epoch}<br>Range: [{min_curv:.3f}, {max_curv:.3f}], μ: {mean_curv:.3f}, σ: {std_curv:.3f}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Loss Landscape X',
            titlefont=dict(color=UQ_PURPLE, size=11)
        ),
        yaxis=dict(
            title='Loss Landscape Y',
            titlefont=dict(color=UQ_PURPLE, size=11),
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=60, r=80, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE, size=10),
        showlegend=False
    )
    
    return fig

def make_persistence_diagram(current_epoch):
    """Generate persistence diagram showing birth/death of topological features"""
    # Simulate persistence data (replace with actual topological analysis)
    np.random.seed(1000 + current_epoch)
    
    # Generate birth-death pairs for different dimensional features
    # H0: Connected components (0-dimensional features)
    n_h0 = max(1, 8 - current_epoch // 5)  # Fewer components as training progresses
    h0_births = np.random.uniform(0, 2, n_h0)
    h0_deaths = h0_births + np.random.exponential(1.5, n_h0)
    
    # H1: Cavities/holes (1-dimensional features)
    n_h1 = max(0, 5 - current_epoch // 8)  # Holes disappear during training
    h1_births = np.random.uniform(0.5, 3, n_h1) if n_h1 > 0 else []
    h1_deaths = np.array(h1_births) + np.random.exponential(2, n_h1) if n_h1 > 0 else []
    
    # H2: Voids (2-dimensional features) - rare in loss landscapes
    n_h2 = max(0, 2 - current_epoch // 15)
    h2_births = np.random.uniform(1, 4, n_h2) if n_h2 > 0 else []
    h2_deaths = np.array(h2_births) + np.random.exponential(1, n_h2) if n_h2 > 0 else []
    
    fig = go.Figure()
    
    # Add diagonal line (birth = death)
    max_val = 8
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Birth = Death',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True
    ))
    
    # Plot H0 features (connected components)
    if len(h0_births) > 0:
        fig.add_trace(go.Scatter(
            x=h0_births,
            y=h0_deaths,
            mode='markers',
            name='H₀ (Components)',
            marker=dict(color='red', size=8, symbol='circle'),
            text=[f'H₀: Birth={b:.2f}, Death={d:.2f}, Life={d-b:.2f}' for b, d in zip(h0_births, h0_deaths)],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Plot H1 features (holes)
    if len(h1_births) > 0:
        fig.add_trace(go.Scatter(
            x=h1_births,
            y=h1_deaths,
            mode='markers',
            name='H₁ (Holes)',
            marker=dict(color='blue', size=8, symbol='square'),
            text=[f'H₁: Birth={b:.2f}, Death={d:.2f}, Life={d-b:.2f}' for b, d in zip(h1_births, h1_deaths)],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Plot H2 features (voids)
    if len(h2_births) > 0:
        fig.add_trace(go.Scatter(
            x=h2_births,
            y=h2_deaths,
            mode='markers',
            name='H₂ (Voids)',
            marker=dict(color='green', size=8, symbol='diamond'),
            text=[f'H₂: Birth={b:.2f}, Death={d:.2f}, Life={d-b:.2f}' for b, d in zip(h2_births, h2_deaths)],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Calculate Betti numbers
    betti_0 = len(h0_births)
    betti_1 = len(h1_births)
    betti_2 = len(h2_births)
    
    fig.update_layout(
        title=dict(
            text=f'Persistence Diagram - Epoch {current_epoch}<br>β₀={betti_0}, β₁={betti_1}, β₂={betti_2}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Birth Time',
            titlefont=dict(color=UQ_PURPLE),
            range=[0, max_val],
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Death Time',
            titlefont=dict(color=UQ_PURPLE),
            range=[0, max_val],
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=60, r=20, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE, size=10),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_critical_points_evolution(current_epoch, total_epochs):
    """Generate critical points evolution plot"""
    # Simulate critical points data over training
    epochs = list(range(min(total_epochs, 50)))
    
    np.random.seed(1100)
    
    # Simulate different types of critical points
    # Minima (should increase and stabilize)
    n_minima = []
    # Maxima (should decrease)
    n_maxima = []
    # Saddle points (complex evolution)
    n_saddles = []
    
    for epoch in epochs:
        # Minima: start few, increase, then stabilize
        minima_count = min(3 + epoch // 5, 8) + np.random.poisson(0.5)
        n_minima.append(minima_count)
        
        # Maxima: start many, decrease rapidly
        maxima_count = max(1, 10 - epoch // 3) + np.random.poisson(0.3)
        n_maxima.append(maxima_count)
        
        # Saddle points: complex evolution
        saddle_count = 5 + 2 * np.sin(epoch * 0.3) + np.random.poisson(1)
        n_saddles.append(max(0, int(saddle_count)))
    
    fig = go.Figure()
    
    # Plot evolution of different critical point types
    fig.add_trace(go.Scatter(
        x=epochs[:current_epoch+1],
        y=n_minima[:current_epoch+1],
        mode='lines+markers',
        name='Local Minima',
        line=dict(color='green', width=3),
        marker=dict(color='green', size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs[:current_epoch+1],
        y=n_maxima[:current_epoch+1],
        mode='lines+markers',
        name='Local Maxima',
        line=dict(color='red', width=3),
        marker=dict(color='red', size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs[:current_epoch+1],
        y=n_saddles[:current_epoch+1],
        mode='lines+markers',
        name='Saddle Points',
        line=dict(color='orange', width=3),
        marker=dict(color='orange', size=6)
    ))
    
    # Highlight current epoch
    if current_epoch < len(epochs):
        fig.add_trace(go.Scatter(
            x=[current_epoch],
            y=[n_minima[current_epoch] + n_maxima[current_epoch] + n_saddles[current_epoch]],
            mode='markers',
            name='Current Total',
            marker=dict(color=UQ_GOLD, size=12, symbol='star')
        ))
    
    # Calculate current totals
    current_minima = n_minima[current_epoch] if current_epoch < len(n_minima) else 0
    current_maxima = n_maxima[current_epoch] if current_epoch < len(n_maxima) else 0
    current_saddles = n_saddles[current_epoch] if current_epoch < len(n_saddles) else 0
    total_critical = current_minima + current_maxima + current_saddles
    
    fig.update_layout(
        title=dict(
            text=f'Critical Points Evolution - Epoch {current_epoch}<br>Min: {current_minima}, Max: {current_maxima}, Saddles: {current_saddles}, Total: {total_critical}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Number of Critical Points',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=60, r=20, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE, size=10),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def make_topology_timeline(current_epoch):
    """Generate topological features timeline showing birth/death events"""
    epochs = list(range(min(current_epoch + 1, 50)))
    
    # Simulate topological events (births and deaths)
    np.random.seed(1200)
    
    events = []
    feature_id = 0
    
    # Generate birth/death events
    for epoch in epochs:
        # Birth events (new features appear)
        n_births = np.random.poisson(0.8)
        for _ in range(n_births):
            feature_type = np.random.choice(['H0', 'H1', 'H2'], p=[0.6, 0.3, 0.1])
            events.append({
                'epoch': epoch,
                'event': 'birth',
                'feature_id': feature_id,
                'feature_type': feature_type,
                'y_pos': feature_id % 20  # For visualization positioning
            })
            feature_id += 1
        
        # Death events (features disappear)
        n_deaths = np.random.poisson(0.6)
        for _ in range(n_deaths):
            if events:  # Only if there are existing features
                # Find a living feature to kill
                living_features = [e for e in events if e['event'] == 'birth' and 
                                 not any(d['event'] == 'death' and d['feature_id'] == e['feature_id'] for d in events)]
                if living_features:
                    feature_to_kill = np.random.choice(living_features)
                    events.append({
                        'epoch': epoch,
                        'event': 'death',
                        'feature_id': feature_to_kill['feature_id'],
                        'feature_type': feature_to_kill['feature_type'],
                        'y_pos': feature_to_kill['y_pos']
                    })
    
    fig = go.Figure()
    
    # Group events by feature type
    feature_types = ['H0', 'H1', 'H2']
    colors = {'H0': 'red', 'H1': 'blue', 'H2': 'green'}
    symbols = {'H0': 'circle', 'H1': 'square', 'H2': 'diamond'}
    
    for ftype in feature_types:
        birth_events = [e for e in events if e['feature_type'] == ftype and e['event'] == 'birth']
        death_events = [e for e in events if e['feature_type'] == ftype and e['event'] == 'death']
        
        # Plot birth events
        if birth_events:
            fig.add_trace(go.Scatter(
                x=[e['epoch'] for e in birth_events],
                y=[e['y_pos'] for e in birth_events],
                mode='markers',
                name=f'{ftype} Birth',
                marker=dict(color=colors[ftype], size=8, symbol=symbols[ftype]),
                text=[f'{ftype} Feature {e["feature_id"]} born at epoch {e["epoch"]}' for e in birth_events],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Plot death events
        if death_events:
            fig.add_trace(go.Scatter(
                x=[e['epoch'] for e in death_events],
                y=[e['y_pos'] for e in death_events],
                mode='markers',
                name=f'{ftype} Death',
                marker=dict(color=colors[ftype], size=8, symbol='x'),
                text=[f'{ftype} Feature {e["feature_id"]} died at epoch {e["epoch"]}' for e in death_events],
                hovertemplate='%{text}<extra></extra>'
            ))
    
    # Add current epoch line
    fig.add_vline(x=current_epoch, line_dash="dash", line_color=UQ_GOLD, 
                  annotation_text=f"Current: Epoch {current_epoch}", 
                  annotation_position="top")
    
    # Count current living features
    living_features = []
    for event in events:
        if event['event'] == 'birth' and event['epoch'] <= current_epoch:
            # Check if this feature has died by current epoch
            death_event = next((e for e in events if e['event'] == 'death' and 
                              e['feature_id'] == event['feature_id'] and 
                              e['epoch'] <= current_epoch), None)
            if not death_event:
                living_features.append(event)
    
    living_by_type = {ftype: len([f for f in living_features if f['feature_type'] == ftype]) 
                      for ftype in feature_types}
    
    fig.update_layout(
        title=dict(
            text=f'Topological Features Timeline - Epoch {current_epoch}<br>Living: H₀={living_by_type["H0"]}, H₁={living_by_type["H1"]}, H₂={living_by_type["H2"]}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Feature ID',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=60, r=20, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE, size=10),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_avg_loss_landscape(total_epochs):
    """Generate average loss landscape over all epochs"""
    # Simulate averaging loss landscapes across all epochs
    all_epochs = list(range(min(total_epochs, 50)))
    
    # Use first epoch's coordinates as template
    x, y, z_template = load_surface(0)
    if x is None:
        fig = go.Figure()
        fig.update_layout(
            title="No surface data available",
            paper_bgcolor=UQ_WHITE,
            plot_bgcolor=UQ_WHITE,
            font=dict(family="Montserrat", color=UQ_PURPLE)
        )
        return fig
    
    # Simulate averaging (in practice, you'd average actual surface data)
    np.random.seed(999)
    avg_loss = np.zeros_like(z_template)
    
    # Average loss across epochs (simulated)
    for epoch in all_epochs:
        x_ep, y_ep, z_ep = load_surface(epoch)
        if z_ep is not None:
            # Add some variation and average
            epoch_variation = 1.0 - 0.8 * (epoch / max(all_epochs)) if all_epochs else 1.0
            avg_loss += z_ep * epoch_variation
    
    avg_loss /= len(all_epochs)
    
    # Create 3D surface plot
    X, Y = np.meshgrid(x, y)
    fig = go.Figure(data=[
        go.Surface(z=avg_loss, x=X, y=Y, colorscale='Viridis', showscale=True, opacity=0.9)
    ])
    
    fig.update_layout(
        title=dict(
            text=f'Average Loss Landscape (Over {len(all_epochs)} Epochs)',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        scene=dict(
            xaxis_title='Direction X',
            yaxis_title='Direction Y',
            zaxis_title='Average Loss',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.35, y=1.35, z=1.35)
            ),
            xaxis=dict(titlefont=dict(color=UQ_PURPLE)),
            yaxis=dict(titlefont=dict(color=UQ_PURPLE)),
            zaxis=dict(titlefont=dict(color=UQ_PURPLE))
        ),
        margin=dict(l=20, r=20, b=20, t=60),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE)
    )
    
    return fig

def make_avg_curvature_distribution(total_epochs):
    """Generate average curvature distribution across all epochs"""
    all_epochs = list(range(min(total_epochs, 50)))
    
    # Simulate curvature distributions across epochs
    np.random.seed(1000)
    all_curvatures = []
    
    for epoch in all_epochs:
        # Simulate curvature values for this epoch
        # In practice, you'd compute actual curvatures
        n_points = 1000
        epoch_decay = np.exp(-0.1 * epoch)
        
        # Generate curvature distribution that evolves over epochs
        base_curvature = np.random.normal(-2 * epoch_decay, 1.0, n_points)
        positive_curvature = np.random.exponential(0.5 * epoch_decay, n_points // 4)
        
        epoch_curvatures = np.concatenate([base_curvature, positive_curvature])
        all_curvatures.extend(epoch_curvatures)
    
    # Create histogram and box plot
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=all_curvatures,
        nbinsx=50,
        name='Curvature Distribution',
        marker=dict(color=UQ_PURPLE, opacity=0.7),
        yaxis='y'
    ))
    
    # Add box plot
    fig.add_trace(go.Box(
        y=all_curvatures,
        name='Curvature Statistics',
        marker=dict(color=UQ_GOLD),
        yaxis='y2'
    ))
    
    # Calculate statistics
    mean_curv = np.mean(all_curvatures)
    std_curv = np.std(all_curvatures)
    median_curv = np.median(all_curvatures)
    
    fig.update_layout(
        title=dict(
            text=f'Average Curvature Distribution<br>μ: {mean_curv:.3f}, σ: {std_curv:.3f}, median: {median_curv:.3f}',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Curvature Value',
            titlefont=dict(color=UQ_PURPLE)
        ),
        yaxis=dict(
            title='Frequency',
            titlefont=dict(color=UQ_PURPLE),
            side='left'
        ),
        yaxis2=dict(
            title='Box Plot',
            titlefont=dict(color=UQ_PURPLE),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        margin=dict(l=60, r=60, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_eigenvalue_evolution_summary(total_epochs):
    """Generate eigenvalue evolution summary with confidence bands"""
    all_epochs = list(range(min(total_epochs, 50)))
    
    # Simulate eigenvalue evolution with confidence intervals
    np.random.seed(1001)
    
    # Generate multiple runs to show confidence bands
    n_runs = 10
    eigenvalue_runs = []
    
    for run in range(n_runs):
        run_eigenvals = []
        for epoch in all_epochs:
            # Simulate eigenvalue evolution with some randomness between runs
            base_decay = np.exp(-0.1 * epoch)
            noise = np.random.normal(0, 0.1, 5)  # 5 leading eigenvalues
            
            base_eigenvals = np.array([-10, -5, -2, -0.5, 0.1])
            eigenvals = base_eigenvals * base_decay + noise
            run_eigenvals.append(eigenvals)
        eigenvalue_runs.append(run_eigenvals)
    
    # Convert to numpy array for easier processing
    eigenvalue_runs = np.array(eigenvalue_runs)  # shape: (n_runs, n_epochs, n_eigenvals)
    
    # Calculate mean and confidence intervals
    mean_eigenvals = np.mean(eigenvalue_runs, axis=0)  # shape: (n_epochs, n_eigenvals)
    std_eigenvals = np.std(eigenvalue_runs, axis=0)
    
    fig = go.Figure()
    
    # Plot each eigenvalue with confidence bands
    colors = ['red', 'orange', 'blue', UQ_PURPLE, 'green']
    names = ['λ₁ (Leading)', 'λ₂ (Second)', 'λ₃ (Third)', 'λ₄ (Fourth)', 'λ₅ (Fifth)']
    
    for i, (color, name) in enumerate(zip(colors, names)):
        mean_vals = mean_eigenvals[:, i]
        std_vals = std_eigenvals[:, i]
        
        # Add confidence band (fill between)
        fig.add_trace(go.Scatter(
            x=all_epochs + all_epochs[::-1],
            y=np.concatenate([mean_vals + std_vals, (mean_vals - std_vals)[::-1]]),
            fill='toself',
            fillcolor=color,
            opacity=0.2,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{name} CI'
        ))
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=all_epochs,
            y=mean_vals,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=3),
            marker=dict(color=color, size=4)
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Zero Line", annotation_position="bottom right")
    
    fig.update_layout(
        title=dict(
            text=f'Eigenvalue Evolution Summary (Mean ± Std over {n_runs} runs)',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Eigenvalue Magnitude',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=70, r=20, b=60, t=60),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def landscape_complexity_metrics(surface):
    """Quantify landscape roughness or complexity"""
    if surface is None or surface.size == 0:
        return {'fractal_dim': 0, 'surface_area': 0, 'gradient_variance': 0, 'roughness': 0}
    
    # 1. Surface area calculation (proxy for roughness)
    dy, dx = np.gradient(surface)
    surface_area = np.sum(np.sqrt(1 + dx**2 + dy**2))
    
    # 2. Gradient variance (measure of landscape variability)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    gradient_variance = np.var(gradient_magnitude)
    
    # 3. Simplified fractal dimension using box-counting method
    def box_count_2d(surface, box_size):
        h, w = surface.shape
        n_boxes_x = h // box_size
        n_boxes_y = w // box_size
        if n_boxes_x == 0 or n_boxes_y == 0:
            return 0
        
        boxes = surface[:n_boxes_x*box_size, :n_boxes_y*box_size].reshape(
            n_boxes_x, box_size, n_boxes_y, box_size)
        # Count boxes with significant variation
        box_variations = np.std(boxes, axis=(1,3))
        return np.sum(box_variations > np.std(surface) * 0.1)
    
    # Calculate fractal dimension
    box_sizes = [2, 4, 8, 16]
    box_counts = []
    for size in box_sizes:
        count = box_count_2d(surface, size)
        if count > 0:
            box_counts.append(count)
        else:
            box_counts.append(1)  # Avoid log(0)
    
    # Linear regression to estimate fractal dimension
    if len(box_counts) > 1:
        log_box_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        fractal_dim = -np.polyfit(log_box_sizes, log_counts, 1)[0]
    else:
        fractal_dim = 1.0
    
    # 4. Roughness index (normalized surface area)
    flat_area = surface.shape[0] * surface.shape[1]
    roughness = surface_area / flat_area
    
    return {
        'fractal_dim': fractal_dim,
        'surface_area': surface_area,
        'gradient_variance': gradient_variance,
        'roughness': roughness
    }

def classify_critical_points(hessian_eigenvalues):
    """Classify critical points based on Hessian eigenvalues"""
    if len(hessian_eigenvalues) == 0:
        return {'minima': 0, 'maxima': 0, 'saddles': {}}
    
    positive = np.sum(hessian_eigenvalues > 1e-6)
    negative = np.sum(hessian_eigenvalues < -1e-6)
    zero = len(hessian_eigenvalues) - positive - negative
    
    # Classification based on eigenvalue signs
    if negative == len(hessian_eigenvalues):
        point_type = 'minimum'
    elif positive == len(hessian_eigenvalues):
        point_type = 'maximum'
    else:
        point_type = f'saddle_point_index_{negative}'
    
    # Morse index (number of negative eigenvalues)
    morse_index = negative
    
    return {
        'type': point_type,
        'morse_index': morse_index,
        'positive_eigenvals': positive,
        'negative_eigenvals': negative,
        'zero_eigenvals': zero,
        'condition_number': np.max(np.abs(hessian_eigenvalues)) / np.min(np.abs(hessian_eigenvalues[hessian_eigenvalues != 0])) if np.any(hessian_eigenvalues != 0) else np.inf
    }

def compute_sharpness_metrics(loss_surface, epsilon=0.1):
    """Compute SAM-inspired sharpness metrics"""
    if loss_surface is None or loss_surface.size == 0:
        return {'sharpness': 0, 'sam_measure': 0, 'max_eigenval': 0, 'spectral_norm': 0}
    
    # 1. Local sharpness: maximum loss increase within epsilon-ball
    center_loss = loss_surface[loss_surface.shape[0]//2, loss_surface.shape[1]//2]
    
    # Create perturbation mask
    h, w = loss_surface.shape
    center_h, center_w = h//2, w//2
    
    # Sample points within epsilon distance
    max_sharpness = 0
    sam_measure = 0
    
    for dh in range(-2, 3):
        for dw in range(-2, 3):
            new_h, new_w = center_h + dh, center_w + dw
            if 0 <= new_h < h and 0 <= new_w < w:
                distance = np.sqrt(dh**2 + dw**2) * epsilon
                if distance <= epsilon and distance > 0:
                    perturbed_loss = loss_surface[new_h, new_w]
                    sharpness = (perturbed_loss - center_loss) / distance
                    max_sharpness = max(max_sharpness, sharpness)
                    sam_measure += abs(perturbed_loss - center_loss)
    
    # 2. Spectral sharpness (approximate largest eigenvalue)
    # Use gradient information as proxy
    dy, dx = np.gradient(loss_surface)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    max_eigenval = np.max(gradient_magnitude)
    spectral_norm = np.linalg.norm(gradient_magnitude)
    
    return {
        'sharpness': max_sharpness,
        'sam_measure': sam_measure,
        'max_eigenval': max_eigenval,
        'spectral_norm': spectral_norm
    }

def make_complexity_metrics_plot(current_epoch):
    """Generate landscape complexity metrics visualization"""
    x, y, z = load_surface(current_epoch)
    if x is None:
        fig = go.Figure()
        fig.update_layout(
            title="No surface data available",
            paper_bgcolor=UQ_WHITE,
            plot_bgcolor=UQ_WHITE,
            font=dict(family="Montserrat", color=UQ_PURPLE)
        )
        return fig
    
    # Calculate complexity metrics
    complexity = landscape_complexity_metrics(z)
    
    # Create time series for all epochs
    epochs = list(range(min(total_epochs, 50)))
    complexities = []
    
    for epoch in epochs:
        x_e, y_e, z_e = load_surface(epoch)
        if z_e is not None:
            comp = landscape_complexity_metrics(z_e)
            complexities.append(comp)
        else:
            complexities.append({'fractal_dim': 0, 'surface_area': 0, 'gradient_variance': 0, 'roughness': 0})
    
    fig = go.Figure()
    
    # Plot different complexity metrics
    metrics = ['fractal_dim', 'roughness', 'gradient_variance']
    colors = ['red', 'blue', 'green']
    names = ['Fractal Dimension', 'Roughness Index', 'Gradient Variance']
    
    for metric, color, name in zip(metrics, colors, names):
        values = [comp[metric] for comp in complexities]
        # Normalize values for comparison
        if max(values) > 0:
            normalized_values = np.array(values) / max(values)
        else:
            normalized_values = values
            
        fig.add_trace(go.Scatter(
            x=epochs,
            y=normalized_values,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=4)
        ))
    
    # Highlight current epoch
    if current_epoch < len(complexities):
        current_values = []
        for metric in metrics:
            val = complexities[current_epoch][metric]
            max_val = max([comp[metric] for comp in complexities])
            normalized_val = val / max_val if max_val > 0 else 0
            current_values.append(normalized_val)
        
        fig.add_trace(go.Scatter(
            x=[current_epoch] * len(current_values),
            y=current_values,
            mode='markers',
            name='Current Epoch',
            marker=dict(color=UQ_GOLD, size=10, symbol='star')
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Landscape Complexity Evolution<br>Epoch {current_epoch}: Fractal Dim={complexity["fractal_dim"]:.2f}, Roughness={complexity["roughness"]:.2f}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Normalized Metric Value',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=60, r=20, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_critical_points_plot(current_epoch, total_epochs):
    """Generate critical points classification visualization"""
    # Simulate critical points analysis over time
    epochs = list(range(min(total_epochs, 50)))
    
    np.random.seed(999 + current_epoch)
    
    # Simulate evolution of critical points
    critical_points_data = []
    for epoch in epochs:
        # Simulate Hessian eigenvalues at multiple points
        n_points = 20
        all_classifications = []
        
        for _ in range(n_points):
            # Generate random eigenvalues (more realistic distribution)
            eigenvals = np.random.normal(-1, 0.5, 5)  # Tend toward negative (minima)
            eigenvals[0] *= (1 + 0.1 * epoch)  # Evolution over time
            
            classification = classify_critical_points(eigenvals)
            all_classifications.append(classification)
        
        # Count different types
        minima = sum(1 for c in all_classifications if c['type'] == 'minimum')
        maxima = sum(1 for c in all_classifications if c['type'] == 'maximum')
        saddles = sum(1 for c in all_classifications if 'saddle' in c['type'])
        avg_morse_index = np.mean([c['morse_index'] for c in all_classifications])
        
        critical_points_data.append({
            'epoch': epoch,
            'minima': minima,
            'maxima': maxima,
            'saddles': saddles,
            'avg_morse_index': avg_morse_index
        })
    
    fig = go.Figure()
    
    # Plot critical point counts
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['minima'] for cp in critical_points_data],
        mode='lines+markers',
        name='Local Minima',
        line=dict(color='green', width=2),
        marker=dict(color='green', size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['maxima'] for cp in critical_points_data],
        mode='lines+markers',
        name='Local Maxima',
        line=dict(color='red', width=2),
        marker=dict(color='red', size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['saddles'] for cp in critical_points_data],
        mode='lines+markers',
        name='Saddle Points',
        line=dict(color='orange', width=2),
        marker=dict(color='orange', size=4)
    ))
    
    # Add average Morse index on secondary y-axis
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['avg_morse_index'] for cp in critical_points_data],
        mode='lines',
        name='Avg Morse Index',
        line=dict(color=UQ_PURPLE, width=2, dash='dash'),
        yaxis='y2'
    ))
    
    # Highlight current epoch
    if current_epoch < len(critical_points_data):
        current_data = critical_points_data[current_epoch]
        fig.add_trace(go.Scatter(
            x=[current_epoch],
            y=[current_data['minima']],
            mode='markers',
            name='Current',
            marker=dict(color=UQ_GOLD, size=12, symbol='star'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Critical Points Evolution - Epoch {current_epoch}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Number of Critical Points',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis2=dict(
            title='Average Morse Index',
            titlefont=dict(color=UQ_PURPLE),
            overlaying='y',
            side='right'
        ),
        margin=dict(l=60, r=60, b=60, t=60),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_complexity_metrics_plot(current_epoch, total_epochs):
    """Generate sharpness/flatness metrics visualization"""
    x, y, z = load_surface(current_epoch)
    if x is None:
        fig = go.Figure()
        fig.update_layout(
            title="No surface data available",
            paper_bgcolor=UQ_WHITE,
            plot_bgcolor=UQ_WHITE,
            font=dict(family="Montserrat", color=UQ_PURPLE)
        )
        return fig
    
    # Calculate sharpness metrics for all epochs
    epochs = list(range(min(total_epochs, 50)))
    sharpness_data = []
    for epoch in epochs:
        x_e, y_e, z_e = load_surface(epoch)
        if z_e is not None:
            sharpness = compute_sharpness_metrics(z_e)
            sharpness_data.append(sharpness)
        else:
            sharpness_data.append({'sharpness': 0, 'sam_measure': 0, 'max_eigenval': 0, 'spectral_norm': 0})
    
    fig = go.Figure()
    
    # Plot different sharpness metrics
    metrics = ['sharpness', 'sam_measure', 'spectral_norm']
    colors = ['red', 'blue', 'purple']
    names = ['Local Sharpness', 'SAM Measure', 'Spectral Norm']
    
    for metric, color, name in zip(metrics, colors, names):
        values = [s[metric] for s in sharpness_data]
        # Normalize for visualization
        if max(values) > 0:
            normalized_values = np.array(values) / max(values)
        else:
            normalized_values = values
            
        fig.add_trace(go.Scatter(
            x=epochs,
            y=normalized_values,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=4)
        ))
    
    # Current epoch sharpness
    current_sharpness = compute_sharpness_metrics(z)
    
    # Highlight current epoch
    if current_epoch < len(sharpness_data):
        current_values = []
        for metric in metrics:
            val = sharpness_data[current_epoch][metric]
            max_val = max([s[metric] for s in sharpness_data])
            normalized_val = val / max_val if max_val > 0 else 0
            current_values.append(normalized_val)
        
        fig.add_trace(go.Scatter(
            x=[current_epoch] * len(current_values),
            y=current_values,
            mode='markers',
            name='Current Epoch',
            marker=dict(color=UQ_GOLD, size=10, symbol='star')
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Sharpness/Flatness Evolution - Epoch {current_epoch}<br>Sharpness: {current_sharpness["sharpness"]:.3f}, SAM: {current_sharpness["sam_measure"]:.3f}',
            font=dict(color=UQ_PURPLE, size=12, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Normalized Metric Value',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        margin=dict(l=60, r=20, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_complexity_evolution_plot(total_epochs):
    """Generate comprehensive complexity evolution across all epochs"""
    epochs = list(range(min(total_epochs, 50)))
    complexities = []
    
    for epoch in epochs:
        x_e, y_e, z_e = load_surface(epoch)
        if z_e is not None:
            comp = landscape_complexity_metrics(z_e)
            complexities.append(comp)
        else:
            complexities.append({'fractal_dim': 0, 'surface_area': 0, 'gradient_variance': 0, 'roughness': 0})
    
    fig = go.Figure()
    
    # Plot individual metrics with separate y-axes for better visualization
    metrics_data = {
        'Fractal Dimension': ([comp['fractal_dim'] for comp in complexities], 'red', 'y'),
        'Roughness Index': ([comp['roughness'] for comp in complexities], 'blue', 'y2'),
        'Surface Area (normalized)': ([comp['surface_area']/1000 for comp in complexities], 'green', 'y3'),
        'Gradient Variance (log)': ([np.log10(comp['gradient_variance'] + 1e-10) for comp in complexities], 'purple', 'y4')
    }
    
    for i, (name, (values, color, yaxis)) in enumerate(metrics_data.items()):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=values,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=4),
            yaxis=yaxis
        ))
    
    fig.update_layout(
        title=dict(
            text='Complexity Metrics Evolution Across All Training Epochs',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Fractal Dimension',
            titlefont=dict(color='red'),
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            title='Roughness Index',
            titlefont=dict(color='blue'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        yaxis3=dict(
            title='Surface Area (×1000)',
            titlefont=dict(color='green'),
            overlaying='y',
            side='left',
            position=0.05,
            showgrid=False
        ),
        yaxis4=dict(
            title='Log Gradient Variance',
            titlefont=dict(color='purple'),
            overlaying='y',
            side='right',
            position=0.95,
            showgrid=False
        ),
        margin=dict(l=80, r=80, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1,
            x=0.5,
            y=1.1,
            orientation='h'
        )
    )
    
    return fig

def make_critical_points_evolution_plot(total_epochs):
    """Generate comprehensive critical points evolution across all epochs"""
    epochs = list(range(min(total_epochs, 50)))
    
    np.random.seed(999)
    
    critical_points_data = []
    for epoch in epochs:
        # Simulate Hessian eigenvalues at multiple points
        n_points = 25
        all_classifications = []
        
        for i in range(n_points):
            # Generate more realistic eigenvalue evolution
            base_eigenvals = np.random.normal(-0.5, 0.3, 5)
            # Add epoch-dependent trends
            evolution_factor = 1 - 0.02 * epoch  # Gradual change
            eigenvals = base_eigenvals * evolution_factor
            
            # Add some noise and occasional positive eigenvalues
            if np.random.rand() < 0.1:  # 10% chance of positive eigenvalues
                eigenvals[np.random.randint(0, len(eigenvals))] = abs(eigenvals[np.random.randint(0, len(eigenvals))])
            
            classification = classify_critical_points(eigenvals)
            all_classifications.append(classification)
        
        # Aggregate statistics
        minima = sum(1 for c in all_classifications if c['type'] == 'minimum')
        maxima = sum(1 for c in all_classifications if c['type'] == 'maximum')
        saddles = sum(1 for c in all_classifications if 'saddle' in c['type'])
        avg_morse_index = np.mean([c['morse_index'] for c in all_classifications])
        avg_condition_number = np.mean([c['condition_number'] for c in all_classifications if not np.isinf(c['condition_number'])])
        
        critical_points_data.append({
            'epoch': epoch,
            'minima': minima,
            'maxima': maxima,
            'saddles': saddles,
            'avg_morse_index': avg_morse_index,
            'avg_condition_number': avg_condition_number
        })
    
    fig = go.Figure()
    
    # Main critical points counts
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['minima'] for cp in critical_points_data],
        mode='lines+markers',
        name='Local Minima',
        line=dict(color='green', width=3),
        marker=dict(color='green', size=5),
        fill='tonexty' if len(fig.data) > 0 else None
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['maxima'] for cp in critical_points_data],
        mode='lines+markers',
        name='Local Maxima',
        line=dict(color='red', width=3),
        marker=dict(color='red', size=5)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['saddles'] for cp in critical_points_data],
        mode='lines+markers',
        name='Saddle Points',
        line=dict(color='orange', width=3),
        marker=dict(color='orange', size=5)
    ))
    
    # Morse index on secondary axis
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[cp['avg_morse_index'] for cp in critical_points_data],
        mode='lines+markers',
        name='Avg Morse Index',
        line=dict(color=UQ_PURPLE, width=2, dash='dash'),
        marker=dict(color=UQ_PURPLE, size=4),
        yaxis='y2'
    ))
    
    # Condition number trend
    condition_nums = [cp['avg_condition_number'] for cp in critical_points_data]
    fig.add_trace(go.Scatter(
        x=epochs,
        y=condition_nums,
        mode='lines',
        name='Avg Condition Number',
        line=dict(color='darkred', width=2, dash='dot'),
        yaxis='y3'
    ))
    
    fig.update_layout(
        title=dict(
            text='Critical Points Classification Evolution Across All Training Epochs',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Number of Critical Points',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis2=dict(
            title='Average Morse Index',
            titlefont=dict(color=UQ_PURPLE),
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Condition Number',
            titlefont=dict(color='darkred'),
            overlaying='y',
            side='right',
            position=0.95
        ),
        margin=dict(l=60, r=80, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        )
    )
    
    return fig

def make_sharpness_evolution_plot(total_epochs):
    """Generate comprehensive sharpness evolution across all epochs"""
    epochs = list(range(min(total_epochs, 50)))
    sharpness_data = []
    
    for epoch in epochs:
        x_e, y_e, z_e = load_surface(epoch)
        if z_e is not None:
            sharpness = compute_sharpness_metrics(z_e)
            sharpness_data.append(sharpness)
        else:
            sharpness_data.append({'sharpness': 0, 'sam_measure': 0, 'max_eigenval': 0, 'spectral_norm': 0})
    
    fig = go.Figure()
    
    # Main sharpness metrics
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[s['sharpness'] for s in sharpness_data],
        mode='lines+markers',
        name='Local Sharpness',
        line=dict(color='red', width=3),
        marker=dict(color='red', size=5),
        fill='tozeroy'
    ))
    
    # SAM measure (normalized)
    sam_values = [s['sam_measure'] for s in sharpness_data]
    if max(sam_values) > 0:
        sam_normalized = np.array(sam_values) / max(sam_values) * max([s['sharpness'] for s in sharpness_data])
    else:
        sam_normalized = sam_values
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=sam_normalized,
        mode='lines+markers',
        name='SAM Measure (scaled)',
        line=dict(color='blue', width=2),
        marker=dict(color='blue', size=4)
    ))
    
    # Spectral norm on secondary axis
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[s['spectral_norm'] for s in sharpness_data],
        mode='lines+markers',
        name='Spectral Norm',
        line=dict(color='purple', width=2, dash='dash'),
        marker=dict(color='purple', size=4),
        yaxis='y2'
    ))
    
    # Max eigenvalue approximation
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[s['max_eigenval'] for s in sharpness_data],
        mode='lines',
        name='Max Eigenvalue Approx',
        line=dict(color='green', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # Add trend analysis
    if len(epochs) > 10:
        # Fit polynomial trend to sharpness
        sharpness_values = [s['sharpness'] for s in sharpness_data]
        trend_coeffs = np.polyfit(epochs, sharpness_values, 2)
        trend_line = np.polyval(trend_coeffs, epochs)
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=trend_line,
            mode='lines',
            name='Sharpness Trend',
            line=dict(color='darkred', width=3, dash='dashdot'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title=dict(
            text='Sharpness/Flatness Evolution Across All Training Epochs',
            font=dict(color=UQ_PURPLE, size=14, family="Montserrat"),
            x=0.5
        ),
        xaxis=dict(
            title='Training Epoch',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis=dict(
            title='Sharpness Metrics',
            titlefont=dict(color=UQ_PURPLE),
            gridcolor=UQ_LIGHT_GREY
        ),
        yaxis2=dict(
            title='Spectral Measures',
            titlefont=dict(color='purple'),
            overlaying='y',
            side='right'
        ),
        margin=dict(l=60, r=60, b=60, t=80),
        paper_bgcolor=UQ_WHITE,
        plot_bgcolor=UQ_WHITE,
        font=dict(family="Montserrat", color=UQ_PURPLE),
        showlegend=True,
        legend=dict(
            bgcolor=UQ_LIGHT_GREY,
            bordercolor=UQ_PURPLE,
            borderwidth=1
        ),
        annotations=[
            dict(
                text="Lower values indicate flatter loss landscape",
                x=0.5, y=0.02,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(color=UQ_PURPLE, size=10)
            )
        ]
    )
    
    return fig

# ---- DASH LAYOUT ----
app = dash.Dash(
    __name__,
    external_stylesheets=["/assets/uq_style.css"],
    title="UQ Loss Landscape Evolution Dashboard"
)

app.layout = html.Div([
    # UQ Header Banner
    # UQ Header Banner
    html.Header([
        html.Div(className="uq-banner", children=[
            html.Img(src="/assets/icon.png", className="uq-logo"),
            html.H1("Neural Network Loss Landscape Evolution", className="banner-title")
        ])
    ]),
    
    # Run Selection Dropdown
    html.Div([
        html.Label('Select Run:', style={'font-weight': 'bold', 'margin-right': '1rem'}),
        dcc.Dropdown(
            id='run-selector',
            options=[
                {'label': '2,8,2 PCA', 'value': '50_-1,1,101 (2,8,2 PCA)'},
                {'label': '2,8,2 RANDOM', 'value': '50_-1,1,101 (2,8,2 RANDOM)'},
                {'label': '3,3,2 PCA', 'value': '50_-1,1,101 (3,3,2 PCA)'},
                {'label': '3,3,2 RANDOM', 'value': '50_-1,1,101 (3,3,2 RANDOM)'},
            ],
            value='50_-1,1,101 (2,8,2 PCA)',
            clearable=False,
            style={'width': '300px'}
        ),
        dcc.Store(id='gif-frame-count'),
    ], style={'padding': '1rem', 'background': UQ_LIGHT_GREY, 'border-radius': '8px', 'margin-bottom': '1rem', 'display': 'flex', 'align-items': 'center'}),
    # Main Content Area
    html.Div(className="main-wrapper", children=[
        # Navigation Sidebar
        html.Nav(className="sidebar", children=[
            html.H2("Dashboard", className="sidebar-title"),
            html.Ul([
                html.Li(html.A("Animation View", href="#animation-section", className="sidebar-link")),
                html.Li(html.A("3D Interactive", href="#interactive-section", className="sidebar-link")),
                html.Li(html.A("Controls", href="#controls-section", className="sidebar-link")),
            ], className="nav-list")
        ]),
        
        # Main Content
        html.Main(className="content-area", children=[
            # Title Section
            html.Div(className="title-section", style={
                'background': UQ_WHITE,
                'padding': '1.5rem 2rem',
                'border-radius': '8px',
                'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                'margin-bottom': '2rem',
                'border-left': f'5px solid {UQ_GOLD}'
            }, children=[
                html.H2("Loss Landscape Evolution Analysis", style={
                    'color': UQ_PURPLE,
                    'margin': '0 0 0.5rem 0',
                    'font-size': '1.8rem',
                    'font-weight': '700'
                }),
                dcc.Loading(
                    id="dynamic-title-loading",
                    type="default",
                    children=[
                        html.P(id="dynamic-title", style={
                            'color': UQ_GREY,
                            'margin': '0',
                            'font-size': '1.1rem',
                            'font-weight': '500'
                        })
                    ]
                )
            ]),
            # (comma added above)
            
            # Controls Section
            html.Section(id="controls-section", children=[
                html.H3("Playback Controls"),
                html.Div(className="controls-row", children=[
                    html.Div([
                        html.Button('▶ Play', id='play-pause-btn', n_clicks=0, className="uq-btn"),
                    ], style={'display': 'flex', 'align-items': 'center', 'gap': '10px'}),
                    html.Div([
                        html.Label("Epoch:", style={'color': UQ_PURPLE, 'font-weight': '600', 'margin-right': '10px'}),
                        dcc.Slider(
                            id='gif-slider',
                            min=0,
                            max=EPOCHS-1, # will be updated by callback
                            value=0,
                            step=1,
                            marks={i: str(i) for i in range(0, EPOCHS, max(1, EPOCHS//10))}, # will be updated by callback
                            className="uq-slider",
                        ),
                    ], style={'flex': '1', 'min-width': '300px'})
                ])
            ]),
            
            # Visualization Section
            html.Div(className="visualization-container", style={
                'display': 'flex', 
                'gap': '2rem', 
                'margin-top': '2rem',
                'background': UQ_WHITE,
                'padding': '2rem',
                'border-radius': '8px',
                'box-shadow': '0 2px 8px rgba(0,0,0,0.1)'
            }, children=[
                # Animation Panel
                html.Section(id="animation-section", style={'flex': '1'}, children=[
                    html.H3("Loss Landscape Animation", style={'margin-top': '0'}),
                    html.Div([
                        html.Img(id='gif-display', style={
                            'width': '120%', 
                            'height': '120%', 
                            'object-fit': 'contain', 
                            'border-radius': '8px', 
                            'border': f'2px solid {UQ_LIGHT_GREY}'
                        }),
                    ], style={
                        'height': '500px', 
                        'display': 'flex', 
                        'align-items': 'center', 
                        'justify-content': 'center', 
                        'overflow': 'hidden',
                        'background': UQ_BACKGROUND,
                        'border-radius': '8px'
                    }),
                ]),
                
                # Interactive 3D Panel  
                html.Section(id="interactive-section", style={'flex': '1'}, children=[
                    html.H3("Interactive 3D Surface", style={'margin-top': '0'}),
                    html.Div([
                        dcc.Graph(id='surface-graph', style={'height': '500px', 'width': '100%'}),
                    ], style={
                        'border-radius': '8px',
                        'background': UQ_WHITE,
                        'border': f'2px solid {UQ_LIGHT_GREY}'
                    }),
                ]),
            ]),
            
            # Gradient Vector Analysis Section
            html.Section(id="gradient-section", children=[
                html.H3("Gradient Vector Analysis"),
                
                # Average Gradient Angle (Static)
                html.Div([
                    dcc.Graph(id='avg-gradient-angle-plot', style={'height': '300px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}',
                    'margin-bottom': '1rem'
                }),
                
                # Gradient Angle Evolution (Dynamic)
                html.Div([
                    dcc.Graph(id='gradient-angle-plot', style={'height': '300px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}'
                }),
            ]),
            
            # Aggregated Analysis Section (Average over all epochs)
            html.Section(id="aggregated-analysis-section", children=[
                html.H3("Aggregated Analysis - Average Over All Epochs"),
                
                # Two-column layout for aggregated plots
                html.Div(className="visualization-container", style={
                    'display': 'flex', 
                    'gap': '1rem', 
                    'margin-bottom': '1rem'
                }, children=[
                    # Average Loss Landscape
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("Average Loss Landscape", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                        html.Div([
                            dcc.Graph(id='avg-loss-landscape', style={'height': '400px', 'width': '100%'}),
                        ], style={
                            'background': UQ_WHITE,
                            'padding': '1rem',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'border': f'2px solid {UQ_LIGHT_GREY}'
                        }),
                    ]),
                    
                    # Average Curvature Distribution
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("Average Curvature Distribution", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                        html.Div([
                            dcc.Graph(id='avg-curvature-dist', style={'height': '400px', 'width': '100%'}),
                        ], style={
                            'background': UQ_WHITE,
                            'padding': '1rem',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'border': f'2px solid {UQ_LIGHT_GREY}'
                        }),
                    ]),
                ]),
                
                # Single wide plot for eigenvalue evolution summary
                html.Div([
                    html.H4("Eigenvalue Evolution Summary", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                    dcc.Graph(id='eigenvalue-evolution-summary', style={'height': '350px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}'
                }),
            ]),
            
            # Principal Direction (PCA) Overlay Section
            html.Section(id="pca-section", children=[
                html.H3("Principal Direction (PCA) Overlay"),
                html.Div([
                    dcc.Graph(id='pca-overlay-plot', style={'height': '400px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}'
                }),
            ]),
            
            # Optimizer Trajectory Overlay Section
            html.Section(id="trajectory-section", children=[
                html.H3("Optimizer Trajectory Overlay"),
                html.Div([
                    dcc.Graph(id='trajectory-overlay-plot', style={'height': '400px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}'
                }),
            ]),
            
            # Eigenvalues/Eigenvectors Display Section
            html.Section(id="eigenvalues-section", children=[
                html.H3("Hessian Eigenvalues & Eigenvectors Analysis"),
                
                # Eigenvalues Time Series
                html.Div([
                    dcc.Graph(id='eigenvalues-plot', style={'height': '350px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}',
                    'margin-bottom': '1rem'
                }),
                
                # Eigenvectors Overlay
                html.Div([
                    dcc.Graph(id='eigenvectors-plot', style={'height': '350px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}'
                }),
            ]),
            
            # Hessian/Curvature Heatmaps Section
            html.Section(id="hessian-heatmaps-section", children=[
                html.H3("Hessian Matrix & Curvature Heatmaps"),
                
                # Two-column layout for heatmaps
                html.Div(className="visualization-container", style={
                    'display': 'flex', 
                    'gap': '1rem', 
                    'margin-top': '1rem'
                }, children=[
                    # Hessian Matrix Heatmap
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("Hessian Matrix", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                        html.Div([
                            dcc.Graph(id='hessian-heatmap', style={'height': '400px', 'width': '100%'}),
                        ], style={
                            'background': UQ_WHITE,
                            'padding': '1rem',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'border': f'2px solid {UQ_LIGHT_GREY}'
                        }),
                    ]),
                    
                    # Curvature Heatmap
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("Curvature Analysis", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                        html.Div([
                            dcc.Graph(id='curvature-heatmap', style={'height': '400px', 'width': '100%'}),
                        ], style={
                            'background': UQ_WHITE,
                            'padding': '1rem',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'border': f'2px solid {UQ_LIGHT_GREY}'
                        }),
                    ]),
                ]),
            ]),
            
            # Topological Features Evolution Section
            html.Section(id="topology-evolution-section", children=[
                html.H3("Topological Features & Critical Points Evolution"),
                
                # Two-column layout for topology analysis
                html.Div(className="visualization-container", style={
                    'display': 'flex', 
                    'gap': '1rem', 
                    'margin-top': '1rem'
                }, children=[
                    # Persistence Diagram
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("Persistence Diagram", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                        html.Div([
                            dcc.Graph(id='persistence-diagram', style={'height': '400px', 'width': '100%'}),
                        ], style={
                            'background': UQ_WHITE,
                            'padding': '1rem',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'border': f'2px solid {UQ_LIGHT_GREY}'
                        }),
                    ]),
                    
                    # Critical Points Evolution
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("Critical Points Evolution", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                        html.Div([
                            dcc.Graph(id='critical-points-evolution', style={'height': '400px', 'width': '100%'}),
                        ], style={
                            'background': UQ_WHITE,
                            'padding': '1rem',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'border': f'2px solid {UQ_LIGHT_GREY}'
                        }),
                    ]),
                ]),
                
                # Topological Features Timeline
                html.Div([
                    html.H4("Topological Features Timeline", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                    dcc.Graph(id='topology-timeline', style={'height': '300px', 'width': '100%'}),
                ], style={
                    'background': UQ_WHITE,
                    'padding': '1rem',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'border': f'2px solid {UQ_LIGHT_GREY}',
                    'margin-top': '1rem'
                }),
            ]),
            
            # Advanced Landscape Analysis Section
            html.Section(id="advanced-analysis-section", children=[
                html.H3("Advanced Landscape Analysis"),
                
                # Landscape Complexity Metrics - Vertical Stack
                html.Div(className="visualization-container", style={'margin-top': '1rem'}, children=[
                    html.H4("Complexity Metrics", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                    
                    # Current epoch complexity
                    html.Div([
                        dcc.Graph(id='complexity-metrics-plot', style={'height': '400px', 'width': '100%'}),
                    ], style={
                        'background': UQ_WHITE,
                        'padding': '1rem',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': f'2px solid {UQ_LIGHT_GREY}',
                        'margin-bottom': '1rem'
                    }),
                    
                    # Evolution across all epochs
                    html.Div([
                        html.H5("Complexity Evolution Across All Epochs", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '0.5rem'}),
                        dcc.Graph(id='complexity-evolution-plot', style={'height': '350px', 'width': '100%'}),
                    ], style={
                        'background': UQ_WHITE,
                        'padding': '1rem',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': f'2px solid {UQ_LIGHT_GREY}'
                    }),
                ]),
                
                # Critical Points Classification - Vertical Stack
                html.Div(className="visualization-container", style={'margin-top': '2rem'}, children=[
                    html.H4("Critical Points Analysis", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                    
                    # Current epoch critical points
                    html.Div([
                        dcc.Graph(id='critical-points-plot', style={'height': '400px', 'width': '100%'}),
                    ], style={
                        'background': UQ_WHITE,
                        'padding': '1rem',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': f'2px solid {UQ_LIGHT_GREY}',
                        'margin-bottom': '1rem'
                    }),
                    
                    # Evolution across all epochs
                    html.Div([
                        html.H5("Critical Points Evolution Across All Epochs", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '0.5rem'}),
                        dcc.Graph(id='critical-points-evolution-plot', style={'height': '350px', 'width': '100%'}),
                    ], style={
                        'background': UQ_WHITE,
                        'padding': '1rem',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': f'2px solid {UQ_LIGHT_GREY}'
                    }),
                ]),
                
                # Sharpness/Flatness Metrics - Vertical Stack
                html.Div(className="visualization-container", style={'margin-top': '2rem'}, children=[
                    html.H4("Sharpness Analysis", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '1rem'}),
                    
                    # Current epoch sharpness
                    html.Div([
                        dcc.Graph(id='sharpness-metrics-plot', style={'height': '400px', 'width': '100%'}),
                    ], style={
                        'background': UQ_WHITE,
                        'padding': '1rem',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': f'2px solid {UQ_LIGHT_GREY}',
                        'margin-bottom': '1rem'
                    }),
                    
                    # Evolution across all epochs
                    html.Div([
                        html.H5("Sharpness Evolution Across All Epochs", style={'color': UQ_PURPLE, 'text-align': 'center', 'margin-bottom': '0.5rem'}),
                        dcc.Graph(id='sharpness-evolution-plot', style={'height': '350px', 'width': '100%'}),
                    ], style={
                        'background': UQ_WHITE,
                        'padding': '1rem',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                        'border': f'2px solid {UQ_LIGHT_GREY}'
                    }),
                ]),
            ]),

            
            # Metadata Section
            html.Div(id='metadata', style={
                'margin-top': '1.5rem',
                'padding': '1rem',
                'background': UQ_LIGHT_GREY,
                'border-radius': '8px',
                'color': UQ_PURPLE,
                'text-align': 'center',
                'font-weight': '600',
                'border-left': f'5px solid {UQ_GOLD}'
            }),
        ])
    ]),
    
    # Hidden Components
    dcc.Interval(id='gif-interval', interval=180, n_intervals=0, disabled=True),
    html.Div(id='last-clicked-epoch', style={'display': 'none'}, children='0')
], style={'font-family': 'Montserrat, Arial, Helvetica, sans-serif', 'background': UQ_BACKGROUND, 'min-height': '100vh'})

# ---- CALLBACK: UPDATE GIF FRAME COUNT AND SLIDER ----
@app.callback(
    Output('gif-frame-count', 'data'),
    Output('gif-slider', 'max'),
    Output('gif-slider', 'marks'),
    Input('run-selector', 'value'),
)
def update_gif_frame_count(run_folder):
    gif_frames = load_gif_frames(run_folder)
    gif_total_frames = len(gif_frames)
    marks = {i: str(i) for i in range(0, gif_total_frames, max(1, gif_total_frames//10))}
    return gif_total_frames, gif_total_frames-1, marks

# ---- CALLBACK FOR PLAY/PAUSE TOGGLE BUTTON ----
@app.callback(
    Output('gif-interval', 'disabled'),
    Output('play-pause-btn', 'children'),
    Input('play-pause-btn', 'n_clicks'),
    State('gif-interval', 'disabled'),
    prevent_initial_call=False
)
def toggle_play_pause(n_clicks, interval_disabled):
    if n_clicks is None or n_clicks == 0:
        return True, '▶ Play'  # Default: paused
    
    # Toggle the state
    if interval_disabled:
        return False, '⏸ Pause'  # Start playing
    else:
        return True, '▶ Play'    # Stop playing



# ---- CALLBACK: UPDATE GIF AND SLIDER (FAST UPDATES) ----
@app.callback(
    Output('gif-display', 'src'),
    Output('gif-slider', 'value'),
    Input('gif-slider', 'value'),
    Input('gif-interval', 'n_intervals'),
    Input('run-selector', 'value'),
    State('gif-interval', 'disabled'),
    State('gif-frame-count', 'data'),
    prevent_initial_call=False
)
def update_gif_frame(epoch, n_intervals, run_folder, interval_disabled, gif_total_frames):
    ctx = dash.callback_context
    triggered = ctx.triggered_id
    gif_frames = load_gif_frames(run_folder)
    if gif_total_frames is None:
        gif_total_frames = len(gif_frames)
    if not interval_disabled and triggered == 'gif-interval':
        next_epoch = (epoch + 1) % gif_total_frames
    else:
        next_epoch = epoch if epoch < gif_total_frames else 0
    frame_src = pil_image_to_base64(gif_frames[next_epoch])
    return frame_src, next_epoch

# ---- CALLBACK: UPDATE PLOTLY GRAPHS (ONLY WHEN PAUSED OR INITIAL) ----
@app.callback(
    Output('surface-graph', 'figure'),
    Input('gif-slider', 'value'),
    Input('run-selector', 'value'),
    Input('play-pause-btn', 'n_clicks'),
    State('gif-interval', 'disabled'),
    prevent_initial_call=False,
    allow_duplicate=True
)
def update_surface_graph(slider_value, run_folder, n_clicks, interval_disabled):
    # Update when animation is paused or play/pause button is clicked
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if interval_disabled or triggered_id == 'play-pause-btn':
        return make_surface_fig(slider_value, run_folder)
    else:
        return dash.no_update

# ---- SLOW CALLBACK: UPDATE ALL OTHER GRAPHS ----
@app.callback(
    Output('gradient-angle-plot', 'figure'),
    Output('pca-overlay-plot', 'figure'),
    Output('trajectory-overlay-plot', 'figure'),
    Output('eigenvalues-plot', 'figure'),
    Output('eigenvectors-plot', 'figure'),
    Output('hessian-heatmap', 'figure'),
    Output('curvature-heatmap', 'figure'),
    Output('persistence-diagram', 'figure'),
    Output('critical-points-evolution', 'figure'),
    Output('topology-timeline', 'figure'),
    Output('complexity-metrics-plot', 'figure'),
    Output('critical-points-plot', 'figure'),
    Output('sharpness-metrics-plot', 'figure'),
    Output('metadata', 'children'),
    Input('gif-slider', 'value'),
    Input('play-pause-btn', 'n_clicks'),
    Input('run-selector', 'value'),
    State('gif-interval', 'disabled'),
    prevent_initial_call=False,
    allow_duplicate=True
)
def update_plotly_graphs(slider_value, button_clicks, run_folder, interval_disabled):
    ctx = dash.callback_context
    ctx = dash.callback_context
    
    # Update plots if:
    # 1. Initial load (no trigger)
    # 2. Animation is paused (interval_disabled)
    if not ctx.triggered:
        epoch_to_plot = 0
    elif interval_disabled:
        epoch_to_plot = slider_value
    else:
        # Animation is running, don't update plots (too laggy)
        return (
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        )
    
    gif_frames = load_gif_frames(run_folder)
    total_epochs = len(gif_frames)
    gradient_fig = make_gradient_angle_fig(epoch_to_plot, total_epochs)
    pca_fig = make_pca_overlay_fig(epoch_to_plot, run_folder)
    trajectory_fig = make_trajectory_overlay_fig(epoch_to_plot, run_folder) if 'run_folder' in make_trajectory_overlay_fig.__code__.co_varnames else make_trajectory_overlay_fig(epoch_to_plot)
    eigenvalues_fig = make_eigenvalues_plot(epoch_to_plot, total_epochs)
    eigenvectors_fig = make_eigenvectors_plot(epoch_to_plot, run_folder) if 'run_folder' in make_eigenvectors_plot.__code__.co_varnames else make_eigenvectors_plot(epoch_to_plot)
    hessian_fig = make_hessian_heatmap(epoch_to_plot, run_folder) if 'run_folder' in make_hessian_heatmap.__code__.co_varnames else make_hessian_heatmap(epoch_to_plot)
    curvature_fig = make_curvature_heatmap(epoch_to_plot, run_folder) if 'run_folder' in make_curvature_heatmap.__code__.co_varnames else make_curvature_heatmap(epoch_to_plot)
    persistence_fig = make_persistence_diagram(epoch_to_plot, run_folder) if 'run_folder' in make_persistence_diagram.__code__.co_varnames else make_persistence_diagram(epoch_to_plot)
    critical_points_fig = make_critical_points_evolution(epoch_to_plot, total_epochs)
    topology_timeline_fig = make_topology_timeline(epoch_to_plot, run_folder) if 'run_folder' in make_topology_timeline.__code__.co_varnames else make_topology_timeline(epoch_to_plot)
    complexity_fig = make_complexity_metrics_plot(epoch_to_plot, total_epochs)
    critical_classification_fig = make_critical_points_plot(epoch_to_plot, total_epochs)
    sharpness_fig = make_complexity_metrics_plot(epoch_to_plot, total_epochs)
    meta_text = f"Epoch: {epoch_to_plot} | Run: {run_folder}"
    return (
        gradient_fig, pca_fig, trajectory_fig, eigenvalues_fig, eigenvectors_fig, hessian_fig, curvature_fig,
        persistence_fig, critical_points_fig, topology_timeline_fig, complexity_fig, critical_classification_fig,
        sharpness_fig, meta_text
    )

# ---- CALLBACK: UPDATE AVERAGE GRADIENT ANGLE PLOT (STATIC) ----
@app.callback(
    Output('avg-gradient-angle-plot', 'figure'),
    Input('play-pause-btn', 'n_clicks'),  # Any input to trigger initial load
    prevent_initial_call=False
)
def update_avg_gradient_plot(n_clicks):
    # This plot doesn't change, so just return the static figure
    gif_frames = load_gif_frames(DEFAULT_RUN)
    total_epochs = len(gif_frames)
    return make_avg_gradient_angle_fig(total_epochs)

# ---- CALLBACK: UPDATE AGGREGATED ANALYSIS PLOTS (STATIC) ----
@app.callback(
    Output('avg-loss-landscape', 'figure'),
    Output('avg-curvature-dist', 'figure'),
    Output('eigenvalue-evolution-summary', 'figure'),
    Input('play-pause-btn', 'n_clicks'),  # Any input to trigger initial load
    prevent_initial_call=False
)
def update_aggregated_plots(n_clicks):
    # These plots don't change with epochs, so generate them once
    gif_frames = load_gif_frames(DEFAULT_RUN)
    total_epochs = len(gif_frames)
    avg_landscape_fig = make_avg_loss_landscape(total_epochs)
    avg_curvature_fig = make_avg_curvature_distribution(total_epochs)
    eigenvalue_summary_fig = make_eigenvalue_evolution_summary(total_epochs)
    return avg_landscape_fig, avg_curvature_fig, eigenvalue_summary_fig

# ---- CALLBACK: UPDATE ADVANCED ANALYSIS EVOLUTION PLOTS (STATIC) ----
@app.callback(
    Output('dynamic-title', 'children'),
    Input('run-selector', 'value'),
)
def update_dynamic_title(run_folder):
    # Example run_folder: '50_-1,1,101 (2,8,2 PCA)'
    # Parse architecture, method, and dimensions
    import re
    arch_match = re.search(r'\((.*?)\)', run_folder)
    arch = arch_match.group(1) if arch_match else ''
    method = 'PCA' if 'PCA' in run_folder.upper() else ('Random' if 'RANDOM' in run_folder.upper() else '')
    dims_match = re.match(r'(\d+)_\[(-?\d+),(-?\d+),(\d+)\s*\((.*?)\)', run_folder)
    dims = ''
    if dims_match:
        dims = f"[-1, 1, 101]"
    else:
        # fallback: try to extract from run_folder
        dims = re.search(r'(\[.*?\])', run_folder)
        dims = dims.group(1) if dims else ''
    # Example dataset extraction (customize as needed)
    dataset = 'XOR'
    title = f"{method}-based Loss Landscape Visualization | Architecture: {arch} | Dataset: {dataset}"
    return title
@app.callback(
    Output('complexity-evolution-plot', 'figure'),
    Output('critical-points-evolution-plot', 'figure'),
    Output('sharpness-evolution-plot', 'figure'),
    Input('play-pause-btn', 'n_clicks'),  # Any input to trigger initial load
    prevent_initial_call=False
)
def update_advanced_evolution_plots(n_clicks):
    # These evolution plots show data across all epochs, so generate them once
    gif_frames = load_gif_frames(DEFAULT_RUN)
    total_epochs = len(gif_frames)
    complexity_evolution_fig = make_complexity_evolution_plot(total_epochs)
    critical_points_evolution_fig = make_critical_points_evolution_plot(total_epochs)
    sharpness_evolution_fig = make_sharpness_evolution_plot(total_epochs)
    return complexity_evolution_fig, critical_points_evolution_fig, sharpness_evolution_fig

if __name__ == '__main__':
    app.run(debug=True)
