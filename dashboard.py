import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import h5py
import os
import plotly.graph_objs as go

SURFACE_DIR = './surfaces'
#old one 

def load_surface(epoch):
    path = os.path.join(SURFACE_DIR, f'surface_epoch_{epoch}.h5')
    with h5py.File(path, 'r') as f:
        x = np.array(f['xcoordinates'])
        y = np.array(f['ycoordinates'])
        loss = np.array(f['train_loss'])
    return x, y, loss

def available_epochs():
    epochs = [
        int(fname.split('_')[-1].replace('.h5', ''))
        for fname in os.listdir(SURFACE_DIR)
        if fname.startswith('surface_epoch_') and fname.endswith('.h5')
    ]
    return sorted(epochs)

app = dash.Dash(__name__, title="Loss Landscape Dashboard")
epochs = available_epochs()
initial_epoch = epochs[0] if epochs else 0

app.layout = html.Div([
    html.H2("Neural Net Loss Landscape Evolution"),
    html.Div([
        html.Label("Epoch:"),
        dcc.Slider(
            id='epoch-slider',
            min=epochs[0],
            max=epochs[-1],
            step=1,
            value=initial_epoch,
            marks={str(ep): str(ep) for ep in epochs},
            updatemode='drag'
        ),
        html.Button('Start/Stop Auto Play', id='auto-btn', n_clicks=0),
        dcc.Interval(id='auto-interval', interval=700, n_intervals=0, disabled=True)
    ], style={'margin': '30px'}),
    dcc.Tabs([
        dcc.Tab(label='3D Surface', children=[
            dcc.Graph(id='surface-plot', style={'height': '700px'})
        ]),
        dcc.Tab(label='2D Contours', children=[
            dcc.Graph(id='contour-plot', style={'height': '700px'})
        ]),
    ]),
    html.Div(id='epoch-info')
])

@app.callback(
    Output('auto-interval', 'disabled'),
    Input('auto-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_auto_play(n_clicks):
    return n_clicks % 2 == 0

@app.callback(
    Output('epoch-slider', 'value'),
    Input('auto-interval', 'n_intervals'),
    State('epoch-slider', 'value'),
    State('auto-interval', 'disabled'),
)
def animate_slider(n_intervals, current_epoch, interval_disabled):
    if interval_disabled:
        return current_epoch
    idx = epochs.index(current_epoch)
    direction = (n_intervals // len(epochs)) % 2  # 0: forward, 1: backward
    if direction == 0:
        new_idx = idx + 1 if idx < len(epochs)-1 else idx - 1
    else:
        new_idx = idx - 1 if idx > 0 else idx + 1
    # Bound check
    if new_idx < 0:
        new_idx = 0
    elif new_idx >= len(epochs):
        new_idx = len(epochs)-1
    return epochs[new_idx]

@app.callback(
    [Output('surface-plot', 'figure'), Output('contour-plot', 'figure'), Output('epoch-info', 'children')],
    [Input('epoch-slider', 'value')]
)
def update_plot(epoch):
    x, y, loss = load_surface(epoch)
    surface_fig = go.Figure(data=[go.Surface(
        z=loss, x=x, y=y,
        colorscale='Viridis',
        colorbar=dict(title="Loss"),
    )])
    surface_fig.update_layout(
        title=f"Loss Surface Epoch {epoch}",
        scene=dict(xaxis_title='Direction 1', yaxis_title='Direction 2', zaxis_title='Loss'),
        margin=dict(l=10, r=10, b=10, t=40)
    )

    contour_fig = go.Figure(data=[go.Contour(
        z=loss, x=x, y=y,
        colorscale='Viridis',
        contours=dict(showlabels=True)
    )])
    contour_fig.update_layout(
        title=f"Loss Contour Epoch {epoch}",
        xaxis_title='Direction 1',
        yaxis_title='Direction 2',
        margin=dict(l=10, r=10, b=10, t=40)
    )

    info = html.P(f"Currently displaying epoch {epoch}. Use the slider or auto-play to see changes.")
    return surface_fig, contour_fig, info

if __name__ == '__main__':
    app.run(debug=True)
