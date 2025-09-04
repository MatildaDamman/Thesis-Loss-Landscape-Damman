import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import h5py
import os
import plotly.graph_objs as go
from gudhi import CubicalComplex
from gudhi.hera import wasserstein_distance

# ----------- CONFIGURE DATA PATHS HERE -----------
SURFACE_DIR = './surfaces'
TRAJ_DIR = './trajectories'
TOPO_DIR = './topology'
EPOCH_MIN = 0
EPOCH_MAX = 9


# ----------- HELPER FUNCTIONS: LOAD/GENERATE DATA ---------

def available_epochs():
    files = [fname for fname in os.listdir(SURFACE_DIR) if fname.endswith('.h5')]
    numbers = []
    for fname in files:
        try:
            ep = int(fname.split('_')[-1].replace('.h5', ''))
            numbers.append(ep)
        except ValueError:
            continue
    return sorted(numbers)

EPOCHS = available_epochs()
if not EPOCHS:
    EPOCHS = list(range(EPOCH_MIN, EPOCH_MAX + 1)) # fallback

def load_surface(epoch):
    path = os.path.join(SURFACE_DIR, f'surface_epoch_{epoch}.h5')
    with h5py.File(path, 'r') as f:
        x = np.array(f['xcoordinates'])
        y = np.array(f['ycoordinates'])
        loss = np.array(f['train_loss'])
    return x, y, loss

def load_trajectory(epoch):
    path = os.path.join(TRAJ_DIR, f'trajectory_epoch_{epoch}.npy')
    if os.path.exists(path):
        return np.load(path)
    return None

def compute_persistence(loss_grid):
    """
    Compute 0-d and 1-d persistent homology of a 2D landscape using Gudhi.
    Returns barcodes (list of tuples), e.g., [(birth, death), ...]
    """
    # Gudhi needs min as filtration: invert for superlevel analysis if wanted
    cube = CubicalComplex(top_dimensional_cells=loss_grid)
    cube.compute_persistence()
    H0 = cube.persistence_intervals_in_dimension(0)  # Connected components
    H1 = cube.persistence_intervals_in_dimension(1)  # Loops/holes
    return H0, H1

def load_topology(epoch):
    _, _, loss = load_surface(epoch)
    # Normalise/reshape if needed; Gudhi expects a 2D grid (not flat).
    H0, H1 = compute_persistence(loss)
    # Dummy minima/saddles logic remains; focus here is persistence bars.
    minima = np.array([[0, 0]])  # TODO: replace with real minima detection if needed
    saddles = np.array([[-0.2, 0.1]])  # Ditto for saddles
    # Format: list of dicts for Dash/Plotly bar plotting
    persistence = []
    for b, d in H0:
        persistence.append({"birth": float(b), "death": float(d), "dimension": 0})
    for b, d in H1:
        persistence.append({"birth": float(b), "death": float(d), "dimension": 1})
    return minima, saddles, persistence


def load_metrics():
    xs = EPOCHS
    tr_loss = np.linspace(8, 0.5, len(xs)) + 0.5 * np.random.randn(len(xs))
    val_loss = tr_loss * np.random.uniform(0.9, 1.1, len(xs))
    return xs, tr_loss, val_loss




# -------------- DASH & UQ-BRANDED UI LAYOUT -----------------

app = dash.Dash(
    __name__,
    external_stylesheets=["/assets/uq_style.css"],
    title="UQ Loss Landscape Evolution Dashboard"
)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.Header([
        html.Div(className="uq-banner", children=[
            html.Img(src="https://static.uq.net.au/v3/logos/uq-lockup-rgb-white.svg", className="uq-logo"),
            html.H1("Neural Network Loss Landscape Evolution", className="banner-title")
        ])
    ]),
    html.Div(className="main-wrapper", children=[
        html.Nav(className="sidebar", children=[
            html.H2("Navigation", className="sidebar-title"),
            html.Ul([
                html.Li(html.A("3D Surface", href="#surface-section", className="sidebar-link")),
                html.Li(html.A("2D Contours", href="#contours-section", className="sidebar-link")),
                html.Li(html.A("Topology", href="#topology-section", className="sidebar-link")),
                html.Li(html.A("Metrics", href="#metrics-section", className="sidebar-link")),
            ], className="nav-list")
        ]),
        html.Main(className="content-area", children=[
            html.Div(className="controls-row", children=[
                html.Label(["Epoch:",
                    dcc.Slider(
                        id='epoch-slider', min=EPOCHS[0], max=EPOCHS[-1], step=1, value=EPOCHS[0],
                        marks={ep: f"{ep}" for ep in EPOCHS},
                        className="uq-slider",
                        tooltip={"always_visible": True},
                        included=True,
                        updatemode="drag",
                    )
                ]),
                html.Button('Start/Stop Auto Play', id='auto-btn', className="uq-btn"),
                dcc.Interval(id='auto-interval', interval=700, n_intervals=0, disabled=True, max_intervals=10000)
            ]),
            html.Section(id="surface-section", children=[
                html.H3("3D Loss Surface (with Trajectory)"),
                dcc.Graph(id="surface-plot", style={"height": "400px"}),
            ]),
            html.Section(id="contours-section", children=[
                html.H3("2D Contour Map (Critical Points)"),
                dcc.Graph(id="contour-plot", style={"height": "400px"}),
            ]),
            html.Section(id="topology-section", children=[
                html.H3("Topological Complexity"),
                dcc.Graph(id="topology-plot", style={"height": "300px"}),
                dcc.Graph(id="diagram-plot", style={"height": "300px"}),
                html.Div(id="topology-summary")
            ]),
            html.Section(id="metrics-section", children=[
                html.H3("Training & Validation Metrics"),
                dcc.Graph(id="metrics-plot", style={"height": "250px"}),
                html.Div(id="metrics-info"),
            ]),
            html.Div(id='epoch-info', style={"margin": "12px 0 0 0"}),
        ])
    ])
])

# -------------------- CALLBACK INTERACTIONS --------------------------------

@app.callback(
    Output('auto-interval', 'disabled'),
    Input('auto-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_auto_play(n_clicks):
    # Even: stopped; Odd: running
    return n_clicks % 2 == 0

@app.callback(
    Output('epoch-slider', 'value'),
    Input('auto-interval', 'n_intervals'),
    State('epoch-slider', 'value'),
    State('auto-interval', 'disabled'),
)
def animate_slider(n_intervals, current_epoch, interval_disabled):
    if interval_disabled: return current_epoch
    idx = EPOCHS.index(current_epoch)
    direction = (n_intervals // len(EPOCHS)) % 2
    if direction == 0:
        idx_new = idx + 1 if idx < len(EPOCHS) - 1 else idx - 1
    else:
        idx_new = idx - 1 if idx > 0 else idx + 1
    return EPOCHS[max(0, min(idx_new, len(EPOCHS)-1))]

@app.callback(
    [Output('surface-plot', 'figure'),
     Output('contour-plot', 'figure'),
     Output('topology-plot', 'figure'),
     Output('diagram-plot', 'figure'),
     Output('metrics-plot', 'figure'),
     Output('epoch-info', 'children'),
     Output('topology-summary', 'children'),
     Output('metrics-info', 'children')],
    [Input('epoch-slider', 'value')]
)
def update_dashboard(epoch):
    # -- 3D Surface + Trajectory --
    x, y, loss = load_surface(epoch)
    traj = load_trajectory(epoch)
    surface_traces = [
        go.Surface(z=loss, x=x, y=y, colorscale='Viridis', colorbar=dict(title="Loss")),
    ]
    if traj is not None and len(traj.shape) == 2:
        zline = np.interp(traj[:, 0], x[:,0], loss[:,0]) if loss.shape == x.shape else np.zeros(len(traj))
        surface_traces.append(
            go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=zline,
                mode='lines+markers',
                line=dict(color='#ffcc00', width=5),
                marker=dict(size=5, color="#51247a"),
                name='Optimizer Trajectory'
            )
        )
    surface_fig = go.Figure(surface_traces)
    surface_fig.update_layout(
        title=f"3D Loss Surface – Epoch {epoch}",
        scene=dict(xaxis_title='Direction 1', yaxis_title='Direction 2', zaxis_title='Loss'),
        margin=dict(l=10, r=10, b=10, t=35)
    )

    # -- 2D Contour with Critical Points --
    minima, saddles, _ = load_topology(epoch)
    contour_fig = go.Figure([
        go.Contour(z=loss, x=x, y=y, colorscale='Viridis', contours=dict(showlabels=True)),
        go.Scatter(x=minima[:, 0], y=minima[:, 1], mode='markers',
                   marker=dict(color='#ffcc00', size=10, symbol='star'),
                   name='Minima'),
        go.Scatter(x=saddles[:, 0], y=saddles[:, 1], mode='markers',
                   marker=dict(color='#51247a', size=11, symbol='x'),
                   name='Saddles')
    ])
    contour_fig.update_layout(
        title=f"2D Contour Map with Critical Points – Epoch {epoch}",
        xaxis_title="Direction 1", yaxis_title="Direction 2",
        margin=dict(l=10, r=10, b=10, t=35)
    )

    # -- Topological Features (Persistence, etc.) --
    minima, saddles, persistence = load_topology(epoch)
    if persistence:
        dimensions = [p["dimension"] for p in persistence]
        barcode_fig = go.Figure()
        for d in [0,1]:
            xs = [f"{i+1}" for i, p in enumerate(persistence) if p["dimension"] == d]
            ys = [p["death"] - p["birth"] for p in persistence if p["dimension"] == d]
            barcode_fig.add_trace(go.Bar(
                x=xs, y=ys, name=f"H{d} (Dim-{d})",
                marker_color=UQ_GOLD if d==0 else "#009CA6"
            ))
        barcode_fig.update_layout(
            title="Persistence Barcode (Evolution of Features)",
            xaxis_title="Feature", yaxis_title="Persistence (death-birth)",
            barmode='stack', legend_title="Homology Dimension",
            margin=dict(l=10, r=10, b=10, t=35)
        )
    else:
        barcode_fig = go.Figure()
        
    # -- Persistence Diagram --
    bd_fig = go.Figure()
    for d in [0, 1]:
        bd = [(p["birth"], p["death"]) for p in persistence if p["dimension"] == d]
        if bd:
            births, deaths = zip(*bd)
            bd_fig.add_trace(go.Scatter(
                x=births, y=deaths, mode='markers',
                name=f"H{d}",
                marker=dict(size=8, color=UQ_GOLD if d == 0 else "#009CA6")
            ))
    bd_fig.add_shape(
        type='line', x0=0, y0=0, x1=1, y1=1,
        line=dict(color="gray", dash='dot'), xref='paper', yref='paper'
    )
    bd_fig.update_layout(
        title="Persistence Diagram",
        xaxis_title="Birth", yaxis_title="Death",
        margin=dict(l=10, r=10, b=10, t=35),
        showlegend=True,
    )

    # -- Metrics (Loss curves etc.) --
    xs, tr_loss, val_loss = load_metrics()
    metrics_fig = go.Figure([
        go.Scatter(x=xs, y=tr_loss, mode='lines+markers', name='Training Loss', line=dict(color=UQ_PURPLE)),
        go.Scatter(x=xs, y=val_loss, mode='lines+markers', name='Val Loss', line=dict(color=UQ_GOLD))
    ])
    metrics_fig.update_layout(
        title=f'Loss Trajectories', xaxis_title='Epoch', yaxis_title='Loss',
        margin=dict(l=10, r=10, b=10, t=35)
    )

    epoch_info = html.Div([
        html.P(f"Currently displaying epoch {epoch}."),
        html.P("Use the slider or Start/Stop Auto Play for temporal exploration.")
    ], style={'color': UQ_GREY})

    topology_info = html.Ul([
        html.Li(f"Number of minima: {minima.shape[0]}"),
        html.Li(f"Number of saddles: {saddles.shape[0]}"),
        html.Li(f"Persistence features tracked: {len(persistence)}"),
    ])

    metrics_info = html.P(
        "Visualize progression of loss/metrics. Link spikes or drops to changes in landscape or topology."
    )
    
    return surface_fig, contour_fig, barcode_fig, bd_fig, metrics_fig, epoch_info, topology_info, metrics_info


UQ_PURPLE = "#51247a"
UQ_GOLD = "#ffcc00"
UQ_GREY = "#808080"

if __name__ == '__main__':
    app.run(debug=True)
