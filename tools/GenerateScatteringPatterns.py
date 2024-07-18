import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ScattBO.utils import generate_structure, calculate_scattering
from pathlib import Path
import os
ROOT_DIR = Path(os.getcwd()).parent.resolve()

# Generate the structure and calculate the scattering
cluster = generate_structure(pH=10, pressure=75, solvent="Ethanol", atom="Au")
q, I = calculate_scattering(cluster, function="Iq")
q, S = calculate_scattering(cluster, function="Sq")
q, F = calculate_scattering(cluster, function="Fq")
r, G = calculate_scattering(cluster, function="Gr")
q_SAXS, SAXS = calculate_scattering(cluster, function="SAXS")

# Define the directories
dirs = ['Data/Iq', 'Data/Sq', 'Data/Fq', 'Data/Gr', 'Data/SAXS']

# Ensure the directories exist
for dir in dirs:
    os.makedirs(os.path.join(ROOT_DIR, dir), exist_ok=True)

# Save files
np.save(os.path.join(ROOT_DIR, 'Data/Iq/Target_Iq_benchmark.npy'), np.vstack((q, I)).T)
np.save(os.path.join(ROOT_DIR, 'Data/Sq/Target_Sq_benchmark.npy'), np.vstack((q, S)).T)
np.save(os.path.join(ROOT_DIR, 'Data/Fq/Target_Fq_benchmark.npy'), np.vstack((q, F)).T)
np.save(os.path.join(ROOT_DIR, 'Data/Gr/Target_PDF_benchmark.npy'), np.vstack((r, G)).T)
np.save(os.path.join(ROOT_DIR, 'Data/SAXS/Target_SAXS_benchmark.npy'), np.vstack((q_SAXS, SAXS)).T)


# Create the subplots
fig = make_subplots(rows=5, cols=1)

# Add the traces
fig.add_trace(go.Scatter(x=q, y=I, mode='lines', name='Iq'), row=1, col=1)
fig.add_trace(go.Scatter(x=q, y=S, mode='lines', name='Sq'), row=2, col=1)
fig.add_trace(go.Scatter(x=q, y=F, mode='lines', name='Fq'), row=3, col=1)
fig.add_trace(go.Scatter(x=r, y=G, mode='lines', name='Gr'), row=4, col=1)
fig.add_trace(go.Scatter(x=q_SAXS, y=SAXS, mode='lines', name='SAXS'), row=5, col=1)

# Set the x-axis and y-axis of the SAXS subplot to log scale
fig.update_xaxes(type='log', row=5, col=1)
fig.update_yaxes(type='log', row=5, col=1)

# Update layout
fig.update_layout(
    title={
        'text': 'Scattering patterns of Icosahedron, noshells=7',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    autosize=False,
    width=800,
    height=1200
)

# Update x and y axis titles for each subplot
fig.update_xaxes(title_text='q (Å⁻¹)', row=1, col=1)
fig.update_yaxes(title_text='I(q) (a.u.)', row=1, col=1)

fig.update_xaxes(title_text='q (Å⁻¹)', row=2, col=1)
fig.update_yaxes(title_text='S(q) (a.u.)', row=2, col=1)

fig.update_xaxes(title_text='q (Å⁻¹)', row=3, col=1)
fig.update_yaxes(title_text='F(q) (a.u.)', row=3, col=1)

fig.update_xaxes(title_text='r (Å)', row=4, col=1)
fig.update_yaxes(title_text='G(r) (a.u.)', row=4, col=1)

fig.update_xaxes(title_text='q (Å⁻¹)', row=5, col=1)
fig.update_yaxes(title_text='SAXS (a.u.)', row=5, col=1)

# Show the plot
fig.show()