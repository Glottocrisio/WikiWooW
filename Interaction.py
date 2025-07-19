import plotly.graph_objects as go
import pandas as pd
import numpy as np


df =  pd.read_csv('updated_data.csv')

# Create the scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Parameter1'],
    y=df['Parameter2'],
    mode='markers',
    marker=dict(
        size=df['Parameter3']*3,
        color=df['Parameter3'],
        colorscale='Viridis',
        showscale=True
    ),
    text=[f'Entity1: {e1}<br>Entity2: {e2}<br>Param3: {p3}' for e1, e2, p3 in zip(df['Entity1'], df['Entity2'], df['Parameter3'])],
    hoverinfo='text'
))

# Update layout
fig.update_layout(
    title='Wikipedia Entity Pairs Visualization',
    xaxis_title='Parameter 1',
    yaxis_title='Parameter 2',
    hovermode='closest'
)

# Show the plot
fig.show()

# If you want to save the plot as an HTML file
fig.write_html("entity_pairs_visualization.html")
