import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def chord_diagram(matrix, names, colors, ax):
    n = len(matrix)
    
    # Create the circular layout
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Draw the nodes
    for i in range(n):
        x = np.cos(theta[i])
        y = np.sin(theta[i])
        ax.add_patch(patches.Circle((x, y), 0.05, fill=True, color=colors[i]))
        ax.text(1.1*x, 1.1*y, names[i], ha='center', va='center', rotation=theta[i]*180/np.pi-90)
    
    # Draw the connections
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] > 0:
                start = theta[i]
                end = theta[j]
                
                # Create the quadratic bezier curve
                verts = [
                    (np.cos(start), np.sin(start)),
                    (0, 0),
                    (np.cos(end), np.sin(end))
                ]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)
                
                patch = patches.PathPatch(path, facecolor='none', edgecolor=colors[i], alpha=0.3, lw=matrix[i][j]/50)
                ax.add_patch(patch)

# Load and preprocess the data
data = pd.read_csv('finalanubi.txt', sep=';')

# Create a matrix of connections
entities = pd.concat([data['Entity1'], data['Entity2']]).unique()
n = len(entities)
matrix = np.zeros((n, n))

for _, row in data.iterrows():
    i = np.where(entities == row['Entity1'])[0][0]
    j = np.where(entities == row['Entity2'])[0][0]
    matrix[i][j] = matrix[j][i] = row['ClickstreamEnt1Ent2']

# Select top connections
top_connections = 15
top_indices = np.argsort(matrix.sum(axis=1))[-top_connections:]
top_entities = entities[top_indices]
top_matrix = matrix[top_indices][:, top_indices]

# Create the chord diagram
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis('off')

colors = plt.cm.viridis(np.linspace(0, 1, len(top_entities)))
chord_diagram(top_matrix, top_entities, colors, ax)

plt.title('Chord Diagram of Top Connections with Anubis', fontsize=16)
plt.tight_layout()
plt.show()
