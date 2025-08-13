#!/usr/bin/env python3
"""
Correlation Matrix Visualization Script
Creates a beautiful correlation matrix with yellow, green, and violet color scheme
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data(filepath):
    """
    Load CSV file and prepare numeric columns for correlation analysis
    """
    # Read CSV with semicolon delimiter
    df = pd.read_csv(filepath, delimiter=',')

    # Define columns to analyze
    columns_to_analyze = [
        'ClickstreamEnt1Ent2',
        'PopularityEnt1',
        'PopularityEnt2',
        'CosineSimilarityEnt1Ent2',
        'DBpediaSimilarityEnt1Ent2',
        'DBpediaRelatednessEnt1Ent2',
        'PalmaInterestingnessEnt1Ent2',
        'Perceived_Relatedness',
        'Subjective_Knowledge',
        'Serendipity'
    ]

    # Filter to existing columns
    valid_columns = [col for col in columns_to_analyze if col in df.columns]

    # Handle PalmaInterestingness special format (dots as thousands separators)
    if 'PalmaInterestingnessEnt1Ent2' in df.columns:
        df['PalmaInterestingnessEnt1Ent2'] = df['PalmaInterestingnessEnt1Ent2'].apply(
            lambda x: float(str(x).replace('.', '')) if pd.notna(x) else np.nan
        )

    # Select only numeric columns
    df_numeric = df[valid_columns].select_dtypes(include=[np.number])

    return df_numeric


def create_custom_colormap():
    """
    Create a custom colormap with violet for negative, yellow for zero, and green for positive correlations
    """
    colors = [
        (0.0, (138 / 255, 43 / 255, 226 / 255)),  # Violet (strong negative)
        (0.3, (147 / 255, 51 / 255, 234 / 255)),  # Blue-violet (weak negative)
        (0.5, (255 / 255, 255 / 255, 100 / 255)),  # Yellow (no correlation)
        (0.7, (124 / 255, 252 / 255, 0 / 255)),  # Green-yellow (weak positive)
        (1.0, (0 / 255, 128 / 255, 0 / 255))  # Green (strong positive)
    ]

    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom_correlation', colors, N=n_bins)
    return cmap


def plot_correlation_matrix(df, method='pearson', cluster=True, figsize=(14, 12)):
    """
    Create a beautiful correlation matrix visualization

    Parameters:
    -----------
    df : DataFrame
        Numeric data for correlation
    method : str
        Correlation method ('pearson', 'spearman', or 'kendall')
    cluster : bool
        Whether to apply hierarchical clustering to reorder variables
    figsize : tuple
        Figure size
    """

    # Calculate correlation matrix
    corr_matrix = df.corr(method=method)

    # Apply hierarchical clustering if requested
    if cluster and len(corr_matrix) > 2:
        # Create distance matrix and perform clustering
        distance_matrix = 1 - np.abs(corr_matrix)
        condensed_distances = squareform(distance_matrix)
        linkage = hierarchy.linkage(condensed_distances, method='average')
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        cluster_order = dendro['leaves']

        # Reorder correlation matrix
        corr_matrix = corr_matrix.iloc[cluster_order, cluster_order]

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # Create gridspec for complex layout
    gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 4, 0.3], width_ratios=[0.5, 4, 0.3],
                          hspace=0.02, wspace=0.02)

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 1])

    # Create custom colormap
    custom_cmap = create_custom_colormap()

    # Plot heatmap
    im = ax_main.imshow(corr_matrix, cmap=custom_cmap, aspect='auto', vmin=-1, vmax=1)

    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            value = corr_matrix.iloc[i, j]
            # Choose text color based on background
            text_color = 'white' if abs(value) > 0.5 else 'black'
            if i != j:  # Don't show 1.00 on diagonal
                ax_main.text(j, i, f'{value:.2f}', ha='center', va='center',
                             fontsize=8, color=text_color, fontweight='bold')

    # Customize main plot
    ax_main.set_xticks(np.arange(len(corr_matrix.columns)))
    ax_main.set_yticks(np.arange(len(corr_matrix.columns)))
    ax_main.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax_main.set_yticklabels(corr_matrix.columns, fontsize=10)

    # Add grid
    ax_main.set_xticks(np.arange(len(corr_matrix.columns) + 1) - 0.5, minor=True)
    ax_main.set_yticks(np.arange(len(corr_matrix.columns) + 1) - 0.5, minor=True)
    ax_main.grid(which='minor', color='white', linestyle='-', linewidth=2)

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[1, 2])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=12)

    # Add dendrograms if clustered
    if cluster and len(corr_matrix) > 2:
        # Top dendrogram
        ax_top = fig.add_subplot(gs[0, 1])
        hierarchy.dendrogram(linkage, ax=ax_top, no_labels=True, color_threshold=0,
                             above_threshold_color='#764ba2')
        ax_top.axis('off')

        # Left dendrogram
        ax_left = fig.add_subplot(gs[1, 0])
        hierarchy.dendrogram(linkage, ax=ax_left, orientation='left', no_labels=True,
                             color_threshold=0, above_threshold_color='#764ba2')
        ax_left.axis('off')

    # Add title
    fig.suptitle(f'Correlation Matrix Heatmap ({method.capitalize()} Correlation)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add statistical summary box
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')

    # Calculate statistics
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)
    stats_text = (
        f"Statistics Summary:\n"
        f"• Strongest Positive: {upper_triangle.max().max():.3f}\n"
        f"• Strongest Negative: {upper_triangle.min().min():.3f}\n"
        f"• Mean Correlation: {upper_triangle.mean().mean():.3f}\n"
        f"• Median Correlation: {upper_triangle.median().median():.3f}"
    )

    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, corr_matrix


def plot_correlation_network(df, threshold=0.3, method='pearson', figsize=(12, 10)):
    """
    Create a network visualization of strong correlations
    """
    import networkx as nx

    corr_matrix = df.corr(method=method)

    # Create graph from correlation matrix
    G = nx.Graph()

    # Add nodes
    for col in corr_matrix.columns:
        G.add_node(col)

    # Add edges for correlations above threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                           weight=abs(corr_value), correlation=corr_value)

    fig, ax = plt.subplots(figsize=figsize)

    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#FFD700', node_size=3000,
                           alpha=0.9, ax=ax)

    # Draw edges with colors based on correlation
    edges = G.edges()
    weights = [G[u][v]['correlation'] for u, v in edges]

    # Create custom colormap for edges
    cmap = create_custom_colormap()

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=3,
                           edge_color=weights, edge_cmap=cmap,
                           edge_vmin=-1, edge_vmax=1, alpha=0.7, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Add edge labels for correlation values
    edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    ax.set_title(f'Correlation Network (|r| ≥ {threshold})', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig


def analyze_serendipity_predictors(df):
    """
    Analyze which variables are most correlated with Serendipity
    """
    if 'Serendipity' not in df.columns:
        print("Warning: 'Serendipity' column not found in data")
        return None

    # Calculate correlations with Serendipity
    serendipity_corr = df.corr()['Serendipity'].sort_values(ascending=False)

    # Remove self-correlation
    serendipity_corr = serendipity_corr[serendipity_corr.index != 'Serendipity']

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars based on correlation strength
    colors = [create_custom_colormap()((c + 1) / 2) for c in serendipity_corr.values]

    bars = ax.barh(range(len(serendipity_corr)), serendipity_corr.values, color=colors)
    ax.set_yticks(range(len(serendipity_corr)))
    ax.set_yticklabels(serendipity_corr.index)
    ax.set_xlabel('Correlation with Serendipity', fontsize=12)
    ax.set_title('Predictors of Serendipity (Ranked by Correlation)', fontsize=14, fontweight='bold')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, serendipity_corr.values)):
        ax.text(value + 0.01 if value > 0 else value - 0.01, i, f'{value:.3f}',
                va='center', ha='left' if value > 0 else 'right', fontweight='bold')

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Add grid
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig, serendipity_corr


def main():
    """
    Main function to run the correlation analysis
    """
    # File path - update this to your file location
    filepath = 'C:\\Users\\palmaco\\Downloads\\enhanced_entity_pairs.csv'

    print("Loading and preparing data...")
    df = load_and_prepare_data(filepath)

    print(f"Data shape: {df.shape}")
    print(f"Columns analyzed: {list(df.columns)}")

    # Create correlation matrix visualization
    print("\nCreating correlation matrix visualization...")
    fig1, corr_matrix = plot_correlation_matrix(df, method='pearson', cluster=True)
    plt.savefig('correlation_matrix_clustered.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_matrix_clustered.png")

    # Create correlation network
    print("\nCreating correlation network...")
    fig2 = plot_correlation_network(df, threshold=0.3, method='pearson')
    plt.savefig('correlation_network.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_network.png")

    # Analyze Serendipity predictors
    print("\nAnalyzing Serendipity predictors...")
    fig3, serendipity_corr = analyze_serendipity_predictors(df)
    if fig3:
        plt.savefig('serendipity_predictors.png', dpi=300, bbox_inches='tight')
        print("Saved: serendipity_predictors.png")

        print("\nTop 5 Serendipity Predictors:")
        print(serendipity_corr.head())

    # Show all plots
    plt.show()

    # Export correlation matrix to CSV
    corr_matrix.to_csv('correlation_matrix.csv')
    print("\nCorrelation matrix saved to: correlation_matrix.csv")

    return corr_matrix


if __name__ == "__main__":
    corr_matrix = main()