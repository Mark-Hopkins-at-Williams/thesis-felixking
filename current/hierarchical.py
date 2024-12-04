import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def perform_hierarchical_clustering(data, n_clusters=None, distance_threshold=None, output_dir=None, dpi=300):
    """
    Perform hierarchical clustering on the data and visualize results.
    
    Parameters:
    data: array-like, shape (n_samples, n_features)
        The input data to cluster
    n_clusters: int, optional
        Number of clusters to find. If None, distance_threshold must be provided
    distance_threshold: float, optional
        The threshold to apply when forming flat clusters
    output_dir: str, optional
        Directory to save the plots
    dpi: int, optional
        Dots per inch for the saved plots
        
    Returns:
    labels: array
        Cluster labels for each point
    """
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Compute the linkage matrix
    # method options: 'single', 'complete', 'average', 'ward'
    linkage_matrix = linkage(data_scaled, method='ward')
    
    # # Plot and save the dendrogram
    # plt.figure(figsize=(15, 10))
    # dendrogram(linkage_matrix)
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Distance')
    # if output_dir:
    #     plt.savefig(
    #         os.path.join(output_dir, 'dendrogram.png'),
    #         dpi=dpi,
    #         bbox_inches='tight'
    #         )
    # plt.close()
    
    # Get cluster labels
    if n_clusters is not None:
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    else:
        labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    sil_score = silhouette_score(data_scaled, labels)
    cluster_scores = silhouette_samples(data_scaled, labels)

    return labels, cluster_scores, sil_score


def plot_clustering_results(data, labels, embed_reductions, output_dir, dpi=300, tag=None, scores=None):
    plt.figure(figsize=(15, 10))

    unique_labels = np.sort(np.unique(labels))
    colors = sns.color_palette('husl', n_colors=len(unique_labels))

    # Create scatter plot with legend
    for l in unique_labels:
        mask = l == labels
        avg_silh = np.mean(scores[mask])
        plt.scatter(
            embed_reductions[mask, 0],
            embed_reductions[mask, 1],
            color=colors[l - 1],
            alpha=0.8,
            label=f'c {l}, avg {avg_silh:.2f}',
            )
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Clusters')
    
    plt.title('Hierarchical Clustering Results')
    plt.xlabel('First component')
    plt.ylabel('Second component')
    
    if output_dir:
        plt.savefig(
            os.path.join(output_dir, f'clustering_results_{tag}.png' if tag is not None else 'clustering_results.png'),
            dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()

# Example usage:
"""
# Assuming your data is stored in a variable called 'data' and t-SNE coordinates in 'tsne_coords'
# and you want to save the plots in a directory called 'output'

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Method 1: Specify number of clusters
labels = perform_hierarchical_clustering(data, n_clusters=5, output_dir=output_dir)

# Method 2: Use distance threshold
# labels = perform_hierarchical_clustering(data, distance_threshold=5.0, output_dir=output_dir)

# Plot results using t-SNE coordinates
plot_clustering_results(data, labels, tsne_coords, output_dir=output_dir)
"""