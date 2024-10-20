import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

def dunn_index(umap_coordinates, labels):
    # Calculate pairwise distances between all samples
    distances = squareform(pdist(umap_coordinates))
    # print("distance calculated")
    # Initialize variables to track the smallest inter-cluster and largest intra-cluster distances
    smallest_inter_cluster_distance = np.inf
    largest_intra_cluster_distance = 0
    
    # Identify unique clusters
    clusters = np.unique(labels)
    
    for cluster in clusters:
        # Indices of samples in the current cluster
        in_cluster = labels == cluster
        # Indices of samples not in the current cluster
        not_in_cluster = ~in_cluster
        
        # Intra-cluster distances for the current cluster
        intra_cluster_distances = distances[in_cluster][:, in_cluster]
        # Update the largest intra-cluster distance if necessary
        largest_intra_cluster = np.max(intra_cluster_distances)
        largest_intra_cluster_distance = max(largest_intra_cluster, largest_intra_cluster_distance)
        
        # Inter-cluster distances between current cluster and other clusters
        inter_cluster_distances = distances[in_cluster][:, not_in_cluster]
        # Update the smallest inter-cluster distance if necessary
        smallest_inter_cluster = np.min(inter_cluster_distances)
        smallest_inter_cluster_distance = min(smallest_inter_cluster, smallest_inter_cluster_distance)
    
    # Calculate Dunn's Index
    dunn_index = smallest_inter_cluster_distance / largest_intra_cluster_distance
    return dunn_index

def calculate_silhouette_score(X, labels, new_point, new_point_label):
    """
    Calculate the silhouette score for a new point given dataset X, binary labels,
    and the new point's label, in a memory-efficient manner.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the dataset coordinates.
    - labels: numpy array of shape (n_samples,), the binary labels for the dataset.
    - new_point: numpy array of shape (n_features,), the new point coordinates.
    - new_point_label: int, the cluster label of the new point.

    Returns:
    - silhouette_score: float, the silhouette score of the new point.
    """

    # Calculate distances from the new point to all other points
    distances_to_new_point = pairwise_distances(X, new_point.reshape(1, -1), metric='euclidean').flatten()

    # Intra-cluster distances
    same_cluster_indices = labels == new_point_label
    if np.any(same_cluster_indices):
        a = np.mean(distances_to_new_point[same_cluster_indices])
    else:
        a = 0.0  # No other points in the same cluster

    # Nearest-cluster distance
    different_cluster_indices = labels != new_point_label
    if np.any(different_cluster_indices):
        b = np.mean(distances_to_new_point[different_cluster_indices])
    else:
        b = a  # No points in a different cluster, unusual but we handle it

    # Silhouette score for the new point
    silhouette_score = (b - a) / max(a, b) if max(a, b) > 0 else 0

    return silhouette_score