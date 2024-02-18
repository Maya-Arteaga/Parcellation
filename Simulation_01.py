import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import RipleysKEstimator
import os

# Define the directory path
i_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/"

# Create the directory if it doesn't exist
os.makedirs(i_path, exist_ok=True)


# Set random state for reproducibility
random_state = 22

# Set the rectangular map dimensions
x_min, x_max = 0, 10
y_min, y_max = 0, 7  # Adjust the height to make it a rectangle

# Generate random data points within the rectangular map
rng = np.random.default_rng()

# Generate random uniform distributed points within the rectangular map for Ripley's K function
z_uniform_ripley = rng.uniform(low=[0, 0], high=[10, 7], size=(100, 2))

# Generate clustered points within the rectangular map, concentrated at the corners for Ripley's K function
z_clustered_ripley = np.concatenate([
    np.concatenate([
        rng.uniform(low=[0, 0], high=[2, 2], size=(20, 2)),  # Decrease size and increase range
        rng.uniform(low=[8, 5], high=[10, 7], size=(30, 2))  # Decrease size and increase range
    ]),
    np.concatenate([
        rng.uniform(low=[8, 0], high=[10, 2], size=(30, 2)),  # Decrease size and increase range
        rng.uniform(low=[0, 5], high=[2, 7], size=(20, 2))  # Decrease size and increase range
    ])
])

# Generate very clustered points within the rectangular map, concentrated at the corners for Ripley's K function

z_very_clustered_ripley = np.concatenate([
    np.concatenate([
        rng.uniform(low=[0, 0], high=[1, 1], size=(40, 2)),  # Increase size and decrease range
        rng.uniform(low=[9, 5], high=[10, 7], size=(40, 2))  # Increase size and decrease range
    ]),
    np.concatenate([
        rng.uniform(low=[9, 0], high=[10, 1], size=(40, 2)),  # Increase size and decrease range
        rng.uniform(low=[0, 5], high=[1, 7], size=(40, 2))  # Increase size and decrease range
    ])
])

# Create instances of RipleysKEstimator
Kest_uniform = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
Kest_clustered = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
Kest_very_clustered = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)

# Define distances (radii) at which to evaluate Ripley's K function
r = np.linspace(0, 5, 100)

# Plot the random data points and Ripley's K function graphs
plt.figure(figsize=(18, 12))

# Plot 1: Random uniform distributed points for Ripley's K function
plt.subplot(3, 2, 1)
plt.scatter(z_uniform_ripley[:, 0], z_uniform_ripley[:, 1], color='blue', label='Uniform Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Uniform Distributed Points (Ripley)')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(r, Kest_uniform(data=z_uniform_ripley, radii=r, mode='none'), color='red', label='K Function')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function (Uniform Points)")
plt.legend()
plt.savefig(os.path.join(i_path, 'Ripley_Uniform.png'))


# Plot 2: Clustered points for Ripley's K function
plt.subplot(3, 2, 3)
plt.scatter(z_clustered_ripley[:, 0], z_clustered_ripley[:, 1], color='green', label='Clustered Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustered Points (Ripley)')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(r, Kest_clustered(data=z_clustered_ripley, radii=r, mode='none'), color='red', label='K Function')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function (Clustered Points)")
plt.legend()
plt.savefig(os.path.join(i_path, 'Ripley_Clustered.png'))

# Plot 3: Very clustered points for Ripley's K function
plt.subplot(3, 2, 5)
plt.scatter(z_very_clustered_ripley[:, 0], z_very_clustered_ripley[:, 1], color='orange', label='Very Clustered Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Very Clustered Points (Ripley)')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(r, Kest_very_clustered(data=z_very_clustered_ripley, radii=r, mode='none'), color='red', label='K Function')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function (Very Clustered Points)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Ripley_Very_Clustered.png'))


# Comparison plot
plt.figure(figsize=(8, 6))
plt.plot(r, Kest_uniform(data=z_uniform_ripley, radii=r, mode='none'), color='blue', label='Uniform Points')
plt.plot(r, Kest_clustered(data=z_clustered_ripley, radii=r, mode='none'), color='green', label='Clustered Points')
plt.plot(r, Kest_very_clustered(data=z_very_clustered_ripley, radii=r, mode='none'), color='orange', label='Very Clustered Points')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function Comparison")
plt.legend()
plt.savefig(os.path.join(i_path, 'Ripley_Comparation.png'))

# Calculate Ripley's K function for each distribution
k_uniform = Kest_uniform(data=z_uniform_ripley, radii=r, mode='none')
k_clustered = Kest_clustered(data=z_clustered_ripley, radii=r, mode='none')
k_very_clustered = Kest_very_clustered(data=z_very_clustered_ripley, radii=r, mode='none')

# Print out the quantities
print("Quantities for Uniformly Distributed Points:")
print("Distances (r):", r)
print("K Function Values:", k_uniform)
print()

print("Quantities for Clustered Points:")
print("Distances (r):", r)
print("K Function Values:", k_clustered)
print()

print("Quantities for Very Clustered Points:")
print("Distances (r):", r)
print("K Function Values:", k_very_clustered)


# Apply HDBSCAN clustering to identify regions of higher density for each case
clusterer_uniform = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
clusterer_clustered = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
clusterer_very_clustered = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)

# Fit the clusters for each case
clusterer_uniform.fit(z_uniform_ripley)
clusterer_clustered.fit(z_clustered_ripley)
clusterer_very_clustered.fit(z_very_clustered_ripley)

# Count the number of clusters found for each case
num_clusters_uniform = len(set(clusterer_uniform.labels_)) - (1 if -1 in clusterer_uniform.labels_ else 0)
num_clusters_clustered = len(set(clusterer_clustered.labels_)) - (1 if -1 in clusterer_clustered.labels_ else 0)
num_clusters_very_clustered = len(set(clusterer_very_clustered.labels_)) - (1 if -1 in clusterer_very_clustered.labels_ else 0)

# Visualize the clustering results for each case
plt.figure(figsize=(18, 18))

# Plot 1: Random uniform distributed points for HDBSCAN clustering
plt.subplot(3, 2, 1)
plt.scatter(z_uniform_ripley[:, 0], z_uniform_ripley[:, 1], c=clusterer_uniform.labels_, cmap='viridis', alpha=0.5)
plt.title(f'Uniformly Distributed Points with HDBSCAN Clustering\nNumber of Clusters: {num_clusters_uniform}')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(i_path, 'HDBSCAN_Uniform.png'))

# Plot 2: Clustered points for HDBSCAN clustering
plt.subplot(3, 2, 3)
plt.scatter(z_clustered_ripley[:, 0], z_clustered_ripley[:, 1], c=clusterer_clustered.labels_, cmap='viridis', alpha=0.5)
plt.title(f'Clustered Points with HDBSCAN Clustering\nNumber of Clusters: {num_clusters_clustered}')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(i_path, 'HDBSCAN_Clustered.png'))


# Plot 3: Very clustered points for HDBSCAN clustering
plt.subplot(3, 2, 5)
plt.scatter(z_very_clustered_ripley[:, 0], z_very_clustered_ripley[:, 1], c=clusterer_very_clustered.labels_, cmap='viridis', alpha=0.5)
plt.title(f'Very Clustered Points with HDBSCAN Clustering\nNumber of Clusters: {num_clusters_very_clustered}')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'HDBSCAN_Very_clustered.png'))

plt.show()
