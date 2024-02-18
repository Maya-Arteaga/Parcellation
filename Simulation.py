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
random_state = 24

# Set the rectangular map dimensions
x_min, x_max = 0, 10
y_min, y_max = 0, 7  # Adjust the height to make it a rectangle

# Generate random data points within the rectangular map
rng = np.random.default_rng()

# Generate random uniform distributed points within the rectangular map for Ripley's K function
z_uniform_ripley = rng.uniform(low=[0, 0], high=[10, 7], size=(20, 2))

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

# Merge the uniform spatial distribution pattern with the very clustered points
z_merged_ripley = np.concatenate([z_uniform_ripley, z_very_clustered_ripley])

# Create instances of RipleysKEstimator
Kest_uniform = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
Kest_clustered = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
Kest_very_clustered = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
Kest_merged = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)

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
plt.savefig(os.path.join(i_path, 'Ripley_Comparison.png'))

# Calculate Ripley's K function for merged points
k_merged = Kest_merged(data=z_merged_ripley, radii=r, mode='none')

# Print out the quantities
print("Quantities for Merged Points:")
print("Distances (r):", r)
print("K Function Values:", k_merged)
print()

# Apply HDBSCAN clustering to identify regions of higher density for merged points
clusterer_merged = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)

# Fit the clusters for merged points
clusterer_merged.fit(z_merged_ripley)

# Count the number of clusters found for merged points
num_clusters_merged = len(set(clusterer_merged.labels_)) - (1 if -1 in clusterer_merged.labels_ else 0)

# Visualize the clustering results for merged points
plt.figure(figsize=(18, 6))





# Plot: Merged points without HDBSCAN clustering
plt.scatter(z_merged_ripley[:, 0], z_merged_ripley[:, 1], alpha=0.5)
plt.title('Merged Points without HDBSCAN Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(i_path, 'Merged_Points.png'))
plt.show()

# Calculate Ripley's K function for merged points
k_merged = Kest_merged(data=z_merged_ripley, radii=r, mode='none')

# Plot Ripley's K function for merged points
plt.figure(figsize=(8, 6))
plt.plot(r, k_merged, color='blue', label='Merged Points')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function for Merged Points")
plt.legend()
plt.savefig(os.path.join(i_path, 'Ripley_Merged.png'))
plt.show()





# Plot: Merged points for HDBSCAN clustering
plt.scatter(z_merged_ripley[:, 0], z_merged_ripley[:, 1], c=clusterer_merged.labels_, cmap='viridis', alpha=0.5)
plt.title(f'Merged Points with HDBSCAN Clustering\nNumber of Clusters: {num_clusters_merged}')
plt.xlabel('X')
plt.ylabel('Y')

# Annotate points with cluster labels, skipping noise (-1) points
for i, label in enumerate(clusterer_merged.labels_):
    if label != -1:
        plt.annotate(label, (z_merged_ripley[i, 0], z_merged_ripley[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig(os.path.join(i_path, 'HDBSCAN_Merged.png'))
plt.show()







# Plot 4: Merged points for Ripley's K function
plt.subplot(3, 2, 6)
plt.plot(r, Kest_merged(data=z_merged_ripley, radii=r, mode='none'), color='red', label='K Function')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function (Merged Points)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Ripley_Merged_HDBSCAN.png'))






# Plot 1: Ripley's K function for all points (excluding noise)
plt.figure(figsize=(12, 8))
plt.plot(r, Kest_merged(data=z_merged_ripley[clusterer_merged.labels_ != -1], radii=r, mode='none'), color='blue', label='Global Ripley\'s K Function (No Noise)')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Global Ripley's K Function (Excluding Noise)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Global_Ripley_Noise_Excluded.png'))
plt.show()


# Plot 1: Ripley's K function for all points (including noise)
plt.figure(figsize=(12, 8))
plt.plot(r, Kest_merged(data=z_merged_ripley, radii=r, mode='none'), color='blue', label='Global Ripley\'s K Function (Including Noise)')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Global Ripley's K Function (Including Noise)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Global_Ripley_Including_Noise.png'))
plt.show()


# Plot: Comparison of Ripley's K function with and without noise
plt.figure(figsize=(12, 8))
plt.plot(r, Kest_merged(data=z_merged_ripley[clusterer_merged.labels_ != -1], radii=r, mode='none'), color='blue', label='Global Ripley\'s K Function (No Noise)')
plt.plot(r, Kest_merged(data=z_merged_ripley, radii=r, mode='none'), color='red', label='Global Ripley\'s K Function (Including Noise)')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Comparison of Global Ripley's K Function")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Global_Ripley_Comparison.png'))
plt.show()




# Plot 2: Ripley's K function for each cluster
# Plot 2: Ripley's K function for each cluster
plt.figure(figsize=(18, 12))
for cluster_label in np.unique(clusterer_merged.labels_):
    if cluster_label == -1:
        continue  # Skip noise points
    
    cluster_points = z_merged_ripley[clusterer_merged.labels_ == cluster_label]
    plt.plot(r, Kest_merged(data=cluster_points, radii=r, mode='none'), label=f'Cluster {cluster_label}')

plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function for Each Cluster (Merged Points)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Ripley_Clusterwise_Merged.png'))
plt.show()




# Set threshold for minimum K function value
min_k_threshold = 0.9

# Plot the Ripley's K function for each cluster in the merged points
plt.figure(figsize=(18, 12))

for cluster_label in np.unique(clusterer_merged.labels_):
    if cluster_label == -1:
        continue  # Skip noise points
    
    cluster_points = z_merged_ripley[clusterer_merged.labels_ == cluster_label]
    k_function_values = Kest_merged(data=cluster_points, radii=r, mode='none')
    
    # Check if the maximum K function value is above the threshold
    if np.max(k_function_values) < min_k_threshold:
        continue  # Skip this cluster if not clustered
    
    # Manually add cluster label, first, and last function value
    label_str = f'Cluster {cluster_label} ({k_function_values[0]:.2f} - {k_function_values[-1]:.2f})'
    plt.plot(r, k_function_values, label=label_str)

plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function for Each Cluster (Merged Points)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Ripley_Clusterwise_Merged_Filtered.png'))








import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory path
i_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N"

# Create the directory if it doesn't exist
os.makedirs(i_path, exist_ok=True)

# Set random state for reproducibility
random_state = 24

# Set the rectangular map dimensions
x_min, x_max = 0, 12
y_min, y_max = 0, 8  # Adjust the height to make it a rectangle

# Number of points for each distribution
n_points_uniform = 100
n_points_clustered = 150
n_points_very_clustered = 200

# Generate random uniform distributed points within the rectangular map
z_uniform = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_points_uniform, 2))

# Generate clustered points within the rectangular map
z_clustered = np.concatenate([
    np.random.uniform(low=[0, 0], high=[3, 3], size=(n_points_clustered, 2)),
    np.random.uniform(low=[8, 5], high=[12, 8], size=(n_points_clustered, 2))
])

# Generate very clustered points within the rectangular map
z_very_clustered = np.concatenate([
    np.random.uniform(low=[0, 0], high=[1, 1], size=(n_points_very_clustered, 2)),
    np.random.uniform(low=[10, 7], high=[12, 8], size=(n_points_very_clustered, 2))
])


# Merge the points
z_merged = np.vstack((z_uniform))

# Plot the merged points
plt.figure(figsize=(10, 8))
plt.scatter(z_merged[:, 0], z_merged[:, 1], alpha=0.5)
plt.title('')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Phase_1.png'))
plt.show()



# Merge the points
z_merged = np.vstack((z_uniform, z_clustered))

# Plot the merged points
plt.figure(figsize=(10, 8))
plt.scatter(z_merged[:, 0], z_merged[:, 1], alpha=0.5)
plt.title('')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Phase_2.png'))
plt.show()


# Merge the points
z_merged = np.vstack((z_uniform, z_clustered, z_very_clustered))

# Plot the merged points
plt.figure(figsize=(10, 8))
plt.scatter(z_merged[:, 0], z_merged[:, 1], alpha=0.5)
plt.title('')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Phase_3.png'))
plt.show()






import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os



# Define a function to create a choropleth map
def plot_choropleth(z, filename):
    # Create a 2D density estimate
    kde = gaussian_kde(z.T)
    xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

    # Plot the density estimate as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(zi.reshape(xi.shape), origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='coolwarm')
    plt.colorbar(label='Density')
    plt.title('Choropleth Mapping')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(i_path, filename))
    plt.show()

# Plot choropleth mapping for each phase
plot_choropleth(z_uniform, 'Phase_1.png')
plot_choropleth(np.vstack((z_uniform, z_clustered)), 'Phase_2.png')
plot_choropleth(np.vstack((z_uniform, z_clustered, z_very_clustered)), 'Phase_3.png')
