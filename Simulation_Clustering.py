import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import RipleysKEstimator
import os


"""
# Define the directory path
i_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N"

# Create the directory if it doesn't exist
os.makedirs(i_path, exist_ok=True)

# Set random state for reproducibility
random_state = 24

# Set the rectangular map dimensions
x_min, x_max = 0, 10
y_min, y_max = 0, 7  # Adjust the height to make it a rectangle

# Function to generate and save plots
def generate_and_save_plot(points, filename, color):
    plt.figure(figsize=(12, 12))  # Set the figure size to 1200x1200 pixels
    plt.scatter(points[:, 0], points[:, 1], s=20, c=color, alpha=0.7)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(os.path.join(i_path, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

# Generate random uniform distributed points within the rectangular map
z_uniform = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(100, 2))
generate_and_save_plot(z_uniform, "uniform.png", 'blue')

# Generate clustered points within the rectangular map
z_clustered = np.concatenate([
    np.random.uniform(low=[0, 0], high=[4, 4], size=(150, 2)),  # Decrease size and decrease range
    np.random.uniform(low=[6, 3], high=[10, 7], size=(250, 2))  # Decrease size and increase range
])
generate_and_save_plot(z_clustered, "clustered.png", 'purple')

# Generate very clustered points within the rectangular map
z_very_clustered = np.concatenate([
    np.random.uniform(low=[0, 0], high=[2, 2], size=(200, 2)),  # Increase size and decrease range
    np.random.uniform(low=[8, 5], high=[10, 7], size=(400, 2))  # Increase size and increase range
])
generate_and_save_plot(z_very_clustered, "very_clustered.png", 'red')


"""




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
plot_choropleth(np.vstack((z_uniform)), 'Phase_1_choroplet.png')
plot_choropleth(np.vstack((z_uniform, z_clustered)), 'Phase_2_choroplet.png')
plot_choropleth(np.vstack((z_uniform, z_clustered, z_very_clustered)), 'Phase_3_choroplet.png')






import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.stats import RipleysKEstimator

# Define the directory path
i_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N"

# Create the directory if it doesn't exist
os.makedirs(i_path, exist_ok=True)

# Define the rectangular map dimensions
x_min, x_max = 0, 12
y_min, y_max = 0, 8

# Generate synthetic data
n_points_uniform = 100
n_points_clustered = 150
n_points_very_clustered = 200

# Generate random uniform distributed points within the rectangular map
x_uniform = np.random.uniform(low=x_min, high=x_max, size=n_points_uniform)
y_uniform = np.random.uniform(low=y_min, high=y_max, size=n_points_uniform)

# Generate clustered points within the rectangular map
x_clustered = np.concatenate([
    np.random.uniform(low=x_min, high=3, size=(n_points_clustered // 2)),
    np.random.uniform(low=8, high=x_max, size=(n_points_clustered // 2))
])
y_clustered = np.concatenate([
    np.random.uniform(low=y_min, high=3, size=(n_points_clustered // 2)),
    np.random.uniform(low=5, high=y_max, size=(n_points_clustered // 2))
])

# Generate very clustered points within the rectangular map
x_very_clustered = np.concatenate([
    np.random.uniform(low=x_min, high=1, size=(n_points_very_clustered // 2)),
    np.random.uniform(low=10, high=x_max, size=(n_points_very_clustered // 2))
])
y_very_clustered = np.concatenate([
    np.random.uniform(low=y_min, high=1, size=(n_points_very_clustered // 2)),
    np.random.uniform(low=7, high=y_max, size=(n_points_very_clustered // 2))
])

# Concatenate the DataFrames
z_uniform = np.column_stack((x_uniform, y_uniform))
z_clustered = np.column_stack((x_clustered, y_clustered))
z_very_clustered = np.column_stack((x_very_clustered, y_very_clustered))

# Define the range of distances for Ripley's K function
r = np.linspace(0, 10, 100)

# Create instances of RipleysKEstimator
Kest_uniform = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
Kest_clustered = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
Kest_very_clustered = RipleysKEstimator(area=(x_max - x_min) * (y_max - y_min), x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)

# Calculate Ripley's K function for each spatial distribution
K_uniform = Kest_uniform(z_uniform, r, mode='ripley')
K_clustered = Kest_clustered(z_clustered, r, mode='ripley')
K_very_clustered = Kest_very_clustered(z_very_clustered, r, mode='ripley')

# Plot Ripley's K function for each spatial distribution
plt.figure(figsize=(12, 8))
plt.plot(r, K_uniform, color='green', label='Ripley\'s K Function (Uniform)')
plt.plot(r, K_clustered, color='orange', label='Ripley\'s K Function (Clustered)')
plt.plot(r, K_very_clustered, color='red', label='Ripley\'s K Function (Very Clustered)')
plt.xlabel('Distance')
plt.ylabel('K Function')
plt.title("Ripley's K Function")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(i_path, 'Ripleys_K_Function.png'))
plt.show()



import pandas as pd

# Create DataFrames for each phase
df_uniform = pd.DataFrame({'Distances (r)': r, 'K Function Values': K_uniform})
df_clustered = pd.DataFrame({'Distances (r)': r, 'K Function Values': K_clustered})
df_very_clustered = pd.DataFrame({'Distances (r)': r, 'K Function Values': K_very_clustered})

# Print out the quantities for each phase
print("Quantities for Uniform Phase:")
print(df_uniform)
print()

print("Quantities for Clustered Phase:")
print(df_clustered)
print()

print("Quantities for Very Clustered Phase:")
print(df_very_clustered)
print()


# Merge the DataFrames for each phase
merged_df = pd.concat([df_uniform, df_clustered, df_very_clustered], keys=['Uniform', 'Clustered', 'Very Clustered'], axis=0)

# Reset the index
merged_df.reset_index(level=0, inplace=True)
merged_df.rename(columns={'level_0': 'Phase'}, inplace=True)

# Print the merged DataFrame
print("Merged DataFrame:")
print(merged_df)


# Transpose the merged DataFrame
transposed_df = merged_df.pivot(index='Distances (r)', columns='Phase', values='K Function Values')

# Print the transposed DataFrame
print("Transposed DataFrame:")
print(transposed_df)










import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import hdbscan
import os

# Define the directory path
i_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N"

# Create the directory if it doesn't exist
os.makedirs(i_path, exist_ok=True)

# Function to parcelate the plot by density levels
def parcelate_and_draw(coordinates, filename):
    # First clustering with very restricted parameters
    clusterer_1 = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    labels_1 = clusterer_1.fit_predict(coordinates)

    # Second clustering with less restricted parameters
    clusterer_2 = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=2)
    labels_2 = clusterer_2.fit_predict(coordinates)

    # Third clustering with parameters to cover the first and second clusters
    clusterer_3 = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=2)
    labels_3 = clusterer_3.fit_predict(coordinates)

    # Set up figure and axis
    f, ax = plt.subplots(1, figsize=(9, 9))

    # Plot individual locations colored by cluster label for each clustering
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='gray', label='Noise', alpha=0.5)
    ax.scatter(coordinates[labels_1 != -1, 0], coordinates[labels_1 != -1, 1], c='blue', label='Cluster 1', alpha=0.5)
    ax.scatter(coordinates[labels_2 != -1, 0], coordinates[labels_2 != -1, 1], c='orange', label='Cluster 2', alpha=0.5)
    ax.scatter(coordinates[labels_3 != -1, 0], coordinates[labels_3 != -1, 1], c='green', label='Cluster 3', alpha=0.5)

    # Plot convex hull polygons for each cluster label for each clustering
    for labels, color in zip([labels_1, labels_2, labels_3], ['red', 'orange', 'purple']):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label != -1:  # Exclude noise points
                # Filter points belonging to the current cluster
                cluster_points = coordinates[labels == label]
                # Calculate convex hull
                hull = ConvexHull(cluster_points)
                # Plot convex hull boundary with dashed lines
                for simplex in hull.simplices:
                    plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], color=color, linestyle='--', lw=2)

    # Add legend
    ax.legend(loc='upper right')

    # Set title
    plt.title('')
    plt.tight_layout()

    # Remove axes
    ax.set_axis_off()

    # Save the plot
    plt.savefig(os.path.join(i_path, filename))
    plt.show()

# Example usage:


# Example usage:
parcelate_and_draw(np.vstack((z_uniform, z_clustered, z_very_clustered)), 'Clustered_Phase_3_parcelated.png')


parcelate_and_draw(np.vstack((z_uniform, z_clustered)), 'Clustered_Phase_2_parcelated.png')


parcelate_and_draw(np.vstack((z_uniform)), 'Clustered_Phase_1_parcelated.png')


parcelate_and_draw((z_very_clustered), 'AAA_Clustered_Phase_4_parcelated.png')
