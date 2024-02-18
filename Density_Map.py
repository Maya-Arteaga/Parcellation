import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from skimage.segmentation import clear_border
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import hdbscan
import os
from morpho import gammaCorrection, set_path

# Define the output directory path
o_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N/images/Output_images/"
os.makedirs(o_path, exist_ok=True)

#image_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N/images/Phase_3.png"
image_path = "/Users/juanpablomayaarteaga/Desktop/Hippocampus.tif"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gammaImg = gammaCorrection(gray, 3)
plt.imshow(gammaImg)

# Apply Otsu's thresholding
ret, Otsu_image = cv2.threshold(gammaImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert the Otsu image
#Otsu_image = 255 - Otsu_image

# Save the Otsu image
cv2.imwrite(os.path.join(o_path, 'output_7.jpg'), Otsu_image)


# Convert the image to a binary array where 1 represents foreground (objects) and 0 represents background
binary_array = (Otsu_image > 0).astype(np.uint8)

# Plot the binary array
plt.figure(figsize=(8, 6))
plt.imshow(binary_array, cmap='gray')
plt.title('Binary Image Representation')
plt.axis('off')
plt.show()




# Get the height and width of the image
height, width = binary_array.shape

# Initialize an empty list to store the coordinates
coordinates = []

# Iterate through each pixel in the binary array
for y in range(height):
    for x in range(width):
        # If the pixel value is 1 (foreground), add its centroid coordinates to the list
        if binary_array[y, x] == 1:
            # Calculate the centroid coordinates
            centroid_x = x + 0.5  # Adding 0.5 to get the centroid of the pixel
            centroid_y = y + 0.5  # Adding 0.5 to get the centroid of the pixel
            # Append the coordinates to the list
            coordinates.append([centroid_x, centroid_y])

# Convert the list of coordinates to a NumPy array
coordinates_array = np.array(coordinates)
# Reverse the y-coordinates of the centroid coordinates
coordinates_array[:, 1] = height - coordinates_array[:, 1]

# Print the shape of the inverted coordinates array


# Print the shape of the coordinates array
print("Shape of coordinates array:", coordinates_array.shape)


# Function to parcelate the plot by density levels
def parcelate_and_draw(coordinates, filename):
    # First clustering with very restricted parameters
    clusterer_1 = hdbscan.HDBSCAN(min_cluster_size=700, min_samples=2)
    labels_1 = clusterer_1.fit_predict(coordinates)
    #800, 2000, 4000
    # Second clustering with less restricted parameters
    clusterer_2 = hdbscan.HDBSCAN(min_cluster_size=2100, min_samples=2)
    labels_2 = clusterer_2.fit_predict(coordinates)

    # Third clustering with parameters to cover the first and second clusters
    clusterer_3 = hdbscan.HDBSCAN(min_cluster_size=4500, min_samples=2)
    labels_3 = clusterer_3.fit_predict(coordinates)

    # Set up figure and axis
    f, ax = plt.subplots(1, figsize=(9, 9))

    # Plot individual locations colored by cluster label for each clustering
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='gray', label='Noise', alpha=0.5)
    ax.scatter(coordinates[labels_3 != -1, 0], coordinates[labels_3 != -1, 1], c='yellow', label='Cluster 3', alpha=0.5)
    ax.scatter(coordinates[labels_2 != -1, 0], coordinates[labels_2 != -1, 1], c='purple', label='Cluster 2', alpha=0.5)
    ax.scatter(coordinates[labels_1 != -1, 0], coordinates[labels_1 != -1, 1], c='green', label='Cluster 1', alpha=0.5)


    # Plot convex hull polygons for each cluster label for each clustering
    for labels, color in zip([labels_1, labels_2, labels_3], ['red', 'orange', 'blue']):
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
    plt.savefig(os.path.join(o_path, filename))
    plt.show()

# Example usage:


# Example usage:
parcelate_and_draw(np.vstack((coordinates_array)), 'ZZZZ_Binary.png')











