#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:25:45 2024

@author: juanpablomayaarteaga
"""

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
import time

# Record start time
start_time = time.time()

# Define the output directory path
o_path = "/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N/images/Output_images/Phase/"
os.makedirs(o_path, exist_ok=True)


name="Phase_1"
image_path = f"/Users/juanpablomayaarteaga/Desktop/Spatial_distribution/N/images/{name}.png"
#image_path = "/Users/juanpablomayaarteaga/Desktop/NAC.tif"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gammaImg = gammaCorrection(gray, 3)
plt.imshow(gammaImg)

# Apply Otsu's thresholding
#ret, Otsu_image = cv2.threshold(gammaImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert the Otsu image
gammaImg = 255 - gammaImg

# Save the Otsu image
#cv2.imwrite(os.path.join(o_path, 'output_7.jpg'), Otsu_image)
cv2.imwrite(os.path.join(o_path, f'{name}.jpg'), gammaImg)


"""
# Convert the image to a binary array where 1 represents foreground (objects) and 0 represents background
binary_array = (Otsu_image > 0).astype(np.uint8)
"""


# Convert the image to a numpy array with values between 0 and 1
#continuous_array = (gammaImg / 255.0).astype(np.float32)
continuous_array = (gammaImg / 255.0).astype(np.float32)

# Round the numbers in the continuous array to five decimal places
continuous_array = np.round(continuous_array, decimals=1) #to make it less expensive, otherwise, it takes 7 decimals

# Get the height and width of the image
height, width = continuous_array.shape  # Adjusted to handle the extra channel dimension

# Print the shape of the continuous array
print("Shape of continuous array:", continuous_array.shape)

# Plot the continuous array
plt.figure(figsize=(8, 6))
plt.imshow(continuous_array, cmap='gray')  # No need to access a specific channel
plt.title('Continuous Image Representation')
plt.axis('off')
plt.show()




# Get the height and width of the image
height, width = continuous_array.shape

# Initialize an empty list to store the coordinates
coordinates = []

# Iterate through each pixel in the continuous array
for y in range(height):
    for x in range(width):
        
        if continuous_array[y, x] > 0:
            # Calculate the centroid coordinates
            centroid_x = x + 0.5  # Adding 0.5 to get the centroid of the pixel
            centroid_y = y + 0.5  # Adding 0.5 to get the centroid of the pixel
            # Append the coordinates to the list
            coordinates.append([centroid_x, centroid_y])

# Convert the list of coordinates to a NumPy array
coordinates_array = np.array(coordinates)

print("Shape of coordinates array:", coordinates_array.shape)


# Convert the list of coordinates to a NumPy array
coordinates_array = np.array(coordinates)

# Convert the centroid coordinates back to integer values
coordinates_array = np.round(coordinates_array).astype(int)

# Reverse the y-coordinates of the centroid coordinates
coordinates_array[:, 1] = height - coordinates_array[:, 1]


# Print the shape of the coordinates array
print("Shape of coordinates array:", coordinates_array.shape)


# Function to parcelate the plot by density levels
def parcelate_and_draw(coordinates, filename):
    # First clustering with very restricted parameters
    clusterer_1 = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=2)
    labels_1 = clusterer_1.fit_predict(coordinates)
    #800, 2000, 4000
    # Second clustering with less restricted parameters
    clusterer_2 = hdbscan.HDBSCAN(min_cluster_size=1000, min_samples=2)
    labels_2 = clusterer_2.fit_predict(coordinates)

    # Third clustering with parameters to cover the first and second clusters
    clusterer_3 = hdbscan.HDBSCAN(min_cluster_size=2000, min_samples=2)
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
                
                # Refine the contour approximation
                epsilon = 0.000001 * cv2.arcLength(cluster_points, True)  # Adjust epsilon as needed
                refined_points = cv2.approxPolyDP(cluster_points, epsilon, True)
                
                # Calculate convex hull based on the refined contour
                hull = cv2.convexHull(refined_points)
                
                # Plot convex hull boundary
                for i in range(len(hull) - 1):
                    plt.plot([hull[i][0][0], hull[i+1][0][0]], [hull[i][0][1], hull[i+1][0][1]], color=color, lw=2)
                # Close the hull polygon
                plt.plot([hull[-1][0][0], hull[0][0][0]], [hull[-1][0][1], hull[0][0][1]], color=color, lw=2)

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
parcelate_and_draw(np.vstack((coordinates_array)), f'{name}.png')






# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")








