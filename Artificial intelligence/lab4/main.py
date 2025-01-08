# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hdbscan
import os

# Loading and Exploring the Dataset
def load_images_from_subfolders(folder, total_images_to_process=100):
    images = []
    total_count = 0  # Counter for total images processed

    for subfolder, _, files in os.walk(folder):  # Traverse subfolders
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):  
                img_path = os.path.join(subfolder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    total_count += 1
                    if total_count >= total_images_to_process:  
                        return images 

    return images  # Return all loaded images if the total limit is not reached

# Set the path to your dataset
folder_path = 'imagedataset'  
total_images = 5749  # Total number of images you want to process
images = load_images_from_subfolders(folder_path, total_images)
print(f"Loaded {len(images)} images.")

# Data Preprocessing
def preprocess_images(images):
    flattened_images = [cv2.resize(img, (100, 100)).flatten() for img in images]  # Resize and flatten
    return np.array(flattened_images)

data = preprocess_images(images)
print(f"Data shape after preprocessing: {data.shape}")

# Standardizing the dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Dimensionality Reduction
pca = PCA(n_components=50)  # number of components
data_reduced = pca.fit_transform(data_scaled)

# Applying HDBSCAN Clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)  # cluster size
cluster_labels = clusterer.fit_predict(data_reduced)

# clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='cosine')  # cosine distance metric
# cluster_labels_cosine = clusterer.fit_predict(reduced_images)  # Predicting cluster labels with new metric

# Displaying cluster laabels
print("Cluster labels assigned to each data point:", cluster_labels)

# Visualize Clustering Results
plt.figure(figsize=(10, 8))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=cluster_labels, cmap='rainbow', s=10)
plt.title('HDBSCAN Clustering of Facial Images')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.show()

# Visualize and Analyze Results
# Identify noise points
noise_points = np.where(cluster_labels == -1)[0]
print(f"Number of noise points: {len(noise_points)}")

# Analyze distribution of noise points
plt.figure(figsize=(10, 5))
plt.hist(cluster_labels[noise_points], bins=np.arange(-1, max(set(cluster_labels)) + 2) - 0.5, edgecolor='black')
plt.title('Distribution of Noise Points')
plt.xlabel('Cluster Label')
plt.ylabel('Frequency')
plt.xticks(np.arange(-1, max(set(cluster_labels)) + 1))
plt.show()

# Evaluating clustering quality using silhouette score
if len(set(cluster_labels)) > 1:  # At least 2 clusters required for silhouette score
    silhouette_avg = silhouette_score(data_reduced, cluster_labels)
    print(f'Silhouette Score: {silhouette_avg:.3f}')
else:
    print('Silhouette Score cannot be calculated with less than 2 clusters.')

# Real-Life Applications
# Test on New Data
def assign_new_image_to_cluster(new_image, clusterer, scaler, pca):
    new_image_resized = cv2.resize(new_image, (100, 100)).flatten().reshape(1, -1)
    new_image_scaled = scaler.transform(new_image_resized)
    new_image_reduced = pca.transform(new_image_scaled)
    cluster_label = clusterer.predict(new_image_reduced)
    return cluster_label

# Representative Images for Clusters
def identify_representative_images(cluster_labels, images, data_reduced):
    unique_labels = set(cluster_labels)
    representative_images = []
    for label in unique_labels:
        if label != -1:  # Exclude noise
            indices = np.where(cluster_labels == label)[0]
            # Calculate the centroid of the cluster
            cluster_center = np.mean(data_reduced[indices], axis=0)
            # Find the image closest to the cluster center
            closest_index = indices[np.argmin(np.linalg.norm(data_reduced[indices] - cluster_center, axis=1))]
            representative_images.append(images[closest_index])
    return representative_images

representative_images = identify_representative_images(cluster_labels, images, data_reduced)

for idx, rep_img in enumerate(representative_images):
    plt.subplot(1, len(representative_images), idx + 1)
    plt.imshow(cv2.cvtColor(rep_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.show()