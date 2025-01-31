Advanced Facial Recognition Clustering Using HDBSCAN


📌 Overview

This project leverages HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to group facial images based on their visual features. The approach effectively handles noise, variable densities, and real-world challenges in clustering tasks.

✨ Key Features

1. 🔍 Data Exploration

Dataset inspection for structure, image dimensions, and visualization.

2. 🛠️ Preprocessing

Flatten Images: Convert 2D images into 1D arrays for clustering.
Standardization: Normalize pixel values to ensure uniform scaling.

3. 📉 Dimensionality Reduction

Use PCA to retain essential variance and optimize clustering performance.

4. 🧠 HDBSCAN Clustering

Group similar facial images while identifying noise and outliers.
Experiment with parameters like min_cluster_size and distance metrics.

5. ⚙️ Real-World Challenges

Simulate noisy data to test algorithm robustness.
Evaluate different distance metrics (e.g., Manhattan, Cosine).

6. 📈 Results Visualization

Scatter plots of clusters.
Analyze noise points and cluster representatives.

7. 📊 Evaluation

Assess clustering quality with metrics like silhouette scores.
Test on new, unseen images for generalization.

📋 Results

Clear, meaningful clusters formed from facial image datasets.
Identification of noise points and representative cluster images.

📦 Dependencies

Python Libraries:
pandas, numpy, matplotlib, sklearn, os, hdbscan

🚀 Applications

1. Cluster Assignment: Generalize clustering on unseen images.

2. Representative Images: Find key images for each cluster.

✅ Conclusion

This project demonstrates the power of HDBSCAN in clustering high-dimensional facial image data. It successfully addresses real-world challenges like noise and variable densities while providing actionable insights into the dataset's structure.

For the full implementation, visit the GitHub Repository.
