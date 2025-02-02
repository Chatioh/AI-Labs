Flask-Based Image Clustering API Using Pre-Trained Model

📌 Overview
This project demonstrates how to deploy a Flask-based REST API for image clustering using a pre-trained machine learning model. The system is designed to predict cluster labels for uploaded images and is optimized for lightweight deployment using Docker.

✨ Key Features
🔍 Image Preprocessing

Convert uploaded images to RGB format for consistency.
Resize images to match the input shape required by the pre-trained model.
🧠 Pre-Trained Model Integration

Load a clustering pipeline (clustering_pipeline.pkl) for predictions.
Process normalized image data for accurate clustering results.
🌐 API-Driven Architecture

Exposes a /predict endpoint for easy integration with external systems.
Accepts image uploads via HTTP POST requests.
⚙️ Lightweight Deployment

Fully containerized with a Docker setup for fast and reliable deployment.
Minimal dependencies to ensure a lean, optimized runtime.
📋 Results
Efficient Image Clustering: Provides accurate cluster labels for input images.
Fast and Scalable API: Handles multiple requests with logging and performance tracking.
Robust Integration: Works seamlessly with the pre-trained model.
📦 Dependencies
Python Libraries:

Flask
Pillow
scikit-learn
numpy
Docker Image:

Python 3.11-slim
🚀 Applications
📈 Clustering Insights:
Analyze uploaded images and assign them to specific clusters based on visual similarities.

📋 Real-World Use Cases:

Organizing image collections.
Identifying patterns or groups within datasets.
Supporting downstream tasks like recommendation systems.
✅ Conclusion
This project highlights the simplicity and power of deploying a pre-trained machine learning model via a lightweight Flask API. It offers robust performance for image clustering tasks while being easy to deploy using Docker.

For more information and setup instructions, refer to the project repository or the provided documentation.