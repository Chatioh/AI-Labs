from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import pickle
import os
import logging
import time

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()          # Log to console
    ]
)

# Load the trained model (update the path to your .pkl file)
MODEL_PATH = 'clustering_pipeline.pkl'

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    logging.info("Trained model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    model = None  # Set to None if loading fails


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict the cluster of an uploaded image.
    """
    start_time = time.time()
    try:
        logging.info("Incoming prediction request")

        if model is None:
            return jsonify({'error': 'Model not loaded. Cannot process request.'}), 500

        # Check if an image file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Load and preprocess the image
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')  # Ensure image is in RGB format

        # Resize the image to match the model input shape
        # (Update this based on your trained model's expected input dimensions)
        expected_shape = (224, 224)  # Example shape (Height, Width)
        image = image.resize((expected_shape[1], expected_shape[0]))  # Resize to (Width, Height)

        # Convert the image to a numpy array and normalize if necessary
        image_array = np.array(image).astype('float32') / 255.0  # Normalize pixel values if needed

        # Add a batch dimension if the model requires it
        image_array = np.expand_dims(image_array, axis=0)

        logging.info("Starting prediction")
        # Predict the cluster using the loaded model
        prediction = model.predict(image_array)

        logging.info(f"Prediction complete: {prediction}")

        response_time = time.time() - start_time
        logging.info(f"Prediction took {response_time:.2f} seconds")

        # Format the response to ensure JSON serializability
        return jsonify({
            'prediction': prediction.tolist(),  # Convert numpy array to list if needed
            'response_time': f"{response_time:.2f} seconds"
        })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
