# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install the necessary Python dependencies
RUN pip install --no-cache-dir flask pillow scikit-learn hdbscan

# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "main.py"]