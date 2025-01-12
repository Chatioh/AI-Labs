# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Step 1: Data Preprocessing
# Load the dataset
data = pd.read_csv('synthetic_traffic_data.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Time-Series Data Handling
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')

# Extract Time-Based Features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

# Step 2: Feature Engineering
# Time-Based Features
data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)

# Lag Features
for lag in range(1, 25):
    data[f'lag_{lag}'] = data['traffic_flow'].shift(lag)
data.dropna(inplace=True)

# Normalization/Standardization
scaler = MinMaxScaler()
feature_columns = ['hour', 'is_weekend', 'is_rush_hour'] + [f'lag_{i}' for i in range(1, 25)]
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Step 3: Data Splitting
X = data[feature_columns]
y = data['traffic_flow']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 4: Model Building Function
def create_model():
    # Neural Network Architecture
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    # Model Compilation
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Wrap the model using KerasRegressor
model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

# Step 5: Model Training and Evaluation
# Train the Model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val))

# Evaluate Model Performance
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
print(f'RMSE: {rmse}')

# Step 6: Prediction for Future Traffic
# Create future feature values with correct dimensions
future_features = np.array([[14, 1, 0] + [0] * 24])  # Example: hour=14, weekend=1, rush_hour=0, followed by 24 lag features

# Ensure the features are scaled the same way as training data
future_features_scaled = scaler.transform(future_features)

# Make the prediction
future_prediction = model.predict(future_features_scaled)
print(f'Future Traffic Prediction: {future_prediction}')

# Step 7: Model Tuning and Optimization
# Example hyperparameter grid for tuning (optional)
param_grid = {
    'batch_size': [16, 32],
    'epochs': [50, 100]
}

# Grid search implementation would go here (optional)

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Scores: {scores}')

# Step 8: Visualization
# Visualize Traffic Predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Traffic Flow')
plt.plot(y_pred, label='Predicted Traffic Flow')
plt.title('Traffic Flow: Actual vs Predicted')
plt.xlabel('Time Index')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()

# Weather vs. Traffic Flow Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['temperature'], data['traffic_flow'], alpha=0.5)
plt.title('Temperature vs. Traffic Flow')
plt.xlabel('Temperature')
plt.ylabel('Traffic Flow')
plt.show()