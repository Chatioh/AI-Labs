# Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV

# Loading and Exploring the Dataset
data = pd.read_csv('synthetic_traffic_data.csv')
print(data.head())
print(data.info())
print(data.describe())

# Checking for missing data
print(data.isnull().sum())

# Time-Series Data Handling
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')

# Extracting time-based features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

# Feature Engineering
# Time-based Features
data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)

# Lag Features
for lag in range(1, 25):  # Last 24 hours
    data[f'lag_{lag}'] = data['traffic_flow'].shift(lag)

data.dropna(inplace=True)  # Drop rows with NaN values from lag features

# Normalization/Standardization
scaler = MinMaxScaler()
feature_columns = ['hour', 'is_weekend', 'is_rush_hour', 'temperature', 'humidity'] + [f'lag_{i}' for i in range(1, 25)]
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Data Splitting
X = data[feature_columns]
y = data['traffic_flow']  # Target variable

# Train-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # Split into training and temporary set
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Building
# Neural Network Architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Regression output

# Model Compilation
model.compile(loss='mean_squared_error', optimizer='adam')

# Model Training and Evaluation
# Training the Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# Evaluation
y_pred = model.predict(X_test)

# Calculating RMSE
rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
print(f'RMSE: {rmse}')

# Prediction
# Training the Model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)  

# Prediction
y_future_pred = model.predict(X_test)  

# #step 6
# # function to create the model (required for GridSearchCV)
# def create_model(learning_rate=0.01):
#     model = Sequential()
#     model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1))  # Regression output
#     model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
#     return model

# # parameter grid
# param_grid = {
#     'learning_rate': [0.001, 0.01, 0.1],
#     'batch_size': [16, 32, 64],
#     'epochs': [50, 100]
# }

# # Using KerasClassifier for GridSearchCV
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# model = KerasRegressor(build_fn=create_model, verbose=0)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)  # 3-fold cross-validation
# grid_result = grid.fit(X_train, y_train)

# # Print best parameters
# print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')

# Visualization
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

# Scatter plot for Temperature vs. Traffic Flow
plt.subplot(1, 2, 1)
plt.scatter(data['temperature'], data['traffic_flow'], alpha=0.5)
plt.title('Temperature vs. Traffic Flow')
plt.xlabel('Temperature')
plt.ylabel('Traffic Flow')

# Scatter plot for Humidity vs. Traffic Flow
plt.subplot(1, 2, 2)
plt.scatter(data['humidity'], data['traffic_flow'], alpha=0.5, color='orange')
plt.title('Humidity vs. Traffic Flow')
plt.xlabel('Humidity')
plt.ylabel('Traffic Flow')

plt.tight_layout()
plt.show()