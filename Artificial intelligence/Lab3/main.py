# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from seaborn
titanic_data = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic_data.head())

# Check for missing values
print(titanic_data.isnull().sum())

#  Drop rows with missing values
titanic_data_cleaned = titanic_data.dropna()

# Identify outliers using box plots for 'fare' and 'age'
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=titanic_data['fare'])
plt.title('Fare Box Plot')

plt.subplot(1, 2, 2)
sns.boxplot(x=titanic_data['age'])
plt.title('Age Box Plot')

plt.show()

# Remove outliers (using IQR method as an example)
Q1 = titanic_data['fare'].quantile(0.25)
Q3 = titanic_data['fare'].quantile(0.75)
IQR = Q3 - Q1
titanic_data_cleaned = titanic_data_cleaned[(titanic_data_cleaned['fare'] >= (Q1 - 1.5 * IQR)) & (titanic_data_cleaned['fare'] <= (Q3 + 1.5 * IQR))]

# Normalizing 'age' and 'fare' using Min-Max scaling
titanic_data_cleaned['age'] = (titanic_data_cleaned['age'] - titanic_data_cleaned['age'].min()) / (titanic_data_cleaned['age'].max() - titanic_data_cleaned['age'].min())
titanic_data_cleaned['fare'] = (titanic_data_cleaned['fare'] - titanic_data_cleaned['fare'].min()) / (titanic_data_cleaned['fare'].max() - titanic_data_cleaned['fare'].min())

# Create 'family_size' feature
titanic_data_cleaned['family_size'] = titanic_data_cleaned['sibsp'] + titanic_data_cleaned['parch']

# Extract title from name like Mr and Mrs
titanic_data_cleaned['title'] = titanic_data_cleaned['name'].str.extract('([A-Za-z]+)\.')

# Correlation analysis to select important features
correlation_matrix = titanic_data_cleaned.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Select features based on correlation or importance (example: 'survived', 'age', 'fare', 'family_size')
features = ['survived', 'age', 'fare', 'family_size']
X = titanic_data_cleaned[features].drop('survived', axis=1)  # Features
y = titanic_data_cleaned['survived']  # Target variable

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)