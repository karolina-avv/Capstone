

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Data Collection and Preprocessing
# Load the dataset
train_data = pd.read_csv('KDDTrain+.csv')
test_data = pd.read_csv('KDDTest+.csv')

# Handle missing values
print("Missing values in train data:", train_data.isnull().sum().sum())

# Encode categorical variables
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Align train and test datasets to ensure same columns
train_data, test_data = train_data.align(test_data, join='inner', axis=1)

# Separate features and labels
X_train = train_data.drop('target_column', axis=1)
y_train = train_data['target_column']
X_test = test_data.drop('target_column', axis=1)
y_test = test_data['target_column']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split into training and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Step 2: Model Training
# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_final, y_train_final)

# Validate the model
y_val_pred = rf_model.predict(X_val)

# Print performance metrics
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Save the trained model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Print final message
print("Model training completed and saved as 'random_forest_model.pkl'")
