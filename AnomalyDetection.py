import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# Load data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Split data into features and target
X = data_train.drop(['id', 'is_anomaly'], axis=1)
y = data_train['is_anomaly']

# Split the data into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(data_test.drop(['id'], axis=1))

# Train the model
model = SVC(kernel='rbf', class_weight='balanced', probability=True)
model.fit(X_train_scaled, y_train)

# Predict probabilities for the validation set
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]

# Calculate ROC-AUC score for the training set
roc_auc_train = roc_auc_score(y_train, y_train_proba)
print("ROC-AUC score for the training set:", roc_auc_train)

# Predict probabilities for the test set
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Define the threshold and make predictions
threshold = 0.2
y_test_pred = [1 if proba > threshold else 0 for proba in y_test_proba]

# Ensure the lengths match
print("Length of data_test['id']:", len(data_test['id']))
print("Length of y_test_pred:", len(y_test_pred))

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': data_test['id'], 'is_anomaly': y_test_pred})

# Save the submission to a CSV file
submission_df.to_csv('submission.csv', index=False)
