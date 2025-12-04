import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import joblib

# Load the Iris dataset (assuming it's available, or use sklearn's iris)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create and train the model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

# Save the model
with open('modelKNN1.pkl', 'wb') as file:
    pickle.dump(model, file)

joblib.dump(model, "knn_model.sav")
print("Model saved as modelKNN1.pkl and knn_model.sav")
