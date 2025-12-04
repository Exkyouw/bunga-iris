import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('knn_model.sav')

# Title
st.title('Iris Flower Classification')

# Input fields
st.header('Input Features')
sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width', 2.0, 5.0, 3.0)
petal_length = st.slider('Petal Length', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width', 0.1, 2.5, 1.0)

# Prediction
if st.button('Predict'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f'Predicted Species: {prediction[0]}')
