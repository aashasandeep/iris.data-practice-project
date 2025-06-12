import streamlit as st
import pickle
import numpy as np
import sklearn

# Load the trained model
with open(r'C:\Users\ASHA SANDEEP\OneDrive\Desktop\iris_data\iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Iris Flower Classifier")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    classes = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Flower: {classes[prediction[0]]}")
