import streamlit as st
import numpy as np
import joblib  # To load the saved model

# Load trained model and label encoders
model = joblib.load("naive_bayes_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Get feature names
feature_names = list(label_encoders.keys())
feature_names.remove("Life Expectancy Category")  # Remove target column

st.title("Life Expectancy Prediction")
st.write("Enter values to predict Life Expectancy Category (Low/Medium/High)")

# Create input fields for user
user_inputs = {}
for col in feature_names:
    user_inputs[col] = st.selectbox(f"Select {col}", label_encoders[col].classes_)

# Predict button
if st.button("Predict Life Expectancy"):
    # Convert user inputs to numerical values
    user_input_values = np.array([label_encoders[col].transform([user_inputs[col]])[0] for col in feature_names]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(user_input_values)[0]
    predicted_category = label_encoders["Life Expectancy Category"].inverse_transform([prediction])[0]
    
    st.success(f"Predicted Life Expectancy Category: {predicted_category}")
