import streamlit as st
import joblib
import numpy as np

# Load models
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
tree_model = joblib.load("tree_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌸 Iris Classification Web App")
st.subheader("SVM | Random Forest | Decision Tree")

# Input fields
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

model_choice = st.selectbox(
    "Choose Model",
    ["SVM", "Random Forest", "Decision Tree"]
)

if st.button("Predict"):

    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if model_choice == "SVM":
        data = scaler.transform(data)
        prediction = svm_model.predict(data)

    elif model_choice == "Random Forest":
        prediction = rf_model.predict(data)

    else:
        prediction = tree_model.predict(data)

    # Convert numeric → actual flower name
    classes = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"🌼 Predicted Flower: {classes[prediction[0]]}")