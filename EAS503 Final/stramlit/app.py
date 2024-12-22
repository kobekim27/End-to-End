import json
import streamlit as st
import requests

user_options = {}

st.title('Income Prediction')

# Define the options for our income prediction model
streamlit_options = {
    "slider_fields": {
        "age": [17, 90],
        "fnlwgt": [12285, 1490400],
        "education-num": [1, 16],
        "capital-gain": [0, 99999],
        "capital-loss": [0, 4356],
        "hours-per-week": [1, 99]
    }
}

# Create sliders for each numerical feature
for field_name, range in streamlit_options["slider_fields"].items():
    min_val, max_val = range
    current_value = round((min_val + max_val)/2)
    user_options[field_name] = st.sidebar.slider(field_name, min_val, max_val, value=current_value)

# Display the selected options
st.write("Selected features:")
st.json(user_options)

if st.button('Predict Income'):
    # Prepare the input data
    features = list(user_options.values())
    
    # Make a request to the deployed API
    api_url = "http://your-deployed-api-url.com/predict"  # Replace with your actual API URL
    response = requests.post(api_url, json={"features": features})
    
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        if prediction == 1:
            st.success("Prediction: Income is greater than $50K")
        else:
            st.info("Prediction: Income is less than or equal to $50K")
    else:
        st.error("Error occurred while making the prediction. Please try again.")

st.write("Note: This app uses a model trained on the Adult Income dataset to predict income based on the provided features.")
