import streamlit as st
import requests
import pandas as pd

# Load the CSV data
@st.cache_data
def load_data():
    return pd.read_csv('adult-all.csv', header=None, names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ])

data = load_data()

st.title('Income Prediction')

# Define the options for our income prediction model
streamlit_options = {
    "slider_fields": {
        "age": [data['age'].min(), data['age'].max()],
        "fnlwgt": [data['fnlwgt'].min(), data['fnlwgt'].max()],
        "education-num": [data['education-num'].min(), data['education-num'].max()],
        "capital-gain": [data['capital-gain'].min(), data['capital-gain'].max()],
        "capital-loss": [data['capital-loss'].min(), data['capital-loss'].max()],
        "hours-per-week": [data['hours-per-week'].min(), data['hours-per-week'].max()]
    },
    "select_fields": {
        "workclass": data['workclass'].unique().tolist(),
        "education": data['education'].unique().tolist(),
        "marital-status": data['marital-status'].unique().tolist(),
        "occupation": data['occupation'].unique().tolist(),
        "relationship": data['relationship'].unique().tolist(),
        "race": data['race'].unique().tolist(),
        "sex": data['sex'].unique().tolist(),
        "native-country": data['native-country'].unique().tolist()
    }
}

user_options = {}

# Create sliders for numerical features
for field_name, range in streamlit_options["slider_fields"].items():
    min_val, max_val = range
    current_value = round((min_val + max_val)/2)
    user_options[field_name] = st.sidebar.slider(field_name, float(min_val), float(max_val), value=float(current_value))

# Create select boxes for categorical features
for field_name, options in streamlit_options["select_fields"].items():
    user_options[field_name] = st.sidebar.selectbox(field_name, options)

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

# Display dataset statistics
st.subheader("Dataset Statistics")
st.write(data.describe())

# Display correlation heatmap
st.subheader("Feature Correlation")
correlation = data.corr()
st.pyplot(correlation.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

# Display income distribution
st.subheader("Income Distribution")
income_distribution = data['income'].value_counts()
st.bar_chart(income_distribution)
