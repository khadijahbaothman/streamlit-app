import streamlit as st
import pandas as pd
import tensorflow as tf

# Load your trained model (using a relative path)
MODEL_PATH = '65my_model.h5'  

model = tf.keras.models.load_model(MODEL_PATH)

# Preprocess descriptions if necessary
def preprocess_descriptions(descriptions):
    # Assuming your model needs numerical data, convert descriptions to numerical data here
    # For example, if your model uses tokenized text, convert text to tokens
    # This is a placeholder for actual preprocessing steps
    # Convert descriptions to a numpy array of floats (this example assumes the descriptions can be converted directly)
    processed_descriptions = descriptions.astype(float)
    processed_descriptions = processed_descriptions.reshape(-1, 1)  # Reshape if necessary
    return processed_descriptions

# Function to make predictions
def make_predictions(data):
    # Check if 'Description' column exists
    if 'Description' not in data.columns:
        st.error("The input data must contain a 'Description' column.")
        return data

    # Preprocess the input data
    descriptions = preprocess_descriptions(data['Description'].astype(str))
    
    # Check the input data format
    st.write("Descriptions for prediction:", descriptions)
    st.write("Shape of descriptions:", descriptions.shape)
    st.write("Type of descriptions:", type(descriptions))
    
    # Make predictions
    try:
        predictions = model.predict(descriptions)
        st.write("Predictions shape:", predictions.shape)
        data['Predicted Label'] = predictions
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        st.write("Stack trace:", e)
    
    return data

# Streamlit App
st.title('Model Deployment with Streamlit')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded Excel file
    input_data = pd.read_excel(uploaded_file)
    
    # Display the uploaded data
    st.write("### Uploaded Data")
    st.write(input_data)
    
    # Make predictions
    output_data = make_predictions(input_data)
    
    # Display the predictions
    st.write("### Predictions")
    st.write(output_data)

    # Download the result
    st.download_button(
        label="Download predictions as CSV",
        data=output_data.to_csv(index=False),
        file_name='predictions.csv',
        mime='text/csv'
    )
