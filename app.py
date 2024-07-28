import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Paths for model and tokenizer
MODEL_PATH = 'my_model.h5'  # Updated model path
TOKENIZER_PATH = 'tokenizerr.pkl'

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the label encoder with the classes provided
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([
    'Body & Chassis',
    'Tyre/Wheel',
    'Electrical (Battery)',
    'General Inspection',
    'Engine Electrical',
    "Operator's Failure",
    'Mix of Safey & Body/Chassis: paint, reflector',
    'Baggage Handling (Dolly/BGG CART/ULD)',
    'Electrical (Low & High Voltage)',
    'Refurbishment',
    'Thermal BGT',
    'Safety (Fire Extinguisher)',
    'Additional Check prior Seasons',
    'Brake System',
    'Load Transfer/Handling',
    'Safety',
    'Description to be more precise',
    'Engine Mechanical',
    'Air Conditioning',
    'Engine Fuel System',
    'Suspension System',
    'Electrical (Beacon Light)',
    'Hydraulic System',
    'Drive Train',
    'Mix of Baggage Handling & Body Chassis',
    'Engine Lubrication System',
    'Electronic System (Sensor)',
    'Pneumatic System',
    'Engine Cooling System',
    'Assembly of new equipment',
    'Safety Failure',
    'Electronic System',
    'Steering System',
    'Safety (Rubber)',
    'Engine Exhaust System',
    'Equipment No Show',
    'Commisioning of new equipment',
    'Safety (Reflector)',
    'Engine Smoking',
    'Engine Intake System',
    'Electrical'
])

# Preprocess descriptions function
def preprocess_descriptions(descriptions):
    descriptions = [desc.lower() for desc in descriptions]
    sequences = tokenizer.texts_to_sequences(descriptions)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)
    return padded_sequences

# Function to make predictions
def make_predictions(descriptions):
    preprocessed_descriptions = preprocess_descriptions(descriptions)
    predictions = model.predict(preprocessed_descriptions)
    return predictions

# Streamlit UI
st.title("Description Classification")

uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data")
    st.write(data)
    
    # Extract descriptions
    descriptions = data['Description'].tolist()
    st.write("Descriptions for prediction:")
    st.write(descriptions)
    
    try:
        # Preprocess and make predictions
        predictions = make_predictions(descriptions)
        
        # Decode predictions
        predicted_class_indices = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_class_indices)
        
        # Add prediction columns to the data DataFrame
        data['Predicted Class Index'] = predicted_class_indices
        data['Predicted Label'] = predicted_labels
        
        st.write("Predictions")
        st.write(data)
    except Exception as e:
        st.error(f"Error during model prediction: {str(e)}")
