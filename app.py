import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Paths for model, tokenizer, and label encoder
MODEL_PATH = 'my_model.h5'
TOKENIZER_PATH = 'tokenizerr.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open(LABEL_ENCODER_PATH, 'rb') as handle:
    label_encoder = pickle.load(handle)

# Preprocess descriptions function
def preprocess_descriptions(descriptions):
    descriptions = [desc.lower() for desc in descriptions]
    sequences = tokenizer.texts_to_sequences(descriptions)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)
    return padded_sequences

# Function to make predictions in batches
def make_predictions_in_batches(descriptions, batch_size=100):
    all_predictions = []
    num_batches = len(descriptions) // batch_size + (1 if len(descriptions) % batch_size != 0 else 0)
    
    for batch_index in range(num_batches):
        batch_descriptions = descriptions[batch_index * batch_size : (batch_index + 1) * batch_size]
        preprocessed_descriptions = preprocess_descriptions(batch_descriptions)
        
        try:
            predictions = model.predict(preprocessed_descriptions)
        except Exception as e:
            st.error(f"Error during batch {batch_index} prediction: {str(e)}")
            st.error(f"Details: {e.__class__.__name__}: {e}")
            return None
        
        all_predictions.extend(predictions)
    
    return np.array(all_predictions)

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
        # Preprocess and make predictions in batches
        st.write("Preprocessing descriptions...")
        st.write("Making predictions in batches...")
        predictions = make_predictions_in_batches(descriptions)
        
        if predictions is not None:
            # Decode predictions
            predicted_class_indices = np.argmax(predictions, axis=1)
            predicted_labels = label_encoder.inverse_transform(predicted_class_indices)
            
            # Add prediction columns to the data DataFrame
            data['Predicted Class Index'] = predicted_class_indices
            data['Predicted Label'] = predicted_labels
            
            st.write("Predictions")
            st.write(data)
        else:
            st.error("Error occurred during batch prediction. Please check the logs.")
    except Exception as e:
        st.error(f"Error during model prediction: {str(e)}")
        st.error(f"Details: {e.__class__.__name__}: {e}")
