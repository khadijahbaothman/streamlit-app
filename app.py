import tensorflow as tf
import numpy as np
import pickle

MODEL_PATH = '65my_model.h5'
TOKENIZER_PATH = 'tokenizerr.pkl'

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocess descriptions if necessary
def preprocess_descriptions(descriptions):
    descriptions = [desc.lower() for desc in descriptions]
    sequences = tokenizer.texts_to_sequences(descriptions)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)
    return padded_sequences

def make_predictions(descriptions):
    preprocessed_descriptions = preprocess_descriptions(descriptions)
    predictions = model.predict(preprocessed_descriptions)
    return predictions

import streamlit as st

st.title("Description Classification")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data")
    st.write(data)
    
    descriptions = data['Description'].tolist()
    st.write("Descriptions for prediction:")
    st.write(descriptions)
    st.write(f"Shape of descriptions: {np.shape(descriptions)}")
    st.write(f"Type of descriptions: {type(descriptions)}")

    try:
        predictions = make_predictions(descriptions)
        st.write("Predictions")
        st.write(predictions)
    except Exception as e:
        st.error(f"Error during model prediction: {str(e)}")
