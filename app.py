import streamlit as st
st.write("Test system if working")
import zipfile
import os
import requests
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np


# Custom headers for the HTTP request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
}

# Debugging: Print current working directory initially
st.write(f"Initial Current Working Directory: {os.getcwd()}")

# Check if the model folder exists
zip_file_path = "my_authorship_model_zip.zip"
if not os.path.exists('my_authorship_model'):
    try:
        # Download the model
        model_url = 'https://jaifar.net/ADS/my_authorship_model_zip.zip'
        r = requests.get(model_url, headers=headers)
        r.raise_for_status()

        # Debugging: Check if download is successful by examining content length
        st.write(f"Downloaded model size: {len(r.content)} bytes")

        # Save the downloaded content
        with open(zip_file_path, "wb") as f:
            f.write(r.content)

        # Debugging: Verify that the zip file exists
        if os.path.exists(zip_file_path):
            st.write("Zip file exists")

            # Extract the model using zipfile
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall('my_authorship_model')
                
            # Debugging: Check if the folder is successfully created
            if os.path.exists('my_authorship_model'):
                st.write("Model folder successfully extracted using zipfile")
                # Debugging: List the directory contents after extraction
                st.write("Listing directory contents:")
                st.write(os.listdir('.'))
            else:
                st.write("Model folder was not extracted successfully using zipfile")
                exit(1)

        else:
            st.write("Zip file does not exist")
            exit(1)
    except Exception as e:
        st.write(f"Failed to download or extract the model: {e}")
        exit(1)
else:
    st.write("Model folder exists")

# Debugging: Print current working directory after extraction
st.write(f"Current Working Directory After Extraction: {os.getcwd()}")

# Debugging: Check if model folder contains required files
try:
    model_files = os.listdir('my_authorship_model')
    st.write(f"Files in model folder: {model_files}")
except Exception as e:
    st.write(f"Could not list files in model folder: {e}")

# Download the required files
file_urls = {
    'tokenizer.pkl': 'https://jaifar.net/ADS/tokenizer.pkl',
    'label_encoder.pkl': 'https://jaifar.net/ADS/label_encoder.pkl'
}

for filename, url in file_urls.items():
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        st.write(f"Failed to download {filename}: {e}")
        exit(1)

# Load the saved model
loaded_model = load_model("my_authorship_model")

# Load the saved tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

max_length = 300  # As defined in the training code

# Function to predict author for new text
def predict_author(new_text, model, tokenizer, label_encoder):
    sequence = tokenizer.texts_to_sequences([new_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)

    predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]
    probabilities = prediction[0]
    author_probabilities = {}
    for idx, prob in enumerate(probabilities):
        author = label_encoder.inverse_transform([idx])[0]
        author_probabilities[author] = prob

    return predicted_label, author_probabilities

st.markdown("CNN : version: 1.2")
new_text = st.text_area("Input your text here")

# Creates a button named 'Press me'
press_me_button = st.button("Which Model Used?")

if press_me_button:
    predicted_author, author_probabilities = predict_author(new_text, loaded_model, tokenizer, label_encoder)
    sorted_probabilities = sorted(author_probabilities.items(), key=lambda x: x[1], reverse=True)

    st.write(f"The text is most likely written by: {predicted_author}")
    st.write("Probabilities for each author are (sorted):")
    for author, prob in sorted_probabilities:
        st.write(f"{author}: {prob * 100:.2f}%")
