import streamlit as st
import zipfile
import os
import requests
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from PIL import Image


# Custom headers for the HTTP request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
}

#################### Load the banner image ##########
# Fetch the image from the URL
banner_image_request = requests.get("https://jaifar.net/ADS/banner.jpg", headers=headers)

# Save the downloaded content
banner_image_path = "banner.jpg"
with open(banner_image_path, "wb") as f:
    f.write(banner_image_request.content)


# Open the image
banner_image = Image.open(banner_image_path)

# Display the image using streamlit
st.image(banner_image, caption='', use_column_width=True)

################ end loading banner image ##################


############# Download Or Check Files/folders exeistince ##############
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
     st.write("Version: 2.1")


# Download the required files
file_urls = {
    'tokenizer.pkl': 'https://jaifar.net/ADS/tokenizer.pkl',
    'label_encoder.pkl': 'https://jaifar.net/ADS/label_encoder.pkl'
}

for filename, url in file_urls.items():
    if not os.path.exists(filename):  # Check if the file doesn't exist
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            st.write(f"Failed to download {filename}: {e}")
            exit(1)
    else:
        st.write(f"File {filename} already exists. Skipping download.")

############### Load CNN Model ############
# Load the saved model
loaded_model = load_model("my_authorship_model")

# Load the saved tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

max_length = 300  # As defined in the training code

############### End Load CNN Model ############

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

new_text = st.text_area("Input Your Text Here:")

# Creates a button named 'Press me'
press_me_button = st.button("Human or Robot?")

if press_me_button:
    predicted_author, author_probabilities = predict_author(new_text, loaded_model, tokenizer, label_encoder)
    sorted_probabilities = sorted(author_probabilities.items(), key=lambda x: x[1], reverse=True)
  
    author_map = {
        "googlebard": "Google Bard",
        "gpt3": "ChatGPT-3",
        "gpt4": "ChatGPT-4",
        "huggingface": "HuggingChat",
        "human": "Human-Written"
    }

    predicted_author_diplay_name =  author_map.get(predicted_author, predicted_author)
    st.write(f"The text is most likely written by: {predicted_author_diplay_name}")
    st.write("Probabilities for each author are (sorted):")
    # Mapping the internal names to display names


    for author, prob in sorted_probabilities:
        display_name = author_map.get(author, author)  # Retrieve the display name, fall back to original if not found
        st.write(f"{display_name}: {prob * 100:.2f}%")
        st.progress(float(prob))

# Using expander to make FAQ sections
st.subheader("Frequently Asked Questions (FAQ)")

# Small Description
with st.expander("What is this project about?"):
    st.write("""
    This project is part of an MSc in Data Analytics at the University of Portsmouth.
    Developed by Jaifar Al Shizawi, it aims to identify whether a text is written by a human or a specific Large Language Model (LLM) like ChatGPT-3, ChatGPT-4, Google Bard, or HuggingChat.
    For inquiries, contact [up2152209@myport.ac.uk](mailto:up2152209@myport.ac.uk).
    Supervised by Dr. Mohamed Bader.
    """)
    
# Aim and Objectives
with st.expander("Aim and Objectives"):
    st.write("""
    The project aims to help staff at the University of Portsmouth distinguish between student-written artifacts and those generated by LLMs. It focuses on text feature extraction, model testing, and implementing a user-friendly dashboard among other objectives.
    """)

# System Details
with st.expander("How does the system work?"):
    st.write("""
    The system is trained using deep learning model on a dataset of 140,546 paragraphs, varying in length from 10 to 1090 words.
    It achieves an accuracy of 0.9964 with a validation loss of 0.094.
    """)

    # Fetch the image from the URL
    accuracy_image_request = requests.get("https://jaifar.net/ADS/best_accuracy.png", headers=headers)
    
    # Save the downloaded content
    image_path = "best_accuracy.png"
    with open(image_path, "wb") as f:
        f.write(accuracy_image_request.content)

    
    # Open the image
    accuracy_image = Image.open(image_path)
    
    # Display the image using streamlit
    st.image(accuracy_image, caption='Best Accuracy', use_column_width=True)

# Data Storage Information
with st.expander("Does the system store my data?"):
    st.write("No, the system does not collect or store any user input data.")

# Use-case Limitation
with st.expander("Can I use this as evidence?"):
    st.write("""
    No, this system is a Proof of Concept (POC) and should not be used as evidence against students or similar entities.
    """)




