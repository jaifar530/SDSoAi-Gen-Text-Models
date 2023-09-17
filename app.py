import streamlit as st
import zipfile
import os
import requests
import re
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from PIL import Image
from joblib import load


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
        # st.write(f"Downloaded model size: {len(r.content)} bytes")

        # Save the downloaded content
        with open(zip_file_path, "wb") as f:
            f.write(r.content)

        # Debugging: Verify that the zip file exists
        if os.path.exists(zip_file_path):
            # st.write("Zip file exists")

            # Extract the model using zipfile
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall('my_authorship_model')
                
            # # Debugging: Check if the folder is successfully created
            # if os.path.exists('my_authorship_model'):
            #     # st.write("Model folder successfully extracted using zipfile")
            #     # Debugging: List the directory contents after extraction
            #     # st.write("Listing directory contents:")
            #     # st.write(os.listdir('.'))
            # else:
            #     st.write("Model folder was not extracted successfully using zipfile")
            #     exit(1)

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
    # else:
    #     st.write(f"File {filename} already exists. Skipping download.")
############ download ridge and ExtraTree stuff

# def has_internet_connection():
#     try:
#         response = requests.get("https://www.google.com/", timeout=5)
#         return True
#     except requests.ConnectionError:
#         return False

def is_zip_file(file_path):
    return zipfile.is_zipfile(file_path)

def are_files_extracted(extracted_files, missing_files):
    for file in missing_files:
        if file not in extracted_files:
            return False
    return True

def check_and_download_files():
    file_names = [
        "truncated_260_to_284.xlsx_vectorizer.pkl",
        "not_trancated_full_paragraph.xlsx_extra_trees_model.pkl",
        "not_trancated_full_paragraph.xlsx_ridge_model.pkl",
        "not_trancated_full_paragraph.xlsx_vectorizer.pkl",
        "truncated_10_to_34.xlsx_extra_trees_model.pkl",
        "truncated_10_to_34.xlsx_ridge_model.pkl",
        "truncated_10_to_34.xlsx_vectorizer.pkl",
        "truncated_35_to_59.xlsx_extra_trees_model.pkl",
        "truncated_35_to_59.xlsx_ridge_model.pkl",
        "truncated_35_to_59.xlsx_vectorizer.pkl",
        "truncated_60_to_84.xlsx_extra_trees_model.pkl",
        "truncated_60_to_84.xlsx_ridge_model.pkl",
        "truncated_60_to_84.xlsx_vectorizer.pkl",
        "truncated_85_to_109.xlsx_extra_trees_model.pkl",
        "truncated_85_to_109.xlsx_ridge_model.pkl",
        "truncated_85_to_109.xlsx_vectorizer.pkl",
        "truncated_110_to_134.xlsx_extra_trees_model.pkl",
        "truncated_110_to_134.xlsx_ridge_model.pkl",
        "truncated_110_to_134.xlsx_vectorizer.pkl",
        "truncated_135_to_159.xlsx_extra_trees_model.pkl",
        "truncated_135_to_159.xlsx_ridge_model.pkl",
        "truncated_135_to_159.xlsx_vectorizer.pkl",
        "truncated_160_to_184.xlsx_extra_trees_model.pkl",
        "truncated_160_to_184.xlsx_ridge_model.pkl",
        "truncated_160_to_184.xlsx_vectorizer.pkl",
        "truncated_185_to_209.xlsx_extra_trees_model.pkl",
        "truncated_185_to_209.xlsx_ridge_model.pkl",
        "truncated_185_to_209.xlsx_vectorizer.pkl",
        "truncated_210_to_234.xlsx_extra_trees_model.pkl",
        "truncated_210_to_234.xlsx_ridge_model.pkl",
        "truncated_210_to_234.xlsx_vectorizer.pkl",
        "truncated_235_to_259.xlsx_extra_trees_model.pkl",
        "truncated_235_to_259.xlsx_ridge_model.pkl",
        "truncated_235_to_259.xlsx_vectorizer.pkl",
        "truncated_260_to_284.xlsx_extra_trees_model.pkl",
        "truncated_260_to_284.xlsx_ridge_model.pkl"
    ]
    missing_files = []

    for file_name in file_names:
        if not os.path.exists(file_name):
            missing_files.append(file_name)

    if missing_files:
        st.write("The following files are missing:")
        for file_name in missing_files:
            st.write(file_name)
        
        # if not has_internet_connection():
        #     st.write("No internet connection. Cannot download missing files.")
        #     return
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            }
            url = 'https://jaifar.net/ADS/content.zip'
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            with open('content.zip', 'wb') as zip_file:
                zip_file.write(response.content)

            if not is_zip_file('content.zip'):
                st.write("Downloaded content is not a ZIP file.")
                return

            with zipfile.ZipFile('content.zip', 'r') as zip_ref:
                zip_ref.extractall()

            extracted_files = os.listdir()
            if not are_files_extracted(extracted_files, missing_files):
                st.write("Not all missing files were extracted.")
                return
            
            st.write("content.zip downloaded and extracted successfully.")
        except Exception as e:
            st.write(f"Error downloading or extracting content.zip: {e}")
    else:
        st.write("All files exist.")

check_and_download_files()

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
word_count = len(re.findall(r'\w+', new_text))
st.write(word_count)

# Creates a button named 'Press me'
press_me_button = st.button("Human or Robot?")

if press_me_button:
    
    ########## ML 

    word_count = len(re.findall(r'\w+', new_text))
    st.write(word_count)

    # Choose the appropriate model based on word count
    if 10 <= word_count <= 34:
        file_prefix = 'truncated_10_to_34.xlsx'
    elif 35 <= word_count <= 59:
        file_prefix = 'truncated_35_to_59.xlsx'
    elif 60 <= word_count <= 84:
        file_prefix = 'truncated_60_to_84.xlsx'
    elif 85 <= word_count <= 109:
        file_prefix = 'truncated_85_to_109.xlsx'
    elif 110 <= word_count <= 134:
        file_prefix = 'truncated_110_to_134.xlsx'
    elif 135 <= word_count <= 159:
        file_prefix = 'truncated_135_to_159.xlsx'
    elif 160 <= word_count <= 184:
        file_prefix = 'truncated_160_to_184.xlsx'
    elif 185 <= word_count <= 209:
        file_prefix = 'truncated_185_to_209.xlsx'
    elif 210 <= word_count <= 234:
        file_prefix = 'truncated_210_to_234.xlsx'
    elif 235 <= word_count <= 259:
        file_prefix = 'truncated_235_to_259.xlsx'
    elif 260 <= word_count <= 284:
        file_prefix = 'truncated_260_to_284.xlsx'
    else:
        file_prefix = 'not_trancated_full_paragraph.xlsx'
    
    # Load the models and vectorizer
    ridge_model = pickle.load(f"{file_prefix}_ridge_model.pkl")
    extra_trees_model = pickle.load(f"{file_prefix}_extra_trees_model.pkl")
    vectorizer = pickle.load(f"{file_prefix}_vectorizer.pkl")
    # Transform the input
    user_input_transformed = vectorizer.transform(new_text)

    # Load the saved models and vectorizer

    # loaded_ridge_classifier = joblib.load(f'{excel_file_to_use}_ridge_model.pkl')
    # loaded_extra_trees_classifier = joblib.load(f'{excel_file_to_use}_extra_trees_model.pkl')
    # loaded_vectorizer = joblib.load(f'{excel_file_to_use}_vectorizer.pkl')

    # user_input_vectorized = loaded_vectorizer.transform(user_input)

    
    # Make predictions
    ridge_prediction = ridge_model.predict(user_input_transformed)
    extra_trees_prediction = extra_trees_model.predict(user_input_transformed)
    
    if ridge_prediction == extra_trees_prediction:
        st.write(f"The author is: {ridge_prediction[0]}")
    else:
        st.write(f"Different predictions. Ridge says: {ridge_prediction[0]}, Extra Trees says: {extra_trees_prediction[0]}")
    
    
    ########## DL
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


# Creates a button named 'Press me'
list_dir = st.button("list")
if list_dir:
    st.write("Listing directory contents:")
    st.write(os.listdir('.'))



