import streamlit as st
import os
import requests
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')  # needed for lemmatization

# Setting up Hugging Face API for NER
API_URL = "https://api-inference.huggingface.co/models/spacy/en_core_web_sm"
headers = {"Authorization": "Bearer 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"}

def get_entities(text):
    data = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=data)
    entities = [item['entity_group'] for item in response.json()[0]]
    return len(entities)

# Set up lemmatizer
lemmatizer = WordNetLemmatizer()

#title 
st.title("Smart Detection System of AI-Generated Text Models")
st.markdown("## This is a POC repo for Smart Detection System of AI Generated Text Models project, it is a pre-trained model that detect the probabilities of using any of the known LLM (chatgpt3, chatgpt4, GoogleBard, HuggingfaceChat)##")

# Check if the file exists
if not os.path.isfile('RandomForestClassifier.pkl'):
    # Download the zip file if it doesn't exist
    url = 'https://jaifar.net/RandomForestClassifier.pkl'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    # Save the file
    with open('RandomForestClassifier.pkl', 'wb') as file:
        file.write(response.content)

with open('RandomForestClassifier.pkl', 'rb') as file:
    clf_loaded = pickle.load(file)

input_paragraph = st.text_area("Input your text here")
df = pd.DataFrame(columns=["paragraph"])
df = df.append({"paragraph": input_paragraph}, ignore_index=True)

num_words = 500
input_paragraph = ' '.join(word_tokenize(input_paragraph)[:num_words])

def extract_features(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    avg_word_length = sum(len(word) for word in words) / len(words)
    avg_sent_length = sum(len(sent) for sent in sentences) / len(sentences)
    punctuation_count = len([char for char in text if char in '.,;:?!'])
    stopword_count = len([word for word in words if word in stopwords.words('english')])
    lemma_count = len(set(lemmatizer.lemmatize(word) for word in words))
    named_entity_count = get_entities(text)
    tagged_words = nltk.pos_tag(words)
    pos_counts = nltk.FreqDist(tag for (word, tag) in tagged_words)
    pos_features = {
        'pos_IN': pos_counts['IN'],
        'pos_DT': pos_counts['DT'],
        'pos_NN': pos_counts['NN'],
        'pos_,': pos_counts[','],
        'pos_VBZ': pos_counts['VBZ'],
        'pos_WDT': pos_counts['WDT'],
        'pos_TO': pos_counts['TO'],
        'pos_VB': pos_counts['VB'],
        'pos_VBG': pos_counts['VBG'],
        'pos_.': pos_counts['.'],
        'pos_JJ': pos_counts['JJ'],
        'pos_NNS': pos_counts['NNS'],
        'pos_RB': pos_counts['RB'],
        'pos_CC': pos_counts['CC'],
        'pos_VBN': pos_counts['VBN'],
    }
    features = {
        'avg_word_length': avg_word_length,
        'avg_sent_length': avg_sent_length,
        'punctuation_count': punctuation_count,
        'stopword_count': stopword_count,
        'lemma_count': lemma_count,
        'named_entity_count': named_entity_count,
    }
    features.update(pos_features)
    return pd.Series(features)

press_me_button = st.button("Press me")

if press_me_button:
    input_features = df['paragraph'].apply(extract_features)
    predicted_llm = clf_loaded.predict(input_features)
    st.write(f"Predicted LLM: {predicted_llm[0]}")
