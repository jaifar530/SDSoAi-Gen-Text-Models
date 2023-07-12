import streamlit as st
import os
import requests
import pickle
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
############
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

#######
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

#version
st.markdown("v1.9")


# URL of the text file
url = 'https://jaifar.net/text.txt'

headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
}

response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Read the content of the file
    content = response.text

    # Print the content of the file
    # print(content)
else:
    # Handle the case when the request fails
    print('Failed to download the file.')



#title 
st.title("Smart Detection System of AI-Generated Text Models")

#subtitle 
st.markdown("This is a POC for Smart Detection System of AI Generated Text Models project (:blue[MSc Data Analytics]), it is a pre-trained model that detect the probablities of using any of the known LLM (chatgpt3, chatgpt4, GoogleBard, HuggingfaceChat)")

#input text 
input_paragraph = st.text_area("Input your text here")
words_counts = word_tokenize(input_paragraph)
final_words = len(words_counts)
st.write('Words counts: ', final_words)

# Define your options
options = ["AI vs AI - RandomForest - 88 Samples", "AI vs AI - Ridge - 2000 Samples", "AI vs Human"]

# Create a dropdown menu with "Option 2" as the default
# selected_option = st.selectbox('Select an Option', options, index=1)
selected_option = st.selectbox('Select an Option', options)





# Check if the file exists
if not os.path.isfile('AI_vs_AI_Ridge_2000_Samples.pkl'):
    # Download the zip file if it doesn't exist
    url = 'https://jaifar.net/AI_vs_AI_Ridge_2000_Samples.pkl'
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }

    response = requests.get(url, headers=headers)

    # Save the file
    with open('AI_vs_AI_Ridge_2000_Samples.pkl', 'wb') as file2:
        file2.write(response.content)



# df = pd.DataFrame(columns=["paragraph"])
# df = df.append({"paragraph": input_paragraph}, ignore_index=True)

df = pd.DataFrame([input_paragraph], columns=["paragraph"])



# Variable to control number of words to retrieve
num_words = 500

# Retrieving only the first num_words words of the paragraph
input_paragraph = ' '.join(word_tokenize(input_paragraph)[:num_words])


# Extracting features
def extract_features_AI_vs_AI_RandomForest_88_Samples(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    avg_word_length = sum(len(word) for word in words if word.isalpha()) / len(words)
    avg_sent_length = sum(len(sent) for sent in sentences) / len(sentences)
    punctuation_count = len([char for char in text if char in '.,;:?!'])
    stopword_count = len([word for word in words if word in stopwords.words('english')])

    lemmatizer = WordNetLemmatizer()
    lemma_count = len(set(lemmatizer.lemmatize(word) for word in words))

    named_entity_count = len([chunk for chunk in ne_chunk(pos_tag(words)) if isinstance(chunk, Tree)])

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



# Extracting features for AI_vs_AI_Ridge_2000_Samples
def extract_features_AI_vs_AI_Ridge_2000_Samples(text):
    
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    avg_word_length = sum(len(word) for word in words if word.isalpha()) / len(words)
    avg_sent_length = sum(len(sent) for sent in sentences) / len(sentences)
    punctuation_count = len([char for char in text if char in '.,;:?!'])
    stopword_count = len([word for word in words if word in stopwords.words('english')])

    lemmatizer = WordNetLemmatizer()
    lemma_count = len(set(lemmatizer.lemmatize(word) for word in words))

    named_entity_count = len([chunk for chunk in ne_chunk(pos_tag(words)) if isinstance(chunk, Tree)])

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
        'pos_PRP': pos_counts['PRP'],
        'pos_VBP': pos_counts['VBP'],
        'pos_VBG': pos_counts['VBG'],
        'pos_.': pos_counts['.'],
        'pos_JJ': pos_counts['JJ'],
        'pos_NNS': pos_counts['NNS'],
        'pos_RB': pos_counts['RB'],
        'pos_PRP$': pos_counts['PRP$'],
        'pos_CC': pos_counts['CC'],
        'pos_MD': pos_counts['MD'],
        'pos_VBN': pos_counts['VBN'],
        'pos_NNP': pos_counts['NNP'],
    }

    features = {
        'avg_word_length': avg_word_length,
        'avg_sent_length': avg_sent_length,
        'punctuation_count': punctuation_count,
        'stopword_count': stopword_count,
        'lemma_count': lemma_count,
        'named_entity_count': named_entity_count,
    }
    # features.update(pos_features)
    features = pd.concat([features, pd.DataFrame(pos_features, index=[0])], axis=1)

    return pd.Series(features)

# Function from Code(2)
def add_vectorized_features(df):
    vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()
    X_bow = vectorizer.fit_transform(df['paragraph'])
    X_tfidf = tfidf_vectorizer.fit_transform(df['paragraph'])
    df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df = pd.concat([df, df_bow, df_tfidf], axis=1)
    return df


# Function define AI_vs_AI_RandomForest_88_Samples
def AI_vs_AI_RandomForest_88_Samples(df):
    



    # input_features = df['paragraph'].apply(extract_features_AI_vs_AI_RandomForest_88_Samples)
    # try:
    #     predicted_llm = clf_loaded.predict(input_features)
    #     st.write(f"Predicted LLM: {predicted_llm[0]}")
    #     predicted_proba = clf_loaded.predict_proba(input_features)
    # except Exception as e:
    #     st.write(f"An error occurred: {str(e)}")

    # labels = clf_loaded.classes_

    # # Create a mapping from old labels to new labels
    # label_mapping = {1: 'gpt3', 2: 'gpt4', 3: 'googlebard', 4: 'huggingface'}

    # # Apply the mapping to the labels
    # new_labels = [label_mapping[label] for label in labels]

    # # Create a dictionary that maps new labels to probabilities
    # prob_dict = {k: v for k, v in zip(new_labels, probabilities)}

    # # Convert probabilities to percentages and sort the dictionary in descending order
    # prob_dict = {k: f'{v*100:.2f}%' for k, v in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)}

    # # Print the dictionary
    # #st.write(prob_dict)

    # # Create a progress bar and a bar chart for each LLM
    # for llm, prob in prob_dict.items():
    #     st.write(llm + ': ' + prob)
    #     st.progress(float(prob.strip('%'))/100)
    # return 

def AI_vs_AI_Ridge_2000_Samples(df):

    # At this point, the pickle file should exist, either it was already there, or it has been downloaded and extracted.
    with open('AI_vs_AI_Ridge_2000_Samples.pkl', 'rb') as file2:
        clf_loaded = pickle.load(file2)

    
    input_features = df['paragraph'].apply(extract_features_AI_vs_AI_Ridge_2000_Samples)

    # Here, input_features is a DataFrame, not a Series
    input_features = pd.concat(input_features.values, ignore_index=True)

    # Add new vectorized features
    df = add_vectorized_features(df)

    # Concatenate input_features and df along columns
    final_features = pd.concat([input_features, df], axis=1)

    predicted_llm = clf_loaded.predict(final_features)
    st.write(f"Predicted LLM: {predicted_llm[0]}")

    return



# Check if the file exists
if not os.path.isfile('AI_vs_AI_RandomForest_88_Samples.pkl'):
# Download the zip file if it doesn't exist
    url = 'https://jaifar.net/AI_vs_AI_RandomForest_88_Samples.pkl'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }

    response = requests.get(url, headers=headers)

    # Save the file
    try:
        with open('AI_vs_AI_RandomForest_88_Samples.pkl', 'wb') as file:
            file.write(response.content)
    except Exception as e:
        st.write(f"An error occurred while writing AI_vs_AI_RandomForest_88_Samples.pkl: {str(e)}") 

try:
    with open('AI_vs_AI_RandomForest_88_Samples.pkl', 'rb') as file:
        clf_loaded = pickle.load(file)
except Exception as e:
    st.write(f"An error occurred while loading AI_vs_AI_RandomForest_88_Samples.pkl: {str(e)}")
    return  # This will exit the function

# Creates a button
press_me_button = st.button("Which Model Used?")

if press_me_button:

    input_features = df['paragraph'].apply(extract_features_AI_vs_AI_RandomForest_88_Samples)
    
    try:
        predicted_llm = clf_loaded.predict(input_features)
        st.write(f"Predicted LLM: {predicted_llm[0]}")
        predicted_proba = clf_loaded.predict_proba(input_features)
    except Exception as e:
        st.write(f"An error occurred: {str(e)}")

    # # Use the selected option to control the flow of your application
    # if selected_option == "AI vs AI - RandomForest - 88 Samples":
    #     AI_vs_AI_RandomForest_88_Samples(df)

    # elif selected_option == "AI vs AI - Ridge - 2000 Samples":
    #     AI_vs_AI_Ridge_2000_Samples(df)

    # elif selected_option == "AI vs Human":
    #     st.write("You selected AI vs Human!")






