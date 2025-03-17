import streamlit as st
import pandas as pd
import joblib
import os
import gdown
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Point to local NLTK data
nltk.data.path.append("./nltk_data")

# Load NLTK components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load Model & Vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Google Drive data
DATA_FILE_ID = "1AsdUWNsA981I0GXty9r345IBC4Ly_D1X"

@st.cache_resource 
def load_nltk_resources(): 
    download_nltk_resources()

@st.cache_data
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return output_path

data_path = download_from_gdrive(DATA_FILE_ID, "data.csv")
df = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

st.title(":newspaper: Fake News Detection App")
st.write("### Dataset Overview:")
st.write(df.head())

user_input = st.text_area("Enter a news headline or article:")
if st.button("Check News"):
    if not user_input.strip():
        st.warning(":warning: Please enter a news headline or article.")
    else:
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]
        st.write("### Prediction:")
        st.success(":white_check_mark: Real News") if prediction == 1 else st.error(":rotating_light: Fake News")
