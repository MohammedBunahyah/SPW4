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

# 🔧 Set the NLTK Data Directory
NLTK_DIR = os.path.expanduser("~/nltk_data")
nltk.data.path.append(NLTK_DIR)

# ✅ Function to Ensure NLTK Resource is Available
def ensure_nltk_resource(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DIR)

# ✅ Download Missing NLTK Resources
ensure_nltk_resource("tokenizers/punkt")
ensure_nltk_resource("corpora/stopwords")
ensure_nltk_resource("corpora/wordnet")
ensure_nltk_resource("corpora/omw-1.4")
ensure_nltk_resource("taggers/averaged_perceptron_tagger")

# ✅ Reload NLTK Data
nltk.data.path.append(NLTK_DIR)

# ✅ Initialize Stopwords & Lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ✅ Load Model & Vectorizer from GitHub
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("🚨 Model or vectorizer file not found! Make sure they are uploaded to GitHub.")

# ✅ Load Dataset from Google Drive
DATA_FILE_ID = "1AsdUWNsA981I0GXty9r345IBC4Ly_D1X"  # Your Google Drive dataset file ID

@st.cache_data
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return output_path

# Check if data.csv exists
data_path = download_from_gdrive(DATA_FILE_ID, "data.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.error("🚨 Dataset file not found! Check Google Drive file ID or manually upload it.")
    df = pd.DataFrame()  # Prevents errors by creating an empty DataFrame

# 🔎 Preprocessing Function (Same as Used in Training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# 🌐 Streamlit App UI
st.title("📰 Fake News Detection App")

st.write("### Dataset Overview:")
if not df.empty:
    st.write(df.head())  # Show first rows of dataset
else:
    st.write("No dataset available.")

# 📝 User Input
user_input = st.text_area("Enter a news headline or article:")

if st.button("Check News"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a news headline or article.")
    else:
        cleaned_input = clean_text(user_input)  # Clean input text
        input_vector = vectorizer.transform([cleaned_input])  # Convert text to TF-IDF
        
        prediction = model.predict(input_vector)[0]  # Predict
        
        # 🎯 Show result
        st.write("### Prediction:")
        st.success("✅ Real News") if prediction == 1 else st.error("🚨 Fake News")
