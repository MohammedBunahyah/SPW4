import streamlit as st
import pandas as pd
import joblib
import gdown
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 🔧 Ensure required NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 🔄 Reload NLTK resources after downloading
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Re-import after downloads to ensure functionality
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# 🛠 Google Drive File IDs
MODEL_FILE_ID = "1431m5bn3RJ0SAOpy3zuRRPJMW_LwpVBo"  # Replace with your model file ID
VECTORIZER_FILE_ID = "1HliHGc-mq_q3CvAVzkKrubUv61S8I2Bp"  # Replace with your vectorizer file ID
DATA_FILE_ID = "1AsdUWNsA981I0GXty9r345IBC4Ly_D1X"  # Your data.csv file ID

# 📥 Function to download files from Google Drive
@st.cache_data
def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return output_path

# ✅ Download and Load Model & Vectorizer
model_path = download_from_gdrive(MODEL_FILE_ID, "random_forest_model.pkl")
vectorizer_path = download_from_gdrive(VECTORIZER_FILE_ID, "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ✅ Load Dataset for Display (Optional)
data_path = download_from_gdrive(DATA_FILE_ID, "data.csv")
df = pd.read_csv(data_path)

# 🔎 Preprocessing Function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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
st.write(df.head())  # Show first rows of dataset

# 📝 User Input
user_input = st.text_area("Enter a news headline or article:")

if st.button("Check News"):
    cleaned_input = clean_text(user_input)  # Clean input text
    input_vector = vectorizer.transform([cleaned_input])  # Convert text to TF-IDF
    
    prediction = model.predict(input_vector)[0]  # Predict
    
    # 🎯 Show result
    st.write("### Prediction:")
    st.success("✅ Real News") if prediction == 1 else st.error("🚨 Fake News")
