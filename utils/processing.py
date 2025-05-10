import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib

# Inisialisasi Stemmer sekali saja
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_comments(df, alay_dict, stopwords):
    df = df.rename(columns=lambda x: x.lower())
    if "comment" not in df.columns:
        raise ValueError("Kolom 'comment' tidak ditemukan pada file yang diunggah.")
    
    df["cleaned_comment"] = df["comment"].apply(lambda x: clean_text(x, alay_dict, stopwords))
    return df

def clean_text(text, alay_dict, stopwords):
    text = str(text)
    
    # 1. Casefolding
    text = text.lower()
    
    # 2. Cleansing
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # Hapus URL
    text = re.sub(r'<.*?>', ' ', text)              # Hapus HTML tag
    text = re.sub(r'@\w+', ' ', text)               # Hapus mention
    text = re.sub(r'&quot;|&gt;|&lt;|&amp;', ' ', text)  # Hapus entitas HTML
    text = re.sub(r'&#39;', '', text)               # Hapus karakter spesial
    text = re.sub(r'[^0-9a-zA-Z\s]', ' ', text)      # Hapus tanda baca dan karakter aneh
    
    # 3. Normalization
    text = ' '.join([alay_dict.get(word, word) for word in text.split()])
    
    # 4. Removing Repetition Character
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 5. Tokenizing
    words = text.split()
    
    # 6. Stemming
    words = [stemmer.stem(word) for word in words]
    
    # 7. Stopword Removal
    words = [word for word in words if word not in stopwords]
    
    return ' '.join(words)

# Fungsi untuk memuat model TF-IDF, PCA, dan klasifikasi
def load_models(tfidf_path, pca_path, model_path):
    tfidf = joblib.load(tfidf_path)
    pca = joblib.load(pca_path)
    model = joblib.load(model_path)
    return tfidf, pca, model

# Fungsi preprocessing lanjutan: TF-IDF + PCA
def preprocess_features(text_series, tfidf, pca):
    tfidf_features = tfidf.transform(text_series)
    reduced_features = pca.transform(tfidf_features.toarray())
    return reduced_features

# Fungsi klasifikasi
def classify_comments(features, model):
    predictions = model.predict(features)
    return predictions
