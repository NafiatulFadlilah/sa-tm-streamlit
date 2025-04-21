import pandas as pd
import re

def preprocess_comments(df):
    df = df.rename(columns=lambda x: x.lower())
    if "comment" not in df.columns:
        raise ValueError("Kolom 'comment' tidak ditemukan pada file yang diunggah.")
    
    df["cleaned_comment"] = df["comment"].apply(clean_text)
    return df

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text
