<!-- # 🎈 SA-TM Streamlit App

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ``` -->
# 📑 SA-TM (Sentiment Analysis - Topic Modeling) Streamlit App

Sistem **SA-TM** adalah aplikasi berbasis Streamlit yang memungkinkan pengguna untuk melakukan **analisis sentimen** dan **pemodelan topik** terhadap komentar pengguna dari platform seperti YouTube. Aplikasi ini mendukung pengunggahan file Excel berisi komentar, melakukan preprocessing teks, klasifikasi sentimen (negatif, netral, positif), serta visualisasi topik menggunakan LDA dan pyLDAvis.

## 🔍 Fitur Utama

- **Dashboard Data**: Menampilkan ringkasan data komentar termasuk jumlah komentar, persebaran like, dan distribusi label sentimen.
- **Wordcloud per Sentimen**: Visualisasi kata-kata dominan untuk masing-masing sentimen.
- **Klasifikasi Sentimen**:
  - Input: File Excel komentar
  - Proses: Preprocessing lanjutan (TF-IDF + PCA)
  - Model: Multinomial Logistic Regression
  - Output: Prediksi sentimen dan opsi untuk mengunduh hasil
- **Pemodelan Topik**:
  - Visualisasi interaktif topik menggunakan pyLDAvis
  - Interpretasi manual setiap topik dalam bentuk deskriptif

## 📁 Struktur Direktori

📦SA-TM/ <br>
├── data/ <br>
│ └── additional_dict-alay.csv # File additional dictionary kata gaul <br>
│ └── after_preprocessing.xlsx # Data komentar hasil preprocessing awal <br>
│ └── dataset_penelitian.xlsx # Raw data komentar <br>
│ └── kamusalay2.csv # File kamus alay <br>
│ └── stopwordbahasa.csv # Daftar istilah kata umum <br>
├── model/ <br>
│ └── pca_transformer.joblib # PCA reducer <br>
│ └── sentiment_model.joblib # Trained sentiment classification model (Multinomial Logistic Regression) <br>
│ └── tfidf_vectorizer.joblib # TF-IDF vectorizer <br>
├── topic_modeling/ <br>
│ └── output_lda_neg.html # Hasil pemodelan topik negatif <br>
│ └── output_lda_pos.html # Hasil pemodelan topik positif <br>
├── utils/ <br>
│ └── loader.py # Modul untuk memuat kamus istilah yang ada di file csv <br>
│ └── processing.py # Modul preprocessing & klasifikasi <br>
├── requirements.txt # Modul berisi dependensi / pustaka yang dibutuhkan <br>
├── streamlit_app.py # File utama Streamlit <br>
└── README.md <br>

## 🛠 Cara Menjalankan Aplikasi di Lokal

1. **Clone repositori** dan masuk ke direktori:
   ```bash
   git clone https://github.com/NafiatulFadlilah/sa-tm-streamlit.git
   cd sa-tm-streamlit
   ```
2. Install dependensi
   ```
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi
   ```
   streamlit run streamlit_app.py
   ```

## 📤 Format Input File Excel

- File input harus memiliki kolom bernama comment (case-insensitive).
- Untuk wordcloud dan klasifikasi, file bisa berisi komentar mentah atau yang telah dipreprocessing tergantung fitur yang digunakan.

## 📦 Teknologi & Library
- Python
- Streamlit
- Scikit-learn
- Gensim
- Sastrawi
- Joblib
- pyLDAvis
- Pandas, Matplotlib, WordCloud

## 🔗 Akses Online
Jika tersedia versi online, bisa diakses melalui: <br>
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sa-tm-79v0n9cp9mu.streamlit.app/)