import streamlit as st
import pandas as pd
import os
import pickle

from streamlit_option_menu import option_menu

# untuk menggunakan fungsi yang ada di processing.py
from utils.processing import preprocess_comments, clean_text

# untuk menggunakan fungsi yang ada di loader.py
from utils.loader import load_alay_dictionary, load_stopwords

from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# --- LOAD RESOURCE ---
# kamus alay
alay_dict = load_alay_dictionary(
    kamusalay_filepath = os.path.join("data", "kamusalay2.csv"),
    additional_filepath = os.path.join("data", "additional_dict-alay.csv")
)

# stopwords
stopwords = load_stopwords(
    stopwords_filepath= os.path.join("data", "stopwordbahasa.csv"),
    additional_stopwords=["lah", "nya", "kalau", "the", "of", "and", "i", "aku", "gue", "kak",
    "kamu", "a", "to", "ku", "rela", "kakak", "eh", "for", "did", "is", "ah", "cui", "nge"],  # extend manual
    excluded_stopwords=["tidak", "kok", "serta", "peserta", "harus", "lagi", "dong", "doang", 
    "tolong", "kenapa", "apa", "kapan", "bagaimana", "dimana", "berapa", "tahun", "ada", "mana", 
    "siapa", "terus", "penerus", "gitu", "begitu", "gini", "begini","bisa", "dapat", "ingin", 
    "mungkin", "jadi", "atur", "pengaturan", "sudah", "udah","diri", "sendiri", "memang", "agak", 
    "sedikit", "kurang", "boleh", "juga", "kembali", "balik", "soal", "ya","sudah", "ingin", 
    "tanya", "saja", "pada", "ayo", "keluar", "lalu", "tiap","hari", "bulan", "kalau", "kalian", 
    "masih", "kira", "masalah", "sekarang", "belum", "pasti", "sebelum", "sesudah", "terlalu", 
    "lebih", "tangis", "pernah", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", 
    "delapan", "sembilan", "sepuluh", "sebelas", "season", "lain"]   # exclude
)

# --- SIDEBAR NAVIGATION dengan option_menu ---
with st.sidebar:
    selected = option_menu(
        "Menu", ["Home", "Klasifikasi Sentimen", "Pemodelan Topik"],
        icons=['house', 'clipboard-check', 'files'],
        menu_icon="cast", default_index=0, 
        styles={
            "container": {
                "padding": "5px", 
                "background-color": "#1E1E1E",  # sidebar background
            },
            "icon": {
                "color": "#C0C0C0",  # soft grey icon
                "font-size": "18px"
            }, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px",
                "font-family": "Helvetica",
                "color": "#CCCCCC",  # font abu muda
                "background-color": "#2C2C2C",  # tombol background
                "border-radius": "8px"
            },
            "nav-link-hover": {
                "background-color": "#3E3E3E",  # hover background lebih terang sedikit
                "color": "#FFFFFF",
            },
            "nav-link-selected": {
                "background-color": "#0E76A8",  # warna biru saat aktif
                "font-size": "16px",
                "font-family": "Helvetica",
                "color": "white",
                "border-radius": "8px",
            }
        }
    )

# --- HALAMAN ---
if selected == "Home":
    # halaman dashboard home
    st.title("📊 Dashboard Sentimen Komentar")
    # konten
    data_path = os.path.join("data", "dataset_penelitian.xlsx")
    if os.path.exists(data_path):
        df = pd.read_excel(data_path)

        # --- Pra-pemrosesan awal ---
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        df["label_text"] = df["label"].map({0: "Negatif", 1: "Netral", 2: "Positif"})

        # --- 1. Statistik Umum ---
        st.subheader("📌 Statistik Umum")
        total_komentar = df["comment"].count()
        total_user = df["username"].nunique()

        col1, col2 = st.columns(2)
        col1.metric("Total Komentar", total_komentar)
        col2.metric("Total Pengguna", total_user)

        # --- 2. Proporsi Sentimen ---
        st.subheader("📊 Distribusi Sentimen")
        label_counts = df["label_text"].value_counts().reindex(["Positif", "Netral", "Negatif"])
        st.plotly_chart(px.pie(values=label_counts.values, names=label_counts.index, title="Proporsi Sentimen"))

        # --- 3. Tren Komentar per Hari ---
        st.subheader("📈 Tren Komentar per Hari")
        trend_df = df.groupby([df["publishedAt"].dt.date, "label_text"]).size().reset_index(name="count")
        fig_trend = px.line(trend_df, x="publishedAt", y="count", color="label_text", markers=True,
                            labels={"publishedAt": "Tanggal", "count": "Jumlah Komentar", "label_text": "Sentimen"},
                            title="Tren Jumlah Komentar Harian per Sentimen")
        st.plotly_chart(fig_trend)

        # --- 4. Komentar dengan Like Terbanyak ---
        st.subheader("👍 Komentar Paling Disukai")
        top_liked = df.sort_values(by="likeCount", ascending=False).head(5)[["username", "comment", "likeCount"]]
        for idx, row in top_liked.iterrows():
            st.markdown(f"""
            **{row['username']}** ❤️ {row['likeCount']} likes  
            _"{row['comment']}"_
            """)

        # --- 5. Wordcloud per Sentimen ---
        st.subheader("☁️ Wordcloud Kata Umum per Sentimen")

        cleaned_data_path = os.path.join("data", "after_preprocessing.xlsx")
        df_clean = pd.read_excel(cleaned_data_path)

        # Mapping label angka ke teks
        label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}
        df_clean["label_text"] = df_clean["label"].map(label_mapping)

        for label_name in ["Positif", "Netral", "Negatif"]:
            text = " ".join(df_clean[df_clean["label_text"] == label_name]["after"].astype(str).tolist())
            wordcloud = WordCloud(width=800, height=300, background_color="black").generate(text)

            st.markdown(f"**{label_name}**")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    else:
        st.error(f"File {data_path} belum ditemukan di folder data.")

elif selected == "Klasifikasi Sentimen":
    # halaman klasifikasi sentimen
    st.title("📝 Klasifikasi Sentimen Komentar")
    # konten
    uploaded_file = st.file_uploader("Upload file komentar (format Excel)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("📄 Data yang diunggah:")
        st.dataframe(df.head())

        if st.button("🔍 Jalankan Klasifikasi"):
            with st.spinner("Sedang memproses..."):
                # preprocessing
                cleaned_df = preprocess_comments(df)
                # load model
                with open("model/sentiment_model.pkl", "rb") as f:
                    model_pipeline = pickle.load(f)
                # klasifikasi
                cleaned_df["predicted_label"] = model_pipeline.predict(cleaned_df["cleaned_comment"])
                st.success("Klasifikasi selesai!")
                st.dataframe(cleaned_df[["comment", "predicted_label"]])

                # tombol unduh
                output_file = "uploads/hasil_klasifikasi.xlsx"
                cleaned_df.to_excel(output_file, index=False)
                with open(output_file, "rb") as f:
                    st.download_button("⬇️ Unduh Hasil Klasifikasi", f, file_name="hasil_klasifikasi.xlsx")

elif selected == "Pemodelan Topik":
    # halaman pemodelan topik
    st.title("📚 Hasil Pemodelan Topik (LDA)")
    # konten
    st.subheader("👍🏼 Hasil Pemodelan Topik Komentar Positif")
    html_path = os.path.join("data", "output_lda_pos.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning("File HTML hasil pemodelan topik tidak ditemukan.")

    st.subheader("👎🏼 Hasil Pemodelan Topik Komentar Negatif")
    html_path = os.path.join("data", "output_lda_neg.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning("File HTML hasil pemodelan topik tidak ditemukan.")
    