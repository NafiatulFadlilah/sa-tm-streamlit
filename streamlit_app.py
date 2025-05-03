import streamlit as st
import pandas as pd
import os
import joblib
import sklearn
import io

from streamlit_option_menu import option_menu

# untuk menggunakan fungsi yang ada di processing.py
from utils.processing import preprocess_comments, clean_text, preprocess_features, classify_comments

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
                # --- 1. Preprocessing Awal (cleaned_comment) ---
                cleaned_df = preprocess_comments(df, alay_dict, stopwords)

                BASE_DIR = os.path.dirname(os.path.abspath(__file__))

                # --- 2. Load Model dan Transformator ---
                tfidf = joblib.load(os.path.join(BASE_DIR, "model", "tfidf_vectorizer.joblib"))
                pca = joblib.load(os.path.join(BASE_DIR, "model", "pca_transformer.joblib"))
                model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.joblib"))

                # --- 3. Preprocessing Lanjutan (TF-IDF + PCA) ---
                features = preprocess_features(cleaned_df["cleaned_comment"], tfidf, pca)

                # --- 4. Klasifikasi ---
                predictions = classify_comments(features, model)
                cleaned_df["predicted_label"] = predictions

                # --- 5. Mapping Label Angka ke Teks (Opsional) ---
                label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
                cleaned_df["label_text"] = cleaned_df["predicted_label"].map(label_map)

                st.success("Klasifikasi selesai!")
                st.dataframe(cleaned_df[["comment", "cleaned_comment", "label_text"]])

                # --- 6. Tombol Unduh ---
                output_buffer = io.BytesIO()
                cleaned_df.to_excel(output_buffer, index=False, engine='openpyxl')
                output_buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Unduh Hasil Klasifikasi",
                    data=output_buffer,
                    file_name="hasil_klasifikasi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
elif selected == "Pemodelan Topik":
    # halaman pemodelan topik
    st.title("📚 Hasil Pemodelan Topik (LDA)")
    # konten
    st.subheader("👍🏼 Hasil Pemodelan Topik Komentar Positif")
    html_path = os.path.join("topic_modeling", "output_lda_pos.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning("File HTML hasil pemodelan topik tidak ditemukan.")
    
    st.subheader("📝 Interpretasi Topik Positif")
    
    st.markdown("""
    **Topik 1: Dukungan terhadap Program & Harapan Lanjutan**  
    > Penonton sangat mengapresiasi acara ini karena memberikan inspirasi dan motivasi belajar, terutama kepada anak-anak Indonesia. 
      Mereka merasa bahwa guru dan pengajaran dalam program ini menarik.  
      Ada harapan agar acara seperti ini terus berlanjut, dan penonton menantikan episode selanjutnya.  
      Sentimen positif juga muncul dari rasa terima kasih atas tayangan ini yang menginspirasi untuk belajar lebih giat.

    ---

    **Topik 2: Apresiasi Umum & Semangat Kompetisi**  
    > Penonton menunjukkan apresiasi tinggi terhadap para peserta, khususnya Kadit dan tim lainnya, karena penampilan mereka yang dianggap keren, hebat, dan membanggakan. 
      Ada nuansa emosional juga, seperti merinding dan sedih saat eliminasi, namun tetap memberi semangat dan dukungan. 
      Penonton merasa terinspirasi oleh semangat peserta yang tetap semangat dan terus berjuang.
                
    ---

    **Topik 3: Dukungan untuk Peserta & Atmosfer Persaingan**  
    > Fokus penonton tertuju pada dukungan personal terhadap peserta seperti Shakira, Axel, dan Sandy, yang mereka anggap tampil luar biasa. 
      Ada semangat kompetisi yang sportif, terlihat dari dukungan untuk yang menang maupun yang kalah. 
      Penonton memuji mental yang kuat dan sikap tenang para peserta, serta merasa puas dengan hasil pertandingan yang jujur dan fair. 
      Atmosfer kompetitif yang positif menjadi daya tarik tersendiri bagi penonton.
    """)

    st.subheader("👎🏼 Hasil Pemodelan Topik Komentar Negatif")
    html_path = os.path.join("topic_modeling", "output_lda_neg.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning("File HTML hasil pemodelan topik tidak ditemukan.")
    
    st.subheader("📝 Interpretasi Topik Negatif dan Rekomendasi Perbaikannya")

    st.markdown("""
    **Topik 1: Kritik Penilaian dan Performa**
    > Ada ketidakpuasan pada soal/penilaian, atau menganggap hasil eliminasi tidak adil.   
    > Rekomendasi: Pastikan sistem penilaian transparan dan bisa dipahami oleh penonton umum.
    
    ---
    
    **Topik 2: Spoiler & Kurangnya Kepuasan**
    > Penonton merasa terganggu dengan spoiler (mungkin dari komentar atau media sosial), konten kurang greget, atau terlalu banyak iklan.   
    > Rekomendasi: Batasi spoiler di komentar dengan filter otomatis, dan perhatikan proporsi konten dibanding iklan.

    ---

    **Topik 3: Keluhan terhadap Durasi & Jadwal Tayang**
    > Penonton merasa kesal karena episode baru keluar terlalu lama, durasinya tanggung, dan menunggu terlalu lama setiap minggunya.   
    > Rekomendasi: Pertimbangkan mempercepat jadwal rilis atau memberikan pengumuman jadwal tayang yang jelas agar ekspektasi penonton terkelola. 
    
    ---
                
    **Topik 4: Kekecewaan terhadap Keputusan & Peserta**
    > Penonton kecewa pada keputusan juri atau produser, dan perasaan sayang terhadap peserta yang tidak lolos.   
    > Rekomendasi: Berikan ruang kepada penonton untuk menyampaikan dukungan (misalnya vote atau polling), dan beri apresiasi kepada peserta yang gugur.
    
    ---
    
    **Topik 5: Ketidakterimaan Hasil Pertandingan**
    > Banyak yang tidak puas dengan hasil (terutama kekalahan peserta favorit), atau merasa peserta yang menang terlalu sombong.  
    > Rekomendasi: Berikan penjelasan hasil pertandingan secara adil dan highlight sportivitas.

    ---
    
    **Topik 6: Keluhan Teknis & Alur Acara**
    > Beberapa penonton merasa terganggu dengan teknis seperti video tidak bisa diunduh, gantung ceritanya, dan acara terasa "tidak teratur".   
    > Rekomendasi: Perhatikan teknis penyajian (seperti kualitas video & alur cerita), dan pertimbangkan menyediakan recap atau versi ringkas.
                
    ---
    
    **Topik 7: Kritik Terhadap Peserta**
    > Audiens mengkritik sikap beberapa peserta yang dianggap sombong. Terdapat kekecewaan terhadap peserta yang tidak memenuhi harapan.
    Juga ada pertanyaan tentang kelayakan beberapa peserta untuk menang atau bertahan dalam kompetisi.   
    > Rekomendasi: Apresiasi semua peserta secara merata.
    
    ---

    **Topik 8: Drama & Emosi Penonton**
    > Ada reaksi emosional terhadap peserta tertentu (seperti Kadit dan Xaviera), termasuk rasa sedih atau curiga terhadap eliminasi.   
    > Rekomendasi: Mungkin bisa lebih transparan dalam sistem eliminasi atau memberi highlight di media sosial terhadap perjuangan peserta favorit.               

    """)