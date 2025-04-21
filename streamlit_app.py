import streamlit as st
import pandas as pd
import os
import pickle
from utils.processing import preprocess_comments, clean_text
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📊 Sentiment Analysis App")
page = st.sidebar.radio("Navigasi", ["🏠 Home", "📝 Klasifikasi Sentimen", "📂 Pemodelan Topik"])

# --- PAGE: HOME / DASHBOARD ---
if page == "🏠 Home":
    st.title("📊 Dashboard Sentimen Komentar")

    data_path = Path("D:/Kuliah/SEMESTER 8/App/sa-tm-streamlit/data/dataset_penelitian.xlsx")
    if data_path.exists():
        df = pd.read_excel(data_path)

        # --- Pra-pemrosesan awal ---
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        df["label_text"] = df["label"].map({0: "Negatif", 1: "Netral", 2: "Positif"})

        # --- 1. Statistik Umum ---
        st.subheader("📌 Statistik Umum")
        total_komentar = df["comment"].count()
        total_user = df["username"].nunique()
        total_like = df["likeCount"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komentar", total_komentar)
        col2.metric("Total Pengguna", total_user)
        col3.metric("Total Like", total_like)

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

        # --- 4. Top 5 Pengguna Aktif ---
        st.subheader("👥 Top 5 Pengguna dengan Komentar Terbanyak")
        top_users = df["username"].value_counts().head(5)
        st.bar_chart(top_users)

        # --- 5. Komentar dengan Like Terbanyak ---
        st.subheader("👍 Komentar Paling Disukai")
        top_liked = df.sort_values(by="likeCount", ascending=False).head(5)[["username", "comment", "likeCount"]]
        for idx, row in top_liked.iterrows():
            st.markdown(f"""
            **{row['username']}** ❤️ {row['likeCount']} likes  
            _"{row['comment']}"_
            """)

        # --- 6. Wordcloud per Sentimen ---
        st.subheader("☁️ Wordcloud Kata Umum per Sentimen")

        for label_name in ["Positif", "Netral", "Negatif"]:
            text = " ".join(df[df["label_text"] == label_name]["comment"].astype(str).tolist())
            wordcloud = WordCloud(width=800, height=300, background_color="white").generate(text)

            st.markdown(f"**{label_name}**")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    else:
        st.warning("File dataset_penelitian.xlsx belum ditemukan di folder data.")


# --- PAGE: KLASIFIKASI SENTIMEN ---
elif page == "📝 Klasifikasi Sentimen":
    st.title("📝 Klasifikasi Sentimen Komentar")

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

# --- PAGE: PEMODELAN TOPIK ---
elif page == "📂 Pemodelan Topik":
    st.title("📂 Hasil Pemodelan Topik (LDA)")

    html_path = "topic_modeling/lda_viz.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning("File HTML hasil pemodelan topik tidak ditemukan.")

