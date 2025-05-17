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
    > Penonton sangat mengapresiasi acara Clash of Champions (COC) karena memberikan inspirasi dan motivasi belajar terutama kepada anak-anak Indonesia. 
      Penonton mengapresiasi bahwa program produksi Ruangguru ini menarik. Terdapat harapan agar program seperti COC terus berlanjut dimana penonton sangat menantikan episode selanjutnya. 
      Sentimen positif juga muncul dari rasa terima kasih atas tayangan ini yang menginspirasi untuk belajar lebih giat.

    ---

    **Topik 2: Apresiasi Umum & Semangat Kompetisi**  
    > Penonton menunjukkan apresiasi tinggi terhadap para peserta, khususnya peserta top 3 dan tim lainnya, karena penampilan mereka yang dianggap keren, hebat, dan membanggakan. 
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
    **Topik 1: Eliminasi Peserta Potensial Terlalu Dini**
    > Beberapa audiens menyatakan kekecewaan ketika dua peserta potensial harus saling berhadapan dalam elimination challenge episode 6. 
      Mereka berharap peserta unggulan bisa bertemu di babak akhir, bukan tersingkir lebih awal, karena hal tersebut dianggap mengurangi ketegangan dan daya tarik kompetisi di episode berikutnya.
                 
    > **Rekomendasi**: Sistem penentuan lawan dalam elimination challenge yang dilakukan secara sukarela dan acak melalui pengambilan bola dalam fishbowl sudah mencerminkan keadilan dan transparansi. 
      Oleh karena itu, penting untuk memperkuat komunikasi bahwa mekanisme ini bersifat adil dan tidak direkayasa, agar audiens memahami bahwa pertemuan antar peserta kuat di awal adalah hasil dari sistem yang objektif, bukan keputusan sepihak. 
      Selain itu, untuk menjaga ekspektasi audiens, dapat dipertimbangkan adanya penekanan sejak awal bahwa setiap peserta memiliki peluang yang sama untuk bertemu siapa pun di babak elimination, serta memperjelas bahwa acara ini tidak mengutamakan drama, tetapi murni menguji kemampuan peserta secara kompetitif. 
    
    ---
    
    **Topik 2: Keluhan terhadap Spoiler, Iklan, dan Revival**
    > Penonton merasa terganggu dengan banyaknya spoiler, iklan, dan sistem revival yang mengurangi ketegangan dan kejutan acara. 
      Hal ini membuat beberapa dari mereka menjadi malas menonton.  
                 
    > **Rekomendasi**: Sosialisasi lebih masif mengenai perbedaan jadwal tayang antara aplikasi Ruangguru dan YouTube dapat membantu mengurangi keluhan spoiler. 
      Audiens yang tidak ingin terkena spoiler dapat diarahkan dengan sopan untuk menonton versi lebih awal di aplikasi. 
      Untuk iklan, pendekatan integrasi yang lebih halus di dalam konten bisa menjadi alternatif. Revival sudah menjadi bagian dari desain acara, sehingga penting untuk dijelaskan secara transparan sejak awal agar tidak dianggap “plot twist” yang membingungkan.

    ---

    **Topik 3: Penayangan Gantung, Durasi, dan Subtitle**
    > Audiens mengeluhkan lamanya waktu tunggu antar episode, durasi yang singkat dan menggantung, serta tidak adanya subtitle untuk penonton non-Indonesia yang ikut menonton.  
                
    > **Rekomendasi**: Ending menggantung merupakan formula umum di serial kompetisi untuk menjaga ketertarikan audiens. 
      Namun, agar tetap adil, Ruangguru bisa menambahkan pengantar atau cuplikan kecil episode berikutnya untuk memberikan sedikit konteks sebagai "penyeimbang" bagi penonton yang kurang menyukai ketegangan berlebih. 
      Mengenai subtitle, jika permintaan semakin tinggi dari audiens internasional, penambahan subtitle Bahasa Inggris secara bertahap bisa menjadi opsi ekspansi audiens yang strategis.
    
    ---
                
    **Topik 4: Kualitas Video dan Penjelasan Soal**
    > Penonton meminta agar penjelasan soal tidak dipotong atau dipercepat. Mereka juga mengeluhkan video yang patah-patah dan kualitas tayangan yang menurun.   
                
    > **Rekomendasi**: Karena soal dapat diakses melalui fitur Adapto di aplikasi resmi penyelenggara, maka perlu dipertimbangkan agar penjelasan cara mengerjakan soal juga tersedia di fitur tersebut dalam bentuk video. 
      Selain itu, penjelasan soal dapat diunggah sebagai klip atau video terpisah di media sosial resmi, serta diperkuat penyampaiannya melalui tayangan episode atau caption episode, agar penonton mengetahui bahwa pembahasan soal tidak hanya tersedia di dalam tayangan. 
      Untuk masalah kualitas video yang patah-patah, perlu evaluasi teknis dan konsistensi output produksi agar pengalaman menonton tetap nyaman, apalagi untuk penonton yang ingin belajar dari tayangan tersebut.
    
    ---
    
    **Topik 5: Perilaku Peserta yang Dinilai Negatif**
    > Audiens memberikan komentar negatif terhadap perilaku beberapa peserta yang dinilai sombong atau kurang sportif, serta mempertanyakan keputusan pertandingan yang tidak sesuai ekspektasi mereka.
                  
    > **Rekomendasi**: Karakter manusiawi peserta bisa menjadi daya tarik selama tidak melampaui batas etika. Tetap penting untuk memberi pembekalan etika sejak awal, namun biarkan dinamika personal terlihat alami karena hal tersebut justru memberikan warna dalam kompetisi. 
      Audiens juga bisa diarahkan untuk melihat peserta sebagai individu utuh, bukan hanya berdasarkan potongan klip emosi sesaat.

    ---
    
    **Topik 6: Kendala Teknis dan Ending Menggantung**
    > Audiens kesulitan menonton ulang karena video tidak bisa diunduh, serta terganggu dengan video yang ngelag dan episode yang diakhiri secara tidak memuaskan (menggantung).
                   
    > **Rekomendasi**: Untuk aksesibilitas, dapat dipertimbangkan opsi streaming lebih ringan atau pengingat bahwa episode tersedia di aplikasi, agar penonton yang mengalami kendala jaringan bisa memilih alternatif yang sesuai. 
      Ending menggantung, seperti dibahas di Topik 3, sebenarnya efektif menjaga ketertarikan, namun komunikasi yang transparan mengenai jadwal tayang dan teaser episode mendatang bisa membantu meredam rasa frustrasi audiens.
                
    ---
    
    **Topik 7: Performa dalam Tim**
    > Beberapa peserta dinilai hanya menumpang di tim yang kuat oleh penonton. Penonton merasa ada peserta yang lebih layak untuk lanjut dibandingkan mereka yang "digendong" oleh rekan timnya.
                  
    > **Rekomendasi**: Transparansi sistem penilaian (misalnya, perolehan skor individual dalam tim) bisa ditampilkan secara eksplisit saat atau setelah pertandingan. Hal ini bisa menjawab keraguan penonton terhadap siapa yang berkontribusi lebih, tanpa perlu menilai secara subjektif dari interaksi visual semata. 
      Selain itu, screen time yang diberikan untuk setiap peserta dapat dilakukan secara adil supaya semua penonton dapat melihat performa masing-masing peserta secara objektif.
    
    ---

    **Topik 8: Asumsi dan Drama yang Merugikan**
    > Audiens membuat asumsi-asumsi yang tidak berdasar seperti hubungan keluarga antarpeserta dan menyebarkan drama yang bisa berdampak negatif bagi pihak yang terlibat dalam acara.
                 
    > **Rekomendasi**: Topik ini cenderung berasal dari persepsi atau spekulasi individu yang sulit dikontrol. 
      Meskipun demikian, pendekatan edukatif secara halus seperti menyisipkan pengingat bahwa kompetisi murni berdasarkan hasil, tanpa intervensi eksternal, bisa membantu menjaga suasana tetap positif.             

    """)