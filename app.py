import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Chatbot Layanan Publik",
    page_icon="🤖",
    layout="centered"
)

# =========================
# DATASET SEDERHANA
# =========================
data = {
    "text": [
        "cara membuat ktp",
        "syarat pembuatan ktp",
        "daftar bpjs",
        "cara daftar bpjs kesehatan",
        "cek bantuan sosial",
        "syarat bansos",
        "jam buka kelurahan",
        "kantor buka jam berapa"
    ],
    "intent": [
        "ktp",
        "ktp",
        "bpjs",
        "bpjs",
        "bansos",
        "bansos",
        "jam_kantor",
        "jam_kantor"
    ]
}

df = pd.DataFrame(data)

# =========================
# MODEL NLP
# =========================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["intent"]

model = MultinomialNB()
model.fit(X, y)

def predict_intent(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# =========================
# RESPON CHATBOT
# =========================
responses = {
    "ktp": "📄 Untuk membuat KTP, silakan datang ke Disdukcapil dengan membawa KK.",
    "bpjs": "🏥 Pendaftaran BPJS bisa dilakukan melalui aplikasi Mobile JKN.",
    "bansos": "🆘 Bantuan sosial diberikan kepada warga yang terdaftar di DTKS.",
    "jam_kantor": "⏰ Kantor pelayanan buka Senin–Jumat pukul 08.00–15.00"
}

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("📌 Menu")

menu = st.sidebar.radio(
    "Pilih Halaman",
    ["🤖 Chatbot", "ℹ️ Tentang", "🧾 Layanan"]
)

# =========================
# HALAMAN CHATBOT
# =========================
if menu == "🤖 Chatbot":
    st.title("🤖 Chatbot Layanan Publik")
    st.write("Tanyakan informasi seputar layanan publik di sini.")

    user_input = st.text_input("💬 Ketik pertanyaan:")

    if user_input:
        intent = predict_intent(user_input)
        st.success(responses[intent])

# =========================
# HALAMAN TENTANG
# =========================
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")

    st.write("""
    Aplikasi ini merupakan chatbot layanan publik berbasis NLP
    yang dapat membantu masyarakat mendapatkan informasi seperti
    pembuatan KTP, BPJS, bantuan sosial, dan jam operasional kantor.
    """)

# =========================
# HALAMAN LAYANAN
# =========================
elif menu == "🧾 Layanan":
    st.title("🧾 Daftar Layanan")

    st.markdown("""
    - 📄 Informasi Pembuatan KTP  
    - 🏥 Informasi BPJS Kesehatan  
    - 🆘 Informasi Bantuan Sosial  
    - ⏰ Jam Operasional Kantor Pelayanan  
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("© 2026 Chatbot Layanan Publik | NLP + Streamlit")
