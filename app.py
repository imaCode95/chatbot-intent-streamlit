import streamlit as st
import pandas as pd
import sqlite3
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
# DATABASE SQLITE
# =========================
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================
# DATASET NLP
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
    "bpjs": "🏥 Pendaftaran BPJS bisa melalui aplikasi Mobile JKN.",
    "bansos": "🆘 Bantuan sosial diberikan kepada warga yang terdaftar di DTKS.",
    "jam_kantor": "⏰ Kantor pelayanan buka Senin–Jumat pukul 08.00–15.00"
}

# =========================
# FUNGSI LOGIN
# =========================
def register_user(username, password):
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )
    return c.fetchone() is not None

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Menu")

if st.session_state.logged_in:
    menu = st.sidebar.radio(
        "Pilih Halaman",
        ["🤖 Chatbot", "ℹ️ Tentang", "🧾 Layanan"]
    )

    st.sidebar.write(f"👤 Login sebagai **{st.session_state.username}**")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
else:
    menu = st.sidebar.radio(
        "Pilih Halaman",
        ["🔐 Login", "📝 Sign Up"]
    )

# =========================
# LOGIN & SIGN UP PAGE
# =========================
if not st.session_state.logged_in:

    if menu == "🔐 Login":
        st.title("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login berhasil")
                st.rerun()
            else:
                st.error("Username atau password salah")

    elif menu == "📝 Sign Up":
        st.title("📝 Sign Up")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Daftar"):
            if register_user(new_user, new_pass):
                st.success("Pendaftaran berhasil, silakan login")
            else:
                st.warning("Username sudah digunakan")

# =========================
# HALAMAN SETELAH LOGIN
# =========================
if st.session_state.logged_in:

    if menu == "🤖 Chatbot":
        st.title("🤖 Chatbot Layanan Publik")
        st.write("Silakan tanyakan informasi layanan publik.")

        user_input = st.text_input("💬 Ketik pertanyaan:")

        if user_input:
            intent = predict_intent(user_input)
            st.success(responses[intent])

    elif menu == "ℹ️ Tentang":
        st.title("ℹ️ Tentang Aplikasi")
        st.write("""
        Chatbot layanan publik berbasis NLP
        untuk membantu masyarakat mendapatkan
        informasi layanan secara cepat dan mudah.
        """)

    elif menu == "🧾 Layanan":
        st.title("🧾 Daftar Layanan")
        st.markdown("""
        - 📄 Pembuatan KTP  
        - 🏥 BPJS Kesehatan  
        - 🆘 Bantuan Sosial  
        - ⏰ Jam Operasional Kantor  
        """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("© 2026 Chatbot Layanan Publik | SQLite + NLP + Streamlit")
