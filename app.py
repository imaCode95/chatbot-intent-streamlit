import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ======================
# DATASET
# ======================
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

# ======================
# MODEL NLP
# ======================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["intent"]

model = MultinomialNB()
model.fit(X, y)

def predict_intent(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# ======================
# RESPON
# ======================
responses = {
    "ktp": "📄 Untuk membuat KTP, silakan datang ke Disdukcapil dengan KK.",
    "bpjs": "🏥 Pendaftaran BPJS bisa melalui aplikasi Mobile JKN.",
    "bansos": "🆘 Bansos diberikan kepada warga yang terdaftar di DTKS.",
    "jam_kantor": "⏰ Kantor pelayanan buka Senin–Jumat pukul 08.00–15.00"
}

# ======================
# STREAMLIT
# ======================
st.title("🤖 Chatbot Layanan Publik")

user_input = st.text_input("Tanyakan layanan publik:")

if user_input:
    intent = predict_intent(user_input)
    st.success(responses[intent])
