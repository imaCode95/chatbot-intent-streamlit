import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import LinearSVC
import tensorflow as tf
import tensorflow_decision_forests as tfdf

nltk.download("stopwords")
from nltk.corpus import stopwords

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Analisis Sentimen NLP", layout="wide")
st.title("📊 Analisis Sentimen Review (TF-IDF & TensorFlow)")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()
st.subheader("📁 Dataset")
st.dataframe(df.head())

# ===============================
# LABELING SENTIMEN
# ===============================
def label_sentiment(rating):
    if rating <= 2:
        return "negatif"
    elif rating == 3:
        return "netral"
    else:
        return "positif"

df["sentiment"] = df["rating"].apply(label_sentiment)

# ===============================
# PREPROCESSING
# ===============================
stop_words = stopwords.words("indonesian")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

df["clean_text"] = df["content"].apply(clean_text)

# ===============================
# ENCODE LABEL
# ===============================
le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])

# ===============================
# PILIH SKEMA
# ===============================
st.subheader("⚙️ Skema Pelatihan")

skema = st.selectbox(
    "Pilih Skema",
    [
        "SVM + TF-IDF + Split 80/20",
        "Random Forest (TF) + Embedding + Split 80/20",
        "SVM + TF-IDF + Split 70/30"
    ]
)

# ===============================
# TRAINING
# ===============================
if st.button("🚀 Jalankan Training"):
    if "80/20" in skema:
        test_size = 0.2
    else:
        test_size = 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    # ===========================
    # SKEMA 1 & 3 (SVM + TF-IDF)
    # ===========================
    if "TF-IDF" in skema:
        tfidf = TfidfVectorizer(max_features=5000)
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        model = LinearSVC()
        model.fit(X_train_vec, y_train)

        train_acc = model.score(X_train_vec, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test_vec))

        st.success("Training selesai!")
        st.write(f"🎯 Akurasi Training: **{train_acc:.4f}**")
        st.write(f"🧪 Akurasi Testing: **{test_acc:.4f}**")

        st.text("Classification Report")
        st.text(classification_report(y_test, model.predict(X_test_vec), target_names=le.classes_))

        st.session_state["model"] = model
        st.session_state["vectorizer"] = tfidf

    # ===========================
    # SKEMA 2 (RF TensorFlow)
    # ===========================
    else:
        def make_dataset(texts, labels):
            return tf.data.Dataset.from_tensor_slices((texts, labels)).batch(32)

        train_ds = make_dataset(X_train, y_train)
        test_ds = make_dataset(X_test, y_test)

        model = tfdf.keras.RandomForestModel(num_trees=200)
        model.compile(metrics=["accuracy"])
        model.fit(train_ds)

        train_eval = model.evaluate(train_ds, verbose=0)
        test_eval = model.evaluate(test_ds, verbose=0)

        st.success("Training selesai!")
        st.write(f"🎯 Akurasi Training: **{train_eval[1]:.4f}**")
        st.write(f"🧪 Akurasi Testing: **{test_eval[1]:.4f}**")

        st.session_state["model"] = model
        st.session_state["vectorizer"] = None

# ===============================
# INFERENCE
# ===============================
st.subheader("🧠 Inference Sentimen")

input_text = st.text_area("Masukkan teks ulasan:")

if st.button("🔍 Prediksi"):
    if "model" not in st.session_state:
        st.warning("⚠️ Jalankan training dulu")
    else:
        text = clean_text(input_text)

        if st.session_state["vectorizer"] is not None:
            vec = st.session_state["vectorizer"].transform([text])
            pred = st.session_state["model"].predict(vec)
        else:
            pred = st.session_state["model"].predict(pd.Series([text]))

        label = le.inverse_transform([pred[0]])[0]
        st.success(f"📌 Hasil Sentimen: **{label.upper()}**")
