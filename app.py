import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import re

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Analisis Sentimen TensorFlow",
    layout="centered"
)

st.title("📊 Analisis Sentimen Review")
st.caption("TensorFlow + Streamlit | Negatif • Netral • Positif")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Dataset_Scrapping_raw.csv")

df = load_data()

st.subheader("📁 Dataset")
st.dataframe(df.head())

# =========================
# LABELING SENTIMEN
# =========================
def label_sentiment(rating):
    if rating <= 2:
        return 0   # negatif
    elif rating == 3:
        return 1   # netral
    else:
        return 2   # positif

df["label"] = df["rating"].apply(label_sentiment)

# =========================
# PREPROCESSING TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_text"] = df["content"].apply(clean_text)

# =========================
# PILIH SKEMA
# =========================
st.subheader("⚙️ Skema Pelatihan")

skema = st.selectbox(
    "Pilih Skema Training",
    [
        "Skema 1: Dense NN + TF-IDF (80/20)",
        "Skema 2: LSTM + Embedding (80/20)",
        "Skema 3: Dense NN + TF-IDF (70/30)"
    ]
)

test_size = 0.2 if "80/20" in skema else 0.3

# =========================
# SPLIT DATA
# =========================
X = df["clean_text"].values
y = df["label"].values

split_index = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# =========================
# TEXT VECTORIZATION
# =========================
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_sequence_length=100
)
vectorizer.adapt(X_train)

# =========================
# MODEL
# =========================
def build_dense_model():
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(5000, 128),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    return model

def build_lstm_model():
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(5000, 128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    return model

# =========================
# TRAINING
# =========================
if st.button("🚀 Jalankan Training"):
    if "LSTM" in skema:
        model = build_lstm_model()
    else:
        model = build_dense_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    st.success("✅ Training selesai")
    st.write(f"🎯 Akurasi Training: **{train_acc:.4f}**")
    st.write(f"🧪 Akurasi Testing: **{val_acc:.4f}**")

    st.session_state["model"] = model

# =========================
# INFERENCE
# =========================
st.subheader("🧠 Inference Sentimen")

input_text = st.text_area("Masukkan teks ulasan:")

if st.button("🔍 Prediksi"):
    if "model" not in st.session_state:
        st.warning("⚠️ Jalankan training terlebih dahulu")
    else:
        cleaned = clean_text(input_text)
        pred = st.session_state["model"].predict([cleaned])
        label = np.argmax(pred)

        label_map = {
            0: "NEGATIF",
            1: "NETRAL",
            2: "POSITIF"
        }

        st.success(f"📌 Hasil Sentimen: **{label_map[label]}**")
