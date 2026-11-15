import os
import sys
import time
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.append("src")
from preprocessing import get_class_names, train_dir, model_path, img_size
from model import load_trained_model, retrain
from prediction import predict

db_path = os.path.join("data", "uploads.db")


def init_db():
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS uploads ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "filename TEXT, label TEXT, created TEXT)"
    )
    con.commit()
    con.close()


def record_upload(filename, label):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO uploads (filename, label, created) VALUES (?, ?, ?)",
        (filename, label, datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()
    con.close()


def count_uploads():
    con = sqlite3.connect(db_path)
    n = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    con.close()
    return n


@st.cache_resource
def get_model():
    return load_trained_model()


@st.cache_data
def class_distribution():
    names = get_class_names()
    counts = {}
    for n in names:
        folder = os.path.join(train_dir, n)
        if os.path.isdir(folder):
            counts[n] = len(os.listdir(folder))
    return pd.Series(counts)


@st.cache_data
def color_and_brightness(sample=100):
    import tensorflow as tf
    names = get_class_names()
    rows = []
    for n in names:
        folder = os.path.join(train_dir, n)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder)[:sample]:
            img = tf.keras.utils.load_img(os.path.join(folder, f), target_size=img_size)
            arr = tf.keras.utils.img_to_array(img)
            rows.append((n, arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean(), arr.mean()))
    return pd.DataFrame(rows, columns=["class", "red", "green", "blue", "brightness"])


st.set_page_config(page_title="Maize Leaf Disease", layout="centered")
init_db()
if "start" not in st.session_state:
    st.session_state["start"] = time.time()

st.title("Maize Leaf Disease Classifier")
page = st.sidebar.radio("go to", ["Predict", "Insights", "Retrain", "Status"])


if page == "Predict":
    st.header("Predict")
    uploaded = st.file_uploader("upload a maize leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, width=260)
        if st.button("predict"):
            try:
                uploaded.seek(0)
                result = predict(get_model(), uploaded)
                st.success(f"{result['label']}   {result['confidence'] * 100:.1f}%")
                st.bar_chart(pd.Series(result["probabilities"]))
            except Exception as e:
                st.error(f"could not predict: {e}")



elif page == "Insights":
    st.header("Dataset insights")
    plots = [
        ("static/Image_per_class.png", "Images per class",
         "The classes are not balanced. Gray Leaf Spot has the fewest images, so we read macro F1 alongside accuracy."),
        ("static/RGB_colors.png", "Average colour per class",
         "Healthy leaves sit greener. Rust and Blight push red up and green down as the tissue browns."),
        ("static/Image_dimensions.png", "Image dimensions",
         "The raw images come in mixed resolutions, which is why every image is resized to 160x160 before training."),
    ]
    for path, title, caption in plots:
        st.subheader(title)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
            st.caption(caption)
        else:
            st.info(f"missing {path}")


elif page == "Retrain":
    st.header("Retrain")
    st.write("Upload new labelled images, then trigger retraining on the saved model.")

    label = st.selectbox("class", get_class_names())
    files = st.file_uploader("images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("save uploads"):
        if not files:
            st.warning("pick some images first")
        else:
            folder = os.path.join(train_dir, label)
            os.makedirs(folder, exist_ok=True)
            for f in files:
                with open(os.path.join(folder, f.name), "wb") as out:
                    out.write(f.getbuffer())
                record_upload(f.name, label)
            class_distribution.clear()
            color_and_brightness.clear()
            st.success(f"saved {len(files)} images into {label}")

    st.divider()
    epochs = st.slider("epochs", 1, 10, 3)
    if st.button("retrain model"):
        with st.spinner("retraining, this takes a moment"):
            hist = retrain(epochs=epochs)
        get_model.clear()
        st.success(f"done. validation accuracy {hist['val_accuracy'][-1]:.3f}")


elif page == "Status":
    st.header("Model status")

    st.metric("model file", "present" if os.path.exists(model_path) else "missing")
    if os.path.exists(model_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        st.write("last updated:", mtime.strftime("%Y-%m-%d %H:%M"))

    st.metric("uploaded images logged", count_uploads())
    st.metric("app uptime (min)", f"{(time.time() - st.session_state['start']) / 60:.1f}")