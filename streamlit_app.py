import os
import io
import ftplib
import certifi
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# ⚙️ LOAD CONFIGURATION FROM JSON
# ================================

FTP_HOST = "161.97.87.182"
FTP_PORT = 21
FTP_USER = "ftpuser4"
FTP_PASS = "EV9UE6B}n2!2"
FTP_PATH = "/api.shaista.top/Image"

# SSL certificate setup
os.environ['SSL_CERT_FILE'] = certifi.where()


@st.cache_resource
def load_model():
    """Load pre-trained ResNet50 model."""
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')


def ftp_list_images() -> list[str]:
    """Get list of .jpg images from FTP directory."""
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(FTP_HOST, FTP_PORT, timeout=10)
            ftp.login(FTP_USER, FTP_PASS)
            ftp.cwd(FTP_PATH)
            files = ftp.nlst()
            return [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    except Exception as e:
        st.error(f"FTP connection failed: {e}")
        return []


def ftp_download_image(filename: str) -> io.BytesIO:
    """Download a single image from FTP into memory."""
    bio = io.BytesIO()
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(FTP_HOST, FTP_PORT, timeout=10)
            ftp.login(FTP_USER, FTP_PASS)
            ftp.cwd(FTP_PATH)
            ftp.retrbinary(f"RETR {filename}", bio.write)
        bio.seek(0)
        return bio
    except Exception as e:
        st.error(f"Error downloading {filename}: {e}")
        return None


def extract_features(img_source, model):
    """Extract ResNet50 features from an image."""
    try:
        img = Image.open(img_source).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed).flatten()
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None


def find_similar_images(upload_img, model, abbr, threshold=0.90, top_n=2):
    """Compare uploaded image with filtered remote FTP images."""
    uploaded_features = extract_features(upload_img, model)
    if uploaded_features is None:
        return []

    all_images = ftp_list_images()
    if not all_images:
        st.warning("No images found on FTP.")
        return []

    abbr = abbr.lower()
    filtered_images = [f for f in all_images if f.lower().startswith(abbr)]
    if not filtered_images:
        st.warning(f"No images found starting with '{abbr}'.")
        return []

    results = []
    with st.spinner(f"Checking similarity among {len(filtered_images)} image(s)..."):
        for fname in filtered_images:
            img_data = ftp_download_image(fname)
            if img_data is None:
                continue

            db_features = extract_features(img_data, model)
            if db_features is None:
                continue

            sim = cosine_similarity([uploaded_features], [db_features])[0][0]
            if sim >= threshold:
                results.append((fname, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ================================
# 🚀 STREAMLIT UI
# ================================
def main():
    st.title("🔍 Find Similar Image With Tagno")
    st.write(
        "Upload an image and find visually similar images and Tag Number."
        "You **must enter an abbreviation** (e.g., 'bg') to search only those image names starting with it."
    )

    model = load_model()
    uploaded_img_file = st.file_uploader("📤 Upload image (.jpg, .png)", type=["jpg", "jpeg", "png"])
    abbr_filter = st.text_input("🔑 Abbreviation (required)", placeholder="e.g. bg").strip()
    threshold = st.slider("🎯 Similarity Threshold", 0.5, 1.0, 0.75, 0.01)

    if uploaded_img_file is not None:
        img = Image.open(uploaded_img_file)
        st.image(img.resize((300, 300)), caption="Uploaded Image")

        if st.button("Find Similar Images"):
            if not abbr_filter:
                st.error("❗ Please enter an abbreviation before searching.")
                return

            matches = find_similar_images(uploaded_img_file, model, abbr=abbr_filter, threshold=threshold, top_n=2)

            if matches:
                st.success(f"✅ Found {len(matches)} similar image(s).")
                cols = st.columns(len(matches))
                for i, (fname, sim) in enumerate(matches):
                    with cols[i]:
                        img_data = ftp_download_image(fname)
                        if img_data:
                            img_name = os.path.splitext(fname)[0]  # filename without extension
                            st.markdown(f"**📁 {img_name}**")
                            st.image(Image.open(img_data))
                            image_url = f"http://{FTP_PATH}/{fname}"
                            st.markdown(f"[🌐 View Full Image]({image_url})")
                            st.caption(f"Similarity: {sim:.2f}")
            else:
                st.warning("No similar image found.")


if __name__ == "__main__":
    main()
