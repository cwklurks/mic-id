import io, os, tempfile, joblib, numpy as np, pandas as pd, streamlit as st, librosa
from features import extract_features

st.set_page_config(page_title="Mic-ID (MVP)", layout="centered")
st.title("Mic-ID (MVP)")
st.caption("Upload ~5s audio → guess the recording device (demo)")


@st.cache_resource
def load_model():
    try:
        clf = joblib.load("models/model.pkl"); le = joblib.load("models/label_encoder.pkl")
        return clf, le
    except Exception:
        return None, None


clf, le = load_model()
file = st.file_uploader("Upload WAV/MP3/M4A", type=["wav","mp3","m4a"])

if file and clf:
    data = file.read()
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
    except Exception:
        suffix = os.path.splitext(file.name or "upload")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data); tmp.flush()
            y, sr = librosa.load(tmp.name, sr=16000, mono=True)
    rms = np.sqrt(np.mean(y**2)) + 1e-8; y = y * (0.05 / rms)  # simple RMS norm
    feats = extract_features(y, 16000).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]; idx = np.argsort(proba)[::-1]
    st.subheader("Prediction")
    for i in idx[:3]:
        st.write(f"{le.classes_[i]} — **{proba[i]*100:.1f}%**")
    st.bar_chart(pd.Series(proba, index=le.classes_))
elif file and not clf:
    st.warning("No trained model found. Run `python train.py` first.")
