"""
Speech Emotion Recognition - Streamlit App (Runtime-Safe)

- No librosa dependency (uses TensorFlow + SciPy for spectrograms)
- Safe for Streamlit Cloud / Lambda Labs where librosa wheels may be missing
- Expects trained models: cnn_model.h5, cnn_lstm_model.h5, yamnet_model.h5
"""

from __future__ import annotations

import io
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import tensorflow as tf
from scipy import signal


# -----------------------------------------------------
# App Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main-header {font-size: 2.4rem; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
      .sub-header {font-size: 1.3rem; color: #ff7f0e; margin-top: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------
# Constants
# -----------------------------------------------------
EMOTIONS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]
EMOTION_COLORS = {
    "Neutral": "#95a5a6",
    "Calm": "#3498db",
    "Happy": "#f39c12",
    "Sad": "#34495e",
    "Angry": "#e74c3c",
    "Fearful": "#9b59b6",
    "Disgust": "#16a085",
    "Surprised": "#e67e22",
}
EMOTION_EMOJI = {
    "Neutral": "😐",
    "Calm": "😌",
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fearful": "😨",
    "Disgust": "🤢",
    "Surprised": "😲",
}
TARGET_SR = 16000
TARGET_DURATION = 3  # seconds
EXPECTED_FRAMES = 94  # ~3s at hop=512


# -----------------------------------------------------
# Model Loading
# -----------------------------------------------------
@st.cache_resource
def load_models() -> Dict[str, tf.keras.Model]:
    models: Dict[str, tf.keras.Model] = {}
    model_paths = {
        "CNN": "cnn_model.h5",
        "CNN-LSTM": "cnn_lstm_model.h5",
        "YAMNet": "yamnet_model.h5",
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
                st.sidebar.success(f"✓ {name} loaded")
            except Exception as exc:  # noqa: PERF203
                st.sidebar.error(f"Failed to load {name}: {exc}")
        else:
            st.sidebar.warning(f"⚠ {name} not found at {path}")

    if not models:
        st.sidebar.error("No models found. Place .h5 files in the working directory.")
    return models


# -----------------------------------------------------
# Audio Utilities (no librosa)
# -----------------------------------------------------
def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    return signal.resample(audio, target_len)


def _fix_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmax: int = 8000,
) -> np.ndarray:
    """Compute log-mel spectrogram using TensorFlow (no librosa).

    Returns array shaped (n_mels, time_frames).
    """
    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    stft = tf.signal.stft(audio_tf, frame_length=n_fft, frame_step=hop_length)
    magnitude = tf.abs(stft)
    power = tf.square(magnitude)

    num_spectrogram_bins = power.shape[-1]
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sr,
        lower_edge_hertz=0,
        upper_edge_hertz=fmax,
    )

    mel_spec = tf.tensordot(power, mel_matrix, axes=1)
    mel_spec.set_shape(power.shape[:-1].concatenate(mel_matrix.shape[-1:]))
    mel_spec = tf.math.log(mel_spec + 1e-9)
    mel_np = mel_spec.numpy()

    # Min-max normalize per sample
    mel_np = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-9)

    # Transpose to (n_mels, frames)
    mel_np = mel_np.T

    # Ensure fixed frame count
    if mel_np.shape[1] > EXPECTED_FRAMES:
        mel_np = mel_np[:, :EXPECTED_FRAMES]
    elif mel_np.shape[1] < EXPECTED_FRAMES:
        pad = EXPECTED_FRAMES - mel_np.shape[1]
        mel_np = np.pad(mel_np, ((0, 0), (0, pad)), mode="edge")

    return mel_np.astype(np.float32)


def preprocess_audio(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load bytes, resample, pad/truncate, and produce model-ready mel spec."""
    audio, sr = sf.read(io.BytesIO(file_bytes))
    audio = _to_mono(np.asarray(audio, dtype=np.float32))
    audio = _resample(audio, sr, TARGET_SR)
    audio = _fix_length(audio, TARGET_SR * TARGET_DURATION)

    mel = compute_mel_spectrogram(audio, sr=TARGET_SR)
    mel_input = mel[np.newaxis, ..., np.newaxis]  # (1, 128, 94, 1)

    return mel_input, audio, TARGET_SR


# -----------------------------------------------------
# Prediction + Visualization
# -----------------------------------------------------
def predict_emotion(models: Dict[str, tf.keras.Model], mel_spec: np.ndarray):
    predictions = {}
    for name, model in models.items():
        try:
            probs = model.predict(mel_spec, verbose=0)[0]
            idx = int(np.argmax(probs))
            predictions[name] = {
                "emotion": EMOTIONS[idx],
                "confidence": float(probs[idx] * 100),
                "probabilities": probs,
            }
        except Exception as exc:  # noqa: PERF203
            st.warning(f"Prediction failed for {name}: {exc}")
    return predictions


def plot_waveform(audio: np.ndarray, sr: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio, color="#1f77b4", linewidth=0.7)
    ax.fill_between(t, audio, alpha=0.3, color="#1f77b4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mel(mel_spec: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(mel_spec, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def plot_probs(predictions: Dict[str, dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(predictions), figsize=(14, 4))
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, predictions.items()):
        probs = pred["probabilities"]
        colors = [EMOTION_COLORS[e] for e in EMOTIONS]
        ax.bar(range(len(EMOTIONS)), probs, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}: {pred['emotion']}")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def summary_table(predictions: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, pred in predictions.items():
        probs = pred["probabilities"]
        top2 = np.argsort(probs)[-2:][::-1]
        rows.append(
            {
                "Model": name,
                "Prediction": f"{EMOTION_EMOJI[pred['emotion']]} {pred['emotion']}",
                "Confidence": f"{pred['confidence']:.2f}%",
                "Top-2": EMOTIONS[top2[1]],
                "Top-2 Conf": f"{probs[top2[1]] * 100:.2f}%",
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">🎤 Speech Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem; color: #555;'>Upload audio to detect emotions using your trained models.</p>",
        unsafe_allow_html=True,
    )

    # Sidebar - models
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.subheader("Models")
    models = load_models()
    if not models:
        st.stop()

    selected_models = [m for m in models.keys() if st.sidebar.checkbox(f"Use {m}", value=True)]
    if not selected_models:
        st.warning("Select at least one model to proceed.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.info("Expected files: cnn_model.h5, cnn_lstm_model.h5, yamnet_model.h5")

    # Main input
    st.markdown('<h2 class="sub-header">📂 Input Audio</h2>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an audio file (WAV/MP3/OGG/FLAC)", type=["wav", "mp3", "ogg", "flac"])

    if uploaded is None:
        st.info("Upload a 3-second clip to see predictions.")
        st.stop()

    file_bytes = uploaded.read()
    st.audio(file_bytes, format="audio/wav")

    with st.spinner("Processing audio..."):
        mel_input, audio, sr = preprocess_audio(file_bytes)
        selected = {k: v for k, v in models.items() if k in selected_models}
        predictions = predict_emotion(selected, mel_input)

    if not predictions:
        st.error("No predictions available. Check models and input.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Waveform", "Mel-Spectrogram", "Probabilities"])

    with tab1:
        cols = st.columns(len(predictions))
        for col, (name, pred) in zip(cols, predictions.items()):
            emotion = pred["emotion"]
            col.markdown(
                f"""
                <div style='background:{EMOTION_COLORS[emotion]}; padding:18px; border-radius:10px; color:white; text-align:center;'>
                    <h3>{name}</h3>
                    <h1>{EMOTION_EMOJI[emotion]}</h1>
                    <h2>{emotion}</h2>
                    <p>{pred['confidence']:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.subheader("Details")
        st.dataframe(summary_table(predictions), use_container_width=True)

    with tab2:
        st.pyplot(plot_waveform(audio, sr))

    with tab3:
        st.pyplot(plot_mel(mel_input[0, :, :, 0]))

    with tab4:
        st.pyplot(plot_probs(predictions))

    st.markdown("---")
    st.caption("Speech Emotion Recognition • TensorFlow • Streamlit • No librosa dependency")


if __name__ == "__main__":
    main()
