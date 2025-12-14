"""Minimal Speech Emotion Recognition app (fresh rebuild).

- No librosa dependency; uses TensorFlow ops and SciPy resampling.
- Expects one or more .h5 Keras models in the working directory.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import tensorflow as tf
from scipy import signal


# ---------------------------
# Settings
# ---------------------------
TARGET_SR = 16_000
CLIP_DURATION = 3.0  # seconds
N_FFT = 1024
HOP = 256
N_MELS = 64
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


# ---------------------------
# Helpers
# ---------------------------
def list_model_files() -> List[str]:
    return [f for f in os.listdir(".") if f.lower().endswith(".h5")]


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return audio
    # Use polyphase for speed and quality.
    return signal.resample_poly(audio, TARGET_SR, orig_sr)


def _pad_or_trim(audio: np.ndarray) -> np.ndarray:
    target_len = int(TARGET_SR * CLIP_DURATION)
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode="constant")
    return audio[:target_len]


def compute_log_mel(audio: np.ndarray) -> np.ndarray:
    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    stft = tf.signal.stft(audio_tf, frame_length=N_FFT, frame_step=HOP)
    power = tf.square(tf.abs(stft))
    mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=power.shape[-1],
        sample_rate=TARGET_SR,
        lower_edge_hertz=0.0,
        upper_edge_hertz=TARGET_SR / 2,
    )
    mel_spec = tf.matmul(power, mel)
    log_mel = tf.math.log(mel_spec + 1e-9)
    log_mel = tf.transpose(log_mel)  # (mel, time)
    # Normalize per clip.
    log_mel = (log_mel - tf.reduce_min(log_mel)) / (tf.reduce_max(log_mel) - tf.reduce_min(log_mel) + 1e-9)
    return log_mel.numpy().astype(np.float32)


def preprocess(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    audio, sr = sf.read(io.BytesIO(file_bytes))
    audio = _to_mono(np.asarray(audio, dtype=np.float32))
    audio = _resample(audio, sr)
    audio = _pad_or_trim(audio)
    log_mel = compute_log_mel(audio)
    # Add batch and channel dims: (1, mel, time, 1)
    return log_mel[np.newaxis, ..., np.newaxis], audio


def predict(model: tf.keras.Model, mel: np.ndarray) -> np.ndarray:
    return model.predict(mel, verbose=0)[0]


def plot_wave(audio: np.ndarray) -> plt.Figure:
    t = np.linspace(0, len(audio) / TARGET_SR, len(audio))
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t, audio, color="#1f77b4", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mel(log_mel: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(log_mel, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_probs(probs: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.bar(range(len(EMOTIONS)), probs, color="#1f77b4", alpha=0.8)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Model probabilities")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: np.ndarray


def run_inference(model_path: str, mel: np.ndarray) -> Prediction:
    model = load_model(model_path)
    probs = predict(model, mel)
    idx = int(np.argmax(probs))
    return Prediction(label=EMOTIONS[idx], confidence=float(probs[idx]), probs=probs)


# ---------------------------
# Streamlit UI
# ---------------------------
def main() -> None:
    st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="wide")
    st.title("🎤 Speech Emotion Recognition")
    st.write("Upload audio, select a model, and view predictions. Fresh minimal rebuild.")

    model_files = list_model_files()
    if not model_files:
        st.error("No .h5 models found in the working directory.")
        return

    choice = st.sidebar.selectbox("Model file", model_files)
    uploaded = st.file_uploader("Upload audio (wav/mp3/ogg/flac)", type=["wav", "mp3", "ogg", "flac"])

    if not uploaded:
        st.info("Please upload an audio clip (≈3 seconds recommended).")
        return

    file_bytes = uploaded.read()
    st.audio(file_bytes, format="audio/wav")

    with st.spinner("Processing and running inference..."):
        mel, audio = preprocess(file_bytes)
        pred = run_inference(choice, mel)

    st.subheader("Prediction")
    st.markdown(f"**{pred.label}**  —  confidence: {pred.confidence*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_wave(audio))
    with col2:
        st.pyplot(plot_mel(mel[0, :, :, 0]))

    st.pyplot(plot_probs(pred.probs))


if __name__ == "__main__":
    main()
