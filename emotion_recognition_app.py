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

# Lazy import for YAMNet to avoid startup cost if unused.
try:
    import tensorflow_hub as hub
except Exception:  # noqa: BLE001
    hub = None


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
def validate_models(paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
    ok: List[str] = []
    bad: Dict[str, str] = {}
    for p in paths:
        try:
            _ = tf.keras.models.load_model(
                p,
                custom_objects={
                    "InputLayer": CompatibleInputLayer,
                    "DTypePolicy": tf.keras.mixed_precision.Policy,
                    "Policy": tf.keras.mixed_precision.Policy,
                },
                compile=False,
                safe_mode=False,
            )
            ok.append(p)
        except Exception as exc:  # noqa: BLE001
            bad[p] = str(exc)
    return ok, bad


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects={
                "InputLayer": CompatibleInputLayer,
                "DTypePolicy": tf.keras.mixed_precision.Policy,
                "Policy": tf.keras.mixed_precision.Policy,
            },
            compile=False,
            safe_mode=False,
        )
    except Exception as exc:
        # Common case: TF/Keras cannot deserialize non-Keras .h5 (e.g., TFJS or incompatible export).
        msg = (
            f"Failed to load model '{path}'. This file may not be a Keras .h5 saved with TF 2.x. "
            "Use the notebook-exported models: cnn_final_model.h5, cnn_lstm_final_model.h5, "
            "or yamnet_classifier_final_model.h5.\n\n"
            f"Loader error: {exc}"
        )
        raise RuntimeError(msg) from exc


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


class CompatibleInputLayer(tf.keras.layers.InputLayer):
    """InputLayer that tolerates legacy configs containing 'batch_shape'."""

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        if "batch_shape" in cfg and "batch_input_shape" not in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
        return super().from_config(cfg)


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


def compute_yamnet_embedding(audio: np.ndarray) -> np.ndarray:
    yamnet = load_yamnet_model()
    waveform_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    waveform_tf = tf.reshape(waveform_tf, [-1])  # ensure 1-D as YAMNet expects
    _scores, embeddings, _spec = yamnet(waveform_tf)
    # embeddings shape: (frames, 1024); average across frames to get 1024-d
    emb = tf.reduce_mean(embeddings, axis=0)  # (1024,)
    emb = tf.reshape(emb, [1, -1])  # (1, 1024) for classifier
    return emb.numpy().astype(np.float32)


def preprocess(file_bytes: bytes, model: tf.keras.Model, model_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return model-ready features, raw audio, and the feature type used."""

    audio, sr = sf.read(io.BytesIO(file_bytes))
    audio = _to_mono(np.asarray(audio, dtype=np.float32))
    audio = _resample(audio, sr)
    audio = _pad_or_trim(audio)

    if is_yamnet_classifier(model, model_path=model_path):
        emb = compute_yamnet_embedding(audio)
        return emb, audio, "yamnet"

    log_mel = compute_log_mel(audio)
    return log_mel[np.newaxis, ..., np.newaxis], audio, "mel"


def predict(model: tf.keras.Model, mel: np.ndarray) -> np.ndarray:
    return model.predict(mel, verbose=0)[0]


def is_yamnet_classifier(model: tf.keras.Model, model_path: str = "") -> bool:
    """Detect if the model expects a 1024-d embedding (YAMNet head)."""

    def _is_1024(shape_like) -> bool:
        if not isinstance(shape_like, (list, tuple)):
            return False
        if len(shape_like) == 2 and shape_like[1] == 1024:
            return True
        return False

    try:
        shapes = model.input_shape
    except Exception:  # noqa: BLE001
        shapes = None

    # Check direct input_shape (can be tuple or list of tuples)
    if shapes is not None:
        if _is_1024(shapes):
            return True
        if isinstance(shapes, (list, tuple)):
            for s in shapes:
                if _is_1024(s):
                    return True

    # Fall back to first layer input shapes
    for layer in getattr(model, "layers", []):
        ishape = getattr(layer, "input_shape", None)
        if ishape is None:
            continue
        if _is_1024(ishape):
            return True
        if isinstance(ishape, (list, tuple)):
            for s in ishape if isinstance(ishape, (list, tuple)) else []:
                if _is_1024(s):
                    return True

    # Final fallback: file name heuristic
    return "yamnet" in model_path.lower()


@st.cache_resource(show_spinner=False)
def load_yamnet_model():
    if hub is None:
        raise RuntimeError("tensorflow_hub is not installed; required for YAMNet classifier.")
    return hub.load("https://tfhub.dev/google/yamnet/1")


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


def run_inference(model: tf.keras.Model, features: np.ndarray) -> Prediction:
    probs = predict(model, features)
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

    valid_models, invalid_models = validate_models(model_files)
    if not valid_models:
        st.error("All detected .h5 files failed to load. See details below.")
        with st.expander("Model load errors"):
            for name, err in invalid_models.items():
                st.markdown(f"**{name}** — {err}")
        return

    if invalid_models:
        with st.expander("Skipped incompatible models"):
            for name, err in invalid_models.items():
                st.markdown(f"**{name}** — {err}")

    choice = st.sidebar.selectbox("Model file", valid_models)
    uploaded = st.file_uploader("Upload audio (wav/mp3/ogg/flac)", type=["wav", "mp3", "ogg", "flac"])

    if not uploaded:
        st.info("Please upload an audio clip (≈3 seconds recommended).")
        return

    file_bytes = uploaded.read()
    st.audio(file_bytes, format="audio/wav")

    with st.spinner("Processing and running inference..."):
        try:
            model = load_model(choice)
            features, audio, feat_kind = preprocess(file_bytes, model, choice)
            pred = run_inference(model, features)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Prediction")
    st.markdown(f"**{pred.label}**  —  confidence: {pred.confidence*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_wave(audio))
    with col2:
        if feat_kind == "mel":
            st.pyplot(plot_mel(features[0, :, :, 0]))
        else:
            st.info("YAMNet embedding computed (1024-d); spectrogram skipped.")

    st.pyplot(plot_probs(pred.probs))


if __name__ == "__main__":
    main()
