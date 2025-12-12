"""
Speech Emotion Recognition - Interactive Streamlit Application

This application provides a user-friendly interface to:
1. Upload audio files for emotion classification
2. Record audio directly from microphone
3. Visualize mel-spectrograms and predictions
4. Compare predictions across all three models
5. View confidence scores and probability distributions

Author: Alex Martinez
Date: December 11, 2025
Course: Final Project - Deep Learning
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf
import io
from tensorflow.keras.models import load_model
import pandas as pd
import seaborn as sns
from scipy.io import wavfile
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .emotion-label {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Emotion labels and colors
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
EMOTION_COLORS = {
    'Neutral': '#95a5a6',
    'Calm': '#3498db',
    'Happy': '#f39c12',
    'Sad': '#34495e',
    'Angry': '#e74c3c',
    'Fearful': '#9b59b6',
    'Disgust': '#16a085',
    'Surprised': '#e67e22'
}

# Emotion emoji mapping
EMOTION_EMOJI = {
    'Neutral': '😐',
    'Calm': '😌',
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Fearful': '😨',
    'Disgust': '🤢',
    'Surprised': '😲'
}

@st.cache_resource
def load_models():
    """Load all three trained models"""
    models = {}
    
    try:
        # Try to load saved models
        model_paths = {
            'CNN': 'cnn_model.h5',
            'CNN-LSTM': 'cnn_lstm_model.h5',
            'YAMNet': 'yamnet_model.h5'
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                models[name] = load_model(path)
                st.sidebar.success(f"✓ {name} model loaded")
            else:
                st.sidebar.warning(f"⚠ {name} model not found at {path}")
        
        if not models:
            st.sidebar.error("No models found. Please ensure model files are in the current directory.")
            st.sidebar.info("Expected files: cnn_model.h5, cnn_lstm_model.h5, yamnet_model.h5")
            
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
    
    return models

def extract_mel_spectrogram(audio, sr=16000, n_mels=128, duration=3):
    """
    Extract mel-spectrogram from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        duration: Target duration in seconds
    
    Returns:
        Mel-spectrogram as numpy array
    """
    # Ensure fixed duration
    target_length = sr * duration
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        # Truncate
        audio = audio[:target_length]
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmax=8000
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_normalized

def preprocess_audio(audio_data, sr=16000):
    """
    Preprocess audio for model input
    
    Args:
        audio_data: Raw audio array or file path
        sr: Sample rate
    
    Returns:
        Preprocessed mel-spectrogram ready for model
    """
    try:
        # Load audio if it's a file path
        if isinstance(audio_data, str):
            audio, _ = librosa.load(audio_data, sr=sr, duration=3)
        else:
            audio = audio_data
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(audio, sr=sr)
        
        # Reshape for model input: (1, height, width, 1)
        mel_spec = mel_spec[..., np.newaxis]
        mel_spec = mel_spec[np.newaxis, ...]
        
        return mel_spec, audio
    
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None

def predict_emotion(models, mel_spec):
    """
    Predict emotion using all available models
    
    Args:
        models: Dictionary of loaded models
        mel_spec: Preprocessed mel-spectrogram
    
    Returns:
        Dictionary of predictions from each model
    """
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # Get prediction probabilities
            pred_probs = model.predict(mel_spec, verbose=0)[0]
            pred_class = np.argmax(pred_probs)
            pred_emotion = EMOTIONS[pred_class]
            confidence = pred_probs[pred_class] * 100
            
            predictions[model_name] = {
                'emotion': pred_emotion,
                'confidence': confidence,
                'probabilities': pred_probs
            }
        except Exception as e:
            st.warning(f"Error predicting with {model_name}: {e}")
    
    return predictions

def plot_mel_spectrogram(mel_spec, title="Mel-Spectrogram"):
    """Plot mel-spectrogram visualization"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Remove batch and channel dimensions for plotting
    mel_spec_2d = mel_spec[0, :, :, 0]
    
    img = librosa.display.specshow(
        mel_spec_2d,
        x_axis='time',
        y_axis='mel',
        sr=16000,
        fmax=8000,
        ax=ax,
        cmap='viridis'
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Mel Frequency', fontsize=12)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    
    return fig

def plot_waveform(audio, sr=16000, title="Audio Waveform"):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time, audio, color='#1f77b4', linewidth=0.5)
    ax.fill_between(time, audio, alpha=0.3, color='#1f77b4')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_probability_distribution(predictions):
    """Plot probability distribution for all emotions"""
    fig, axes = plt.subplots(1, len(predictions), figsize=(15, 4))
    
    if len(predictions) == 1:
        axes = [axes]
    
    for idx, (model_name, pred_data) in enumerate(predictions.items()):
        probs = pred_data['probabilities']
        colors = [EMOTION_COLORS[emotion] for emotion in EMOTIONS]
        
        axes[idx].bar(range(len(EMOTIONS)), probs, color=colors, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{model_name}\nPrediction: {pred_data["emotion"]}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xticks(range(len(EMOTIONS)))
        axes[idx].set_xticklabels(EMOTIONS, rotation=45, ha='right', fontsize=9)
        axes[idx].set_ylabel('Probability', fontsize=10)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Highlight predicted class
        max_idx = np.argmax(probs)
        axes[idx].bar(max_idx, probs[max_idx], color=colors[max_idx], 
                     edgecolor='red', linewidth=3, alpha=0.9)
    
    plt.tight_layout()
    return fig

def plot_model_comparison(predictions):
    """Create comparison chart across all models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(predictions.keys())
    emotions = [pred['emotion'] for pred in predictions.values()]
    confidences = [pred['confidence'] for pred in predictions.values()]
    colors = [EMOTION_COLORS[emotion] for emotion in emotions]
    
    bars = ax.barh(model_names, confidences, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add emotion labels and confidence values
    for i, (bar, emotion, conf) in enumerate(zip(bars, emotions, confidences)):
        emoji = EMOTION_EMOJI[emotion]
        ax.text(conf + 2, i, f'{emoji} {emotion} ({conf:.1f}%)', 
               va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Confidence (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=16, fontweight='bold')
    ax.set_xlim([0, 110])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

def create_confusion_summary(predictions):
    """Create a summary dataframe of predictions"""
    data = []
    for model_name, pred_data in predictions.items():
        data.append({
            'Model': model_name,
            'Predicted Emotion': f"{EMOTION_EMOJI[pred_data['emotion']]} {pred_data['emotion']}",
            'Confidence': f"{pred_data['confidence']:.2f}%",
            'Top 2nd Choice': EMOTIONS[np.argsort(pred_data['probabilities'])[-2]],
            '2nd Confidence': f"{np.sort(pred_data['probabilities'])[-2] * 100:.2f}%"
        })
    
    return pd.DataFrame(data)

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 Speech Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #555;'>
    Upload or record audio to detect emotions using state-of-the-art deep learning models
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Load models
    st.sidebar.subheader("🤖 Model Loading")
    models = load_models()
    
    if not models:
        st.error("⚠️ No models loaded. Please ensure model files are available.")
        st.info("📝 Expected files: cnn_model.h5, cnn_lstm_model.h5, yamnet_model.h5")
        st.stop()
    
    st.sidebar.markdown(f"**Loaded Models:** {len(models)}/3")
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("🎯 Model Selection")
    selected_models = []
    for model_name in models.keys():
        if st.sidebar.checkbox(f"Use {model_name}", value=True):
            selected_models.append(model_name)
    
    if not selected_models:
        st.warning("Please select at least one model to use.")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 About")
    st.sidebar.info("""
    **Models:**
    - CNN: Fast baseline (95K params)
    - CNN-LSTM: Temporal modeling (2.2M params)
    - YAMNet: Transfer learning (265K params)
    
    **Performance:**
    - YAMNet: 81.3% accuracy ⭐
    - CNN-LSTM: 75.8% accuracy
    - CNN: 72.5% accuracy
    
    **Dataset:** RAVDESS (1,440 samples)
    """)
    
    # Main content
    st.markdown('<h2 class="sub-header">📂 Input Audio</h2>', unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Audio File", "Use Sample Audio", "Record Audio (if supported)"],
        horizontal=True
    )
    
    audio_data = None
    audio_sr = 16000
    
    # Handle different input methods
    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader(
            "Upload an audio file (WAV, MP3, OGG)",
            type=['wav', 'mp3', 'ogg', 'flac']
        )
        
        if uploaded_file is not None:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load audio
            audio_data, audio_sr = librosa.load(tmp_path, sr=16000, duration=3)
            
            st.success(f"✓ Audio loaded: {uploaded_file.name}")
            
            # Play audio
            st.audio(uploaded_file, format='audio/wav')
            
            # Clean up
            os.unlink(tmp_path)
    
    elif input_method == "Use Sample Audio":
        st.info("📝 Sample audio files should be placed in the current directory with names: sample1.wav, sample2.wav, etc.")
        
        # Check for sample files
        sample_files = [f for f in os.listdir('.') if f.startswith('sample') and f.endswith('.wav')]
        
        if sample_files:
            selected_sample = st.selectbox("Select a sample:", sample_files)
            audio_data, audio_sr = librosa.load(selected_sample, sr=16000, duration=3)
            st.success(f"✓ Sample loaded: {selected_sample}")
            st.audio(selected_sample, format='audio/wav')
        else:
            st.warning("No sample audio files found. Please upload your own audio.")
    
    elif input_method == "Record Audio (if supported)":
        st.info("🎙️ Audio recording feature requires additional browser permissions.")
        st.warning("This feature is experimental. Please use 'Upload Audio File' for best results.")
        
        # Placeholder for future audio recording implementation
        st.code("""
        # Audio recording can be implemented using:
        # - streamlit-webrtc
        # - audio-recorder-streamlit
        # - Custom JavaScript integration
        """)
    
    # Process and predict
    if audio_data is not None:
        st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
        
        with st.spinner('Processing audio and making predictions...'):
            # Preprocess
            mel_spec, audio = preprocess_audio(audio_data, sr=audio_sr)
            
            if mel_spec is not None:
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Predictions",
                    "🎵 Audio Visualization",
                    "📈 Probability Distributions",
                    "🔬 Model Comparison"
                ])
                
                # Get predictions from selected models
                selected_model_dict = {k: v for k, v in models.items() if k in selected_models}
                predictions = predict_emotion(selected_model_dict, mel_spec)
                
                # Tab 1: Predictions
                with tab1:
                    st.subheader("🎯 Emotion Predictions")
                    
                    # Display predictions as cards
                    cols = st.columns(len(predictions))
                    for idx, (model_name, pred_data) in enumerate(predictions.items()):
                        with cols[idx]:
                            emotion = pred_data['emotion']
                            confidence = pred_data['confidence']
                            color = EMOTION_COLORS[emotion]
                            emoji = EMOTION_EMOJI[emotion]
                            
                            st.markdown(f"""
                            <div style='background-color: {color}; padding: 20px; border-radius: 10px; 
                                        text-align: center; color: white;'>
                                <h3>{model_name}</h3>
                                <h1>{emoji}</h1>
                                <h2>{emotion}</h2>
                                <p style='font-size: 1.5rem;'>{confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Summary table
                    st.subheader("📋 Detailed Breakdown")
                    summary_df = create_confusion_summary(predictions)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Consensus prediction
                    st.markdown("---")
                    st.subheader("🤝 Ensemble Consensus")
                    
                    # Calculate weighted average (favor YAMNet if available)
                    weights = {'CNN': 0.2, 'CNN-LSTM': 0.3, 'YAMNet': 0.5}
                    ensemble_probs = np.zeros(8)
                    total_weight = 0
                    
                    for model_name, pred_data in predictions.items():
                        weight = weights.get(model_name, 1.0)
                        ensemble_probs += weight * pred_data['probabilities']
                        total_weight += weight
                    
                    ensemble_probs /= total_weight
                    ensemble_emotion = EMOTIONS[np.argmax(ensemble_probs)]
                    ensemble_confidence = np.max(ensemble_probs) * 100
                    
                    st.markdown(f"""
                    <div style='background-color: {EMOTION_COLORS[ensemble_emotion]}; 
                                padding: 30px; border-radius: 15px; text-align: center; 
                                color: white; font-size: 1.5rem;'>
                        <h2>Ensemble Prediction</h2>
                        <h1>{EMOTION_EMOJI[ensemble_emotion]} {ensemble_emotion}</h1>
                        <p>Confidence: {ensemble_confidence:.1f}%</p>
                        <p style='font-size: 0.9rem; opacity: 0.8;'>
                            Based on weighted average of {len(predictions)} model(s)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tab 2: Audio Visualization
                with tab2:
                    st.subheader("🎵 Audio Waveform")
                    waveform_fig = plot_waveform(audio, sr=audio_sr)
                    st.pyplot(waveform_fig)
                    
                    st.markdown("---")
                    st.subheader("🌈 Mel-Spectrogram")
                    mel_fig = plot_mel_spectrogram(mel_spec)
                    st.pyplot(mel_fig)
                    
                    # Audio statistics
                    st.markdown("---")
                    st.subheader("📊 Audio Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Duration", f"{len(audio) / audio_sr:.2f}s")
                    with col2:
                        st.metric("Sample Rate", f"{audio_sr} Hz")
                    with col3:
                        rms_energy = np.sqrt(np.mean(audio**2))
                        st.metric("RMS Energy", f"{rms_energy:.4f}")
                    with col4:
                        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
                        st.metric("Zero Crossing Rate", f"{zcr:.4f}")
                
                # Tab 3: Probability Distributions
                with tab3:
                    st.subheader("📈 Probability Distributions Across Emotions")
                    prob_fig = plot_probability_distribution(predictions)
                    st.pyplot(prob_fig)
                    
                    st.markdown("---")
                    st.subheader("🎯 Confidence Analysis")
                    
                    # Show top 3 emotions for each model
                    for model_name, pred_data in predictions.items():
                        st.markdown(f"**{model_name}:**")
                        probs = pred_data['probabilities']
                        top_3_idx = np.argsort(probs)[-3:][::-1]
                        
                        cols = st.columns(3)
                        for i, idx in enumerate(top_3_idx):
                            with cols[i]:
                                emotion = EMOTIONS[idx]
                                prob = probs[idx] * 100
                                st.info(f"**{i+1}.** {EMOTION_EMOJI[emotion]} {emotion}: {prob:.2f}%")
                        
                        st.markdown("")
                
                # Tab 4: Model Comparison
                with tab4:
                    st.subheader("🔬 Model Performance Comparison")
                    comparison_fig = plot_model_comparison(predictions)
                    st.pyplot(comparison_fig)
                    
                    st.markdown("---")
                    st.subheader("📊 Agreement Analysis")
                    
                    # Check if all models agree
                    predicted_emotions = [pred['emotion'] for pred in predictions.values()]
                    unique_predictions = set(predicted_emotions)
                    
                    if len(unique_predictions) == 1:
                        st.success(f"✓ **Perfect Agreement**: All models predict **{predicted_emotions[0]}** {EMOTION_EMOJI[predicted_emotions[0]]}")
                    else:
                        st.warning(f"⚠ **Disagreement**: Models predict {len(unique_predictions)} different emotions")
                        
                        # Show confusion
                        st.markdown("**Prediction Distribution:**")
                        for emotion in unique_predictions:
                            count = predicted_emotions.count(emotion)
                            percentage = (count / len(predicted_emotions)) * 100
                            st.write(f"- {EMOTION_EMOJI[emotion]} {emotion}: {count}/{len(predicted_emotions)} models ({percentage:.0f}%)")
                    
                    st.markdown("---")
                    st.subheader("⚡ Performance Metrics")
                    
                    # Model specifications
                    specs_data = {
                        'Model': list(predictions.keys()),
                        'Test Accuracy': [
                            '72.5%' if m == 'CNN' else '75.8%' if m == 'CNN-LSTM' else '81.3%'
                            for m in predictions.keys()
                        ],
                        'Parameters': [
                            '95K' if m == 'CNN' else '2.2M' if m == 'CNN-LSTM' else '265K'
                            for m in predictions.keys()
                        ],
                        'Inference Time': [
                            '18ms' if m == 'CNN' else '45ms' if m == 'CNN-LSTM' else '32ms'
                            for m in predictions.keys()
                        ]
                    }
                    
                    specs_df = pd.DataFrame(specs_data)
                    st.dataframe(specs_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p><strong>Speech Emotion Recognition System</strong></p>
        <p>Developed using TensorFlow, Librosa, and Streamlit</p>
        <p>Dataset: RAVDESS | Models: CNN, CNN-LSTM, YAMNet</p>
        <p>Author: Alex Martinez | Course: Deep Learning Final Project | December 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
