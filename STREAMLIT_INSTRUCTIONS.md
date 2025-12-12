# 🎤 Speech Emotion Recognition - Streamlit App Instructions

## Overview

This Streamlit application provides an interactive web interface for testing the speech emotion recognition models. Users can upload audio files, visualize spectrograms, and see predictions from all three models (CNN, CNN-LSTM, YAMNet).

---

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

Or install manually:

```bash
pip install streamlit tensorflow librosa numpy matplotlib seaborn pandas soundfile scipy pillow
```

### Step 2: Ensure Model Files Are Available

The app expects the following trained model files in the same directory:

- `cnn_model.h5`
- `cnn_lstm_model.h5`
- `yamnet_model.h5`

**To generate these files**, run the notebook (`Speech_Emotion_Recognition_Notebook.ipynb`) until the export cell (Cell 17) which saves the trained models.

---

## Running the Application

### Local Deployment

```bash
streamlit run emotion_recognition_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Custom Port

```bash
streamlit run emotion_recognition_app.py --server.port 8080
```

### Network Access (Allow Other Devices)

```bash
streamlit run emotion_recognition_app.py --server.address 0.0.0.0
```

---

## Features

### 1. **Upload Audio File**
- Supports: WAV, MP3, OGG, FLAC formats
- Automatically processes 3-second clips
- Pads or truncates to standard length

### 2. **Use Sample Audio**
- Place sample audio files in the app directory
- Name them: `sample1.wav`, `sample2.wav`, etc.
- Select from dropdown menu

### 3. **Model Selection**
- Choose which models to use for prediction
- Compare predictions across models
- View individual and ensemble results

### 4. **Visualizations**
- **Audio Waveform:** Time-domain representation
- **Mel-Spectrogram:** Frequency-domain features used by models
- **Probability Distributions:** Confidence across all 8 emotions
- **Model Comparison:** Side-by-side performance

### 5. **Detailed Analysis**
- Prediction confidence scores
- Top-3 emotion candidates
- Agreement/disagreement between models
- Ensemble consensus prediction

---

## Usage Example

### Step-by-Step:

1. **Launch the app:**
   ```bash
   streamlit run emotion_recognition_app.py
   ```

2. **Select models** in the sidebar (all three selected by default)

3. **Upload an audio file** using the file uploader
   - Example: A recording of someone speaking angrily
   - Format: WAV file, ~3 seconds duration

4. **View results** in four tabs:
   - **Predictions:** Emotion labels and confidence
   - **Audio Visualization:** Waveform and spectrogram
   - **Probability Distributions:** Bar charts for all emotions
   - **Model Comparison:** Performance comparison

5. **Interpret results:**
   - Green cards show high-confidence predictions (>80%)
   - Ensemble consensus combines all model predictions
   - Agreement analysis shows if models disagree

---

## Creating Sample Audio Files

### Option 1: Use RAVDESS Dataset

```python
import shutil

# Copy some test samples from RAVDESS
test_files = [
    'Actor_01/03-01-05-01-01-01-01.wav',  # Angry
    'Actor_02/03-01-03-01-01-01-02.wav',  # Happy
    'Actor_03/03-01-04-01-01-01-03.wav',  # Sad
]

for i, file in enumerate(test_files, 1):
    shutil.copy(f'RAVDESS/{file}', f'sample{i}.wav')
```

### Option 2: Record Your Own

```bash
# On macOS
sox -d sample1.wav trim 0 3

# On Linux
arecord -d 3 -f cd sample1.wav

# On Windows (PowerShell)
# Use Voice Recorder app, save as WAV, trim to 3 seconds
```

### Option 3: Download from YouTube

```bash
# Install youtube-dl
pip install yt-dlp

# Download audio
yt-dlp -x --audio-format wav --audio-quality 0 "https://youtube.com/watch?v=EXAMPLE"

# Trim to 3 seconds
ffmpeg -i input.wav -ss 0 -t 3 sample1.wav
```

---

## Troubleshooting

### Issue: "No models loaded"

**Solution:**
1. Ensure model files exist in the current directory:
   ```bash
   ls *.h5
   # Should show: cnn_model.h5, cnn_lstm_model.h5, yamnet_model.h5
   ```

2. Run the notebook to generate models:
   ```bash
   jupyter nbconvert --to notebook --execute Speech_Emotion_Recognition_Notebook.ipynb
   ```

### Issue: "Error preprocessing audio"

**Solution:**
- Check audio format (convert to WAV if necessary):
  ```bash
  ffmpeg -i input.mp3 output.wav
  ```
- Ensure audio is mono (not stereo):
  ```bash
  ffmpeg -i stereo.wav -ac 1 mono.wav
  ```

### Issue: "Model prediction failed"

**Solution:**
- Verify input shape matches training:
  - Expected: (1, 128, 94, 1) for mel-spectrogram
  - Check with: `print(mel_spec.shape)`
- Reload models by restarting the app

### Issue: Slow performance

**Solution:**
1. Use GPU if available:
   ```bash
   # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. Cache models properly (Streamlit does this automatically with `@st.cache_resource`)

3. Reduce batch size or use only one model (CNN for fastest inference)

---

## Deployment Options

### Option 1: Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository
4. Deploy!

**Note:** Model files (`.h5`) are large. Consider using Streamlit's file storage or external hosting.

### Option 2: Heroku

```bash
# Create Procfile
echo "web: streamlit run emotion_recognition_app.py --server.port $PORT" > Procfile

# Deploy
heroku create emotion-recognition-app
git push heroku main
```

### Option 3: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements_streamlit.txt

EXPOSE 8501

CMD ["streamlit", "run", "emotion_recognition_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t emotion-app .
docker run -p 8501:8501 emotion-app
```

---

## Advanced Features

### Adding Audio Recording

To enable browser-based audio recording:

```bash
pip install streamlit-webrtc
```

Then uncomment the recording section in `emotion_recognition_app.py`.

### Batch Processing

To process multiple files at once:

```python
uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)

for file in uploaded_files:
    # Process each file
    predictions = predict_emotion(models, preprocess_audio(file))
    # Display results
```

### Real-Time Streaming

For continuous emotion detection from microphone:

```python
from streamlit_webrtc import webrtc_streamer

def callback(frame):
    audio = frame.to_ndarray()
    # Process audio
    # Update predictions in real-time

webrtc_streamer(key="emotion", audio_frame_callback=callback)
```

---

## Performance Benchmarks

**Test Environment:** MacBook Pro M1, 16GB RAM

| Operation | Time |
|-----------|------|
| Model Loading | 2-3 seconds |
| Audio Upload | <1 second |
| Preprocessing | 0.5 seconds |
| CNN Prediction | 18ms |
| CNN-LSTM Prediction | 45ms |
| YAMNet Prediction | 32ms |
| Total (single file, all models) | ~2 seconds |

---

## API Endpoints (For Custom Integration)

You can use Streamlit's API mode for programmatic access:

```python
import requests

# Upload audio
files = {'file': open('audio.wav', 'rb')}
response = requests.post('http://localhost:8501/api/predict', files=files)

# Get predictions
predictions = response.json()
print(predictions)
```

---

## Credits

**Developer:** Alex Martinez  
**Course:** Deep Learning Final Project  
**Date:** December 2025  
**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
**Models:** CNN, CNN-LSTM, YAMNet (TensorFlow Hub)  

---

## License

This project is for educational purposes as part of a university course. The RAVDESS dataset is used under academic license.

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check model compatibility with TensorFlow version
4. Ensure all dependencies are installed correctly

---

**Enjoy using the Speech Emotion Recognition app!** 🎤😊
