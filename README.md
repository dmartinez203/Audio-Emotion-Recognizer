# Speech Emotion Recognition (Streamlit Submission)

Minimal submission package with three trained models (CNN, CNN-LSTM, YAMNet transfer) and a Streamlit demo.

## Contents
- `emotion_recognition_app.py` — Streamlit UI for inference (handles YAMNet embeddings automatically).
- Models: `cnn_final_model.h5`, `cnn_lstm_final_model.h5`, `yamnet_classifier_final_model.h5`.
- Notebook: `Speech_Emotion_Recognition_Notebook.ipynb` (full training workflow, analysis, plots).
- Support files: `requirements.txt`, `requirements_streamlit.txt`, `runtime.txt`, `model_comparison.csv`, `training_results.json`, `training_summary.txt`, `final_comparison_all_models.png`.

## Setup
1) Python 3.11 recommended. Create/activate a virtual env.
2) Install deps (macOS Metal-optimized TF included):
```bash
pip install -r requirements.txt
```

## Run the demo
```bash
streamlit run emotion_recognition_app.py
```
- Upload an audio clip (~3s, wav/mp3/ogg/flac).
- Choose a model; YAMNet classifier uses built-in embeddings automatically.

## Data
- RAVDESS dataset should be placed under `Audio_Speech_Actors_01-24/` (kept for reference). Not required for inference.

## Notes
- Extra artifacts, duplicate models, and intermediate exports were removed for a clean submission.
- If TF Hub is missing, install: `pip install tensorflow-hub`.
