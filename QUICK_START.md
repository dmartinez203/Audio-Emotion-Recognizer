# 🚀 QUICK START - Lambda Labs Training

## Upload & Run (3 Steps)

1. **Upload** `Speech_Emotion_Recognition_Notebook.ipynb` to Lambda Labs
2. **Click** "Run All Cells" (or Ctrl+Shift+Enter)  
3. **Wait** ~1-2 hours → **DONE!** ✅

---

## What Happens Automatically

✅ Install all packages (TensorFlow, librosa, etc.)  
✅ Download RAVDESS dataset (1.4 GB)  
✅ Train CNN model (50 epochs, ~30 min)  
✅ Train CNN-LSTM model (50 epochs, ~60 min)  
✅ Train YAMNet model (100 epochs, ~15 min)  
✅ Generate confusion matrices  
✅ Export all results & models  

---

## Expected Results

| Model | Accuracy | Time |
|-------|----------|------|
| CNN | 65-75% | 30m |
| CNN-LSTM | 68-78% | 60m |
| **YAMNet** | **75-85%** | 15m |

**Best**: YAMNet (75-85%)

---

## Files to Download

**Required:**
- `Speech_Emotion_Recognition_Notebook.ipynb` (with outputs)
- `training_summary.txt`
- `final_comparison_all_models.png`

**Optional:**
- All `.h5` model files
- `training_results.json`

---

## Troubleshooting

**Out of Memory?** → Reduce batch size to 16 in Cell 3  
**No GPU?** → Check Cell 1 output for "GPU detected"  
**Download fails?** → Manual download from Zenodo link in Cell 3  

---

## Success = See This

```
='=80
TRAINING COMPLETED SUCCESSFULLY! ✅
='=80
```

---

## Lambda Labs Cost

RTX 3090: ~$1.00 total (1.5-2 hours @ $0.50/hr)

---

**THAT'S IT!** Just upload and run. Everything is automated.

Good luck! 🎓

---

## 🌐 Frontend (React + GitHub Pages)

- Location: `frontend/` (Vite + React, no Streamlit).
- Local dev: `cd frontend && npm install && npm run dev`.
- Build: `npm run build` → outputs to `frontend/dist`.
- Deploy to GitHub Pages: `npm run deploy` (uses `gh-pages`, base set to `/Audio-Emotion-Recognizer/`).
- Hook to backend: wire `App.jsx` to POST your audio file to `/predict` (see suggested contract in the UI). For client-only, convert a `.h5` model to TF.js and load it in `App.jsx`.
