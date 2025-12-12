# Lambda Labs Training Guide - Speech Emotion Recognition

## 🚀 Quick Start (5 Steps)

1. **Upload** `Speech_Emotion_Recognition_Notebook.ipynb` to Lambda Labs
2. **Open** notebook in Jupyter
3. **Run All Cells** (Ctrl+Shift+Enter or Cell → Run All)
4. **Wait** ~1-2 hours for complete training
5. **Download** all output files

## ⏱️ Expected Timeline

```
[0:00]  Start - Package installation
[0:05]  Dataset download begins (1.4 GB)
[0:15]  CNN training starts
[0:45]  CNN-LSTM training starts
[1:15]  YAMNet training starts
[1:30]  Results export and visualization
[1:35]  COMPLETE ✅
```

## 📊 What You'll Get

### Performance Metrics
- **CNN**: 65-75% accuracy (~95K parameters)
- **CNN-LSTM**: 68-78% accuracy (~2.2M parameters)
- **YAMNet**: 75-85% accuracy (~265K trainable parameters)

### Output Files (Download These!)
```
cnn_final_model.h5                    # Trained models
cnn_lstm_final_model.h5
yamnet_classifier_final_model.h5

training_results.json                  # Detailed metrics
model_comparison.csv                   # Comparison table
training_summary.txt                   # Text report
final_comparison_all_models.png        # Visualization

predictions.npz                        # All predictions
logs/                                  # TensorBoard logs
```

## 🖥️ Lambda Labs Setup

### Option 1: Web Interface (Easiest)

1. Go to Lambda Labs dashboard
2. Launch GPU instance (RTX 3090 or better recommended)
3. Open JupyterLab from dashboard
4. Upload notebook via drag-and-drop
5. Run all cells!

### Option 2: SSH/SCP

```bash
# Upload notebook
scp Speech_Emotion_Recognition_Notebook.ipynb ubuntu@<IP>:~/

# SSH into instance
ssh ubuntu@<IP>

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Access at: http://<IP>:8888
```

## 📈 Monitoring Training

### View Progress in Notebook
Just scroll down to see live training progress with progress bars!

### TensorBoard (Optional)

```bash
# In Lambda Labs terminal
tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# Visit: http://<IP>:6006
```

### GPU Monitoring

```bash
# Check GPU usage
watch -n 1 nvidia-smi
```

## 🔧 Optimization Tips

### For Faster Training
- Use RTX 3090 or A6000 instances
- Increase batch size to 64 (if you have 24GB+ VRAM)
- Already optimized: Memory growth enabled

### If Out of Memory
Edit Cell 3 and Cell 12:
```python
BATCH_SIZE = 16  # Change from 32 to 16
```

### Skip Long Training (Quick Test)
Edit Cell 12, 14, 15:
```python
epochs=10,  # Change from 50/100 to 10
```
(Will reduce accuracy but finish in ~20 minutes)

## ❗ Troubleshooting

### Dataset Download Fails
**Solution**: Manual download
1. Download: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
2. Upload to Lambda Labs
3. Extract: `unzip Audio_Speech_Actors_01-24.zip`
4. Restart notebook

### No GPU Detected
**Check**: Run `nvidia-smi` in terminal
- If no output: Contact Lambda Labs support
- If working: Restart kernel and run Cell 1

### LibROSA Audio Errors
**Solution**: Already handled in code
- Code has fallback mechanisms
- Should auto-recover from corrupted files

### Training Seems Stuck
**Normal behavior**: 
- Dataset creation: Takes 2-3 minutes (librosa loading)
- First epoch: Slower (data pipeline warmup)
- YAMNet embedding extraction: 10-15 minutes (appears frozen but working)

## 💰 Cost Estimation

| Instance | Cost/hr | Total Cost |
|----------|---------|------------|
| RTX 3090 | ~$0.50 | ~$1.00 |
| A6000 | ~$0.80 | ~$1.60 |
| A100 | ~$1.50 | ~$3.00 |

**Recommended**: RTX 3090 (best price/performance)

## ✅ Verification Checklist

After training completes, verify:

- [ ] Cell 1: "✓ GPU(s) detected"
- [ ] Cell 3: "✓ Found 1440 audio files"
- [ ] Cell 12: CNN test accuracy shown
- [ ] Cell 14: CNN-LSTM test accuracy shown
- [ ] Cell 15: YAMNet test accuracy shown
- [ ] Cell 16: Bar charts displayed
- [ ] Cell 17: All files saved confirmation
- [ ] Files exist in directory

## 📥 What to Submit

**Minimum Required**:
1. `Speech_Emotion_Recognition_Notebook.ipynb` (with all outputs visible)
2. `training_summary.txt`
3. `final_comparison_all_models.png`

**Complete Submission** (Recommended):
- All above files
- `training_results.json`
- `model_comparison.csv`
- All three `.h5` model files

## 🎓 Grading Rubric Mapping

| Requirement | Location in Notebook |
|-------------|---------------------|
| Problem Statement | Cells 1-3 (markdown) |
| Three Algorithms | CNN (Cell 12), LSTM (Cell 14), YAMNet (Cell 15) |
| Mathematical Theory | Cells 26-27 (markdown) |
| Data Description | Cell 25 (markdown) |
| Preprocessing | Cells 5, 10 + markdown |
| Training Results | Cells 12, 14, 15 outputs |
| Confusion Matrix | Cell 13 output |
| Model Comparison | Cell 16 output |
| References | Cell 31 (markdown) |

## 🆘 Need Help?

**Common Issues Solved**:
- ✅ All packages auto-install
- ✅ Dataset auto-downloads
- ✅ GPU auto-detected
- ✅ Models auto-save
- ✅ Results auto-export

**Still Stuck?**
1. Check Cell 1 output for errors
2. Restart kernel: Kernel → Restart
3. Clear outputs: Cell → All Output → Clear
4. Run all cells again

## 🎉 Success Indicators

You'll know it worked when you see:

```
='=80
TRAINING COMPLETED SUCCESSFULLY! ✅
='=80
```

And you have all output files in your directory!

---

**Ready to Train?** Just upload the notebook and click "Run All"! 🚀

**Estimated Total Time**: 1-2 hours on GPU  
**Estimated Cost**: $1-2 on Lambda Labs  
**Expected Best Accuracy**: 75-85% (YAMNet)

Good luck with your training and graduation! 🎓
