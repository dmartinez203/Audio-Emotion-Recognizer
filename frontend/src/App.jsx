import React, { useEffect, useMemo, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import Meyda from 'meyda';

const EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];
const SAMPLE_RATE = 16000;
const N_MELS = 128;
const N_FFT = 1024;
const HOP = 256;
const TIME_FRAMES = 188; // 3s @16k, hop=256
const MODEL_PATH = '/Audio-Emotion-Recognizer/models/best/model.json'; // place converted TF.js model here

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${sizes[i]}`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [notes, setNotes] = useState(
    'Static frontend only — connect a backend API or TF.js model to enable predictions.'
  );
  const [model, setModel] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [loading, setLoading] = useState(false);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    // Warm up tfjs; ignore failure if model file is missing
    tf.ready().then(() => {
      // no-op
    });
  }, []);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    if (f) {
      setNotes('Ready to run local inference if a TF.js model is present.');
    } else {
      setNotes('Static frontend only — connect a backend API or TF.js model to enable predictions.');
    }
  };

  const clearSelection = () => {
    setFile(null);
    setNotes('Static frontend only — connect a backend API or TF.js model to enable predictions.');
    setProbabilities(null);
  };

  async function ensureModel() {
    if (model) return model;
    try {
      const m = await tf.loadLayersModel(MODEL_PATH);
      setModel(m);
      return m;
    } catch (err) {
      console.warn('Model not found at', MODEL_PATH, err);
      throw new Error('Model file missing. Place converted TF.js model at public/models/best/model.json');
    }
  }

  async function decodeAndResample(fileObj) {
    const arrayBuffer = await fileObj.arrayBuffer();
    const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    const audio = await audioCtx.decodeAudioData(arrayBuffer);
    // mono
    const channel = audio.numberOfChannels > 1 ? audio.getChannelData(0) : audio.getChannelData(0);
    // already at SAMPLE_RATE due to AudioContext config; ensure length 3s
    const targetLength = SAMPLE_RATE * 3;
    let data = channel;
    if (data.length > targetLength) {
      data = data.slice(0, targetLength);
    } else if (data.length < targetLength) {
      const pad = new Float32Array(targetLength);
      pad.set(data);
      data = pad;
    }
    return data;
  }

  function computeMel(data) {
    const mel = Meyda.extract('melSpectrogram', data, {
      bufferSize: N_FFT,
      hopSize: HOP,
      sampleRate: SAMPLE_RATE,
      melBands: N_MELS,
      // disable power conversion; Meyda returns power spectrum by default
    });
    // mel shape: [frames][mel]
    const frames = mel.length;
    // pad/crop to TIME_FRAMES
    let arr = mel;
    if (frames < TIME_FRAMES) {
      const pad = new Array(TIME_FRAMES - frames).fill(new Array(N_MELS).fill(0));
      arr = mel.concat(pad);
    } else if (frames > TIME_FRAMES) {
      arr = mel.slice(0, TIME_FRAMES);
    }
    // transpose to [mel, time]
    const transposed = Array.from({ length: N_MELS }, (_, m) => arr.map((f) => f[m]));
    // min-max normalize
    let min = Infinity;
    let max = -Infinity;
    for (const row of transposed) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const range = max - min || 1e-6;
    const norm = transposed.map((row) => row.map((v) => (v - min) / range));
    return norm; // shape [128][TIME_FRAMES]
  }

  async function runInference() {
    if (!file) {
      setNotes('Select an audio file first.');
      return;
    }
    setLoading(true);
    setNotes('Running on-device inference...');
    try {
      const m = await ensureModel();
      const wav = await decodeAndResample(file);
      const mel = computeMel(wav);
      const input = tf.tensor(mel).reshape([1, N_MELS, TIME_FRAMES]);
      const preds = tf.tidy(() => m.predict(input));
      const probs = Array.from(preds.dataSync());
      const normalized = probs.map((p) => p / probs.reduce((a, b) => a + b, 0));
      const pairs = EMOTIONS.map((e, i) => ({ emotion: e, prob: normalized[i] || 0 }));
      pairs.sort((a, b) => b.prob - a.prob);
      setProbabilities(pairs);
      setNotes(`Top prediction: ${pairs[0].emotion} (${(pairs[0].prob * 100).toFixed(1)}%)`);
    } catch (err) {
      console.error(err);
      setNotes(err.message || 'Inference failed. Ensure model files exist.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="pill">GitHub Pages friendly · React + Vite</p>
        <h1>Audio Emotion Recognizer (UI)</h1>
        <p className="lede">
          A lightweight React front-end you can deploy on GitHub Pages. Upload an audio clip, preview it,
          and forward it to your own API or a TensorFlow.js model. No Streamlit or server required.
        </p>
        <div className="cta-row">
          <a className="btn primary" href="https://github.com/dmartinez203/Audio-Emotion-Recognizer" target="_blank" rel="noreferrer">
            View Repo
          </a>
          <a className="btn ghost" href="https://vitejs.dev/guide/" target="_blank" rel="noreferrer">
            Vite Guide
          </a>
        </div>
      </header>

      <main className="grid">
        <section className="card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Try it</p>
              <h2>Upload & Preview</h2>
            </div>
            <span className="tag">Frontend-only</span>
          </div>
          <p className="muted">Supported types: wav, mp3, ogg, flac (3-10s works best).</p>

          <label className="dropzone">
            <input type="file" accept="audio/*" onChange={onFileChange} />
            <div>
              <p className="drop-title">Drop audio here or click to browse</p>
              <p className="muted">Your file stays in-browser until you wire a backend.</p>
            </div>
          </label>

          {file && (
            <div className="file-summary">
              <div>
                <p className="label">File</p>
                <p className="value">{file.name}</p>
              </div>
              <div>
                <p className="label">Size</p>
                <p className="value">{formatBytes(file.size)}</p>
              </div>
              <div>
                <p className="label">Type</p>
                <p className="value">{file.type || 'audio/*'}</p>
              </div>
              <button className="btn tiny" onClick={clearSelection}>Clear</button>
            </div>
          )}

          {previewUrl && (
            <div className="player">
              <audio src={previewUrl} controls preload="metadata" />
              <p className="muted">Preview only. Add your inference call in code to get predictions.</p>
            </div>
          )}

          <div className="notice">
            <strong>Status:</strong> {notes}
          </div>

          <div className="actions">
            <button className="btn primary" onClick={runInference} disabled={!file || loading}>
              {loading ? 'Running…' : 'Run on-device inference'}
            </button>
          </div>

          <div className="next">
            <p className="label">Next actions</p>
            <ul>
              <li>Place your converted TF.js model at <code>public/models/best/model.json</code> (keeps weights locally).</li>
              <li>Or switch to a backend /predict endpoint if you prefer server inference.</li>
              <li>Add confidence bars for the eight emotions: {EMOTIONS.join(', ')}.</li>
            </ul>
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Hook it up</p>
              <h2>Suggested API contract</h2>
            </div>
          </div>
          <p className="muted">Drop this into your backend and point the frontend to it.</p>
          <pre className="code">
{`POST https://<your-api>/predict
Content-Type: multipart/form-data

file: <audio-file>

Response 200
{
  "emotion": "happy",
  "confidences": {
    "happy": 0.87,
    "surprised": 0.06,
    "neutral": 0.04,
    "angry": 0.03,
    "sad": 0.00,
    "fearful": 0.00,
    "disgust": 0.00,
    "calm": 0.00
  }
}`}
          </pre>
          <div className="note-box">
            <p className="label">Backend options</p>
            <ul>
              <li>Python FastAPI + TensorFlow: load your .h5 models and expose /predict.</li>
              <li>Node/Express + TF.js: host converted tfjs_graph_model files.</li>
              <li>Serverless (Cloudflare Workers / AWS Lambda): small tfjs-lite model.</li>
            </ul>
          </div>
        </section>

        {probabilities && (
          <section className="card">
            <div className="card-head">
              <div>
                <p className="eyebrow">Result</p>
                <h2>Predicted emotions</h2>
              </div>
            </div>
            <ul className="probs">
              {probabilities.map((p) => (
                <li key={p.emotion}>
                  <span>{p.emotion}</span>
                  <div className="bar">
                    <div
                      className="fill"
                      style={{ width: `${(p.prob * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="pct">{(p.prob * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Deploy</p>
              <h2>Publish to GitHub Pages</h2>
            </div>
            <span className="tag">Base: /Audio-Emotion-Recognizer/</span>
          </div>
          <ol className="steps">
            <li>cd frontend && npm install</li>
            <li>npm run build</li>
            <li>npm run deploy (uses gh-pages, pushes to gh-pages branch)</li>
            <li>GitHub → Settings → Pages → Source: gh-pages branch, / root</li>
          </ol>
          <p className="muted">
            Vite base is already set to <code>/Audio-Emotion-Recognizer/</code> for GitHub Pages.
            If you fork/rename the repo, update <code>vite.config.js</code> accordingly.
          </p>
        </section>

        <section className="card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Models</p>
              <h2>Where the models live</h2>
            </div>
          </div>
          <p className="muted">
            Training artifacts (.h5) and the notebook stay in the repo root. This UI is decoupled so you
            can evolve the backend independently. For client-side inference, convert a selected model to
            TensorFlow.js with:
          </p>
          <pre className="code small">
{`pip install tensorflowjs
tensorflowjs_converter \
  --input_format=keras \
  cnn_final_model.h5 \
  ./frontend/public/models/cnn_tfjs/`}
          </pre>
          <p className="muted">
            Then load the converted model in this React app with <code>@tensorflow/tfjs</code> and add inference
            logic inside <code>App.jsx</code>.
          </p>
        </section>
      </main>
    </div>
  );
}
