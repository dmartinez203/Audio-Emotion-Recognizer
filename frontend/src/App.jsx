import React, { useMemo, useState } from 'react';

const EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];

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

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    if (f) {
      setNotes('Ready to send this clip to your API or process offline.');
    } else {
      setNotes('Static frontend only — connect a backend API or TF.js model to enable predictions.');
    }
  };

  const clearSelection = () => {
    setFile(null);
    setNotes('Static frontend only — connect a backend API or TF.js model to enable predictions.');
  };

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

          <div className="next">
            <p className="label">Next actions</p>
            <ul>
              <li>Expose a POST /predict endpoint (e.g., FastAPI) and call it from this UI.</li>
              <li>Convert your Keras model to TF.js and run inference fully client-side.</li>
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
