"""
Toxic Comment Classification — Flask Deployment App
=====================================================
Serves predictions from the models trained in
Toxic_Comment_Classification_NLP_Pipeline.ipynb

Models wired up:
  - Logistic Regression (TF-IDF)   -> models/final_lr_models.pkl
  - XGBoost (TF-IDF)                -> models/xgb_models.pkl
  - BiLSTM (Keras embeddings)       -> models/bilstm_model.h5 + keras_tokenizer.json
  - BERT (transformers)             -> models/bert_toxic_model/

NOTE ON RoBERTa: the RoBERTa checkpoint is NOT included. The uploaded
RoBERTa files (config.json, tokenizer_config.json, special_tokens_map.json,
model.safetensors) shared the exact same filenames as the BERT files and
were overwritten on upload — only BERT's weights survived on disk. If you
re-export the RoBERTa model into its own folder (e.g.
models/roberta_toxic_model/), see the commented-out ROBERTA section below
to re-enable it.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model as load_keras_model
import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# ── NLTK (preprocessing) ──
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ── Keras / TensorFlow (BiLSTM) ──

# ── Transformers / PyTorch (BERT) ──

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

LABEL_COLS = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate']

BERT_MAXLEN = 128
BILSTM_MAXLEN = 200
DEFAULT_THRESHOLD = 0.5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Flask looks for templates in  <project_root>/templates/
# Make sure index.html is at:   templates/index.html
app = Flask(__name__, template_folder='templates')

# ─────────────────────────────────────────────────────────────────────────
# Preprocessing — EXACT same function used in the training notebook
# (Step 5). All models (including BERT) were trained on this cleaned
# text, so inference must use the identical pipeline for consistency.
# ─────────────────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(text, use_stemming=False):
    """Full NLP preprocessing pipeline (mirrors the training notebook)."""
    if text is None:
        return ''
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'<.*?>', '', text)                    # remove HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)                  # remove punctuation
    text = re.sub(r'\d+', '', text)                       # remove numbers
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────────────
# Load models once at startup
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading models — this may take a minute (BERT is large)...")
print("=" * 60)

# TF-IDF vectorizer (shared by LR and XGBoost)
with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)
print("✓ TF-IDF vectorizer loaded")

# Logistic Regression — one fitted model per label
with open(os.path.join(MODELS_DIR, 'final_lr_models.pkl'), 'rb') as f:
    lr_models = pickle.load(f)
print("✓ Logistic Regression models loaded:", list(lr_models.keys()))

# XGBoost — one fitted model per label
try:
    with open(os.path.join(MODELS_DIR, 'xgb_models.pkl'), 'rb') as f:
        xgb_models = pickle.load(f)
    print("✓ XGBoost models loaded:", list(xgb_models.keys()))
    XGB_AVAILABLE = True
except Exception as e:
    print(
        f"⚠️  XGBoost models could not be loaded ({e}) — disabling XGBoost route")
    xgb_models = None
    XGB_AVAILABLE = False

# BiLSTM + its Keras tokenizer
try:
    bilstm_model = load_keras_model(
        os.path.join(MODELS_DIR, 'bilstm_model.h5'))
    with open(os.path.join(MODELS_DIR, 'keras_tokenizer.json'), 'r', encoding='utf-8') as f:
        keras_tokenizer = tokenizer_from_json(f.read())
    print("✓ BiLSTM model + tokenizer loaded")
    BILSTM_AVAILABLE = True
except Exception as e:
    print(f"⚠️  BiLSTM could not be loaded ({e}) — disabling BiLSTM route")
    bilstm_model = None
    keras_tokenizer = None
    BILSTM_AVAILABLE = False

# BERT
try:
    BERT_DIR = os.path.join(MODELS_DIR, 'bert_toxic_model')
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_DIR).to(DEVICE)
    bert_model.eval()
    print(f"✓ BERT model loaded on {DEVICE}")
    BERT_AVAILABLE = True
except Exception as e:
    print(f"⚠️  BERT could not be loaded ({e}) — disabling BERT route")
    bert_tokenizer = None
    bert_model = None
    BERT_AVAILABLE = False

# ── RoBERTa ──
try:
    ROBERTA_DIR = os.path.join(MODELS_DIR, 'roberta_toxic_model')
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_DIR).to(DEVICE)
    roberta_model.eval()
    print(f"✓ RoBERTa model loaded on {DEVICE}")
    ROBERTA_AVAILABLE = True
except Exception as e:
    print(f"⚠️  RoBERTa could not be loaded ({e}) — disabling RoBERTa route")
    roberta_tokenizer = None
    roberta_model = None
    ROBERTA_AVAILABLE = False

print("=" * 60)
print("All available models loaded. Starting Flask app...")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────
# Inference helpers — one per model type, all return {label: probability}
# ─────────────────────────────────────────────────────────────────────────
def predict_lr(clean_text):
    vec = tfidf.transform([clean_text])
    return {col: float(lr_models[col].predict_proba(vec)[0][1]) for col in LABEL_COLS}


def predict_xgb(clean_text):
    vec = tfidf.transform([clean_text]).astype('float32')
    return {col: float(xgb_models[col].predict_proba(vec)[0][1]) for col in LABEL_COLS}


def predict_bilstm(clean_text):
    seq = keras_tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=BILSTM_MAXLEN,
                           padding='post', truncating='post')
    probs = bilstm_model.predict(padded, verbose=0)[0]
    return {col: float(p) for col, p in zip(LABEL_COLS, probs)}


def predict_bert(clean_text):
    enc = bert_tokenizer(
        clean_text, max_length=BERT_MAXLEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert_model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {col: float(p) for col, p in zip(LABEL_COLS, probs)}


def predict_roberta(clean_text):
    enc = roberta_tokenizer(
        clean_text, max_length=BERT_MAXLEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = roberta_model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {col: float(p) for col, p in zip(LABEL_COLS, probs)}


MODEL_REGISTRY = {
    'lr':      {'name': 'Logistic Regression', 'fn': predict_lr,     'available': True},
    'xgboost': {'name': 'XGBoost',             'fn': predict_xgb,    'available': XGB_AVAILABLE},
    'bilstm':  {'name': 'BiLSTM',              'fn': predict_bilstm, 'available': BILSTM_AVAILABLE},
    'bert':    {'name': 'BERT',                'fn': predict_bert,   'available': BERT_AVAILABLE},
    # RoBERTa is listed so the UI can show it as disabled when not loaded.
    # Re-enable by placing the model in models/roberta_toxic_model/ and
    # un-commenting the loading block at the top of this file.
    'roberta': {'name': 'RoBERTa ★', 'fn': predict_roberta, 'available': ROBERTA_AVAILABLE},
}


def run_prediction(comment, model_key, threshold=DEFAULT_THRESHOLD):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose from: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[model_key]
    if not entry['available']:
        raise RuntimeError(
            f"Model '{model_key}' is not available on this server.")

    clean_text = preprocess_text(comment)
    probs = entry['fn'](clean_text)
    labels = {col: int(p >= threshold) for col, p in probs.items()}
    return {
        'model_used': entry['name'],
        'comment': comment,
        'cleaned_text': clean_text,
        'threshold': threshold,
        'probabilities': {k: round(v, 4) for k, v in probs.items()},
        'predictions': labels,
        'is_toxic_any': int(any(labels.values())),
    }


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    available_models = [v['name']
                        for k, v in MODEL_REGISTRY.items() if v['available']]
    return render_template('index.html', models=MODEL_REGISTRY, available=available_models)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_available': {k: v['available'] for k, v in MODEL_REGISTRY.items()},
        'roberta_available': ROBERTA_AVAILABLE,
        'device': DEVICE,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON API.
    Body: { "comment": "some text", "model": "lr" | "xgboost" | "bilstm" | "bert",
            "threshold": 0.5 (optional) }
    """
    data = request.get_json(silent=True) or request.form
    comment = (data.get('comment') or '').strip()
    model_key = (data.get('model') or 'lr').strip().lower()
    threshold = float(data.get('threshold', DEFAULT_THRESHOLD))

    if not comment:
        return jsonify({'error': 'Field "comment" is required.'}), 400

    try:
        result = run_prediction(comment, model_key, threshold)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503


@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Run the comment through every available model for side-by-side comparison."""
    data = request.get_json(silent=True) or request.form
    comment = (data.get('comment') or '').strip()
    threshold = float(data.get('threshold', DEFAULT_THRESHOLD))

    if not comment:
        return jsonify({'error': 'Field "comment" is required.'}), 400

    results = {}
    for key, entry in MODEL_REGISTRY.items():
        if entry['available']:
            try:
                results[key] = run_prediction(comment, key, threshold)
            except Exception as e:
                results[key] = {'error': str(e)}
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
