# ✍️ ShoiliChinno — Personal Writing Style Fingerprinting
<!-- Top Banner / Title -->
<h1 align="center">✍️ ShoiliChinno</h1>
<p align="center">
  <strong>শৈলীচিহ্ন</strong> — <em>"The Signature of Your Style"</em>
</p>
<p align="center">
  An end-to-end NLP system that identifies <strong>who</strong> wrote a text based entirely on <strong>how</strong> they write — not what they say.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Sentence--BERT-0A0A0A?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="SBERT">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

**"শৈলীচিহ্ন" (Shoili Chinno) — "Style Mark/Signature"**

An end-to-end NLP system that identifies **who wrote a piece of text based on how they write**, not what they write about. Combines classical stylometry (punctuation habits, function-word frequency, sentence structure, readability) with modern transformer embeddings to perform **author attribution**, **author verification**, **unknown-author detection** and **style-change detection**, wrapped in a full interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Author Attribution** | Given a piece of text, ranks the most likely author out of all known writers in the dataset. Long texts are split into the same 150-word chunks the model was trained on and the chunk predictions are averaged |
| ❓ **Unknown-Author Detection** | Instead of always forcing a guess, the app answers **"no confident match"** when no known author fits strongly enough, and **"outside what the model knows"** when the text is unlike the news writing it was trained on (blogs, fiction, email…). Adjustable strictness slider |
| 🧭 **Style-Change Timeline** | For longer texts, shows which author each chunk resembles and flags points where the writing style shifts sharply — a lead for mixed authorship or an inserted passage |
| 🔍 **Author Verification** | Given two texts, estimates whether they were written by the same person using cosine similarity on style embeddings |
| 🧬 **Stylometric Profiling** | Standalone inspection tool — extracts and visualizes a text's punctuation habits, function-word usage, POS balance, readability scores, and vocabulary richness, independent of the classifier |
| 📂 **Batch Analysis** | Upload a CSV of many texts and get predictions for all of them at once, with the known-author check per row, optional accuracy scoring against known labels and CSV export |
| 📊 **Model Dashboard** | Live view of model type, test accuracy, document-level accuracy, embedding model, and all known authors |
| 🎨 **Polished Dark UI** | Custom-styled Streamlit dashboard with gauges, bar charts, donut charts, and verdict cards |

---

## 🏗️ How It Works — The Full Pipeline

```
Raw Articles (Reuters 50-50 dataset)
        │
        ▼
 ① Load & Consolidate ─── walks C50train/C50test folders → single structured DataFrame
        │
        ▼
 ② Clean ─── light-touch only (keeps punctuation/casing — that IS the style signal)
        │
        ▼
 ③ Chunk ─── splits articles into overlapping 150-word chunks (more training samples per author)
        │
        ▼
 ④ Stylometric Features ─── ~100+ handcrafted features per chunk:
        • Sentence length & word length stats     • Punctuation frequency (11+ characters)
        • 60+ function-word frequencies            • POS-tag ratios (noun/verb/adj/adv/pronoun)
        • Vocabulary richness (type-token ratio)   • Readability (Flesch, Gunning Fog, SMOG)
        • Character trigram frequencies            • Stopword ratio
        │
        ▼
 ⑤ Semantic Embeddings ─── Sentence-BERT (all-MiniLM-L6-v2) — 384-dim dense vectors
        │
        ▼
 ⑥ Feature Selection + Combine ─── SelectKBest keeps top-40 stylometric features,
        concatenated with embeddings → final feature vector
        │
        ▼
 ⑦ Train / Tune ─── GridSearchCV over Logistic Regression, Linear SVM, Random Forest
        │
        ▼
 ⑧ Evaluate ─── baseline comparison, confusion matrix, document-level (majority-vote) accuracy
        │
        ▼
 ⑨ Verification Model ─── cosine similarity between embeddings + ROC-tuned threshold
        │
        ▼
 ⑩ Save Pipeline ─── scaler, selector, label encoder, model, config → .joblib / .json
        │
        ▼
 ⑪ Calibrate ─── style-change threshold + unknown-author thresholds, measured on held-out test articles
        │
        ▼
 ⑫ Streamlit App ─── interactive UI wrapping the entire trained pipeline
```

### What happens when you paste a text into the app

```
Text ──► split into 150-word chunks (40-word overlap, same as training)
           │
           ├─► features + embeddings per chunk ──► author probabilities per chunk
           │                                            │
           │                     ┌──────────────────────┼─────────────────────────┐
           │                     ▼                      ▼                         ▼
           │            average over chunks    shift between neighbouring    domain check:
           │            → author ranking       chunks (Jensen–Shannon)       does this look like
           │                     │             → style-change timeline       the training data?
           │                     ▼                                                │
           │            top probability ≥ threshold? ◄────────────────────────────┘
           ▼
   ✅ Top match   |   ❓ No confident match   |   🌐 Outside what the model knows
```

---

## 📊 Model Performance

Evaluated on the full 50-author Reuters benchmark (50 authors, held-out test articles). The shipped model is **Logistic Regression** (C=10):

| Metric | Score |
|---|---|
| Random-guess baseline | ~1.9% |
| **Chunk-level test accuracy** | **~56.7%** |
| **Document-level accuracy (notebook, majority vote over chunks)** | **~65.0%** |
| **Document-level accuracy (app, averaged chunk probabilities)** | **~67.7%** |
| Same model, whole article as one input (old app behaviour) | ~61.3% |

> The last three rows were measured with the app's own code on 400 randomly sampled test articles; the first rows come from `pipeline_config.json`. A 50-way classification task is genuinely hard — ~57–68% with classical ML + embeddings is a strong, honest result. Some authors with distinctive style score 90%+ individually; others with more generic writing score lower — see the confusion matrix in the notebook.

### Unknown-author & style-change detection

All thresholds are measured on held-out Reuters test articles, never guessed. The scripts `calibrate_open_set.py` and `calibrate_style_shift.py` reproduce them.

| Check | Result |
|---|---|
| **Domain check** — real Reuters articles wrongly flagged as "outside what the model knows" | 2% |
| **Unknown news author** — texts by authors hidden from a retrained model that get "no confident match" (default strictness) | 37% |
| Accuracy of answers that pass the known-author check (default strictness) | 77% (vs 70% with no check) |
| **Style change** — two-author spliced documents caught | 81% |
| Style change — single-author documents wrongly flagged | 14% |

> Unknown-author detection was simulated by retraining the model with 10 of the 50 authors held out (5 folds) and checking whether their articles were rejected. Catching an unseen *news* writer is the hardest case, because they share the same house style; clearly non-news text (blogs, fiction, email, academic prose) is caught by the domain check.

---

## 📁 Project Structure

```
Shoilichinno/
├── app.py                              # 🎨 Streamlit interactive dashboard
├── pipeline.py                         # 🧠 Core logic: features, chunking, prediction, style shift, domain check
├── requirements.txt                    # 📦 Python dependencies
├── reuters_50_50.csv                   # 📄 Consolidated Reuters dataset
├── ShoiliChinno.ipynb                  # 📓 Main training & evaluation notebook
│
├── calibrate_style_shift.py            # 📏 Calibrates the style-change threshold
├── calibrate_open_set.py               # 📏 Calibrates unknown-author + domain checks
├── style_shift_calibration.json        # ⚙️ Output of calibrate_style_shift.py
├── open_set_calibration.json           # ⚙️ Output of calibrate_open_set.py
├── domain_centroids.npy                # ⚙️ Per-author embedding centroids for the domain check
│
├── pipeline_config.json                # ⚙️ Pipeline config (used by the app)
├── pipeline_config_v2.json             # ⚙️ Pipeline config (v2)
│
├── author_attribution_model.joblib     # 🤖 Trained classifier (used by the app)
├── author_attribution_model_v2.joblib  # 🤖 Trained classifier (v2)
├── author_label_encoder.joblib         # 🏷️ Author name ↔ class index
├── author_label_encoder_v2.joblib      # 🏷️ Author name ↔ class index (v2)
├── stylo_feature_selector.joblib       # 🔍 SelectKBest feature selector
├── stylo_feature_selector_v2.joblib    # 🔍 SelectKBest feature selector (v2)
├── stylo_scaler.joblib                 # 📊 Fitted StandardScaler
├── stylo_scaler_v2.joblib              # 📊 Fitted StandardScaler (v2)
│
├── .cache/                             # 🗃️ Cached chunk features from calibration (ignored by Git)
└── .venv/                              # 🐍 Virtual environment (ignored by Git)
```
## 📸 Live Demo Preview:
<div align="center">
  <img src="Screenshot 2026-09-04 121342.png" alt="Attribution" width="400"/>
  <img src="Screenshot 2026-09-04 121305.png" alt="Verification" width="400"/>
  <br>
  <em>Left: Predicting Author | Right: A small Bargraph</em>
</div>
<div align="center">
  <img src="Screenshot 2026-09-04 121250.png" alt="Attribution" width="400"/>
  <img src="Screenshot 2026-09-04 121305.png" alt="Verification" width="400"/>
  <br>
  <em>Left: Verifying Authorship| Right: Figure </em>
</div>
<div align="center">
  <img src="Screenshot 2026-09-04 121649.png" alt="Attribution" width="32%"/>
  <img src="Screenshot 2026-09-04 121710.png" alt="Verification" width="32%"/>
  <img src="Screenshot 2026-09-04 121758.png" alt="Profiling" width="32%"/>
  <br>
  <em> Stylometric Profile (Left) |   Analyzing Profile (Center) | Full Feature Breaddown (Right)</em>
</div>
<div align="center">
  <img src="Screenshot 2026-09-04 121931.png" alt="Attribution" width="400"/>
  <img src="Screenshot 2026-09-04 121948.png" alt="Verification" width="400"/>
  <br>
  <em>Left: Batch Analysis| Right: Breakdown </em>
</div>
<div align="center">
  <img src="Screenshot 2026-09-04 122007.png" alt="Attribution" width="400"/>
  <img src="Screenshot 2026-09-04 122034.png" alt="Verification" width="400"/>
  <br>
  <em>Left: About Section | Right: Details </em>
</div>



---

## 🧰 Tech Stack

- **[Streamlit](https://streamlit.io/)** — interactive web UI
- **[scikit-learn](https://scikit-learn.org/)** — Logistic Regression, Linear SVM, Random Forest, `GridSearchCV`, `SelectKBest`, `StandardScaler`, `LabelEncoder`
- **[Sentence-Transformers](https://www.sbert.net/)** — `all-MiniLM-L6-v2` for semantic embeddings
- **[NLTK](https://www.nltk.org/)** — tokenization, sentence splitting, POS tagging
- **[textstat](https://github.com/textstat/textstat)** — readability metrics (Flesch, Gunning Fog, SMOG)
- **[Plotly](https://plotly.com/python/)** — interactive bar charts, gauges, and donut charts in the UI
- **pandas / NumPy** — data wrangling and feature matrices
- **joblib** — model/artifact persistence

---

## 📦 Dataset

**Reuters 50-50 (C50)** — the standard academic benchmark for authorship attribution:

| Property | Value |
|---|---|
| Authors | 50 |
| Articles per author | 100 (50 train + 50 test) |
| Total articles | 5,000 |
| Total chunks (150-word, overlapping) | ~12,700 |
| Avg. words/article | ~506 |
| Format | Plain `.txt`, one article per file, pre-split into `C50train/` and `C50test/` |
| Genre | Reuters newswire (formal, edited journalistic writing) |

> **Note:** News writing is fairly formal and edited, which flattens some personal quirks. For a more personal fingerprinting demo, swap in informal data (WhatsApp exports, emails, blog posts) using the same `text` / `author` / `split` schema — the entire pipeline works unchanged.

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install streamlit plotly joblib numpy pandas scikit-learn sentence-transformers nltk textstat
```

### 2. Get the dataset
Use the included `reuters_50_50.csv`. If it's missing, the notebook falls back to reading the raw `C50train/` and `C50test/` folders from `reuters_extracted/`.

### 3. Run the notebook to train the pipeline
The trained artifacts are already included, so you can skip this step unless you want to retrain.
Open `ShoiliChinno.ipynb` in Jupyter/Colab and **Run All** (top to bottom, fresh kernel). This will:
- Load and chunk the dataset
- Extract stylometric features + embeddings
- Train and tune the attribution models
- Build the verification model
- Save all pipeline artifacts (`.joblib` + `.json`) needed by the app

> ⚠️ First run needs internet access — it downloads NLTK tokenizer data and the Sentence-BERT model weights. Both are cached locally afterward.

### 4. Recalibrate (only after retraining)
The calibration files are included too. If you retrain the model, regenerate them so the thresholds match the new model:
```bash
python calibrate_style_shift.py   # a few minutes
python calibrate_open_set.py      # ~20–30 min the first time (features are cached in .cache/ for reruns)
```
The app still runs without these files; the unknown-author check is then disabled and the style-change threshold falls back to an uncalibrated default.

### 5. Launch the app
Make sure the generated artifact files sit next to `app.py`, then:
```bash
streamlit run app.py
```
Open the printed URL (usually `http://localhost:8501`).

---

## 🚀 Usage Guide

### Predict Author
Paste any text (60+ words recommended; 150+ for the timeline) → get:
- a verdict: **✅ top match**, **❓ no confident match**, or **🌐 outside what the model knows**
- a ranked bar chart of the closest authors, with probabilities averaged over chunks
- for texts over 150 words, a **style timeline**: a heatmap of which author each chunk resembles, a style-shift chart, and a table of segments when a style change is detected

Use the **unknown-author strictness** slider to trade off between turning away genuine known-author texts and catching unknown writers; the caption under the result shows the measured error rates for the chosen setting.

### Verify Authorship
Paste two texts → get a similarity gauge and a clear "same author / different author" verdict, with an adjustable threshold slider.

### Stylometric Profile
Paste any text → see its raw stylistic fingerprint: top function words, POS balance, readability scores, and the full feature table — no model prediction needed, pure feature inspection.

### Batch Analysis
Upload a CSV with a `text` column (optionally a `label` column for ground truth) → get predictions for every row with its known-author check result, an accuracy score if labels are provided (overall and on rows that passed the check), and a downloadable results CSV.

---

## 🔬 Stylometric Features Used

- **Structure:** average sentence length, sentence-length std. dev., average word length (+ std. dev.)
- **Punctuation:** frequency of `. , ; : ! ? - " ' ( )`, ellipses, and em-dashes (per 100 words)
- **Function words:** frequency of 60+ common words (the, and, but, however, therefore, etc.) — subconscious and hard to fake
- **Part-of-speech ratios:** noun / verb / adjective / adverb / pronoun balance
- **Vocabulary richness:** type-token ratio
- **Readability:** Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG Index, avg. syllables/word
- **Character trigrams:** frequency of common 3-letter sequences (ing, the, ent, ion, etc.)
- **Habits:** capitalization ratio, all-caps word ratio, stopword ratio

Combined with **384-dimensional Sentence-BERT embeddings** for semantic/subtle style signal.

---

## ⚠️ Known Limitations

- Trained on **formal news writing** — informal text (chat, casual email) is usually flagged as "outside what the model knows" rather than attributed.
- Catching an **unknown news writer** is only moderately reliable (~37% at default strictness): they share Reuters' house style with the known authors.
- The unknown-author threshold was calibrated on a 40-author model and applied to the 50-author one, so its error rates are estimates.
- **Style-change flags are leads, not proof** — ~14% of single-author texts get a false flag.
- The verification threshold is tuned on the Reuters test-set similarity distribution; it may need recalibration for very different text types or lengths.
- This is a research/portfolio demo, **not** a forensic or legally admissible authorship tool.
- Short texts (under ~60 words) produce less reliable predictions, since the model was trained on chunks of that minimum length.

---

## 🗺️ Roadmap Ideas

- [x] Chunk-level prediction with probability averaging
- [x] Style-change timeline for mixed-authorship detection
- [x] Unknown-author ("none of the above") detection with a domain check
- [ ] Make verification style-based (e.g. Burrows' Delta) instead of embedding-only, which partly measures topic
- [ ] Fine-tune on personal data (WhatsApp/email exports) for a truly personalized fingerprint
- [ ] Replace cosine-similarity verification with a trained Siamese network for a learned similarity metric
- [ ] Add SHAP/coefficient-based explainability — show *which* features drove a given prediction
- [ ] Support larger embedding models (e.g. `all-mpnet-base-v2`) as an accuracy/speed tradeoff option
- [ ] Add multi-language support for non-English writing style fingerprinting

---

## 📄 License

This project is licensed under the MIT License — free to use, modify, and distribute, with attribution.

---

## 🙌 Acknowledgements

Built with scikit-learn, Sentence-Transformers, NLTK, textstat, and Streamlit. Dataset: Reuters 50-50 (C50), a standard authorship-attribution benchmark.

> *ShoiliChinno — every writer leaves a signature, even without a name.*
