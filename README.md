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

An end-to-end NLP system that identifies **who wrote a piece of text based on how they write**, not what they write about. Combines classical stylometry (punctuation habits, function-word frequency, sentence structure, readability) with modern transformer embeddings to perform **author attribution** and **author verification**, wrapped in a full interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Author Attribution** | Given a piece of text, ranks the most likely author out of all known writers in the dataset, with confidence scores |
| 🔍 **Author Verification** | Given two texts, estimates whether they were written by the same person using cosine similarity on style embeddings |
| 🧬 **Stylometric Profiling** | Standalone inspection tool — extracts and visualizes a text's punctuation habits, function-word usage, POS balance, readability scores, and vocabulary richness, independent of the classifier |
| 📂 **Batch Analysis** | Upload a CSV of many texts and get predictions for all of them at once, with optional accuracy scoring against known labels and CSV export |
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
 ⑪ Streamlit App ─── interactive UI wrapping the entire trained pipeline
```

---

## 📊 Model Performance

Evaluated on the full 50-author Reuters benchmark (50 authors, 2,500 test chunks):

| Metric | Score |
|---|---|
| Random-guess baseline | ~2.0% |
| Majority-class baseline | ~2.3% |
| **Chunk-level test accuracy (Linear SVM)** | **~57.3%** |
| **Document-level accuracy (majority vote)** | **~63.7%** |
| Improvement over random guessing | **~27.6x** |

> A 50-way classification task is genuinely hard — for reference, published academic baselines on this exact dataset typically fall in the 70–85% range with heavy tuning. ~57–64% with classical ML + embeddings is a strong, honest result, not a failure. Some authors (e.g. those with very distinctive style) score 90%+ individually; others with more generic writing score lower — see the confusion matrix in the notebook for the full breakdown.

---

## 📁 Project Structure

```
shoilichinno/
├── app.py                              # Streamlit UI (attribution, verification, profiling, batch)
├── requirements.txt                    # Python dependencies for the app
├── SETUP.md                            # App setup & run instructions
│
├── load_reuters_dataset.py             # Script: consolidates raw C50train/C50test → one CSV
├── reuters_50_50.csv                   # Consolidated dataset (5,000 articles, 50 authors)
│
├── writing_style_fingerprinting.ipynb     # v1: baseline pipeline notebook
├── writing_style_fingerprinting_v2.ipynb  # v2: tuned pipeline (baselines, GridSearchCV, feature selection)
├── ShoiliChinno_fixed.ipynb               # Final consolidated & bug-fixed notebook
│
└── (generated after running the notebook)
    ├── stylo_scaler.joblib              # Fitted StandardScaler for stylometric features
    ├── stylo_feature_selector.joblib    # Fitted SelectKBest feature selector
    ├── author_label_encoder.joblib      # LabelEncoder mapping author names ↔ class indices
    ├── author_attribution_model.joblib  # Trained best classifier (Logistic Reg / SVM / RF)
    └── pipeline_config.json             # Config: embedding model, chunk size, threshold, etc.
```

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
Either use the included `reuters_50_50.csv`, or regenerate it from raw folders:
```bash
python load_reuters_dataset.py --dataset_root /path/to/reuters_extracted --out reuters_50_50.csv
```

### 3. Run the notebook to train the pipeline
Open `ShoiliChinno_fixed.ipynb` in Jupyter/Colab and **Run All** (top to bottom, fresh kernel). This will:
- Load and chunk the dataset
- Extract stylometric features + embeddings
- Train and tune the attribution models
- Build the verification model
- Save all pipeline artifacts (`.joblib` + `.json`) needed by the app

> ⚠️ First run needs internet access — it downloads NLTK tokenizer data and the Sentence-BERT model weights. Both are cached locally afterward.

### 4. Launch the app
Make sure the generated artifact files sit next to `app.py`, then:
```bash
streamlit run app.py
```
Open the printed URL (usually `http://localhost:8501`).

---

## 🚀 Usage Guide

### Predict Author
Paste any text (60+ words recommended) → get a ranked bar chart of the most likely authors with confidence scores.

### Verify Authorship
Paste two texts → get a similarity gauge and a clear "same author / different author" verdict, with an adjustable threshold slider.

### Stylometric Profile
Paste any text → see its raw stylistic fingerprint: top function words, POS balance, readability scores, and the full feature table — no model prediction needed, pure feature inspection.

### Batch Analysis
Upload a CSV with a `text` column (optionally a `label` column for ground truth) → get predictions for every row, an accuracy score if labels are provided, and a downloadable results CSV.

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

- Trained on **formal news writing** — informal text (chat, casual email) may behave differently until fine-tuned on that domain.
- `LinearSVC` (the best-performing model in testing) has no native probability output — the app shows normalized decision scores for ranking, not calibrated probabilities.
- The verification threshold is tuned on the Reuters test-set similarity distribution; it may need recalibration for very different text types or lengths.
- This is a research/portfolio demo, **not** a forensic or legally admissible authorship tool.
- Short texts (under ~60 words) produce less reliable predictions, since the model was trained on chunks of that minimum length.

---

## 🗺️ Roadmap Ideas

- [ ] Fine-tune on personal data (WhatsApp/email exports) for a truly personalized fingerprint
- [ ] Replace cosine-similarity verification with a trained Siamese network for a learned similarity metric
- [ ] Add SHAP/coefficient-based explainability — show *which* features drove a given prediction
- [ ] Support larger embedding models (e.g. `all-mpnet-base-v2`) as an accuracy/speed tradeoff option
- [ ] Add multi-language support for non-English writing style fingerprinting

---

## 📄 License

This project currently has no explicit license. Add a `LICENSE` file (MIT, Apache 2.0, etc.) before sharing or distributing.

---

## 🙌 Acknowledgements

Built with scikit-learn, Sentence-Transformers, NLTK, textstat, and Streamlit. Dataset: Reuters 50-50 (C50), a standard authorship-attribution benchmark.

> *ShoiliChinno — every writer leaves a signature, even without a name.*
