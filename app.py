import json
import re
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="ShoiliChinno — Writing Style Fingerprinting",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E6E6E6;
    }
    section[data-testid="stSidebar"] {
        background-color: #14161F;
        border-right: 1px solid #262838;
    }
    h1, h2, h3 { color: #F5F5F7; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C29;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: #B0B0C0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6C63FF !important;
        color: white !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1A1C29;
        border: 1px solid #2A2C3D;
        border-radius: 12px;
        padding: 14px 18px;
    }
    .fingerprint-card {
        background-color: #1A1C29;
        border: 1px solid #2A2C3D;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 14px;
    }
    .verdict-same {
        background: linear-gradient(135deg, #1f3d2e, #14261c);
        border: 1px solid #3fae6a;
        border-radius: 14px;
        padding: 18px 22px;
    }
    .verdict-diff {
        background: linear-gradient(135deg, #3d1f1f, #261414);
        border: 1px solid #ae3f3f;
        border-radius: 14px;
        padding: 18px 22px;
    }
    .stButton>button {
        background-color: #6C63FF;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.2em;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #5a52e0; }
</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = Path(__file__).parent

REQUIRED_FILES = {
    "scaler": "stylo_scaler.joblib",
    "selector": "stylo_feature_selector.joblib",
    "label_encoder": "author_label_encoder.joblib",
    "model": "author_attribution_model.joblib",
    "config": "pipeline_config.json",
}


FUNCTION_WORDS = [
    "the", "a", "an", "and", "but", "or", "so", "yet", "for", "nor",
    "in", "on", "at", "by", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "of", "off", "over", "under",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "not", "no", "very", "just", "only", "also", "however", "therefore",
]
PUNCT_CHARS = [".", ",", ";", ":", "!", "?", "-", '"', "'", "(", ")"]
COMMON_TRIGRAMS = [
    "ing", "the", "ent", "ion", "and", "for", "her", "his", "tha", "was",
    "ere", "ate", "all", "you", "ear", "str", "men", "con", "pro",
]


@st.cache_resource(show_spinner=False)
def get_nltk_ready():
    """Download NLTK data once, quietly. Requires internet on first run."""
    import nltk
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True


def extract_stylometric_features(text: str) -> dict:
    get_nltk_ready()
    from nltk import word_tokenize, sent_tokenize, pos_tag

    try:
        import textstat
        has_textstat = True
    except ImportError:
        has_textstat = False

    words = word_tokenize(text)
    words_lower = [w.lower() for w in words if w.isalpha()]
    sentences = sent_tokenize(text)

    n_words = max(len(words_lower), 1)
    n_sentences = max(len(sentences), 1)
    sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]

    features = {}
    features["avg_sentence_length"] = n_words / n_sentences
    features["avg_word_length"] = np.mean([len(w) for w in words_lower]) if words_lower else 0
    features["type_token_ratio"] = len(set(words_lower)) / n_words
    features["sentence_length_std"] = np.std(sentence_lengths) if sentence_lengths else 0

    for p in PUNCT_CHARS:
        features[f"punct_{p}"] = (text.count(p) / n_words) * 100
    features["punct_ellipsis"] = (text.count("...") / n_words) * 100
    features["punct_emdash"] = (text.count("—") / n_words) * 100

    word_counts = Counter(words_lower)
    for fw in FUNCTION_WORDS:
        features[f"fw_{fw}"] = (word_counts.get(fw, 0) / n_words) * 100

    try:
        tags = pos_tag(words)
        tag_counts = Counter(tag for _, tag in tags)
        total_tags = max(sum(tag_counts.values()), 1)
        pos_groups = {
            "pos_noun": ["NN", "NNS", "NNP", "NNPS"],
            "pos_verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            "pos_adj": ["JJ", "JJR", "JJS"],
            "pos_adv": ["RB", "RBR", "RBS"],
            "pos_pron": ["PRP", "PRP$"],
        }
        for group_name, tag_list in pos_groups.items():
            features[group_name] = sum(tag_counts.get(t, 0) for t in tag_list) / total_tags
    except Exception:
        for group_name in ["pos_noun", "pos_verb", "pos_adj", "pos_adv", "pos_pron"]:
            features[group_name] = 0.0

    features["avg_word_length_std"] = np.std([len(w) for w in words_lower]) if words_lower else 0
    features["capitalized_word_ratio"] = sum(1 for w in words if w.istitle()) / n_words
    features["uppercase_word_ratio"] = sum(1 for w in words if w.isupper() and len(w) > 1) / n_words

    if has_textstat:
        try:
            features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            features["gunning_fog"] = textstat.gunning_fog(text)
            features["smog_index"] = textstat.smog_index(text)
            features["avg_syllables_per_word"] = textstat.avg_syllables_per_word(text)
        except Exception:
            has_textstat = False
    if not has_textstat:
        avg_word_len = features["avg_word_length"]
        avg_sent_len = features["avg_sentence_length"]
        est_syllables = 1 + avg_word_len / 4
        flesch = 206.835 - 1.015 * avg_sent_len - 84.6 * est_syllables
        flesch_kincaid = 0.39 * avg_sent_len + 11.8 * est_syllables - 15.59
        features["flesch_reading_ease"] = max(0, min(100, flesch))
        features["flesch_kincaid_grade"] = max(0, flesch_kincaid)
        features["gunning_fog"] = 0.4 * (avg_sent_len + 100 * (1 - features["type_token_ratio"]))
        features["smog_index"] = 1.043 * np.sqrt(max(0, 30 * (1 - features["type_token_ratio"])) + 3.1291)
        features["avg_syllables_per_word"] = est_syllables

    clean = re.sub(r"[^a-zA-Z]", " ", text).lower()
    trigram_counts = {}
    for i in range(len(clean) - 2):
        tri = clean[i:i + 3]
        if " " not in tri:
            trigram_counts[tri] = trigram_counts.get(tri, 0) + 1
    for tri in COMMON_TRIGRAMS:
        features[f"tri_{tri}"] = (trigram_counts.get(tri, 0) / n_words) * 100

    stopwords = set(FUNCTION_WORDS)
    features["stopword_ratio"] = sum(1 for w in words_lower if w in stopwords) / n_words

    return features



def missing_files():
    return [name for name, fname in REQUIRED_FILES.items() if not (ARTIFACT_DIR / fname).exists()]


@st.cache_resource(show_spinner="Loading trained pipeline...")
def load_artifacts():
    scaler = joblib.load(ARTIFACT_DIR / REQUIRED_FILES["scaler"])
    selector = joblib.load(ARTIFACT_DIR / REQUIRED_FILES["selector"])
    label_encoder = joblib.load(ARTIFACT_DIR / REQUIRED_FILES["label_encoder"])
    model = joblib.load(ARTIFACT_DIR / REQUIRED_FILES["model"])
    with open(ARTIFACT_DIR / REQUIRED_FILES["config"]) as f:
        config = json.load(f)
    return scaler, selector, label_encoder, model, config


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it, please wait)...")
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def featurize_text(text: str, scaler, selector, label_encoder, config, embedder) -> np.ndarray:
    stylo = extract_stylometric_features(text)
    columns = config.get("stylometric_feature_columns")
    if columns is None:
        
        columns = list(stylo.keys())
    stylo_vec = np.array([[stylo.get(col, 0.0) for col in columns]])
    stylo_scaled = scaler.transform(stylo_vec)
    stylo_selected = selector.transform(stylo_scaled)
    emb = embedder.encode([text], convert_to_numpy=True)
    return np.hstack([stylo_selected, emb])


def predict_author(text: str, scaler, selector, label_encoder, model, config, embedder, top_k: int = 5):
    vec = featurize_text(text, scaler, selector, label_encoder, config, embedder)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        kind = "probability"
        return [(label_encoder.inverse_transform([i])[0], float(proba[i])) for i in top_idx], kind
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        s_min, s_max = scores.min(), scores.max()
        norm = (scores - s_min) / (s_max - s_min + 1e-9)
        top_idx = np.argsort(scores)[::-1][:top_k]
        kind = "score"
        return [(label_encoder.inverse_transform([i])[0], float(norm[i])) for i in top_idx], kind
    pred = model.predict(vec)[0]
    return [(label_encoder.inverse_transform([pred])[0], None)], "label_only"


def verify_authorship(text_a: str, text_b: str, config, embedder, threshold: float = None):
    threshold = threshold if threshold is not None else config.get("verification_threshold", 0.5)
    emb_a = embedder.encode([text_a], convert_to_numpy=True)
    emb_b = embedder.encode([text_b], convert_to_numpy=True)
    similarity = float(cosine_similarity(emb_a, emb_b)[0][0])
    return {
        "similarity": similarity,
        "threshold": float(threshold),
        "same_author": similarity >= threshold,
    }


def word_count(text: str) -> int:
    return len(text.split())


MIN_RECOMMENDED_WORDS = 60 



with st.sidebar:
    st.markdown("## ✍️ ShoiliChinno")
    st.caption("Personal Writing Style Fingerprinting")
    st.divider()

    missing = missing_files()
    if missing:
        st.error("⚠️ Missing trained artifacts")
        st.markdown(
            "This app needs the following files saved next to `app.py` "
            "(generated by running your notebook's save-pipeline cell):"
        )
        for m in missing:
            st.code(REQUIRED_FILES[m])
        st.stop()

    scaler, selector, label_encoder, model, config = load_artifacts()
    embedder = load_embedder(config.get("embedding_model", "all-MiniLM-L6-v2"))

    st.success("✅ Pipeline loaded")
    st.metric("Known Authors", len(label_encoder.classes_))
    st.metric("Model", config.get("best_model_name", "Unknown"))
    if "test_accuracy" in config:
        st.metric("Test Accuracy", f"{config['test_accuracy']*100:.1f}%")
    if "document_level_accuracy" in config:
        st.metric("Document-Level Accuracy", f"{config['document_level_accuracy']*100:.1f}%")

    st.divider()
    st.caption(f"Embedding model: `{config.get('embedding_model', '—')}`")
    st.caption(f"Chunk size: {config.get('chunk_size_words', '—')} words")
    st.caption(f"Verification threshold: {config.get('verification_threshold', 0.5):.3f}")

    with st.expander("📋 All known authors"):
        st.write(", ".join(sorted(label_encoder.classes_)))


st.markdown("# ✍️ Writing Style Fingerprinting")
st.caption(
    "Identify authors from writing style alone, verify whether two texts share an author, "
    "and inspect the stylometric fingerprint of any piece of text."
)

tab_predict, tab_verify, tab_profile, tab_batch, tab_about = st.tabs(
    ["🔮 Predict Author", "🔍 Verify Authorship", "🧬 Stylometric Profile", "📂 Batch Analysis", "ℹ️ About"]
)


with tab_predict:
    st.markdown("### Who wrote this?")
    st.write(
        f"Paste a piece of text below and the model will rank the most likely authors "
        f"out of its {len(label_encoder.classes_)} known writers. "
        f"For best results, use at least **{MIN_RECOMMENDED_WORDS} words**."
    )

    col_input, col_settings = st.columns([3, 1])
    with col_input:
        predict_text = st.text_area(
            "Text to analyze",
            height=220,
            placeholder="Paste an article, essay, or any piece of writing here...",
            key="predict_text",
        )
    with col_settings:
        top_k = st.slider("Show top N authors", min_value=1, max_value=10, value=5)
        st.write("")
        run_predict = st.button("🔮 Predict Author", use_container_width=True)

    if predict_text:
        wc = word_count(predict_text)
        if wc < MIN_RECOMMENDED_WORDS:
            st.warning(
                f"This text is only {wc} words. Predictions on short text are less reliable — "
                f"the model was trained on chunks of at least {MIN_RECOMMENDED_WORDS} words."
            )

    if run_predict:
        if not predict_text or not predict_text.strip():
            st.error("Please paste some text first.")
        else:
            with st.spinner("Analyzing writing style..."):
                results, kind = predict_author(
                    predict_text, scaler, selector, label_encoder, model, config, embedder, top_k=top_k
                )

            if kind == "label_only":
                st.info(f"**Predicted author:** {results[0][0]}")
            else:
                label_name = "Probability" if kind == "probability" else "Relative confidence"
                df_res = pd.DataFrame(results, columns=["Author", label_name])

                st.markdown(
                    f'<div class="fingerprint-card">'
                    f'<h4 style="margin-top:0">Top match: {results[0][0]}</h4>'
                    f'<p style="color:#9a9ab0;margin-bottom:0">'
                    f'{label_name}: {results[0][1]*100:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )

                fig = px.bar(
                    df_res.sort_values(label_name),
                    x=label_name, y="Author", orientation="h",
                    color=label_name, color_continuous_scale="Purples",
                )
                fig.update_layout(
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                    font_color="#E6E6E6", height=80 + 40 * len(df_res),
                    xaxis_tickformat=".0%", coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                if kind == "score":
                    st.caption(
                        "ℹ️ This model (Linear SVM) doesn't output true probabilities — bars show "
                        "normalized decision scores for relative ranking, not calibrated confidence."
                    )


with tab_verify:
    st.markdown("### Did the same person write both?")
    st.write("Paste two pieces of text — the model compares their style embeddings and estimates whether they share an author.")

    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area("Text A", height=200, key="verify_a", placeholder="First piece of text...")
    with col_b:
        text_b = st.text_area("Text B", height=200, key="verify_b", placeholder="Second piece of text...")

    custom_threshold = st.slider(
        "Similarity threshold (higher = stricter)",
        min_value=0.0, max_value=1.0,
        value=float(config.get("verification_threshold", 0.5)), step=0.01,
    )

    if st.button("🔍 Verify Authorship", use_container_width=False):
        if not text_a.strip() or not text_b.strip():
            st.error("Please paste text into both boxes.")
        else:
            wc_a, wc_b = word_count(text_a), word_count(text_b)
            if wc_a < MIN_RECOMMENDED_WORDS or wc_b < MIN_RECOMMENDED_WORDS:
                st.warning(
                    f"Text A: {wc_a} words · Text B: {wc_b} words. "
                    f"Results are more reliable above {MIN_RECOMMENDED_WORDS} words each."
                )
            with st.spinner("Comparing writing styles..."):
                result = verify_authorship(text_a, text_b, config, embedder, threshold=custom_threshold)

            verdict_class = "verdict-same" if result["same_author"] else "verdict-diff"
            verdict_text = "✅ Likely the SAME author" if result["same_author"] else "❌ Likely DIFFERENT authors"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["similarity"] * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#6C63FF"},
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": result["threshold"] * 100,
                    },
                    "steps": [
                        {"range": [0, result["threshold"] * 100], "color": "#3d1f1f"},
                        {"range": [result["threshold"] * 100, 100], "color": "#1f3d2e"},
                    ],
                },
                title={"text": "Style Similarity"},
            ))
            fig.update_layout(
                paper_bgcolor="#0E1117", font_color="#E6E6E6", height=280,
                margin=dict(l=20, r=20, t=50, b=10),
            )

            col_gauge, col_verdict = st.columns([1, 1])
            with col_gauge:
                st.plotly_chart(fig, use_container_width=True)
            with col_verdict:
                st.markdown(
                    f'<div class="{verdict_class}">'
                    f'<h3 style="margin-top:0">{verdict_text}</h3>'
                    f'<p>Similarity: <b>{result["similarity"]:.3f}</b></p>'
                    f'<p>Threshold: <b>{result["threshold"]:.3f}</b></p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "This is a cosine-similarity check on style embeddings, tuned via ROC analysis "
                    "on the test set — not a legal or forensic determination."
                )


with tab_profile:
    st.markdown("### Inspect the stylometric fingerprint of any text")
    st.write("This shows the raw stylometric signals the model uses — independent of any prediction.")

    profile_text = st.text_area(
        "Text to profile", height=200, key="profile_text",
        placeholder="Paste text to see its punctuation habits, function-word usage, readability, and more...",
    )

    if st.button("🧬 Analyze Style", use_container_width=False):
        if not profile_text.strip():
            st.error("Please paste some text first.")
        else:
            with st.spinner("Extracting stylometric features..."):
                feats = extract_stylometric_features(profile_text)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg. sentence length", f"{feats['avg_sentence_length']:.1f} words")
            c2.metric("Avg. word length", f"{feats['avg_word_length']:.2f} chars")
            c3.metric("Vocabulary richness", f"{feats['type_token_ratio']:.2f}")
            c4.metric("Flesch reading ease", f"{feats.get('flesch_reading_ease', 0):.0f}")

            col_fw, col_pos = st.columns(2)

            with col_fw:
                st.markdown("**Top function words (per 100 words)**")
                fw_items = sorted(
                    [(k.replace("fw_", ""), v) for k, v in feats.items() if k.startswith("fw_")],
                    key=lambda x: x[1], reverse=True,
                )[:10]
                fw_df = pd.DataFrame(fw_items, columns=["Word", "Frequency"])
                fig_fw = px.bar(fw_df, x="Frequency", y="Word", orientation="h", color_discrete_sequence=["#6C63FF"])
                fig_fw.update_layout(
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="#E6E6E6",
                    height=350, margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_fw, use_container_width=True)

            with col_pos:
                st.markdown("**Part-of-speech balance**")
                pos_items = [
                    ("Noun", feats.get("pos_noun", 0)), ("Verb", feats.get("pos_verb", 0)),
                    ("Adjective", feats.get("pos_adj", 0)), ("Adverb", feats.get("pos_adv", 0)),
                    ("Pronoun", feats.get("pos_pron", 0)),
                ]
                pos_df = pd.DataFrame(pos_items, columns=["POS", "Ratio"])
                fig_pos = px.pie(pos_df, names="POS", values="Ratio", hole=0.5,
                                  color_discrete_sequence=px.colors.sequential.Purples_r)
                fig_pos.update_layout(
                    paper_bgcolor="#0E1117", font_color="#E6E6E6", height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_pos, use_container_width=True)

            with st.expander("📄 Full feature breakdown"):
                full_df = pd.DataFrame(sorted(feats.items()), columns=["Feature", "Value"])
                st.dataframe(full_df, use_container_width=True, height=350)


with tab_batch:
    st.markdown("### Predict authors for many texts at once")
    st.write(
        "Upload a CSV with a `text` column (an optional `label` column is shown alongside "
        "predictions for comparison). Each row is treated as one document."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            batch_df = None

        if batch_df is not None:
            if "text" not in batch_df.columns:
                st.error("CSV must have a column named `text`.")
            else:
                st.write(f"Loaded {len(batch_df)} rows.")
                max_rows = st.slider("Rows to process (capped to keep this fast)", 1, min(200, len(batch_df)), min(20, len(batch_df)))
                if st.button("📂 Run Batch Prediction", use_container_width=False):
                    progress = st.progress(0, text="Starting...")
                    rows = []
                    subset = batch_df.head(max_rows)
                    for i, (_, row) in enumerate(subset.iterrows()):
                        text = str(row["text"])
                        try:
                            preds, kind = predict_author(
                                text, scaler, selector, label_encoder, model, config, embedder, top_k=1
                            )
                            pred_author, score = preds[0]
                        except Exception as e:
                            pred_author, score = f"ERROR: {e}", None
                        result_row = {"text_preview": text[:80] + ("..." if len(text) > 80 else ""), "predicted_author": pred_author}
                        if "label" in batch_df.columns:
                            result_row["true_author"] = row["label"]
                            result_row["correct"] = (row["label"] == pred_author)
                        rows.append(result_row)
                        progress.progress((i + 1) / max_rows, text=f"Processing {i+1}/{max_rows}")
                    progress.empty()

                    results_df = pd.DataFrame(rows)
                    st.dataframe(results_df, use_container_width=True)

                    if "correct" in results_df.columns:
                        acc = results_df["correct"].mean()
                        st.metric("Batch accuracy", f"{acc*100:.1f}%")

                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download results as CSV", data=csv_bytes,
                        file_name="batch_predictions.csv", mime="text/csv",
                    )


with tab_about:
    st.markdown("""
### About ShoiliChinno

**ShoiliChinno** ("style fingerprint") identifies who wrote a piece of text based on *how*
they write, not *what* they write about. It combines two signal types:

- **Stylometric features** — sentence length, punctuation habits, function-word frequency,
  part-of-speech balance, vocabulary richness, and readability scores.
- **Semantic embeddings** — dense sentence representations from a pretrained transformer,
  capturing subtler stylistic patterns handcrafted rules miss.

#### Two capabilities
1. **Author Attribution** — given text, rank which of the known authors most likely wrote it.
2. **Author Verification** — given two texts, estimate whether they share an author, using
   cosine similarity between style embeddings and a threshold tuned via ROC analysis.

#### Model details
""")
    info_cols = st.columns(3)
    info_cols[0].metric("Model type", config.get("best_model_name", "—"))
    info_cols[1].metric("Embedding model", config.get("embedding_model", "—"))
    info_cols[2].metric("Chunk size", f"{config.get('chunk_size_words', '—')} words")

    st.markdown("""
#### Honest limitations
- Trained on news-style writing (Reuters articles) — informal writing (chat, email) may perform differently.
- Verification threshold is tuned on the test set distribution; it may need recalibration for very different text types.
- This is a demo/portfolio project, not a forensic or legal authorship tool.
""")