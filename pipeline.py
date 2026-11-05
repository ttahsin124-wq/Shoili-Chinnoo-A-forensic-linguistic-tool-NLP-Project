"""Core ShoiliChinno pipeline: feature extraction, chunking, attribution and style-shift scoring.

Kept free of Streamlit so it can be shared by app.py and offline scripts
(e.g. calibrate_style_shift.py).
"""
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.spatial.distance import jensenshannon

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

# Same defaults as the training notebook (CONFIG cell).
DEFAULT_CHUNK_SIZE = 150
DEFAULT_CHUNK_OVERLAP = 40
MIN_CHUNK_WORDS = 60


@lru_cache(maxsize=1)
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


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP,
               min_words: int = MIN_CHUNK_WORDS):
    """Split text into overlapping word windows, matching the training notebook.

    Returns a list of (start_word, end_word, chunk_text). Text no longer than one
    chunk is returned whole. A too-short trailing fragment is replaced by a full
    window aligned to the end of the text, so the ending is still analysed.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [(0, len(words), " ".join(words))]

    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if len(chunk_words) < min_words:
            tail_start = len(words) - chunk_size
            if tail_start > chunks[-1][0]:
                chunks.append((tail_start, len(words), " ".join(words[tail_start:])))
            break
        chunks.append((start, start + len(chunk_words), " ".join(chunk_words)))
        if start + chunk_size >= len(words):
            break
    return chunks


def featurize_texts(texts, scaler, selector, config, embedder) -> np.ndarray:
    columns = config.get("stylometric_feature_columns")
    rows = []
    for t in texts:
        stylo = extract_stylometric_features(t)
        if columns is None:
            columns = list(stylo.keys())
        rows.append([stylo.get(col, 0.0) for col in columns])
    stylo_selected = selector.transform(scaler.transform(np.array(rows)))
    emb = embedder.encode(list(texts), convert_to_numpy=True)
    return np.hstack([stylo_selected, emb])


def class_scores(X: np.ndarray, model):
    """Per-row score distribution over classes, plus what kind of score it is."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X), "probability"
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True), "score"
    preds = model.predict(X)
    idx = np.searchsorted(model.classes_, preds)
    out = np.zeros((len(preds), len(model.classes_)))
    out[np.arange(len(preds)), idx] = 1.0
    return out, "vote"


def analyze_text(text: str, scaler, selector, label_encoder, model, config, embedder) -> dict:
    """Chunk the text, score every chunk, and aggregate to a document-level ranking.

    Document score = mean of chunk score vectors (soft voting).
    """
    chunks = chunk_text(
        text,
        config.get("chunk_size_words", DEFAULT_CHUNK_SIZE),
        config.get("chunk_overlap_words", DEFAULT_CHUNK_OVERLAP),
    )
    X = featurize_texts([c[2] for c in chunks], scaler, selector, config, embedder)
    chunk_probs, kind = class_scores(X, model)
    doc_probs = chunk_probs.mean(axis=0)
    chunk_top = chunk_probs.argmax(axis=1)
    doc_top = int(doc_probs.argmax())
    return {
        "authors": np.asarray(label_encoder.classes_),
        "chunks": chunks,
        "features": X,
        "chunk_probs": chunk_probs,
        "doc_probs": doc_probs,
        "kind": kind,
        "agreement": float((chunk_top == doc_top).mean()),
        "shifts": style_shifts(chunk_probs),
    }


def top_authors(result: dict, k: int = 5):
    idx = np.argsort(result["doc_probs"])[::-1][:k]
    return [(result["authors"][i], float(result["doc_probs"][i])) for i in idx]


def confidence_scores(doc_probs: np.ndarray) -> dict:
    """Document-level confidence scores (higher = more likely written by a known author)."""
    top2 = np.sort(doc_probs)[::-1][:2]
    return {"max_prob": float(top2[0]), "margin": float(top2[0] - top2[1])}


def domain_scores(X_chunks: np.ndarray, n_stylo: int, centroids: np.ndarray) -> dict:
    """How much a document resembles the training material (averaged over its chunks).

    news_similarity:   mean over chunks of the max cosine similarity between the chunk's
                       embedding and any known author's centroid (low = unlike training data).
    style_unusualness: mean absolute z-score of the selected stylometric features
                       (high = unusual style compared to training data).
    """
    E = X_chunks[:, n_stylo:]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return {
        "news_similarity": float((E @ centroids.T).max(axis=1).mean()),
        "style_unusualness": float(np.abs(X_chunks[:, :n_stylo]).mean()),
    }


def style_shifts(chunk_probs: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance (base 2, in [0, 1]) between each pair of consecutive chunks."""
    return np.array([
        float(jensenshannon(chunk_probs[i], chunk_probs[i + 1], base=2))
        for i in range(len(chunk_probs) - 1)
    ])


def find_segments(chunk_probs: np.ndarray, shifts: np.ndarray, threshold: float):
    """Split the chunk sequence wherever the style shift exceeds the threshold.

    Returns a list of (first_chunk, last_chunk, mean_probs) per segment.
    """
    boundaries = [i + 1 for i, s in enumerate(shifts) if s >= threshold]
    starts = [0] + boundaries
    ends = boundaries + [len(chunk_probs)]
    return [(s, e - 1, chunk_probs[s:e].mean(axis=0)) for s, e in zip(starts, ends)]
