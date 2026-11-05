"""Calibrate the "unknown author" (open-set) rejection used by the Predict tab.

The production model knows all 50 authors, so we can't test it on unknown writers directly.
Instead we simulate: retrain a clone of the model with 10 authors held out, then score
held-out TEST articles from both the 40 known and the 10 unknown authors. A good confidence
score should be high for known authors and low for unknown ones. Repeated over 5 folds.

It also calibrates a separate DOMAIN check. The model is over-confident on writing unlike
its Reuters training data (blogs, fiction, email...), so we additionally measure how far a
text is from the training material: similarity of its embeddings to the per-author
centroids, and how unusual its stylometric features are. Thresholds are low/high
percentiles over Reuters test articles.

Run:  .venv/Scripts/python.exe calibrate_open_set.py
Writes: open_set_calibration.json, domain_centroids.npy
        (chunk features are cached in .cache/ for reruns)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from pipeline import chunk_text, class_scores, confidence_scores, domain_scores, featurize_texts

ROOT = Path(__file__).parent
CACHE = ROOT / ".cache"
N_FOLDS = 5
SEED = 42
# Operating points offered in the app: fraction of known-author documents we keep.
KEEP_RATES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_KEEP_RATE = 0.85
DOMAIN_PERCENTILE = 1   # domain check rejects ~this % of real Reuters articles per signal


def build_chunk_features(df, scaler, selector, config, embedder):
    """Features for every chunk of every document, cached to disk."""
    CACHE.mkdir(exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n")   # keep the cache out of git
    path =CACHE / "chunk_features.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return d["X"], d["doc_idx"]
    size = config.get("chunk_size_words", 150)
    overlap = config.get("chunk_overlap_words", 40)
    texts, doc_idx = [], []
    for i, text in enumerate(df["text"]):
        for _, _, chunk in chunk_text(text, size, overlap):
            texts.append(chunk)
            doc_idx.append(i)
    print(f"Featurizing {len(texts)} chunks from {len(df)} documents (one-off, cached)...")
    parts = []
    for start in range(0, len(texts), 500):
        parts.append(featurize_texts(texts[start:start + 500], scaler, selector, config, embedder))
        print(f"  {min(start + 500, len(texts))}/{len(texts)}")
    X, doc_idx = np.vstack(parts), np.array(doc_idx)
    np.savez_compressed(path, X=X, doc_idx=doc_idx)
    return X, doc_idx


def main():
    scaler = joblib.load(ROOT / "stylo_scaler.joblib")
    selector = joblib.load(ROOT / "stylo_feature_selector.joblib")
    model = joblib.load(ROOT / "author_attribution_model.joblib")
    config = json.loads((ROOT / "pipeline_config.json").read_text())
    embedder = SentenceTransformer(config.get("embedding_model", "all-MiniLM-L6-v2"))

    df = pd.read_csv(ROOT / "reuters_50_50.csv")
    X, doc_idx = build_chunk_features(df, scaler, selector, config, embedder)
    chunk_author = df["author"].to_numpy()[doc_idx]
    chunk_split = df["split"].to_numpy()[doc_idx]

    rng = np.random.default_rng(SEED)
    authors = rng.permutation(sorted(df["author"].unique()))
    folds = np.array_split(authors, N_FOLDS)

    records = []  # one row per test document per fold
    for f, held_out in enumerate(folds):
        held_out = set(held_out)
        train_mask = (chunk_split == "train") & ~np.isin(chunk_author, list(held_out))
        clf = clone(model).fit(X[train_mask], chunk_author[train_mask])
        test_docs = df.index[df["split"] == "test"]
        for d in test_docs:
            rows = doc_idx == d
            probs, _ = class_scores(X[rows], clf)
            doc_probs = probs.mean(axis=0)
            known = df.at[d, "author"] not in held_out
            rec = {"fold": f, "known": known,
                   "correct": known and clf.classes_[doc_probs.argmax()] == df.at[d, "author"]}
            rec.update(confidence_scores(doc_probs))
            records.append(rec)
        print(f"fold {f + 1}/{N_FOLDS} done")

    res = pd.DataFrame(records)
    aucs = {s: float(roc_auc_score(res["known"], res[s])) for s in ["max_prob", "margin"]}
    score = max(aucs, key=aucs.get)
    known_scores = res.loc[res["known"], score]

    points = []
    for keep in KEEP_RATES:
        thr = float(np.quantile(known_scores, 1 - keep))
        accepted = res[score] >= thr
        acc_known = res[accepted & res["known"]]
        points.append({
            "keep_rate_known": keep,
            "threshold": thr,
            "unknown_rejected": float((~accepted[~res["known"]]).mean()),
            "accuracy_when_accepted": float(acc_known["correct"].mean()),
        })

    # Domain check: per-author embedding centroids from TRAIN chunks, thresholds from TEST docs.
    n_stylo = int(selector.get_support().sum())
    E = X[:, n_stylo:]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    train_chunks = chunk_split == "train"
    centroids = np.stack([E[train_chunks & (chunk_author == a)].mean(axis=0) for a in sorted(set(chunk_author))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    np.save(ROOT / "domain_centroids.npy", centroids.astype(np.float32))
    test_docs = df.index[df["split"] == "test"]
    dom = pd.DataFrame([domain_scores(X[doc_idx == d], n_stylo, centroids) for d in test_docs])
    sim_thr = float(np.percentile(dom["news_similarity"], DOMAIN_PERCENTILE))
    unusual_thr = float(np.percentile(dom["style_unusualness"], 100 - DOMAIN_PERCENTILE))
    domain = {
        "news_similarity_min": sim_thr,
        "style_unusualness_max": unusual_thr,
        "n_stylometric_features": n_stylo,
        "reuters_rejected": float(((dom["news_similarity"] < sim_thr) | (dom["style_unusualness"] > unusual_thr)).mean()),
    }

    out = {
        "domain_check": domain,
        "score": score,
        "auroc": aucs,
        "default_keep_rate": DEFAULT_KEEP_RATE,
        "operating_points": points,
        "accuracy_without_rejection": float(res.loc[res["known"], "correct"].mean()),
        "n_folds": N_FOLDS,
        "authors_held_out_per_fold": len(folds[0]),
        "source": "Reuters C50 test split, leave-10-authors-out",
    }
    (ROOT / "open_set_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
