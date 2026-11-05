"""Calibrate the style-change threshold used by the Predict tab's timeline.

Uses held-out TEST articles only:
  * single-author docs  -> distribution of chunk-to-chunk shifts with no real change
  * spliced docs (first half of author A + first half of author B) -> shifts at a true change

The threshold is the chosen percentile of single-author shifts; we report how often
spliced boundaries are caught and how often clean docs raise a false alarm.

Run:  .venv/Scripts/python.exe calibrate_style_shift.py
Writes: style_shift_calibration.json
"""
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from pipeline import analyze_text

ROOT = Path(__file__).parent
N_DOCS = 300
N_SPLICED = 300
SPLICE_HALF_WORDS = 220   # 2 x 220 = 440 words -> 4 chunks, change in the middle
PERCENTILE = 95
SEED = 42


def main():
    rng = random.Random(SEED)
    scaler = joblib.load(ROOT / "stylo_scaler.joblib")
    selector = joblib.load(ROOT / "stylo_feature_selector.joblib")
    label_encoder = joblib.load(ROOT / "author_label_encoder.joblib")
    model = joblib.load(ROOT / "author_attribution_model.joblib")
    config = json.loads((ROOT / "pipeline_config.json").read_text())
    embedder = SentenceTransformer(config.get("embedding_model", "all-MiniLM-L6-v2"))

    def run(text):
        return analyze_text(text, scaler, selector, label_encoder, model, config, embedder)

    df = pd.read_csv(ROOT / "reuters_50_50.csv")
    test = df[(df["split"] == "test") & (df["word_count"] >= 2 * SPLICE_HALF_WORDS)].reset_index(drop=True)

    # 1) Single-author documents: every consecutive shift is a "no change" sample.
    same_shifts, same_doc_max = [], []
    for i, row in enumerate(test.sample(N_DOCS, random_state=SEED).itertuples()):
        words = row.text.split()[: 2 * SPLICE_HALF_WORDS]   # same length as spliced docs
        shifts = run(" ".join(words))["shifts"]
        same_shifts.extend(shifts.tolist())
        same_doc_max.append(float(shifts.max()))
        if (i + 1) % 50 == 0:
            print(f"single-author docs: {i + 1}/{N_DOCS}")

    # 2) Spliced documents: true change at word SPLICE_HALF_WORDS.
    spliced_doc_max = []
    by_author = {a: g for a, g in test.groupby("author")}
    authors = list(by_author)
    for i in range(N_SPLICED):
        a, b = rng.sample(authors, 2)
        ta = by_author[a].sample(1, random_state=rng.randint(0, 10**9)).iloc[0]["text"].split()
        tb = by_author[b].sample(1, random_state=rng.randint(0, 10**9)).iloc[0]["text"].split()
        text = " ".join(ta[:SPLICE_HALF_WORDS] + tb[:SPLICE_HALF_WORDS])
        spliced_doc_max.append(float(run(text)["shifts"].max()))
        if (i + 1) % 50 == 0:
            print(f"spliced docs: {i + 1}/{N_SPLICED}")

    threshold = float(np.percentile(same_shifts, PERCENTILE))
    detection = float(np.mean(np.array(spliced_doc_max) >= threshold))
    false_alarm = float(np.mean(np.array(same_doc_max) >= threshold))

    out = {
        "threshold": threshold,
        "percentile_of_same_author_shifts": PERCENTILE,
        "detection_rate_spliced_docs": detection,
        "false_alarm_rate_single_author_docs": false_alarm,
        "doc_length_words": 2 * SPLICE_HALF_WORDS,
        "n_single_author_docs": N_DOCS,
        "n_spliced_docs": N_SPLICED,
        "same_author_shift_median": float(np.median(same_shifts)),
        "spliced_max_shift_median": float(np.median(spliced_doc_max)),
        "source": "Reuters C50 test split",
    }
    (ROOT / "style_shift_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
