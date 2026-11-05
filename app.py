import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from pipeline import (
    analyze_text, confidence_scores, domain_scores, extract_stylometric_features, find_segments,
    top_authors,
)

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
CALIBRATION_FILE = "style_shift_calibration.json"
OPEN_SET_FILE = "open_set_calibration.json"
DOMAIN_CENTROIDS_FILE = "domain_centroids.npy"
DEFAULT_SHIFT_THRESHOLD = 0.8  # used only if calibrate_style_shift.py hasn't been run


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


@st.cache_data(show_spinner=False)
def load_shift_calibration():
    path = ARTIFACT_DIR / CALIBRATION_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_open_set_calibration():
    path = ARTIFACT_DIR / OPEN_SET_FILE
    if not path.exists():
        return None
    with open(path) as f:
        calib = json.load(f)
    centroids_path = ARTIFACT_DIR / DOMAIN_CENTROIDS_FILE
    calib["centroids"] = np.load(centroids_path) if centroids_path.exists() else None
    return calib


def open_set_point(calib: dict, keep_rate: float) -> dict:
    return min(calib["operating_points"], key=lambda p: abs(p["keep_rate_known"] - keep_rate))


def known_author_check(analysis: dict, calib: dict, point: dict) -> dict:
    """Two-stage open-set check.

    status: "out_of_domain" (text unlike the training data — no attribution attempted),
            "no_match"      (in domain, but no known author fits strongly enough),
            "match", or None when no calibration is available.
    """
    if not calib or analysis["kind"] == "vote":
        return {"status": None}
    out = {"confidence": confidence_scores(analysis["doc_probs"])[calib["score"]]}
    dom_cfg = calib.get("domain_check")
    if dom_cfg and calib.get("centroids") is not None:
        dom = domain_scores(analysis["features"], dom_cfg["n_stylometric_features"], calib["centroids"])
        out.update(dom)
        if (dom["news_similarity"] < dom_cfg["news_similarity_min"]
                or dom["style_unusualness"] > dom_cfg["style_unusualness_max"]):
            out["status"] = "out_of_domain"
            return out
    out["status"] = "match" if out["confidence"] >= point["threshold"] else "no_match"
    return out


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
CHART_LAYOUT = dict(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="#E6E6E6")


def render_style_timeline(result: dict, top_k: int):
    """Per-chunk attribution heatmap + style-shift line, with segments split at large shifts."""
    chunks, probs, shifts = result["chunks"], result["chunk_probs"], result["shifts"]
    authors = result["authors"]
    calib = load_shift_calibration()
    threshold = calib["threshold"] if calib else DEFAULT_SHIFT_THRESHOLD

    st.markdown("#### 🧭 Style timeline")
    st.caption(
        f"The text was split into {len(chunks)} overlapping chunks of "
        f"~{config.get('chunk_size_words', 150)} words (the size the model was trained on). "
        f"{result['agreement']*100:.0f}% of chunks agree with the overall top match."
    )

    labels = [f"#{i+1}<br>w{s+1}–{e}" for i, (s, e, _) in enumerate(chunks)]
    shown_idx = np.argsort(result["doc_probs"])[::-1][:top_k]
    heat = probs[:, shown_idx].T
    fig = go.Figure(go.Heatmap(
        z=heat, x=labels, y=[authors[i] for i in shown_idx],
        colorscale="Purples", zmin=0, zmax=max(float(heat.max()), 1e-6),
        hovertemplate="Chunk %{x}<br>%{y}: %{z:.1%}<extra></extra>",
        colorbar=dict(title="P", tickformat=".0%"),
    ))
    fig.update_layout(**CHART_LAYOUT, height=90 + 32 * len(shown_idx),
                      margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    mids = [f"#{i+1}→#{i+2}" for i in range(len(shifts))]
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=mids, y=shifts, mode="lines+markers", line=dict(color="#6C63FF"),
        marker=dict(size=9, color=["#ae3f3f" if s >= threshold else "#6C63FF" for s in shifts]),
        hovertemplate="%{x}: shift %{y:.2f}<extra></extra>", name="Style shift",
    ))
    fig_s.add_hline(y=threshold, line_dash="dash", line_color="#ae3f3f",
                    annotation_text="change threshold", annotation_font_color="#E6E6E6")
    fig_s.update_layout(**CHART_LAYOUT, height=220, margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(range=[0, 1], title="Shift"), showlegend=False)
    st.plotly_chart(fig_s, use_container_width=True)

    segments = find_segments(probs, shifts, threshold)
    if len(segments) == 1:
        st.success("No strong style change detected — the text reads as consistent throughout.")
    else:
        st.warning(
            f"⚠️ {len(segments) - 1} possible style change(s) detected. The text may combine "
            f"writing from more than one person, or contain an inserted or edited passage."
        )
        seg_rows = []
        for n, (a, b, mean_p) in enumerate(segments, 1):
            best = int(mean_p.argmax())
            seg_rows.append({
                "Segment": n,
                "Chunks": f"#{a+1}–#{b+1}" if a != b else f"#{a+1}",
                "Words": f"{chunks[a][0]+1}–{chunks[b][1]}",
                "Closest author": authors[best],
                "Confidence": f"{mean_p[best]*100:.1f}%",
                "Starts with": " ".join(chunks[a][2].split()[:12]) + "…",
            })
        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

    if calib:
        st.caption(
            f"Shift = Jensen–Shannon distance between neighbouring chunks' author distributions. "
            f"Threshold {threshold:.2f} = {calib['percentile_of_same_author_shifts']}th percentile of "
            f"shifts inside single-author Reuters test articles. On {calib['doc_length_words']}-word test "
            f"documents it caught {calib['detection_rate_spliced_docs']*100:.0f}% of two-author splices and "
            f"raised a false alarm on {calib['false_alarm_rate_single_author_docs']*100:.0f}% of "
            f"single-author texts — treat flags as leads to investigate, not proof."
        )
    else:
        st.caption(f"⚠️ Uncalibrated threshold ({threshold}). Run `calibrate_style_shift.py` to calibrate it.")

    with st.expander("📄 Per-chunk predictions"):
        rows = []
        for i, (s, e, txt) in enumerate(chunks):
            best = int(probs[i].argmax())
            rows.append({
                "Chunk": i + 1, "Words": f"{s+1}–{e}",
                "Top author": authors[best], "Confidence": f"{probs[i, best]*100:.1f}%",
                "Shift from previous": "" if i == 0 else f"{shifts[i-1]:.2f}",
                "Preview": " ".join(txt.split()[:15]) + "…",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



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


st.markdown("# ✍️Linguistic Authorship Analysis")
st.caption(
    "Identify authors from writing style alone, verify whether two texts share an author "
    "and uncover the Linguistic Patterns and Authorship Signals Behind Any Text."
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
        open_calib = load_open_set_calibration()
        if open_calib:
            keep_options = sorted((p["keep_rate_known"] for p in open_calib["operating_points"]), reverse=True)
            keep_rate = st.select_slider(
                "Unknown-author strictness",
                options=keep_options,
                value=min(keep_options, key=lambda k: abs(k - open_calib["default_keep_rate"])),
                format_func=lambda k: f"{(1 - k) * 100:.0f}%",
                help="Higher = more willing to answer 'none of the known authors'. "
                     "The value is the share of genuine known-author texts that would be "
                     "turned away, measured on held-out test articles.",
            )
            open_point = open_set_point(open_calib, keep_rate)
        else:
            open_point = None
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
                analysis = analyze_text(
                    predict_text, scaler, selector, label_encoder, model, config, embedder
                )
            results, kind = top_authors(analysis, top_k), analysis["kind"]
            n_chunks = len(analysis["chunks"])

            if kind == "vote":
                st.info(f"**Predicted author:** {results[0][0]}")
            else:
                label_name = "Probability" if kind == "probability" else "Relative confidence"
                df_res = pd.DataFrame(results, columns=["Author", label_name])

                basis = (f"averaged over {n_chunks} chunks" if n_chunks > 1 else "single chunk")
                check = known_author_check(analysis, open_calib, open_point)
                status = check["status"]
                if status == "out_of_domain":
                    st.markdown(
                        f'<div class="verdict-diff">'
                        f'<h4 style="margin-top:0">🌐 This text is outside what the model knows</h4>'
                        f'<p style="margin-bottom:0">Its content and style are very different from the '
                        f'Reuters news articles the model was trained on, so it can\'t reliably say '
                        f'whether any of the {len(label_encoder.classes_)} known authors wrote it — '
                        f'and most likely none of them did. The ranking below is shown for '
                        f'reference only.</p></div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.markdown("**Closest known authors** (not a reliable attribution)")
                elif status == "no_match":
                    st.markdown(
                        f'<div class="verdict-diff">'
                        f'<h4 style="margin-top:0">❓ No confident match among the '
                        f'{len(label_encoder.classes_)} known authors</h4>'
                        f'<p style="margin-bottom:0">The writing style doesn\'t fit any known author '
                        f'strongly enough — it may be by someone the model has never seen. '
                        f'Closest: <b>{results[0][0]}</b> ({label_name.lower()} '
                        f'{results[0][1]*100:.1f}%, {basis}).</p></div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.markdown("**Closest known authors** (none passed the known-author check)")
                else:
                    passed = " · ✅ passes known-author check" if status == "match" else ""
                    st.markdown(
                        f'<div class="fingerprint-card">'
                        f'<h4 style="margin-top:0">Top match: {results[0][0]}</h4>'
                        f'<p style="color:#9a9ab0;margin-bottom:0">'
                        f'{label_name}: {results[0][1]*100:.1f}% ({basis}){passed}</p></div>',
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

                if status == "out_of_domain":
                    dom_cfg = open_calib["domain_check"]
                    st.caption(
                        f"Domain check: similarity to the training material = {check['news_similarity']:.3f} "
                        f"(minimum {dom_cfg['news_similarity_min']:.3f}); style unusualness = "
                        f"{check['style_unusualness']:.2f} (maximum {dom_cfg['style_unusualness_max']:.2f}). "
                        f"Only {dom_cfg['reuters_rejected']*100:.0f}% of real Reuters test articles fail this check."
                    )
                elif status is not None:
                    score_name = {"max_prob": "top probability", "margin": "gap between top two authors"}
                    st.caption(
                        f"Known-author check: {score_name.get(open_calib['score'], open_calib['score'])} = "
                        f"{check['confidence']:.3f} vs threshold {open_point['threshold']:.3f}. In a simulation where 10 "
                        f"authors were hidden from the model, this setting turned away "
                        f"{open_point['unknown_rejected']*100:.0f}% of texts by unseen authors and "
                        f"{(1 - open_point['keep_rate_known'])*100:.0f}% of texts by known ones; answers "
                        f"that passed were {open_point['accuracy_when_accepted']*100:.0f}% correct "
                        f"(vs {open_calib['accuracy_without_rejection']*100:.0f}% with no check)."
                    )
                elif open_calib is None:
                    st.caption("⚠️ Known-author check unavailable — run `calibrate_open_set.py` to enable it.")

            if n_chunks > 1:
                st.divider()
                render_style_timeline(analysis, top_k)
            else:
                st.caption(
                    f"💡 Paste more than {config.get('chunk_size_words', 150)} words to get a "
                    f"chunk-by-chunk style timeline that can flag mixed authorship."
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
    st.markdown("### Analyze the Linguistic Patterns Behind Any Text")
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
                            analysis = analyze_text(
                                text, scaler, selector, label_encoder, model, config, embedder
                            )
                            pred_author, score = top_authors(analysis, 1)[0]
                            status = known_author_check(analysis, open_calib, open_point)["status"]
                        except Exception as e:
                            pred_author, score, status = f"ERROR: {e}", None, None
                        result_row = {"text_preview": text[:80] + ("..." if len(text) > 80 else ""), "predicted_author": pred_author}
                        result_row["confidence"] = score
                        if status is not None:
                            result_row["known_author_check"] = {
                                "match": "pass", "no_match": "no confident match",
                                "out_of_domain": "outside training domain",
                            }[status]
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
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Batch accuracy (top match)", f"{acc*100:.1f}%")
                        if "known_author_check" in results_df.columns:
                            passed = results_df["known_author_check"] == "pass"
                            m2.metric("Passed known-author check", f"{passed.mean()*100:.0f}%")
                            if passed.any():
                                m3.metric("Accuracy on passed rows", f"{results_df.loc[passed, 'correct'].mean()*100:.1f}%")
                    elif "known_author_check" in results_df.columns:
                        passed = results_df["known_author_check"] == "pass"
                        st.metric("Passed known-author check", f"{passed.mean()*100:.0f}%")

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

#### Capabilities
1. **Author Attribution** — given text, rank which of the known authors most likely wrote it.
   Long texts are split into 150-word chunks (as in training) and chunk predictions are averaged.
2. **Unknown-author detection** — if the style doesn't fit any known author strongly enough,
   the app says "no confident match" instead of forcing a guess. Calibrated by hiding 10
   authors from a retrained copy of the model and checking it rejects them.
3. **Style-change timeline** — flags points in a long text where the writing style shifts,
   which can indicate mixed authorship or an inserted passage.
4. **Author Verification** — given two texts, estimate whether they share an author, using
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
- Unknown-author detection is only moderately reliable when the unknown writer is another news
  journalist (the simulation caught roughly a third to a half of them, depending on strictness). Text that
  is clearly not news writing (blogs, fiction, email, academic prose) is caught by a separate domain check.
- The unknown-author threshold was calibrated with a 40-author model and applied to the 50-author one,
  so its error rates are estimates.
- Style-change flags are leads to investigate, not proof — single-author texts are sometimes flagged too.
- This is a demo/portfolio project, not a forensic or legal authorship tool.
""")