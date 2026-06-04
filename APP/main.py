import pickle
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ============================
# ADD PROJECT ROOT TO PATH
# ============================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================
# IMPORT SHAP IMPLEMENTATIONS
# ============================
from EXPLAINABILITY.svm_lr_shap import (
    shap_kernel_instance as shap_kernel_classic,
    plot_shap_values as plot_shap_classic,
)
from EXPLAINABILITY.bert_shap import (
    predict_proba_bert,
    shap_kernel_instance_bert_fast,
    # plot_shap_values as plot_shap_bert,
)
from DATA.data_netral import background_dataset as back_data

# ============================================================
# SESSION STATE & STYLING
# ============================================================
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "cleaned_text" not in st.session_state:
    st.session_state.cleaned_text = ""

st.markdown(
    """
    <style>
    .highlight-container {
        margin-top: 25px;
        margin-bottom: 25px;
        padding: 15px;
        border-radius: 10px;
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LABEL PRETTY & DATA LOAD
# ============================================================
LABEL_PRETTY = {
    "HS_Individual": "Individu",
    "HS_Group": "Kelompok",
    "HS_Religion": "Agama",
    "HS_Physical": "Fisik",
    "HS_Race": "Ras",
    "HS_Gender": "Gender",
}


@st.cache_data
def load_cleaning_resources():
    alay_path = project_root / "DATA" / "raw" / "new_kamusalay.csv"
    alay_df = pd.read_csv(alay_path)
    alay_dict = dict(zip(alay_df["alay"], alay_df["normal"]))
    return alay_dict

alay_dict = load_cleaning_resources()

# Compiled regex for speed
RE_CLEAN = re.compile(r"http\S+|www\S+|https\S+|@\w+|\\n|\n|\brt\b|[^a-zA-Z\s]")
RE_MULTISPACE = re.compile(r"\s+")
RE_REPEATED = re.compile(r"(.)\1{2,}")

def clean_for_shap(text, alay_map, max_words=40):
    if pd.isna(text):
        return ""
    # Hanya hapus simbol/URL, JANGAN hapus kata (stopwords)
    s = RE_CLEAN.sub(" ", str(text).lower())
    s = RE_MULTISPACE.sub(" ", s).strip()

    tokens = s.split()
    # Mapping alay tetap dilakukan agar model paham konteksnya
    s = " ".join([alay_map.get(t, t) for t in tokens])

    # Repeated char reduction (misal: 'tidakkkk' -> 'tidak')
    s = RE_REPEATED.sub(r"\1", s)

    return " ".join(s.split()[:max_words])

def normalize_shap(shap_vals):
    if len(shap_vals) == 0:
        return shap_vals
    max_val = np.max(np.abs(shap_vals))
    return shap_vals if max_val == 0 else shap_vals / max_val

def shap_text_highlight(tokens, shap_values):
    if len(tokens) == 0 or len(shap_values) == 0:
        return "<i>Tidak ada kontribusi SHAP.</i>"

    shap_norm = normalize_shap(np.array(shap_values))
    spans = [
        f'<span style="background-color:rgba({255 if val > 0 else 0},0,{0 if val > 0 else 255},{min(abs(val), 1.0)});'
        f'padding:4px;margin:2px;border-radius:4px;display:inline-block;font-weight:500;">{tok}</span>'
        for tok, val in zip(tokens, shap_norm)
    ]
    return (
        f'<div class="highlight-container" style="line-height:2.4;">'
        + " ".join(spans)
        + "</div>"
    )

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_bert(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = TFAutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model

bert_tokenizer, bert_model = load_bert(str(project_root / "MODELS" / "model_hs_indobert"))
BERT_THRESHOLDS = np.array([0.16, 0.64, 0.18, 0.66, 0.12, 0.34])


@st.cache_resource
def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)

artifact = load_artifact(str(project_root / "MODELS" / "baseline" / "model_artifact_2.pkl"))
vectorizer = artifact["vectorizer"]
labels = artifact["label_columns"]
features_list = vectorizer.get_feature_names_out()

MODEL_MAP = {
    "BR Logistic Regression": (
        artifact["ovr_lr"],
        np.array(artifact["thresh"]["ovr_lr"]),
    ),
    "BR Calibrated SVM": (
        artifact["ovr_svm"],
        np.array(artifact["thresh"]["ovr_svm"]),
    ),
    "CC Logistic Regression": (
        artifact["cc_lr"],
        np.array(artifact["thresh"]["cc_lr"]),
    ),
    "CC Calibrated SVM": (artifact["cc_svm"], np.array(artifact["thresh"]["cc_svm"])),
}

# ============================================================
# PREDICTION LOGIC (OPTIMIZED)
# ============================================================
def predict_bert_batch(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="tf", truncation=True, padding=True, max_length=128
    )
    logits = model(inputs).logits
    probs = tf.sigmoid(logits).numpy()[0]
    return probs

# ============================================================
# UI
# ============================================================
st.title("🛡️ Hate speech Classification + SHAP")

text_input = st.text_area("Masukkan kalimat yang ingin dianalisis:", height=150)

col1, col2 = st.columns(2)
with col1:
    model_type = st.radio(
        "Arsitektur Model:", ["Klasik (TF-IDF + ML)", "BERT"], horizontal=True
    )
with col2:
    num_samples = st.number_input("Sampel KernelSHAP:", 64, 2048, 256, 64)

if model_type == "Klasik (TF-IDF + ML)":
    model_name = st.selectbox("Model Spesifik:", list(MODEL_MAP.keys()))

if st.button("Mulai Analisis Prediksi", type="primary"):
    st.session_state.cleaned_text = clean_for_shap(text_input, alay_dict)
    st.session_state.predicted = True

# ============================================================
# MAIN LOGIC
# ============================================================
if st.session_state.predicted:
    current_text = st.session_state.cleaned_text
    st.markdown("---")
    st.subheader("📊 Hasil Prediksi Model")

    if model_type == "Klasik (TF-IDF + ML)":
        clf, thresholds = MODEL_MAP[model_name]
        vec = vectorizer.transform([current_text])
        probs = clf.predict_proba(vec)[0]
        pred = (probs >= thresholds).astype(int)
    else:
        # Optimized: Only one forward pass for all labels
        probs = predict_bert_batch(current_text, bert_model, bert_tokenizer)
        pred = (probs >= BERT_THRESHOLDS).astype(int)

    any_hs = np.any(pred == 1)
    if any_hs:
        det = [
            LABEL_PRETTY.get(labels[i], labels[i]) for i, v in enumerate(pred) if v == 1
        ]
        st.error(f"⚠️ **Terdeteksi Hate speech!** Kategori: **{', '.join(det)}**.")
    else:
        st.success("✅ **Teks Aman.** Tidak terdeteksi ujaran kebencian.")

    df_res = pd.DataFrame(
        {
            "Label": [LABEL_PRETTY.get(l, l) for l in labels],
            "Probabilitas (%)": [f"{p * 100:.2f}%" for p in probs],
            "Threshold": thresholds
            if model_type == "Klasik (TF-IDF + ML)"
            else BERT_THRESHOLDS,
            "Hasil": ["⚠️ Terdeteksi" if x == 1 else "✅ Aman" for x in pred],
        }
    )
    st.table(df_res)

    st.subheader("🔍 Penjelasan Kontribusi Kata (SHAP)")
    tabs = st.tabs([LABEL_PRETTY.get(l, l) for l in labels])
    for i, label in enumerate(labels):
        with tabs[i]:
            t_start = time.time()
            if model_type == "Klasik (TF-IDF + ML)":
                base, shap_vals, feats = shap_kernel_classic(
                    current_text,
                    vectorizer,
                    MODEL_MAP[model_name][0],
                    features_list,
                    labels,
                    label,
                    num_samples=num_samples,
                )
                # Highlight Text
                st.markdown(
                    shap_text_highlight(feats, shap_vals),
                    unsafe_allow_html=True,
                )
                # Plot
                fig_classic = plot_shap_classic(base, shap_vals, feats, current_text, label)
                st.pyplot(fig_classic)
            else:
                # Optimized: Calculate once, use for both highlight and plot
                base, feats, shap_vals, _ = shap_kernel_instance_bert_fast(
                    current_text, 
                    bert_tokenizer, 
                    bert_model, 
                    labels, 
                    label, 
                    num_samples,
                    background_dataset=back_data # Gunakan dataset netral yang diimpor
                )
                
                # Highlight Text
                st.markdown(
                    shap_text_highlight(feats, shap_vals),
                    unsafe_allow_html=True,
                )
                
                # Plot
                fig_bert = plot_shap_classic(base, np.array(shap_vals), feats, current_text, label)
                st.pyplot(fig_bert)

            st.caption(f"⏱️ Komputasi: {time.time() - t_start:.2f} detik")
