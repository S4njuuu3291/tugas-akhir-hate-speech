import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

try:
    from manual_shap.bert.data_netral import background_dataset as back_data
except:
    back_data = ["neutral text example"]


def predict_proba_bert(sentence, clf_bert, tokenizer, target_index):
    inputs = tokenizer(sentence, return_tensors="tf", truncation=True, padding=True)
    logits = clf_bert(inputs).logits
    probs = tf.sigmoid(logits).numpy()
    return probs[0, target_index]


def kernel_weight(M, k):
    if k == 0 or k == M:
        return 1e3
    return (M - 1) / (scipy.special.comb(M, k) * k * (M - k))


def tokenize_with_word_mapping(sentence, tokenizer):
    words = sentence.split()
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="tf",
        truncation=True,
        padding=False,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    word_ids = encoded.word_ids()
    return words, tokens, word_ids


def aggregate_shap_to_words(words, word_ids, idx_features, shap_values):
    word_shap_list = [0.0] * len(words)
    for shap_idx, token_idx in enumerate(idx_features):
        word_idx = word_ids[token_idx]
        if word_idx is not None:
            word_shap_list[word_idx] += shap_values[shap_idx]
    return word_shap_list


def calculate_expected_baseline(background_texts, tokenizer, clf_bert, target_index):
    probs = []
    for text in background_texts:
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
        logits = clf_bert(inputs).logits
        prob = tf.sigmoid(logits)[0, target_index].numpy()
        probs.append(prob)
    return np.mean(probs)


def shap_kernel_instance_bert_fast(
    sentence,
    tokenizer,
    clf_bert,
    labels,
    target_label,
    num_samples=32,
    background_dataset=back_data,
):
    start_time = time.time()
    target_index = labels.index(target_label)
    base_value = calculate_expected_baseline(
        background_dataset, tokenizer, clf_bert, target_index
    )

    words, tokens, word_ids = tokenize_with_word_mapping(sentence, tokenizer)
    idx_features = np.array([i for i, w in enumerate(word_ids) if w is not None])
    M = len(idx_features)

    if M == 0:
        return base_value, words, [0.0] * len(words), 0.0

    pos_map = {tok: i for i, tok in enumerate(idx_features)}
    subsets = [[], list(idx_features)]
    seen = {(), tuple(idx_features)}
    ks = np.arange(1, M)

    if len(ks) > 0:
        probs_k = np.array([kernel_weight(M, k) for k in ks])
        probs_k /= probs_k.sum()
        while len(subsets) < min(num_samples, 2**M):
            k = np.random.choice(ks, p=probs_k)
            s = tuple(sorted(np.random.choice(idx_features, size=k, replace=False)))
            if s not in seen:
                subsets.append(list(s))
                seen.add(s)

    n = len(subsets)
    Z = np.zeros((n, M), dtype=np.float32)
    sizes = np.zeros(n, dtype=int)
    for i, subset in enumerate(subsets):
        for tok in subset:
            Z[i, pos_map[tok]] = 1.0
        sizes[i] = len(subset)

    encoded = tokenizer(sentence, return_tensors="tf", truncation=True, padding=False)
    input_ids = encoded["input_ids"][0].numpy()
    attention_mask = encoded["attention_mask"][0].numpy()

    input_ids_batch = []
    for i in range(n):
        ids = input_ids.copy()
        inactive = idx_features[Z[i] == 0]
        ids[inactive] = tokenizer.mask_token_id
        input_ids_batch.append(ids)

    inputs = {
        "input_ids": tf.constant(input_ids_batch),
        "attention_mask": tf.constant([attention_mask] * n),
    }
    logits = clf_bert(inputs).logits
    probs_out = tf.sigmoid(logits)[:, target_index].numpy()

    y_adj = probs_out - base_value
    weights = np.array([kernel_weight(M, k) for k in sizes])
    WX = Z * weights[:, None]
    Wy = y_adj * weights
    beta = np.linalg.pinv(Z.T @ WX) @ (Z.T @ Wy)

    word_shap_values = aggregate_shap_to_words(words, word_ids, idx_features, beta)
    return base_value, words, word_shap_values, time.time() - start_time


def plot_shap_values(
    base_value, shap_values, fitur_aktif, sentence, target_label, top_n=50
):
    df = pd.DataFrame({"Fitur": fitur_aktif, "SHAP Value": shap_values})
    df["abs"] = df["SHAP Value"].abs()
    df_plot = df.sort_values("abs", ascending=False).head(top_n).iloc[::-1]

    colors = ["red" if v > 0 else "blue" for v in df_plot["SHAP Value"]]
    fig, ax = plt.subplots(figsize=(10, len(df_plot) * 0.6 + 1.5))
    ax.barh(df_plot["Fitur"], df_plot["SHAP Value"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f'SHAP (BERT – Word Level) — {target_label}\n"{sentence[:50]}..."')
    ax.set_xlabel("Kontribusi terhadap Probabilitas")
    fig.tight_layout()
    return fig
