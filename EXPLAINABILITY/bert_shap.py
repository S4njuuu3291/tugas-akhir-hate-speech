import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

try:
    from DATA.data_netral import background_dataset as back_data
except:
    try:
        from manual_shap.bert.data_netral import background_dataset as back_data
    except:
        back_data = ["neutral text example"]


def predict_proba_bert(sentence, clf_bert, tokenizer, target_index):
    """
    Mendapatkan probabilitas prediksi untuk label tertentu dari model BERT.

    Parameters
    ----------
    sentence : str
        Teks input yang akan diprediksi.
    clf_bert : TFAutoModelForSequenceClassification
        Model BERT yang sudah di-fine-tune.
    tokenizer : BertTokenizerFast
        Tokenizer yang sesuai dengan model.
    target_index : int
        Index label target.

    Returns
    -------
    float
        Probabilitas kelas target setelah sigmoid.
    """
    inputs = tokenizer(sentence, return_tensors="tf", truncation=True, padding=True)
    logits = clf_bert(inputs).logits
    probs = tf.sigmoid(logits).numpy()
    return probs[0, target_index]


def kernel_weight(M, k):
    """
    Bobot Kernel SHAP untuk subset berukuran `k` dari total `M` token.
    Rumus: (M-1) / (C(M,k) * k * (M-k))

    Parameters
    ----------
    M : int
        Total jumlah token aktif dalam instance.
    k : int
        Ukuran subset yang dievaluasi.

    Returns
    -------
    float
        Bobot Kernel SHAP untuk subset tsb.
    """
    # SHAP paper: subset kosong (k=0) dan penuh (k=M) diberi bobot ~0
    # karena kurang informatif; digunakan 1e-6 agar tidak division by zero
    if k == 0 or k == M:
        return 1e6
    return (M - 1) / (scipy.special.comb(M, k) * k * (M - k))


def tokenize_with_word_mapping(sentence, tokenizer):
    """
    Tokenisasi kalimat per-kata dan mapping token ke word index.
    Berguna untuk aggregasi SHAP dari token-level ke word-level.

    Parameters
    ----------
    sentence : str
        Kalimat input.
    tokenizer : BertTokenizerFast
        Tokenizer BERT.

    Returns
    -------
    words : list of str
        Daftar kata hasil split spasi.
    tokens : list of str
        Daftar token BERT (termasuk subword tokens).
    word_ids : list of int or None
        Untuk setiap token, index kata asalnya (None untuk [CLS]/[SEP]).
    """
    words = sentence.split()
    encoded = tokenizer(
        words,
        is_split_into_words=True,  # penting: tokenizer tahu input sudah per-kata
        return_tensors="tf",
        truncation=True,
        padding=False,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    word_ids = encoded.word_ids()  # mapping: token_index -> word_index
    return words, tokens, word_ids


def aggregate_shap_to_words(words, word_ids, idx_features, shap_values):
    """
    Agregasi SHAP values dari level token (subword) ke level word.
    Subword tokens dari kata yang sama dijumlahkan kontribusinya.

    Parameters
    ----------
    words : list of str
        Daftar kata asli.
    word_ids : list of int or None
        Mapping token_index -> word_index.
    idx_features : np.array
        Index token yang aktif (bukan [CLS]/[SEP]/[PAD]).
    shap_values : np.array
        SHAP values per token.

    Returns
    -------
    list of float
        SHAP values per word (panjang = len(words)).
    """
    word_shap_list = [0.0] * len(words)
    for shap_idx, token_idx in enumerate(idx_features):
        word_idx = word_ids[token_idx]
        if word_idx is not None:
            # Jumlahkan kontribusi semua subword token dari kata yang sama
            word_shap_list[word_idx] += shap_values[shap_idx]
    return word_shap_list


def calculate_expected_baseline(background_texts, tokenizer, clf_bert, target_index):
    """
    Menghitung expected value (baseline) dengan merata-rata probabilitas
    prediksi pada background dataset (teks netral).

    Parameters
    ----------
    background_texts : list of str
        Dataset latar belakang (teks netral).
    tokenizer : BertTokenizerFast
        Tokenizer BERT.
    clf_bert : TFAutoModelForSequenceClassification
        Model BERT.
    target_index : int
        Index label target.

    Returns
    -------
    float
        Rata-rata probabilitas (baseline / E[f(x)]).
    """
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
    """
    Menghitung SHAP values untuk satu instance teks menggunakan Kernel SHAP
    dengan model BERT. Berbeda dengan versi SVM/LR, masking fitur dilakukan
    dengan mengganti token yang tidak aktif menjadi [MASK] token.

    Alur:
    1. Hitung baseline dari background dataset (teks netral).
    2. Tokenisasi kalimat dan mapping token-to-word.
    3. Sampling subset token berdasarkan bobot Kernel SHAP.
    4. Batch predict: untuk tiap subset, token di luar subset diganti [MASK].
    5. Weighted linear regression untuk mendapatkan SHAP values.
    6. Agregasi SHAP dari token-level ke word-level.

    Parameters
    ----------
    sentence : str
        Kalimat yang ingin dijelaskan.
    tokenizer : BertTokenizerFast
        Tokenizer BERT.
    clf_bert : TFAutoModelForSequenceClassification
        Model BERT yang sudah di-fine-tune.
    labels : list of str
        Daftar nama label.
    target_label : str
        Label target yang dijelaskan.
    num_samples : int, optional
        Jumlah subset yang di-sampling (default 32).
    background_dataset : list of str, optional
        Teks netral untuk menghitung baseline.

    Returns
    -------
    base_value : float
        Nilai baseline (E[f(x)]).
    words : list of str
        Daftar kata dalam kalimat.
    word_shap_values : list of float
        Kontribusi SHAP per kata.
    elapsed : float
        Waktu eksekusi dalam detik.
    """
    start_time = time.time()
    target_index = labels.index(target_label)

    # 1. Hitung baseline dari background dataset (teks netral)
    base_value = calculate_expected_baseline(
        background_dataset, tokenizer, clf_bert, target_index
    )

    # 2. Tokenisasi & mapping token ke word
    words, tokens, word_ids = tokenize_with_word_mapping(sentence, tokenizer)
    # Ambil index token yang punya word (bukan [CLS], [SEP], atau padding)
    idx_features = np.array([i for i, w in enumerate(word_ids) if w is not None])
    M = len(idx_features)

    if M == 0:
        return base_value, words, [0.0] * len(words), 0.0

    # 3. Sampling subset token
    pos_map = {tok: i for i, tok in enumerate(idx_features)}
    subsets = [[], list(idx_features)]  # selalu sertakan empty & full set
    seen = {(), tuple(idx_features)}
    ks = np.arange(1, M)

    if len(ks) > 0:
        # Sampling dengan probabilitas proporsional terhadap Kernel weight
        probs_k = np.array([kernel_weight(M, k) for k in ks])
        probs_k /= probs_k.sum()
        while len(subsets) < min(num_samples, 2**M):
            k = np.random.choice(ks, p=probs_k)
            s = tuple(sorted(np.random.choice(idx_features, size=k, replace=False)))
            if s not in seen:
                subsets.append(list(s))
                seen.add(s)

    n = len(subsets)
    # Matriks Z: 1 jika token ada di subset, 0 jika tidak
    Z = np.zeros((n, M), dtype=np.float32)
    sizes = np.zeros(n, dtype=int)
    for i, subset in enumerate(subsets):
        for tok in subset:
            Z[i, pos_map[tok]] = 1.0
        sizes[i] = len(subset)

    # 4. Batch predict: ganti token tidak aktif dengan [MASK] token
    encoded = tokenizer(sentence, return_tensors="tf", truncation=True, padding=False)
    input_ids = encoded["input_ids"][0].numpy()
    attention_mask = encoded["attention_mask"][0].numpy()

    input_ids_batch = []
    for i in range(n):
        ids = input_ids.copy()
        inactive = idx_features[Z[i] == 0]
        ids[inactive] = tokenizer.mask_token_id  # masking token yg tidak aktif
        input_ids_batch.append(ids)

    inputs = {
        "input_ids": tf.constant(input_ids_batch),
        "attention_mask": tf.constant([attention_mask] * n),
    }
    logits = clf_bert(inputs).logits
    probs_out = tf.sigmoid(logits)[:, target_index].numpy()

    # 5. Weighted Linear Regression: β = (Z^T W Z)^{-1} Z^T W y
    y_adj = probs_out - base_value  # output - baseline
    weights = np.array([kernel_weight(M, k) for k in sizes])
    WX = Z * weights[:, None]
    Wy = y_adj * weights
    beta = np.linalg.pinv(Z.T @ WX) @ (Z.T @ Wy)

    # 6. Agregasi SHAP dari token-level ke word-level
    word_shap_values = aggregate_shap_to_words(words, word_ids, idx_features, beta)

    # ── Print hasil SHAP ──
    word_shap_pairs = list(zip(words, word_shap_values))
    word_shap_sorted = sorted(word_shap_pairs, key=lambda x: abs(x[1]), reverse=True)

    print("-" * 50)
    print(f"Kalimat: {sentence}")
    print(f"Penjelasan untuk: {target_label}")
    print("-" * 50)

    for w, val in word_shap_sorted:
        print(f"{w}: {val:.4f}")

    # Verifikasi additive property
    original_proba = predict_proba_bert(sentence, clf_bert, tokenizer, target_index)
    total_shap = sum(word_shap_values)

    print("\nBaseline (E[f(x)]):", base_value)
    print("Baseline + ΣSHAP:", base_value + total_shap)
    print("Prediksi Asli f(x):", original_proba)

    elapsed = time.time() - start_time
    print(f"Total waktu SHAP: {elapsed:.4f} detik")

    return base_value, words, word_shap_values, elapsed


def plot_shap_values(
    base_value, shap_values, fitur_aktif, sentence, target_label, top_n=50
):
    """
    Visualisasi SHAP values dalam bentuk horizontal bar chart (word-level).

    Parameters
    ----------
    base_value : float
        Nilai baseline (output saat semua token = [MASK]).
    shap_values : list of float
        Kontribusi SHAP per kata.
    fitur_aktif : list of str
        Daftar kata dalam kalimat.
    sentence : str
        Kalimat asli yang dijelaskan.
    target_label : str
        Label target yang dijelaskan.
    top_n : int, optional
        Jumlah kata top yang ditampilkan (default 50).

    Returns
    -------
    matplotlib.figure.Figure
        Figure objek untuk ditampilkan/disimpan.
    """
    df = pd.DataFrame({"Fitur": fitur_aktif, "SHAP Value": shap_values})
    df["abs"] = df["SHAP Value"].abs()
    # Urutkan berdasarkan magnitude, ambil top_n, balik agar terbesar di atas
    df_plot = df.sort_values("abs", ascending=False).head(top_n).iloc[::-1]

    # Warna: merah = mendorong (+), biru = menurunkan (-)
    colors = ["red" if v > 0 else "blue" for v in df_plot["SHAP Value"]]
    fig, ax = plt.subplots(figsize=(10, len(df_plot) * 0.6 + 1.5))
    ax.barh(df_plot["Fitur"], df_plot["SHAP Value"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)  # garis nol referensi
    ax.set_title(f'SHAP (BERT – Word Level) — {target_label}\n"{sentence[:50]}..."')
    ax.set_xlabel("Kontribusi terhadap Probabilitas")
    fig.tight_layout()
    return fig
