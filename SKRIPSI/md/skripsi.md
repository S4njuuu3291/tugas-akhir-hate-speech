

KESIMPULAN DAN SARAN

## Kesimpulan

Dari seluruh rangkaian penelitian yang telah dilakukan yakni mulai dari pengolahan data mentah, eksperimen model, hingga analisis transparansi menggunakan _Explainable AI_ (XAI), dapat ditarik beberapa kesimpulan utama:

- Model deteksi ujaran kebencian _multilabel_ berhasil dibangun menggunakan arsitektur IndoBERT yang mampu mengidentifikasi enam kategori kebencian secara simultan. Hasil eksperimen membuktikan keunggulan IndoBERT dengan raihan _Macro_ _F1-score_ sebesar 0,69, melampaui model klasik CC-SVM yang mencapai 0,61. Secara komprehensif, IndoBERT mencapai _Micro_ _F1-score_ sebesar 0,73 dan Hamming Loss sebesar 0,0493, yang menunjukkan tingkat kesalahan prediksi label yang rendah. Optimisasi _threshold_ per label terbukti krusial dalam meningkatkan sensitivitas terhadap kategori minoritas, dengan peningkatan paling signifikan pada label HS*Physical (+12,7% atau +0,127 poin \_F1-score*) dan HS*Gender (+6,5% atau +0,065 poin \_F1-score*). Hasil ini mengonfirmasi bahwa penggunaan _threshold_ tetap (0,5) tidak optimal untuk _dataset multilabel_ dengan distribusi tidak seimbang.
- Pendekatan _Explainable AI_ (XAI) melalui implementasi manual KernelSHAP berhasil memberikan transparansi pada tingkat kata. Analisis menunjukkan perbedaan signifikan pola pikir model, di mana model klasik cenderung bersifat leksikal (menghafal kata kasar), sedangkan IndoBERT lebih cerdas secara kontekstual dengan menangkap makna kata ganti target dan narasi dehumanisasi.
- Evaluasi menunjukkan bahwa optimasi _threshold_ per label sangat vital dalam meningkatkan sensitivitas model terhadap kategori minoritas seperti fisik dan gender. Validitas hasil ini diperkuat melalui implementasi prototipe aplikasi _web_ Streamlit yang mampu memvisualisasikan kontribusi kata secara _real-time_ bagi pengguna. Visualisasi ini memberikan transparansi penuh kepada pengguna dan memastikan sistem dapat dipertanggungjawabkan karena proses deteksi tidak lagi bersifat _black-box_, melainkan memiliki landasan bukti linguistik yang terukur.
- Penerapan KernelSHAP sebagai metode _Explainable_ AI terbukti efektif dalam memberikan transparansi terhadap keputusan model. Implementasi manual algoritma KernelSHAP memungkinkan kontrol penuh terhadap proses atribusi fitur dan menghasilkan visualisasi yang intuitif bagi pengguna non-teknis. Analisis studi kasus menunjukkan perbedaan fundamental antara model klasik yang bersifat leksikal (mengandalkan kata kunci) dengan IndoBERT yang kontekstual (memahami peran sintaksis dan narasi). Meskipun demikian, penelitian ini belum melakukan evaluasi kuantitatif terhadap keterpakaian dan tingkat kepercayaan dari penjelasan yang dihasilkan, yang dapat menjadi fokus penelitian lanjutan.

## Saran

Penelitian ini tentu masih memiliki ruang untuk penyempurnaan. Agar pengembangan sistem deteksi ujaran kebencian ke depannya bisa lebih relevan, akurat, dan efisien, penulis menyarankan beberapa hal berikut:

- Pembaruan _dataset_ (Data Terkini): Bahasa di media sosial berkembang sangat cepat; kata _slang_ atau konteks politik tahun lalu mungkin sudah tidak relevan hari ini. Penelitian selanjutnya sangat disarankan untuk menggunakan atau menambahkan _dataset_ yang lebih mutakhir (terbaru) agar model dapat mengenali pola ujaran kebencian yang sedang tren saat ini.
- Perluasan Sumber Data (Platform Lain): Saat ini model hanya dilatih menggunakan data dari Twitter/X. Mengingat karakteristik netizen berbeda-beda di setiap platform, sebaiknya penelitian mendatang memperluas cakupan data ke media sosial lain seperti Instagram (kolom komentar), TikTok, atau Facebook. Hal ini penting agar model tidak bias pada gaya bahasa satu platform saja dan lebih siap menghadapi variasi serangan siber.
- Eksplorasi Teknik XAI Lain: Meskipun SHAP memberikan interpretasi yang akurat, proses komputasinya cukup berat. Penulis menyarankan untuk mengeksplorasi dan membandingkan teknik _Explainable AI_ lainnya, seperti LIME (Local Interpretable Model-agnostic Explanations) atau Attention Rollout (khusus _Transformer_). Tujuannya adalah mencari metode yang mungkin bisa memberikan penjelasan sama baiknya namun dengan waktu komputasi yang lebih ringan.

DAFTAR PUSTAKA

Adadi, A., & Berrada, M. (2018). Peeking Inside the _Black-box_: A Survey on _Explainable_ _Artificial Intelligence_ (XAI). _IEEE Access_, _6_, 52138-52160. <https://doi.org/10.1109/ACCESS.2018.2870052>

Arrieta, A. B., Díaz-Rodríguez, N., Ser, J. Del, Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2019). _Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI_.

<https://arxiv.org/abs/1910.10045>

Boutell, M. R., Luo, J., Shen, X., & Brown, C. M. (2004). _Learning_ multi-label scene _classification_. _Pattern Recognition_, _37_(9), 1757-1771.

[https://doi.org/10.1016/j.patcog.2004.03.009](https://doi.org/https://doi.org/10.1016/j.patcog.2004.03.009)

Cortes, C., & Vapnik, V. (1995). Support-vector networks. _Machine learning_, _20_(3), 273-297. <https://doi.org/10.1007/BF00994018>

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. <https://arxiv.org/abs/1810.04805>

Fortuna, P., & Nunes, S. (2018). A Survey on Automatic Detection of _Hate speech_ in Text. _ACM Computing Surveys (CSUR)_, _51_, 1-30.

<https://api.semanticscholar.org/CorpusID:52184457>

Gibaja, E. L., & Ventura, S. (2015). A Tutorial on Multi-Label _Learning_. _ACM Computing Surveys_, _47_. <https://doi.org/10.1145/2716262>

Graves, A. (2014). _Generating Sequences With Recurrent Neural Networks_. <https://arxiv.org/abs/1308.0850>

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. _Neural Computation_, _9_(8), 1735-1780. <https://doi.org/10.1162/neco.1997.9.8.1735>

Ibrahim, M. A., Arifin, S., Yudistira, I. G. A. A., Nariswari, R., Abdillah, A. A., Murnaka, N. P., & Prasetyo, P. W. (2022). An _Explainable AI_ Model for _Hate speech_ Detection on Indonesian Twitter. _CommIT (Communication and Information Technology) Journal_, _16_(2), 175-182.

<https://doi.org/10.21512/commit.v16i2.8343>

Ibrohim, M. O., & Budi, I. (2019). Multi-label _Hate speech_ and _Abusive_ _Language_ Detection in Indonesian Twitter. In S. T. Roberts, J. Tetreault, V. Prabhakaran, & Z. Waseem (Eds.), _Proceedings of the Third Workshop on Abusive Language Online_ (pp. 46-57). Association for Computational Linguistics. <https://doi.org/10.18653/v1/W19-3506>

Jurafsky, D., & Martin, J. H. (2025). _Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, with Language Models_ (3rd ed.).

[https://_web_.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)

Lundberg, S., & Lee, S.-I. (2017). _A Unified Approach to Interpreting Model Predictions_. <https://arxiv.org/abs/1705.07874>

Madjarov, G., Kocev, D., Gjorgjevikj, D., & Džeroski, S. (2012). An extensive experimental comparison of methods for multi-label _learning_. _Pattern Recognition_, _45_(9), 3084-3104. [https://doi.org/10.1016/j.patcog.2012.03.004](https://doi.org/https://doi.org/10.1016/j.patcog.2012.03.004)

Meltwater, W. A. S. &. (2025). _Digital 2025: Indonesia_.

<https://datareportal.com/reports/digital-2025-indonesia>

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). _Efficient Estimation of Word Representations in Vector Space_. <https://arxiv.org/abs/1301.3781>

Ousidhoum, N., Lin, Z., Zhang, H., Song, Y., & Yeung, D.-Y. (2019). Multilingual and Multi-Aspect _Hate speech_ Analysis. _CoRR_, _abs/1908.11049_.

<http://arxiv.org/abs/1908.11049>

Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. In A. Moschitti, B. Pang, & W. Daelemans (Eds.), _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ (pp. 1532-1543). Association for Computational Linguistics. <https://doi.org/10.3115/v1/D14-1162>

Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2009). _Classifier Chains_ for _Multi-label classification_. In W. Buntine, M. Grobelnik, D. Mladenic, & J. Shawe-Taylor (Eds.), _Machine learning and Knowledge Discovery in Databases_ (pp. 254-269). Springer Berlin Heidelberg.

Saputra, R. A., & Sibaroni, Y. (2025). _Multilabel_ _Hate speech_ Classification in Indonesian Political Discourse on X using Combined _Deep learning_ Models with Considering Sentence Length. _Jurnal Ilmu Komputer Dan Informasi_, _18_(1), 113-125. [https://doi.org/10.21609/jiki.v18i1.1440](https://doi.org/https://doi.org/10.21609/jiki.v18i1.1440)

Tsoumakas, G., & Katakis, I. M. (2007). _Multi-label classification_: An Overview. _Int. J. Data Warehous. Min._, _3_, 1-13.

<https://api.semanticscholar.org/CorpusID:11608263>

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). _Attention Is All You Need_.

<https://arxiv.org/abs/1706.03762>

Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., Soleman, S., Mahendra, R., Fung, P., Bahar, S., & Purwarianti, A. (2020). _IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding_. <https://arxiv.org/abs/2009.05387>

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T. Le, Gugger, S., … Rush, A. M. (2020). _HuggingFace's Transformers: State-of-the-art Natural Language Processing_.

<https://arxiv.org/abs/1910.03771>

Zhang, M.-L., & Zhou, Z.-H. (2014). A review on multi-label _learning_ algorithms. _IEEE Transactions on Knowledge and Data Engineering_, _26_(8), 1819-1837.

<https://doi.org/10.1109/TKDE.2013.39>

&nbsp;

LAMPIRAN

Lampiran 1 Kode Algoritma KernelSHAP untuk Model Klasik

import itertools

import time

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy

def predict_proba(x_vec, clf, target_index):

if hasattr(x_vec, "shape") and len(x_vec.shape) == 1:

x_vec = x_vec.reshape(1, -1)

return clf.predict_proba(x_vec)\[:, target_index\]\[0\]

def predict*on_subset(subset, x\_\_instance*, clf, target_index):

x**masked\_= x**instance\_.copy().toarray().flatten()

keep = set(subset)

all*features_indices = list(range(x\_\_instance*.shape\[1\]))

for j in all_features_indices:

if j not in keep:

x\__masked_\[j\] = 0.0

return predict*proba(x\_\_masked*.reshape(1, -1), clf, target_index)

def kernel_weight(M, k):

if k == 0 or k == M:

return 1e6

return (M - 1) / (scipy.special.comb(M, k) \* k \* (M - k))

def shap*kernel\_\_instance*(

sentence,

vectorizer,

clf,

features_list,

labels,

target_label,

num_samples=128,

):

start_time = time.time()

target_index = labels.index(target_label)

\# --------------------------------------------------------

\# Vectorize input (ONCE)

\# --------------------------------------------------------

x_sparse = vectorizer.transform(\[sentence\])

x_dense = x_sparse.toarray().ravel()

idx_features = x_sparse.nonzero()\[1\]

fitur_aktif = \[features_list\[i\] for i in idx_features\]

M = len(idx_features)

if M == 0:

print("Tidak ada fitur aktif.")

return 0.0, \[\], \[\]

print(f"Fitur aktif ({M}): {fitur_aktif}")

print(f"Menjelaskan label: {target_label}")

\# --------------------------------------------------------

\# Generate Subsets

\# --------------------------------------------------------

total_subsets = 2\*\*M

subsets = \[\]

subsets_seen = set()

if num_samples >= total_subsets:

subsets = \[

list(s) for k in range(M + 1) for s in itertools.combinations(range(M), k)

\]

else:

\# Mandatory: empty & full

subsets = \[\[\], list(range(M))\]

subsets_seen = {(), tuple(range(M))}

possible_ks = list(range(1, M))

raw_weights = np.array(\[kernel_weight(M, k) for k in possible_ks\])

k_probs = raw_weights / raw_weights.sum()

while len(subsets) < num_samples:

k = np.random.choice(possible_ks, p=k_probs)

s = tuple(sorted(np.random.choice(M, size=k, replace=_False_)))

if s not in subsets_seen:

subsets.append(list(s))

subsets_seen.add(s)

num_subsets = len(subsets)

\# --------------------------------------------------------

\# Build Mask Matrix (Z)

\# --------------------------------------------------------

Z = np.zeros((num_subsets, M))

subset_sizes = np.zeros(num_subsets, dtype=int)

for i, subset in enumerate(subsets):

Z\[i, subset\] = 1

subset_sizes\[i\] = len(subset)

\# --------------------------------------------------------

\# Build _Masked_ Feature Matrix (BATCH)

\# --------------------------------------------------------

X_batch = np.zeros((num_subsets, x_dense.shape\[0\]))

X_batch\[:\] = x_dense

\# Zero-out inactive features

for i in range(num_subsets):

inactive = idx_features\[Z\[i\] == 0\]

X_batch\[i, inactive\] = 0.0

\# --------------------------------------------------------

\# Batch Prediction (CRITICAL SPEEDUP)

\# --------------------------------------------------------

probs = clf.predict_proba(X_batch)\[:, target_index\]

\# Baseline = subset kosong

base_value = probs\[subset_sizes == 0\]\[0\]

y_adjusted = probs - base_value

weights = np.array(\[kernel_weight(M, k) for k in subset_sizes\])

\# --------------------------------------------------------

\# _Weighted_ Least Squares WITHOUT diag(W)

\# --------------------------------------------------------

WX = Z \* weights\[:, None\]

Wy = y_adjusted \* weights

try:

beta = np.linalg.pinv(Z.T @ WX) @ (Z.T @ Wy)

except np.linalg.LinAlgError:

beta = np.zeros(M)

shap_values = beta

\# --------------------------------------------------------

\# Output

\# --------------------------------------------------------

shap_results = sorted(

zip(fitur_aktif, shap_values),

key=lambda x: abs(x\[1\]),

reverse=_True_,

)

print("-" \* 50)

for f, val in shap_results:

print(f"{f}: {val:.4f}")

original_proba = clf.predict_proba(x_sparse)\[:, target_index\]\[0\]

sum_shap = base_value + shap_values.sum()

print(f"\\nBaseline (E\[f(x)\]): {base_value:.4f}")

print(f"Baseline + ΣSHAP: {sum_shap:.4f}")

print(f"Prediksi Asli f(x): {original_proba:.4f}")

print(f"Total waktu SHAP: {time.time() - start_time:.4f} detik")

return base_value, shap_values, fitur_aktif

def plot_shap_values(

base_value, shap_values, fitur_aktif, sentence, target_label, top_n=50

):

df_shap = pd.DataFrame({"Fitur": fitur_aktif, "SHAP Value": shap_values})

df_shap\["Abs SHAP"\] = df_shap\["SHAP Value"\].abs()

df*plot = df_shap.sort_values(by="Abs SHAP", ascending=\_False*).head(top_n).iloc\[::-1\]

colors = \["#ff0051" if v > 0 else "#008bfb" for v in df_plot\["SHAP Value"\]\]

fig, ax = plt.subplots(figsize=(10, len(df_plot) \* 0.5 + 2))

ax.barh(

df_plot\["Fitur"\],

df_plot\["SHAP Value"\],

color=colors,

edgecolor="black",

alpha=0.8,

)

ax.axvline(0, color="black", linewidth=0.8)

ax.set_title(

f'Interpretasi SHAP: {target_label}\\n"{sentence\[:50\]}..."', fontsize=12

)

ax.set_xlabel("Kontribusi terhadap Probabilitas")

ax.grid(axis="x", linestyle=":", alpha=0.5)

fig.tight_layout()

return fig

Lampiran 2 Kode Algoritma KernelSHAP untuk Model IndoBERT

import itertools

import time

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.special

import tensorflow as tf

from _Transformers_ import AutoTokenizer, TFAutoModelForSequenceClassification

\# Asumsi back_data diimpor dari file lokal lo

try:

from manual_shap.bert.data_netral import background\_\_Dataset_as back_data

except:

back_data = \["neutral text example"\]

def predict_proba_bert(sentence, clf_bert, tokenizer, target_index):

inputs = tokenizer(sentence, return*tensors="tf", truncation=\_True*, padding=_True_)

logits = clf_bert(inputs).logits

probs = tf.sigmoid(logits).numpy()

return probs\[0, target_index\]

def kernel_weight(M, k):

if k == 0 or k == M:

return 1e3

return (M - 1) / (scipy.special.comb(M, k) \* k \* (M - k))

def tokenize_with_word_mapping(sentence, tokenizer):

words = sentence.split()

encoded = tokenizer(

words,

is*split_into_words=\_True*,

return_tensors="tf",

truncation=_True_,

padding=_False_,

)

tokens = tokenizer.convert_ids_to_tokens(encoded\["input_ids"\]\[0\])

word_ids = encoded.word_ids()

return words, tokens, word_ids

def aggregate_shap_to_words(words, word_ids, idx_features, shap_values):

\# PERBAIKAN: Gunakan list agar urutan terjaga dan kata duplikat tidak hilang

word_shap_list = \[0.0\] \* len(words)

for shap_idx, token_idx in enumerate(idx_features):

word_idx = word_ids\[token_idx\]

if word_idx is not None:

word_shap_list\[word_idx\] += shap_values\[shap_idx\]

return word_shap_list

def calculate_expected_baseline(background_texts, tokenizer, clf_bert, target_index):

probs = \[\]

for text in background_texts:

inputs = tokenizer(text, return*tensors="tf", padding=\_True*, truncation=_True_)

logits = clf_bert(inputs).logits

prob = tf.sigmoid(logits)\[0, target_index\].numpy()

probs.append(prob)

return np.mean(probs)

def shap*kernel\_\_instance*\_bert_fast(

sentence,

tokenizer,

clf_bert,

labels,

target_label,

num_samples=32,

background\__Dataset_\=back_data,

):

start_time = time.time()

target_index = labels.index(target_label)

base_value = calculate_expected_baseline(

background\__Dataset_, tokenizer, clf_bert, target_index

)

words, tokens, word_ids = tokenize_with_word_mapping(sentence, tokenizer)

idx_features = np.array(\[i for i, w in enumerate(word_ids) if w is not None\])

M = len(idx_features)

if M == 0:

return base_value, words, \[0.0\] \* len(words), 0.0

pos_map = {tok: i for i, tok in enumerate(idx_features)}

subsets = \[\[\], list(idx_features)\]

seen = {(), tuple(idx_features)}

ks = np.arange(1, M)

if len(ks) > 0:

probs_k = np.array(\[kernel_weight(M, k) for k in ks\])

probs_k /= probs_k.sum()

while len(subsets) < min(num_samples, 2\*\*M):

k = np.random.choice(ks, p=probs_k)

s = tuple(sorted(np.random.choice(idx*features, size=k, replace=\_False*)))

if s not in seen:

subsets.append(list(s))

seen.add(s)

n = len(subsets)

Z = np.zeros((n, M), dtype=np.float32)

sizes = np.zeros(n, dtype=int)

for i, subset in enumerate(subsets):

for tok in subset:

Z\[i, pos_map\[tok\]\] = 1.0

sizes\[i\] = len(subset)

encoded = tokenizer(sentence, return*tensors="tf", truncation=\_True*, padding=_False_)

input_ids = encoded\["input_ids"\]\[0\].numpy()

attention_mask = encoded\["attention_mask"\]\[0\].numpy()

input_ids_batch = \[\]

for i in range(n):

ids = input_ids.copy()

inactive = idx_features\[Z\[i\] == 0\]

ids\[inactive\] = tokenizer.mask_token_id

input_ids_batch.append(ids)

inputs = {

"input_ids": tf.constant(input_ids_batch),

"attention_mask": tf.constant(\[attention_mask\] \* n),

}

logits = clf_bert(inputs).logits

probs_out = tf.sigmoid(logits)\[:, target_index\].numpy()

y_adj = probs_out - base_value

weights = np.array(\[kernel_weight(M, k) for k in sizes\])

WX = Z \* weights\[:, None\]

Wy = y_adj \* weights

beta = np.linalg.pinv(Z.T @ WX) @ (Z.T @ Wy)

\# OUTPUT: Sekarang mengembalikan list words dan list nilai SHAP yang sinkron

word_shap_values = aggregate_shap_to_words(words, word_ids, idx_features, beta)

return base_value, words, word_shap_values, time.time() - start_time

def plot_shap_values(

base_value, shap_values, fitur_aktif, sentence, target_label, top_n=50

):

df = pd.DataFrame({"Fitur": fitur_aktif, "SHAP Value": shap_values})

df\["abs"\] = df\["SHAP Value"\].abs()

df*plot = df.sort_values("abs", ascending=\_False*).head(top_n).iloc\[::-1\]

colors = \["red" if v > 0 else "blue" for v in df_plot\["SHAP Value"\]\]

fig, ax = plt.subplots(figsize=(10, len(df_plot) \* 0.6 + 1.5))

ax.barh(df_plot\["Fitur"\], df_plot\["SHAP Value"\], color=colors, edgecolor="black")

ax.axvline(0, color="black", linewidth=0.8)

ax.set_title(f'SHAP (BERT - Word Level) - {target_label}\\n"{sentence\[:50\]}..."')

ax.set_xlabel("Kontribusi terhadap Probabilitas")

fig.tight_layout()

return fig

Lampiran 3 Kode Implementasi Antarmuka _Web_ Menggunakan Streamlit

import pickle

import re

import time

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import streamlit as st

import tensorflow as tf

from _Transformers_ import AutoTokenizer, TFAutoModelForSequenceClassification

\# ============================

\# IMPORT SHAP IMPLEMENTATIONS

\# ============================

from manual_shap.baseline.svm_lr_shap import (

plot_shap_values as plot_shap_classic,

)

from manual_shap.baseline.svm_lr_shap import (

shap*kernel\_\_instance* as shap_kernel_classic,

)

from manual_shap.bert.bert_shap import (

plot_shap_values as plot_shap_bert,

)

from manual_shap.bert.bert_shap import (

predict_proba_bert,

shap*kernel\_\_instance*\_bert_fast,

)

from manual_shap.bert.data_netral import background\_\_Dataset_as back_data

\# ============================================================

\# SESSION STATE & STYLING

\# ============================================================

if "predicted" not in st.session_state:

st.session*state.predicted = \_False*

if "cleaned_text" not in st.session_state:

st.session_state.cleaned_text = ""

st.markdown(

"""

&lt;style&gt;

.highlight-container {

margin-top: 25px;

margin-bottom: 25px;

padding: 15px;

border-radius: 10px;

background-color: transparent;

}

&lt;/style&gt;

""",

unsafe*allow_html=\_True*,

)

\# ============================================================

\# LABEL PRETTY & DATA LOAD

\# ============================================================

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

alay_df = pd.read_csv("data\\\\new_kamusalay.csv")

alay_dict = dict(zip(alay_df\["alay"\], alay_df\["normal"\]))

return alay_dict

alay_dict = load_cleaning_resources()

\# Compiled regex for speed

RE_CLEAN = re.compile(r"http\\S+|www\\S+|https\\S+|@\\w+|\\\\n|\\n|\\brt\\b|\[^a-zA-Z\\s\]")

RE_MULTISPACE = re.compile(r"\\s+")

RE_REPEATED = re.compile(r"(.)\\1+")

def clean_for_shap(text, alay_map, max_words=40):

if pd.isna(text):

return ""

\# Hanya hapus simbol/URL, JANGAN hapus kata (stopwords)

s = RE_CLEAN.sub(" ", str(text).lower())

s = RE_MULTISPACE.sub(" ", s).strip()

tokens = s.split()

\# Mapping alay tetap dilakukan agar model paham konteksnya

s = " ".join(\[alay_map.get(t, t) for t in tokens\])

\# Repeated char reduction (misal: 'tidakkkk' -> 'tidak')

s = RE_REPEATED.sub(r"\\1", s)

return " ".join(s.split()\[:max_words\])

def normalize_shap(shap_vals):

if len(shap_vals) == 0:

return shap_vals

max_val = np.max(np.abs(shap_vals))

return shap_vals if max_val == 0 else shap_vals / max_val

def shap_text_highlight(tokens, shap_values):

if len(tokens) == 0 or len(shap_values) == 0:

return "&lt;i&gt;Tidak ada kontribusi SHAP.&lt;/i&gt;"

shap_norm = normalize_shap(np.array(shap_values))

spans = \[

f'&lt;span style="background-color:rgba({255 if val &gt; 0 else 0},0,{0 if val > 0 else 255},{min(abs(val), 1.0)});'

f'padding:4px;margin:2px;border-radius:4px;display:inline-block;font-weight:500;">{tok}&lt;/span&gt;'

for tok, val in zip(tokens, shap_norm)

\]

return (

f'&lt;div class="highlight-container" style="line-height:2.4;"&gt;'

\+ " ".join(spans)

\+ "&lt;/div&gt;"

)

\# ============================================================

\# LOAD MODELS

\# ============================================================

@st.cache_resource

def load_bert(path):

tokenizer = AutoTokenizer.from_pretrained(path)

model = TFAutoModelForSequenceClassification.from_pretrained(path)

return tokenizer, model

bert_tokenizer, bert_model = load_bert("saved_models/model_hs_indobert")

BERT\_\_THRESHOLD_S = np.array(\[0.16, 0.64, 0.18, 0.66, 0.12, 0.34\])

@st.cache_resource

def load_artifact(path):

with open(path, "rb") as f:

return pickle.load(f)

artifact = load_artifact("saved_models/baseline/model_artifact_2.pkl")

vectorizer = artifact\["vectorizer"\]

labels = artifact\["label_columns"\]

features_list = vectorizer.get_feature_names_out()

MODEL_MAP = {

"_BR_ Logistic Regression": (

artifact\["ovr_lr"\],

np.array(artifact\["thresh"\]\["ovr_lr"\]),

),

"_BR_ Calibrated SVM": (

artifact\["ovr_svm"\],

np.array(artifact\["thresh"\]\["ovr_svm"\]),

),

"CC Logistic Regression": (

artifact\["cc_lr"\],

np.array(artifact\["thresh"\]\["cc_lr"\]),

),

"CC Calibrated SVM": (artifact\["cc_svm"\], np.array(artifact\["thresh"\]\["cc_svm"\])),

}

\# ============================================================

\# PREDICTION LOGIC (OPTIMIZED)

\# ============================================================

def predict_bert_batch(text, model, tokenizer):

inputs = tokenizer(

text, return*tensors="tf", truncation=\_True*, padding=_True_, max_length=128

)

logits = model(inputs).logits

probs = tf.sigmoid(logits).numpy()\[0\]

return probs

\# ============================================================

\# UI

\# ============================================================

st.title("🛡️ _Hate speech_ Classification + SHAP")

text_input = st.text_area("Masukkan kalimat yang ingin dianalisis:", height=150)

col1, col2 = st.columns(2)

with col1:

model_type = st.radio(

"Arsitektur Model:", \["Klasik (TF-IDF + ML)", "BERT"\], horizontal=_True_

)

with col2:

num_samples = st.number_input("Sampel KernelSHAP:", 128, 2048, 256, 64)

if model_type == "Klasik (TF-IDF + ML)":

model_name = st.selectbox("Model Spesifik:", list(MODEL_MAP.keys()))

if st.button("Mulai Analisis Prediksi", type="primary"):

st.session_state.cleaned_text = clean_for_shap(text_input, alay_dict)

st.session*state.predicted = \_True*

\# ============================================================

\# MAIN LOGIC

\# ============================================================

if st.session_state.predicted:

current_text = st.session_state.cleaned_text

st.markdown("---")

st.subheader("📊 Hasil Prediksi Model")

if model_type == "Klasik (TF-IDF + ML)":

clf, \_threshold_s = MODEL_MAP\[model_name\]

vec = vectorizer.transform(\[current_text\])

probs = clf.predict_proba(vec)\[0\]

pred = (probs >= \_threshold_s).astype(int)

else:

\# Optimized: Only one forward pass for all labels

probs = predict_bert_batch(current_text, bert_model, bert_tokenizer)

pred = (probs >= BERT\_\_THRESHOLD_S).astype(int)

any_hs = np.any(pred == 1)

if any_hs:

det = \[

LABEL_PRETTY.get(labels\[i\], labels\[i\]) for i, v in enumerate(pred) if v == 1

\]

st.error(f"⚠️ \*\*Terdeteksi _Hate speech_!\*\* Kategori: \*\*{', '.join(det)}\*\*.")

else:

st.success("✅ \*\*Teks Aman.\*\* Tidak terdeteksi ujaran kebencian.")

df_res = pd.DataFrame(

{

"Label": \[LABEL_PRETTY.get(l, l) for l in labels\],

"Probabilitas (%)": \[f"{p \* 100:.2f}%" for p in probs\],

"_Threshold_": \_threshold_s

if model_type == "Klasik (TF-IDF + ML)"

else BERT\_\_THRESHOLD_S,

"Hasil": \["⚠️ Terdeteksi" if x == 1 else "✅ Aman" for x in pred\],

}

)

st.table(df_res)

st.subheader("🔍 Penjelasan Kontribusi Kata (SHAP)")

tabs = st.tabs(\[LABEL_PRETTY.get(l, l) for l in labels\])

for i, label in enumerate(labels):

with tabs\[i\]:

t_start = time.time()

if model_type == "Klasik (TF-IDF + ML)":

base, shap_vals, feats = shap_kernel_classic(

current_text,

vectorizer,

MODEL_MAP\[model_name\]\[0\],

features_list,

labels,

label,

num_samples=num_samples,

)

else:

\# Perhatikan urutan variabel: base, feats (list kata), shap_vals (list nilai)

base, feats, shap*vals,* = shap*kernel\_\_instance*\_bert_fast(

current_text, bert_tokenizer, bert_model, labels, label, num_samples

)

\# Karena sudah list, panjang pasti sama dengan jumlah kata di kalimat asli

st.markdown(

shap_text_highlight(feats, shap_vals),

unsafe*allow_html=\_True*,

)

if model_type == "Klasik (TF-IDF + ML)":

st.pyplot(

plot_shap_classic(base, shap_vals, feats, current_text, label)

)

else:

\# Pastikan shap_vals dikonversi ke numpy array untuk plotting

st.pyplot(

plot_shap_bert(

base, np.array(shap_vals), feats, current_text, label

)

)

st.caption(f"⏱️ Komputasi: {time.time() - t_start:.2f} detik")
