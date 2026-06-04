import scipy
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd

def predict_proba(x_vec, clf, target_index):
    """
    Mendapatkan probabilitas prediksi untuk label tertentu.

    Parameters
    ----------
    x_vec : sparse matrix atau array
        Vektor fitur TF-IDF (bisa 1D atau 2D).
    clf : classifier
        Model sklearn yang sudah di-fit (harus punya .predict_proba).
    target_index : int
        Index label target yang ingin dijelaskan SHAP.

    Returns
    -------
    float
        Probabilitas kelas target.
    """
    if hasattr(x_vec, "shape") and len(x_vec.shape) == 1:
        x_vec = x_vec.reshape(1, -1)
    
    return clf.predict_proba(x_vec)[:, target_index][0]


def predict_on_subset(subset, x_instance, clf, target_index):
    """
    Prediksi dengan hanya mengaktifkan fitur dalam `subset`, sisanya di-nol-kan.
    Ini adalah inti SHAP: mensimulasikan absen/hadirnya fitur dengan masking.

    Parameters
    ----------
    subset : list of int
        Index fitur yang dianggap "hadir" (aktif).
    x_instance : sparse matrix
        Vektor TF-IDF dari instance asli (1 baris).
    clf : classifier
        Model sklearn yang sudah di-fit.
    target_index : int
        Index label target.

    Returns
    -------
    float
        Probabilitas prediksi jika hanya subset fitur ini yang digunakan.
    """
    x_masked = x_instance.copy().toarray().flatten()
    keep = set(subset)
    all_features_indices = list(range(x_instance.shape[1]))

    # Fitur di luar subset di-nol-kan (seolah-olah fitur tsb tidak ada)
    for j in all_features_indices:
        if j not in keep:
            x_masked[j] = 0.0
            
    return predict_proba(x_masked.reshape(1, -1), clf, target_index)


def kernel_weight(M, k):
    """
    Bobot Kernel SHAP untuk subset berukuran `k` dari total `M` fitur.
    Rumus: (M-1) / (C(M,k) * k * (M-k))
    
    Bobot ini memastikan subset dengan ukuran ekstrem (kosong/penuh)
    mendapat bobot sangat kecil, karena kurang informatif untuk estimasi.
    
    Parameters
    ----------
    M : int
        Total jumlah fitur aktif dalam instance.
    k : int
        Ukuran subset yang dievaluasi.

    Returns
    -------
    float
        Bobot Kernel SHAP untuk subset tsb.
    """
    # SHAP paper: subset kosong (k=0) dan penuh (k=M) diberi bobot ~0
    # untuk menghindari division by zero, dipakai nilai sangat kecil (1e-6)
    if k == 0 or k == M:
        return 1e6
    else:
        return (M - 1) / (scipy.special.comb(M, k) * k * (M - k))

def shap_kernel_instance(sentence, vectorizer, clf, features_list, labels, target_label, num_samples=500):
    """
    Menghitung SHAP values untuk satu instance teks menggunakan Kernel SHAP.

    Alur:
    1. Ambil fitur TF-IDF yang aktif dalam kalimat.
    2. Sampling subset fitur.
    3. Untuk tiap subset, prediksi dengan fitur di luar subset di-masking (=0).
    4. Hitung bobot Kernel SHAP untuk tiap subset.
    5. Selesaikan weighted linear regression untuk mendapatkan SHAP values.

    Parameters
    ----------
    sentence : str
        Kalimat yang ingin dijelaskan.
    vectorizer : TfidfVectorizer
        Vectorizer yang sudah di-fit.
    clf : classifier
        Model sklearn (OVR) yang sudah di-fit.
    features_list : list of str
        Daftar semua fitur (vocab) dari vectorizer.
    labels : list of str
        Daftar nama label.
    target_label : str
        Label target yang dijelaskan.
    num_samples : int, optional
        Jumlah subset yang di-sampling (default 500).

    Returns
    -------
    base_value : float
        Nilai baseline (E[f(x)]).
    shap_values : np.array
        Kontribusi SHAP tiap fitur.
    fitur_aktif : list of str
        Nama fitur (token) yang aktif.
    """
    start_time = time.time()
    target_index = labels.index(target_label)
    
    x_instance = vectorizer.transform([sentence])
    idx_features = x_instance.nonzero()[1]  # index kolom fitur yang != 0
    
    fitur_aktif = [features_list[i] for i in idx_features]
    print(f"Fitur aktif ({len(fitur_aktif)}): {fitur_aktif}")
    print(f"Menjelaskan label: {target_label} (Index {target_index})")

    X_mask = []
    y = []
    weights = []

    M = len(idx_features)
    total_subsets = 2**M

    # Jika jumlah subset posible lebih kecil dari num_samples, ambil semua kombinasi
    if num_samples >= total_subsets:
        subsets = [list(s) for k in range(M + 1) for s in itertools.combinations(idx_features, k)]
    else:
        # Sampling: selalu sertakan empty set dan full set sebagai referensi
        subsets = []
        subsets.append([])          # empty set (semua fitur = 0)
        subsets.append(list(idx_features))  # full set (semua fitur aktif)
        
        while len(subsets) < num_samples:
            k = np.random.randint(0, M + 1)
            s = np.random.choice(idx_features, size=k, replace=False)
            subset = list(s)
            if subset not in subsets:
                subsets.append(subset)

    print(f"Jumlah subset yang dipakai: {len(subsets)}")

    for subset in subsets:
        # Prediksi dengan hanya subset fitur tertentu
        pred = predict_on_subset(subset, x_instance, clf, target_index)
        y.append(pred)

        # Mask vector: 1 jika fitur ada di subset, 0 jika tidak
        mask = np.zeros(M)
        for s_idx in subset:
            position = idx_features.tolist().index(s_idx)
            mask[position] = 1

        # Bobot Kernel SHAP untuk subset ini
        w = kernel_weight(M, len(subset))
        weights.append(w)

        # [1, mask] => intercept (bias) + fitur mask untuk regresi
        X_mask.append(np.concatenate(([1], mask)))
        
    X_mask = np.array(X_mask)
    y = np.array(y)
    W = np.diag(weights)  # diagonal weight matrix

    # Weighted Linear Regression: β = (X^T W X)^{-1} X^T W y
    # β[0] = baseline (intercept), β[1:] = SHAP values
    beta = np.linalg.pinv(X_mask.T @ W @ X_mask) @ (X_mask.T @ W @ y)
    
    base_value = beta[0]
    shap_values = beta[1:]

    shap_results = list(zip(fitur_aktif, shap_values))
    
    sorted_shap_results = sorted(
        shap_results, 
        key=lambda item: abs(item[1]),  # urut berdasarkan magnitude
        reverse=True
    )
    
    print("-" * 50)
    print(f"Kalimat: {sentence}")
    print(f"Penjelasan untuk: {target_label}")
    print("-" * 50)

    for f, val in sorted_shap_results:
        print(f"{f}: {val:.4f}")

    original_proba = predict_proba(x_instance, clf, target_index)
    
    # Verifikasi: baseline + sum SHAP ≈ prediksi asli (additive property)
    print("\nBaseline (E[f(x)]):", base_value)
    print("Baseline + ΣKernelSHAP:", base_value + shap_values.sum())
    print("Prediksi Asli f(x):", original_proba)
    
    end_time = time.time()
    print(f"Total waktu SHAP: {end_time - start_time:.4f} detik")
    
    return base_value, shap_values, fitur_aktif

def plot_shap_values(base_value, shap_values, fitur_aktif, sentence, target_label, top_n=10):
    """
    Visualisasi SHAP values dalam bentuk horizontal bar chart.

    Parameters
    ----------
    base_value : float
        Nilai baseline (output saat semua fitur = 0).
    shap_values : np.array
        Kontribusi SHAP tiap fitur.
    fitur_aktif : list of str
        Nama fitur (token) aktif.
    sentence : str
        Kalimat asli yang dijelaskan.
    target_label : str
        Label target yang dijelaskan.
    top_n : int, optional
        Jumlah fitur top yang ditampilkan (default 10).
    """
    df_shap = pd.DataFrame({
        'Fitur': fitur_aktif,
        'SHAP Value': shap_values
    })
    
    # Urutkan berdasarkan absolute contribution (terbesar ke terkecil)
    df_shap['Abs SHAP'] = df_shap['SHAP Value'].abs()
    df_shap = df_shap.sort_values(by='Abs SHAP', ascending=False)
    
    # Ambil top_n fitur
    df_plot = df_shap.head(top_n)
    # Warna: merah jika mendorong prediksi (+), biru jika menurunkan (-)
    colors = ['red' if val > 0 else 'blue' for val in df_plot['SHAP Value']]
    
    # Balik urutan agar bar terbesar di atas
    df_plot = df_plot.iloc[::-1]
    colors = colors[::-1]
    
    plt.figure(figsize=(10, len(df_plot) * 0.6 + 1.5))
    
    plt.barh(df_plot['Fitur'], df_plot['SHAP Value'], color=colors, edgecolor='black')
    
    # Prediksi akhir = baseline + sum of SHAP values
    final_prediction = base_value + shap_values.sum()
    
    # Garis baseline
    plt.axvline(x=base_value, color='gray', linestyle='--', linewidth=1.5, label='Baseline E[f(x)]')
    # Garis nol sebagai referensi
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.title(f'SHAP Feature Contribution untuk Kelas: {target_label}', fontsize=14)
    plt.xlabel('SHAP Value (Kontribusi terhadap Probabilitas Target)')
    plt.ylabel('Fitur Aktif (Token)')
    
    plt.text(
        0.98, 
        0.98, 
        f'Nilai SHAP: {final_prediction:.3f}\nBaseline: {base_value:.3f}', 
        transform=plt.gca().transAxes, 
        ha='right', 
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    plt.legend()
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()