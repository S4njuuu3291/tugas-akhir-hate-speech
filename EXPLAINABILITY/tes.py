import numpy as np

def predict_model_universal(text_list):
    """
    Mock fungsi model Black-Box universal (misal prediksi skor sentimen negatif).
    Menerima list berisi string kalimat, mengembalikan array NumPy float (y_pred).
    """
    y_pred = []
    
    # Nilai dasar (Bias/Baseline) ketika inputnya kosong tanpa kata
    baseline_bias = 0.10 
    
    # Nilai kontribusi murni tiap kata yang disembunyikan di dalam model.
    # Nilai-nilai inilah yang harus berhasil ditebak oleh rumus WLS NumPy Anda!
    hidden_weights = {
        "aku": 0.05,
        "benci": 0.70,
        "cina": 0.243,
        "daratan": 0.2
    }
    
    for text in text_list:
        score = baseline_bias
        
        # Pecah kalimat menjadi kata-kata terpisah
        words_in_text = text.split()
        
        # Jika kata kunci aktif di dalam teks sampel, tambahkan skornya
        for word, weight in hidden_weights.items():
            if word in words_in_text:
                score += weight
                
        y_pred.append(score)
        
    return np.array(y_pred)

def factorial(x):
    hasil = 1
    for x in range(1,x+1):
        hasil*=x
    return hasil

def combination(N,k):
    return factorial(N)/(factorial(k)*factorial(N-k))

def kernel_weight(M,k):
    if k == M or k == 0:
        return 1e6
    return (M-1)/(combination(M,k)*k*(M-k))

kalimat = "aku benci cina daratan"

subsets = [
    "aku benci",  
    "aku cina",   
    "cina",       
    "benci",
    "aku",
    "daratan" # <--- Tambahkan ini
]

M = 4
n_sampel = 5
y_pred = predict_model_universal(subsets)
y_shap = y_pred - 0.10

print(y_shap)

X_mask = []
print(X_mask)

weights = []

for subset in subsets:
    mask = np.zeros(M)
    for s_idx in subset.split():
        position = kalimat.split().index(s_idx)
        mask[position] = 1
    X_mask.append(mask)

    weights.append(kernel_weight(M,len(subset.split())))


X_mask = np.array(X_mask)

y_shap = np.array(y_shap)
W = np.diag(weights)
print(X_mask.shape,y_shap.shape,W.shape)

print(np.linalg.pinv(X_mask.T @ W @ X_mask ) @ (X_mask.T @ W @ y_shap))