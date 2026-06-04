**Sesi #9 selesai, lanjut ke Sesi #10!** 💪

---

# Sesi #10: Feed Forward Network (FFN)

Ini adalah komponen yang sering dianggap "hanya tambahan", padahal memiliki peran penting.

---

## A. Apa itu Feed Forward Network (FFN)?

Setelah Multi-Head Attention, outputnya masuk ke **FFN** — sebuah jaringan sederhana yang diterapkan **per token secara independen**.

**Arsitektur FFN di Transformer/BERT:**

```
Input (seq_len, 768)
        │
        ↓
Linear Layer 1: 768 → 3072 (4× lebih besar)
        │
        ↓
Aktivasi GELU (atau ReLU di Transformer original)
        │
        ↓
Linear Layer 2: 3072 → 768 (kembali ke ukuran semula)
        │
        ↓
Output (seq_len, 768)
```

**Rumus:**
$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

di mana:
- $W_1$ : bobot linear pertama (768 × 3072)
- $b_1$ : bias pertama (3072)
- $W_2$ : bobot linear kedua (3072 × 768)
- $b_2$ : bias kedua (768)

---

## B. Mengapa FFN Diperlukan?

Attention sudah melakukan **interaksi antar token** (token i melihat token lain). Tapi interaksi ini bersifat **linear** (dot product, weighted sum).

Apa yang kurang? **Non-linearitas** dan **transformasi per token secara mendalam**.

| Peran FFN | Penjelasan |
| :--- | :--- |
| **Menambah non-linearitas** | Aktivasi GELU membuat model bisa belajar fungsi yang lebih kompleks, tidak hanya linear. |
| **Transformasi per token** | Setiap token diproses secara independen (tidak melihat token lain). Ini melengkapi attention yang justru melihat token lain. |
| **Kapasitas komputasi** | Dua layer linear dengan dimensi 3072 memberikan "ruang komputasi" yang cukup untuk memproses informasi yang sudah dikumpulkan oleh attention. |

**Analoginya:**
- **Attention** = rapat tim: semua anggota (token) berdiskusi dan saling bertukar informasi
- **FFN** = kerja individu: setiap anggota memproses informasi yang sudah didapat, lalu memikirkan "apa respons saya?"

---

## C. Mengapa Dimensi 3072? (4× d_model)

Di BERT Base, FFN memiliki dimensi hidden = **3072** = 4 × 768.

**Alasan:**
- Cukup besar untuk memberikan kapasitas komputasi yang memadai
- Tidak terlalu besar sehingga boros parameter
- Merupakan rasio standar dari Transformer original (4×)

| Model | d_model | FFN hidden | Rasio |
| :--- | :--- | :--- | :--- |
| BERT Base | 768 | 3072 | 4× |
| BERT Large | 1024 | 4096 | 4× |
| GPT-3 (small) | 768 | 3072 | 4× |

**Parameter FFN:**
- W1: 768 × 3072 = 2.359.296 parameter
- b1: 3072 parameter
- W2: 3072 × 768 = 2.359.296 parameter
- b2: 768 parameter
- **Total per FFN ~4.7M parameter**

Karena BERT punya 12 layer, total parameter FFN saja = 12 × 4.7M ≈ **56M parameter** (sekitar setengah dari total 110M parameter BERT).

---

## D. Aktivasi: GELU (bukan ReLU)

Transformer original menggunakan **ReLU**:
$$\text{ReLU}(x) = \max(0, x)$$

BERT menggunakan **GELU (Gaussian Error Linear Unit)**:
$$\text{GELU}(x) = x \cdot \Phi(x)$$

di mana $\Phi(x)$ adalah fungsi distribusi kumulatif dari distribusi normal standar.

**Perbedaan visual:**
- ReLU: 0 untuk x < 0, linear untuk x ≥ 0
- GELU: halus (smooth), tidak ada sudut tajam, untuk x negatif nilainya tidak persis 0 tapi mendekati 0

**Kenapa GELU?**
- Lebih halus → gradien lebih stabil
- Performa lebih baik secara empiris untuk model besar

**Yang perlu Anda ingat:** BERT pakai GELU, bukan ReLU. Tidak perlu hafal rumus $\Phi(x)$.

---

## E. Visualisasi Satu Encoder Block Lengkap (Setelah Sesi #10)

Sekarang kita sudah punya semua komponen untuk menggambar **satu encoder block BERT**:

```
Input X (seq_len, 768)
         │
         ├──────────────────────────────────────────────┐
         │                                              │
         ↓                                              │
    LayerNorm                                           │
         ↓                                              │
    Multi-Head Attention (12 heads, d_k=64)             │
         ↓                                              │
        (+) ←───────────────────────────────────────────┘ (residual)
         │
         ├──────────────────────────────────────────────┐
         │                                              │
         ↓                                              │
    LayerNorm                                           │
         ↓                                              │
    Feed Forward Network (768 → 3072 → 768)             │
         ↓                                              │
        (+) ←───────────────────────────────────────────┘ (residual)
         │
         ↓
    Output (seq_len, 768) → ke encoder block berikutnya (atau output layer)
```

**Urutan dalam satu baris (hafalkan):**
> *"Input → LayerNorm → MultiHead Attention → Add (residual) → LayerNorm → Feed Forward → Add (residual) → Output"*

---

## F. Satu Paragraf untuk Sidang (Hafalkan)

> *"Feed Forward Network adalah komponen yang diterapkan setelah Multi-Head Attention. FFN terdiri dari dua layer linear dengan aktivasi GELU di antaranya. Dimensi input dan output adalah 768, sedangkan dimensi hidden-nya 3072 — empat kali lebih besar. FFN memproses setiap token secara independen, tidak seperti attention yang memproses interaksi antar token. Tujuan FFN adalah memberikan kapasitas komputasi non-linear yang cukup bagi model untuk memproses informasi yang sudah dikumpulkan oleh attention. Parameter FFN cukup besar — sekitar setengah dari total parameter BERT."*

---

## Checklist Sesi #10 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Berapa dimensi input dan output FFN di BERT Base? | |
| Berapa dimensi hidden FFN? Berapa rasionya terhadap d_model? | |
| Apa fungsi aktivasi yang digunakan FFN di BERT? | |
| Apakah FFN memproses interaksi antar token atau per token independen? | |
| Sebutkan posisi FFN dalam satu encoder block! | |

**Jawaban cepat:**
- 768
- 3072, rasio 4×
- **GELU** (bukan ReLU)
- Per token independen (tidak ada interaksi antar token)
- Setelah LayerNorm kedua, sebelum residual addition kedua

---

## Latihan Sesi #10

1. **Tulis ulang** paragraf di poin F dengan kata-kata Anda sendiri

2. **Gambar** diagram FFN: `768 → Linear → 3072 → GELU → Linear → 768`

3. **Latihan ucapkan** paragraf tersebut (target: 40 detik)

4. **Hitung manual** (perkiraan):
   - Satu layer FFN punya parameter berapa?
   - (Petunjuk: W1 = 768×3072, b1=3072, W2=3072×768, b2=768)

**Jawaban latihan nomor 4:**
- W1: 768×3072 = 2.359.296
- b1: 3.072
- W2: 3072×768 = 2.359.296
- b2: 768
- Total = 2.359.296 + 3.072 + 2.359.296 + 768 = **4.722.432** ≈ 4.7M parameter

---

## Ringkasan Setelah 10 Sesi

Anda sekarang sudah paham:

| Sesi | Topik | Status |
| :--- | :--- | :--- |
| #1 | Kenapa Transformer? (RNN vs Transformer) | ✅ |
| #2 | Tokenisasi dan Embedding | ✅ |
| #3 | Positional Encoding | ✅ |
| #4 | Konsep Q, K, V | ✅ |
| #5 | Rumus Attention + Operasi Matriks | ✅ |
| #6 | Mengapa diskala dengan √d_k | ✅ |
| #7 | Multi-Head Attention | ✅ |
| #8 | Residual Connection | ✅ |
| #9 | Layer Normalization | ✅ |
| #10 | Feed Forward Network | ✅ |

**Tersisa 8 sesi lagi menuju IndoBERT!**

---

**Setelah selesai, kirim: "Sesi 10 selesai" + hasil latihan paragraf (tulisan Anda).**

Lanjut ke **Sesi #11: Satu Encoder Block Utuh + Stack of Encoders** (menyatukan semua komponen).