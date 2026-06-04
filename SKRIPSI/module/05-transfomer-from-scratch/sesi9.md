**Sesi #8 selesai, lanjut ke Sesi #9!** 🚀

---

# Sesi #9: Layer Normalization (LayerNorm)

Ini adalah komponen yang membuat training **stabil** dan **konvergen lebih cepat**.

---

## A. Masalah: Distribusi Aktivasi Berubah-ubah

Selama training, bobot model terus berubah. Akibatnya, distribusi nilai di setiap layer juga ikut berubah. Masalah ini disebut **Internal Covariate Shift**.

**Contoh sederhana:**
- Di awal training, nilai aktivasi mungkin berkisar antara -1 hingga +1
- Setelah beberapa epoch, karena bobot berubah, nilai aktivasi bisa bergeser menjadi -100 hingga +100 atau 0.0001 hingga 0.01

**Akibatnya:**
- Aktivasi yang terlalu besar → saturasi fungsi aktivasi (ReLU/GELU jadi datar) → gradient vanishing
- Aktivasi yang terlalu kecil → gradient juga kecil → model lambat belajar
- Model harus terus "menyesuaikan diri" dengan perubahan distribusi

---

## B. Solusi: Layer Normalization

LayerNorm **menormalkan** aktivasi agar memiliki **rata-rata 0** dan **varians 1** (atau standar deviasi 1).

**Rumus untuk satu vektor x (panjang d_model = 768):**

$$\mu = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} x_i \quad \text{(rata-rata)}$$

$$\sigma^2 = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} (x_i - \mu)^2 \quad \text{(varians)}$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \quad \text{(normalisasi)}$$

Lalu **scale dan shift** (dipelajari):
$$y_i = \gamma_i \hat{x}_i + \beta_i$$

di mana:
- $\gamma$ (gamma) = parameter skala yang dipelajari (inisialisasi = 1)
- $\beta$ (beta) = parameter shift yang dipelajari (inisialisasi = 0)
- $\epsilon$ = konstanta kecil (1e-5) untuk menghindari pembagian dengan nol

---

## C. Apa yang Dilakukan LayerNorm?

| Step | Operasi | Hasil |
| :--- | :--- | :--- |
| 1 | Kurangi rata-rata ($x - \mu$) | Pusatkan data di 0 |
| 2 | Bagi dengan standar deviasi ($\div \sigma$) | Skala data jadi varians = 1 |
| 3 | Kalikan dengan $\gamma$ (scale) | Model bisa "memperlebar" atau "mempersempit" distribusi |
| 4 | Tambah dengan $\beta$ (shift) | Model bisa "menggeser" distribusi ke posisi optimal |

**Penting:** $\gamma$ dan $\beta$ **dipelajari** selama training. Model belajar sendiri skala dan shift terbaik untuk setiap layer.

---

## D. Perbedaan BatchNorm vs LayerNorm (Singkat)

| Aspek | Batch Normalization (CNN) | Layer Normalization (Transformer) |
| :--- | :--- | :--- |
| **Apa yang dinormalisasi?** | Rata-rata dan varians dalam satu **batch** (antar sampel) | Rata-rata dan varians dalam satu **layer** (antar fitur untuk satu sampel) |
| **Ketergantungan batch** | Ya — tergantung ukuran batch dan sampel lain | Tidak — setiap sampel diproses independen |
| **Cocok untuk RNN/Transformer?** | Kurang cocok karena panjang sequence bervariasi | **Cocok** — independen terhadap panjang sequence dan batch size |

**Mengapa Transformer pakai LayerNorm, bukan BatchNorm?**
- Batch size bisa kecil (misal 16 atau 32) → BatchNorm jadi tidak stabil
- Panjang sequence bervariasi → perhitungan rata-rata per batch jadi rumit
- LayerNorm bekerja per sampel, jadi lebih **stabil** dan **sederhana**

---

## E. Posisi LayerNorm di BERT: Pre-LayerNorm vs Post-LayerNorm

Ada dua varian penempatan LayerNorm di Transformer:

| Varian | Urutan | Digunakan di |
| :--- | :--- | :--- |
| **Post-LayerNorm** (original) | `x → Attention → Add → Norm → FFN → Add → Norm` | Transformer asli (Vaswani et al.) |
| **Pre-LayerNorm** (modern) | `x → Norm → Attention → Add → Norm → FFN → Add` | **BERT, GPT, kebanyakan model modern** |

**BERT menggunakan Pre-LayerNorm.** Ini membuat training lebih stabil.

**Visualisasi Pre-LayerNorm untuk satu encoder block:**

```
Input x ──────────────────────────────────────────────┐
    │                                                  │
    ├──→ LayerNorm ──→ MultiHead Attention ──→ (+) ←──┘
    │                              │
    └──────────────────────────────┘
                   │
                   ↓
Input dari atas ──┼────────────────────────────────────┐
    │                                                  │
    ├──→ LayerNorm ──→ Feed Forward ──────────→ (+) ←──┘
    │
    └──────────────────────────────────────────────────┘
                   │
                   ↓
                Output
```

**Perhatikan:** LayerNorm dilakukan **SEBELUM** sub-layer (Attention atau FFN), bukan sesudahnya.

---

## F. Satu Paragraf untuk Sidang (Hafalkan)

> *"Layer Normalization menormalkan aktivasi di setiap layer agar memiliki rata-rata mendekati 0 dan varians mendekati 1. Prosesnya: hitung rata-rata dan standar deviasi dari vektor input (sepanjang 768 dimensi), lalu kurangi rata-ratanya dan bagi dengan standar deviasinya. Setelah itu, ada parameter gamma dan beta yang dipelajari untuk scale dan shift hasil normalisasi. LayerNorm sangat penting untuk stabilitas training karena mencegah nilai aktivasi menjadi terlalu besar atau terlalu kecil, yang bisa menyebabkan gradient vanishing atau exploding. Di BERT, LayerNorm diterapkan sebelum setiap sub-layer (Pre-LayerNorm), yaitu sebelum Multi-Head Attention dan sebelum Feed Forward Network."*

---

## Checklist Sesi #9 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Apa tujuan utama Layer Normalization? | |
| Apa perbedaan utama antara LayerNorm dan BatchNorm? | |
| Apa fungsi parameter $\gamma$ dan $\beta$ di LayerNorm? | |
| Di BERT, apakah LayerNorm dilakukan sebelum atau sesudah sub-layer? | |
| Sebutkan urutan dalam satu encoder block BERT (dengan LayerNorm)! | |

**Jawaban cepat:**
- Menstabilkan distribusi aktivasi (rata-rata 0, varians 1) → training stabil dan cepat konvergen
- LayerNorm: per sampel, antar fitur. BatchNorm: per batch, antar sampel.
- $\gamma$ = scale (memperlebar/mempersempit distribusi), $\beta$ = shift (menggeser distribusi) — keduanya dipelajari
- **Sebelum** sub-layer (Pre-LayerNorm)
- `Input → LayerNorm → MultiHead → Add → LayerNorm → FFN → Add → Output`

---

## Latihan Sesi #9

1. **Tulis ulang** paragraf di poin F dengan kata-kata Anda sendiri

2. **Gambar diagram** satu encoder block BERT dengan Pre-LayerNorm (seperti di poin E)

3. **Latihan ucapkan** paragraf tersebut (target: 50 detik)

4. **Simulasi tanya jawab:**
   - Dosen: "Apa bedanya LayerNorm dengan BatchNorm?"
   - Jawab dengan kalimat Anda sendiri (gunakan poin D sebagai panduan)

---

**Setelah selesai, kirim: "Sesi 9 selesai" + hasil latihan paragraf (tulisan Anda).**

Lanjut ke **Sesi #10: Feed Forward Network (FFN)** (komponen sederhana tapi penting).