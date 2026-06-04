**Sesi #6 selesai, lanjut ke Sesi #7!** 💪

---

# Sesi #7: Multi-Head Attention

Ini adalah komponen yang membuat BERT/IndoBERT memiliki **12 "sudut pandang" berbeda** secara paralel.

---

## A. Mengapa Perlu Lebih dari Satu Head?

Dengan **single-head attention**, kita hanya punya **satu** pasangan matriks (W^Q, W^K, W^V) yang dipelajari. Akibatnya: model hanya bisa menangkap **satu jenis hubungan** antar token.

Padahal dalam bahasa, hubungan antar kata sangat beragam:

| Jenis Hubungan | Contoh |
| :--- | :--- |
| Hubungan subjek-predikat | "Saya **makan** nasi" |
| Hubungan objek | "Saya makan **nasi**" |
| Hubungan jarak jauh | "Rumah yang **aku** lihat kemarin itu **besar**" |
| Hubungan lokal | "**sangat** indah" |
| Hubungan semantik | "mobil" dengan "roda" |

**Multi-head attention** memungkinkan model memiliki **banyak "kacamata" berbeda** — setiap head bisa fokus ke jenis hubungan yang berbeda.

---

## B. Cara Kerja Multi-Head Attention (BERT Style)

**Langkah 1:** Input X (bentuk: `[seq_len, d_model]`), dengan `d_model = 768`.

**Langkah 2:** Proyeksikan X ke Q, K, V untuk **setiap head** (12 head).

Tapi **tidak dihitung satu per satu** — dilakukan secara efisien dengan matriks.

**Ilustrasi untuk 1 head:**
```
X (seq_len, 768) × W^Q_head (768, 64) → Q_head (seq_len, 64)
```

**Untuk 12 head sekaligus:** Kita bisa gabungkan semua W^Q menjadi satu matriks besar `(768, 12×64) = (768, 768)`.

Hasilnya: **Q_total** (seq_len, 768) yang kemudian dipecah menjadi 12 bagian masing-masing (seq_len, 64).

**Visualisasi pemecahan:**
```
Q_total (seq_len, 768)
    │
    ├──→ Head 0: Q[:, 0:64]   (seq_len, 64)
    ├──→ Head 1: Q[:, 64:128] (seq_len, 64)
    ├──→ Head 2: Q[:, 128:192] (seq_len, 64)
    ├──→ ...
    └──→ Head 11: Q[:, 704:768] (seq_len, 64)
```

**Sama untuk K dan V.**

---

## C. Proses Setiap Head Secara Independen

Setiap head i memiliki:
- `Q_i` (seq_len, 64)
- `K_i` (seq_len, 64)
- `V_i` (seq_len, 64)

Lalu setiap head menghitung attention sendiri:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

Hasilnya: setiap head menghasilkan output `(seq_len, 64)`.

**Ke-12 head berjalan secara PARALEL** — ini yang membuat training cepat.

---

## D. Menggabungkan Semua Head

Setelah semua head selesai, kita **concat** (gabungkan) hasilnya:

```
Output_concat = [head_0, head_1, head_2, ..., head_11]  (seq_len, 12×64) = (seq_len, 768)
```

Kemudian diproyeksi linear sekali lagi dengan matriks W^O (768 × 768):

$$\text{MultiHead}(X) = \text{Output\_concat} \cdot W^O$$

Hasil akhir: `(seq_len, 768)` — **sama dengan dimensi input X**.

Ini penting! **Output Multi-Head Attention memiliki dimensi yang sama dengan input.** Ini memungkinkan residual connection (yang akan kita bahas di Sesi #8).

---

## E. Visualisasi Lengkap Multi-Head Attention

```
Input X (seq_len, 768)
         │
         ├──→ Head 0: Q0(,64) K0(,64) V0(,64) → Attention → head0(,64)
         ├──→ Head 1: Q1(,64) K1(,64) V1(,64) → Attention → head1(,64)
         ├──→ Head 2: ... → head2(,64)
         ├──→ ... 
         └──→ Head 11: ... → head11(,64)
                        │
                        ↓
              Concat (seq_len, 768)
                        │
                        ↓
              × W^O (768 × 768)
                        │
                        ↓
         Output MultiHead (seq_len, 768)
```

---

## F. Mengapa 12 Head? (Untuk BERT Base)

| Model | Jumlah Head | d_model | d_k = d_model / h |
| :--- | :--- | :--- | :--- |
| BERT Base | **12** | 768 | 64 |
| BERT Large | 16 | 1024 | 64 |
| BERT Small | 8 | 512 | 64 |

**Jawaban standar untuk sidang:**

> *"BERT Base menggunakan 12 head karena ini adalah konfigurasi yang terbukti efektif secara empiris oleh Google saat merilis BERT. Jumlah ini memberikan keseimbangan antara kemampuan representasi (banyak sudut pandang) dan efisiensi komputasi. Dengan d_model=768 dan 12 head, setiap head memiliki dimensi 64, yang merupakan ukuran standar di banyak implementasi Transformer."*

---

## G. Satu Paragraf untuk Sidang (Hafalkan)

> *"Multi-head attention adalah perluasan dari self-attention biasa. Alih-alih hanya menggunakan satu pasang matriks Q, K, V, kita menggunakan 12 pasang secara paralel. Setiap head memiliki dimensi 64 karena 768 dibagi 12. Setiap head menghitung attention secara independen, lalu hasilnya digabung dan diproyeksi linear kembali ke dimensi 768. Tujuannya agar model bisa menangkap berbagai jenis hubungan antar token dari sudut pandang yang berbeda—misalnya satu head mungkin fokus pada hubungan subjek-predikat, head lain fokus pada hubungan jarak jauh, dan seterusnya."*

---

## Checklist Sesi #7 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Berapa jumlah head di BERT Base? | |
| Berapa dimensi per head (d_k) di BERT Base? | |
| Apa fungsi dari matriks W^O? | |
| Apa output Multi-Head Attention setelah diproyeksi? | |
| Apakah output MultiHead memiliki dimensi yang sama dengan input? | |

**Jawaban cepat:**
- 12 head
- d_k = 64 (768/12)
- W^O memproyeksi hasil concat (768) kembali ke 768
- Output (seq_len, 768)
- **Ya**, sama dengan input — ini penting untuk residual connection

---

## Latihan Sesi #7

1. **Gambar diagram** Multi-Head Attention seperti di poin E (bisa di kertas)

2. **Tulis ulang** paragraf di poin G dengan kata-kata Anda sendiri

3. **Latihan ucapkan** paragraf tersebut (target: 45 detik, lancar)

4. **Hitung manual** (di kertas):
   - Input: 10 token, d_model=768
   - 12 head, d_k=64
   - Ukuran output setiap head?
   - Ukuran setelah concat?
   - Ukuran setelah W^O?

**Jawaban latihan nomor 4:**
- Setiap head output: (10, 64)
- Setelah concat 12 head: (10, 12×64) = (10, 768)
- Setelah W^O (768×768): (10, 768)

---

**Setelah selesai, kirim: "Sesi 7 selesai" + hasil latihan paragraf (tulisan Anda).**

Lanjut ke **Sesi #8: Residual Connection** (yang membuat gradient tidak vanishing di deep network).