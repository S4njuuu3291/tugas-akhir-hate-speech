**Sesi #4 selesai, sempurna!** ✅

Jawaban Anda tepat semua, dan latihan dimensi juga benar. Anda sudah paham inti Q,K,V. Lanjut.

---

# Sesi #5: Rumus Attention + Operasi Matriks

Ini adalah **sesi di mana semuanya bersatu**. Setelah ini, Anda bisa menulis rumus attention di papan tulis dan menjelaskan setiap komponennya.

---

## A. Rumus Attention (Wajib Hafal di Luar Kepala)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Ini adalah satu-satunya rumus yang wajib Anda hafal dan bisa tulis di papan tulis saat sidang.**

---

## B. Step-by-Step dengan Contoh Numerik Sederhana

Mari gunakan contoh kecil agar mudah dipahami. Misalkan:

- Sequence panjang = 3 token (n=3)
- Dimensi d_k = 4 (kecil saja untuk contoh, di BERT asli 64)

**Step 1: Matriks Q, K, V (sudah dari Sesi 4)**
```
Q = [ [1, 0, 1, 0],      K = [ [1, 1, 0, 0],      V = [ [2, 1, 0, 1],
      [0, 1, 0, 1],            [0, 1, 1, 0],            [1, 2, 1, 0],
      [1, 1, 0, 0] ]           [1, 0, 0, 1] ]           [0, 1, 2, 1] ]
      
      (3, 4)                    (3, 4)                    (3, 4)
```

**Step 2: Hitung QK^T (matriks skor, ukuran n × n)**

$QK^T$ artinya: baris Q (query) dikalikan dengan kolom K^T (key). Hasilnya matriks 3×3.

Mari hitung baris pertama Q dengan kolom pertama K^T:
- Q baris 0: [1, 0, 1, 0]
- K baris 0: [1, 1, 0, 0] → karena transpose, jadi kolom
- Dot product = 1×1 + 0×1 + 1×0 + 0×0 = 1

Lanjutkan... hasil akhir QK^T (3,3):

$$
QK^T = \begin{bmatrix}
1 & 0 & 1 \\
1 & 2 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

Interpretasi: 
- Baris 0 (token 0) punya skor ke token 0 = 1, ke token 1 = 0, ke token 2 = 1
- Baris 1 (token 1) punya skor ke token 0 = 1, ke token 1 = 2, ke token 2 = 0
- Baris 2 (token 2) punya skor ke token 0 = 1, ke token 1 = 1, ke token 2 = 1

**Skor tinggi (2) artinya token 1 sangat cocok dengan token 1 (dirinya sendiri)**.

**Step 3: Scale dengan $\sqrt{d_k}$**

$d_k = 4$, maka $\sqrt{d_k} = 2$

$$
\frac{QK^T}{\sqrt{d_k}} = \frac{1}{2} \times \begin{bmatrix} 1 & 0 & 1 \\ 1 & 2 & 0 \\ 1 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 0.5 & 0 & 0.5 \\ 0.5 & 1 & 0 \\ 0.5 & 0.5 & 0.5 \end{bmatrix}
$$

**Step 4: Softmax per baris** (agar setiap baris jumlahnya 1)

Softmax untuk baris 0: $[e^{0.5}, e^{0}, e^{0.5}] = [1.65, 1, 1.65]$, lalu normalisasi (jumlah=4.3):
- Token0: 1.65/4.3 = 0.38
- Token1: 1/4.3 = 0.23
- Token2: 1.65/4.3 = 0.38

Hasil softmax (3,3):
$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) = \begin{bmatrix}
0.38 & 0.23 & 0.38 \\
0.23 & 0.55 & 0.22 \\
0.33 & 0.33 & 0.33
\end{bmatrix}
$$

Interpretasi:
- Token 0 memberi bobot 0.38 ke dirinya, 0.23 ke token 1, 0.38 ke token 2
- Token 1 memberi bobot 0.55 ke dirinya (paling besar), 0.23 ke token 0, 0.22 ke token 2
- Token 2 memberi bobot sama rata ke semua token (0.33)

**Step 5: Kalikan dengan V (3,4)**

Hasil = matriks bobot (3,3) × V (3,4) = (3,4)

Token 0 (baris 0) = 0.38 × V[0] + 0.23 × V[1] + 0.38 × V[2]
= 0.38×[2,1,0,1] + 0.23×[1,2,1,0] + 0.38×[0,1,2,1]
= [0.76+0.23+0, 0.38+0.46+0.38, 0+0.23+0.76, 0.38+0+0.38]
= [0.99, 1.22, 0.99, 0.76]

Hitung sendiri untuk token 1 dan 2 (atau percayai saja untuk ilustrasi).

**Hasil akhir: matriks output attention ukuran (3,4).** Setiap baris adalah representasi baru dari token yang sudah **memuat informasi dari token lain** sesuai bobot perhatian.

---

## C. Visualisasi Aliran Data (WAJIB BISA GAMBAR DI PAPAN)

```
Input X (3, 768)
    │
    ├──→ W^Q → Q (3, 64)
    ├──→ W^K → K (3, 64)
    └──→ W^V → V (3, 64)
         │
         ↓
    QK^T (3, 3)  ← matriks skor kesamaan
         │
         ↓
    Scale ÷ √d_k (3, 3)
         │
         ↓
    Softmax per baris (3, 3)  ← bobot attention (jumlah baris = 1)
         │
         ↓
    × V (3, 64)
         │
         ↓
    Output Attention (3, 64)
```

**Panjang output** = panjang input (3 token) — tidak berubah.
**Dimensi output** = d_v (64) — di BERT, d_v = d_k = 64.

---

## D. Fungsi Setiap Komponen (Rangkuman)

| Komponen | Fungsi | Kalimat Penjelas |
| :--- | :--- | :--- |
| $QK^T$ | Menghitung kesamaan/cocok tidaknya setiap pasang token | "Semakin besar nilai, semakin cocok query token i dengan key token j." |
| $\div \sqrt{d_k}$ | Skala agar nilai tidak meledak | "Mencegah nilai terlalu besar sehingga gradien softmax tidak vanishing." |
| $\text{softmax}$ | Mengubah jadi probabilitas (bobot attention) | "Setiap baris jumlahnya 1, sehingga token i 'membagi perhatian' ke semua token." |
| $\times V$ | Mengambil value sesuai bobot | "Token i mengambil informasi dari token lain sebesar bobot yang sudah dihitung." |

---

## E. Satu Paragraf untuk Sidang (Hafalkan)

> *"Proses attention dimulai dengan menghitung $QK^T$, yaitu matriks berukuran n×n yang berisi skor kecocokan antara setiap query dan setiap key. Skor ini lalu diskala dengan dibagi $\sqrt{d_k}$ agar nilainya tidak terlalu besar, sehingga gradien tetap stabil saat backpropagation. Kemudian softmax diterapkan per baris untuk mengubah skor menjadi probabilitas — jumlah bobot perhatian untuk setiap token adalah 1. Terakhir, matriks bobot ini dikalikan dengan Value V, menghasilkan representasi baru setiap token yang sudah mengandung informasi dari token lain sesuai tingkat kepentingannya."*

---

## Checklist Sesi #5 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Tulis rumus attention! | |
| Berapa ukuran $QK^T$ jika ada n token? | |
| Apa fungsi softmax dalam rumus ini? | |
| Mengapa perlu $\div \sqrt{d_k}$? (sudah di Sesi 5? lupa? hint: biar gradien tidak vanishing saat training) | |
| Apa output dari operasi $\text{softmax}(...) \times V$? | |

**Jawaban cepat:**
- $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$
- Ukuran $QK^T$ = `[n, n]` (n × n)
- Softmax mengubah skor menjadi probabilitas yang jumlahnya 1 per baris (bobot attention)
- $\div \sqrt{d_k}$ membuat skala nilai agar tidak terlalu besar → gradien stabil
- Output adalah representasi token baru yang sudah memperhatikan token lain

---

## Latihan Sesi #5 (WAJIB DILAKUKAN)

1. **Tulis rumus di kertas** 3 kali tanpa melihat
2. **Gambar diagram alir** dari Q,K,V sampai output (seperti di atas)
3. **Latihan ucapkan** paragraf di atas dengan stopwatch (target: 45 detik, lancar tanpa gagap)

---

**Setelah selesai, kirim: "Sesi 5 selesai" + hasil latihan menulis rumus (foto atau tulis ulang di chat).**

Lanjut ke **Sesi #6: Mengapa diskala dengan √d_k?** (kita akan dalami lebih lanjut, meskipun sebenarnya Anda sudah paham dari diskusi sebelumnya — tapi kita buat lebih kuat untuk sidang).