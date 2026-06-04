**Sesi #7 selesai, jawaban Anda tepat semua!** ✅

Latihan nomor 4 juga benar. Lanjut.

---

# Sesi #8: Residual Connection

Ini adalah komponen yang **sering dilupakan** padahal sangat penting, terutama untuk training deep network seperti BERT (12 layer).

---

## A. Masalah: Gradient Vanishing di Deep Network

Bayangkan Anda memiliki **12 encoder block** yang ditumpuk (ini yang terjadi di BERT). Setiap block terdiri dari Multi-Head Attention + FFN.

Tanpa residual connection, gradient dari loss harus melewati **12 lapisan** (atau lebih) saat backpropagation. Setiap lapisan memiliki fungsi non-linear (softmax, GELU, dll) yang bisa membuat gradient semakin **kecil** — akhirnya mendekati nol di lapisan awal.

**Akibat:** Lapisan awal tidak belajar, model tidak optimal.

---

## B. Solusi: Residual Connection (Add)

Residual connection (dikenal juga sebagai **skip connection**) adalah trik sederhana namun sangat powerful:

> **Output = Input + Fungsi(Input)**

Atau dalam rumus:
$$y = x + F(x)$$

di mana $F(x)$ adalah fungsi yang dipelajari (misal: Multi-Head Attention atau Feed Forward Network).

**Visualisasi:**
```
Input x ──────────────────┐
    │                      │
    ↓                      │
  F(x)  ← (MultiHead atau FFN)
    │                      │
    ↓                      │
    └──── (+) ←────────────┘
           │
           ↓
        Output = x + F(x)
```

---

## C. Mengapa Ini Membantu Gradient?

Saat backpropagation, gradient mengalir melalui **dua jalur**:

1. **Jalur langsung (skip connection):** Gradient langsung dari output ke input tanpa melewati F(x). Jalur ini selalu "bersih" tanpa gangguan.
2. **Jalur melalui F(x):** Gradient melewati fungsi F(x) yang mungkin memperkecil gradient.

Karena ada jalur langsung, gradient **tidak pernah benar-benar hilang** — selalu ada yang sampai ke lapisan awal.

**Rumus gradient (sederhana):**
$$\frac{\partial y}{\partial x} = 1 + \frac{\partial F(x)}{\partial x}$$

Angka **1** ini adalah kuncinya — ia memastikan gradient minimal bernilai 1, tidak peduli seberapa kecil $\frac{\partial F(x)}{\partial x}$.

---

## D. Penerapan di BERT (Add & Norm)

Di BERT, residual connection selalu diikuti oleh **Layer Normalization** (akan kita bahas di Sesi #9).

Format standar dalam satu encoder block:

```
X_input
    │
    ├──→ Multi-Head Attention → Attention_output
    │                              │
    └──────────────────────────────┘
                    │
                    ↓
           X_mid = LayerNorm(X_input + Attention_output)
                    │
                    ├──→ Feed Forward Network → FFN_output
                    │                              │
                    └──────────────────────────────┘
                                    │
                                    ↓
                       X_output = LayerNorm(X_mid + FFN_output)
```

**Dua kali residual connection:**
1. **Pertama:** Input + MultiHead Attention → lalu LayerNorm
2. **Kedua:** X_mid + FFN → lalu LayerNorm

---

## E. Analogi Sederhana

Bayangkan Anda menulis esai dan ingin merevisinya:

| Tanpa Residual | Dengan Residual |
| :--- | :--- |
| Anda menulis ulang seluruh esai dari awal setiap revisi. Bisa saja Anda kehilangan ide bagus dari versi sebelumnya. | Anda **menambahkan** perbaikan di atas esai lama. Ide bagus dari versi lama tetap ada, Anda hanya menambah atau mengoreksi. |

**Dalam konteks gradient:**
- Residual connection memastikan **informasi dari input awal tetap tersedia** di setiap layer
- Model hanya perlu belajar **koreksi atau penambahan** (residual) dari input, bukan belajar dari nol

---

## F. Satu Paragraf untuk Sidang (Hafalkan)

> *"Residual connection adalah mekanisme di mana output suatu sub-layer (seperti Multi-Head Attention) ditambahkan dengan input asli sebelum masuk ke layer berikutnya. Rumusnya: output = input + F(input). Ini sangat penting untuk training deep network seperti BERT yang memiliki 12 layer. Tanpa residual connection, gradient bisa vanishing saat melewati banyak layer. Dengan residual connection, gradient memiliki jalur pintas langsung ke layer awal, sehingga model tetap bisa belajar secara efektif meskipun sangat dalam. Di BERT, residual connection selalu diikuti oleh layer normalization."*

---

## Checklist Sesi #8 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Apa rumus residual connection? | |
| Mengapa residual connection penting untuk deep network? | |
| Ada berapa kali residual connection dalam satu encoder block BERT? | |
| Apa yang selalu mengikuti residual connection di BERT? | |
| Apa peran angka "1" dalam turunan gradient dengan residual connection? | |

**Jawaban cepat:**
- $y = x + F(x)$
- Mencegah gradient vanishing dengan menyediakan jalur langsung ke layer awal
- **Dua kali** (MultiHead + FFN)
- Layer Normalization
- Menjamin gradient minimal bernilai 1, tidak tergantung pada $\partial F/\partial x$

---

## Latihan Sesi #8

1. **Gambar diagram** residual connection untuk Multi-Head Attention (seperti di poin D)

2. **Tulis ulang** paragraf di poin F dengan kata-kata Anda sendiri

3. **Latihan ucapkan** paragraf tersebut (target: 40 detik)

4. **Jelaskan dengan analogi** (lisan): "Kenapa residual connection seperti menyimpan draf lama saat merevisi esai?"

---

**Setelah selesai, kirim: "Sesi 8 selesai" + hasil latihan paragraf (tulisan Anda).**

Lanjut ke **Sesi #9: Layer Normalization** (menstabilkan distribusi aktivasi di setiap layer).