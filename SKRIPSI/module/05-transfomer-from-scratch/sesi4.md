**Sesi #3 selesai, lanjut ke Sesi #4!** 💪

---

# Sesi #4: Konsep Query, Key, Value

Ini adalah **sesi paling penting** dari seluruh rangkaian. Jika Anda paham ini, Anda sudah paham 60% inti Transformer.

---

## A. Analogi Pencarian di Perpustakaan

Bayangkan Anda ingin mencari buku tentang "Artificial Intelligence" di perpustakaan.

| Konsep | Analogi Perpustakaan | Dalam Transformer |
| :--- | :--- | :--- |
| **Query** | Pertanyaan Anda: "Buku tentang AI?" | Vektor yang mewakili **apa yang sedang dicari** oleh satu token |
| **Key** | Label di punggung buku: "AI", "Machine Learning", "Cooking" | Vektor yang mewakili **identitas atau label** dari setiap token |
| **Value** | Isi buku yang sebenarnya | Vektor yang mewakili **informasi aktual** dari token |

**Prosesnya:**
1. Anda (Query) mencari buku dengan mencocokkan pertanyaan Anda dengan label (Key) setiap buku.
2. Buku dengan label yang paling cocok dengan pertanyaan Anda akan diberi perhatian lebih.
3. Anda membaca isi (Value) dari buku-buku yang relevan tersebut.

Dalam Transformer:
> Setiap token punya **Query** (apa yang saya cari), **Key** (ini identitas saya), dan **Value** (ini informasi saya). Query token A akan "cocok" dengan Key token B jika token B relevan untuk token A. Lalu Value token B akan diambil sesuai tingkat kecocokan tersebut.

---

## B. Dari Input ke Q, K, V (Operasi Linear)

Input kita adalah matriks **X** dengan bentuk `[seq_len, d_model]` (contoh: 3 token × 768).

Untuk mendapatkan Q, K, V, kita kalikan X dengan **tiga matriks bobot yang berbeda**:

```
Q = X · W^Q    (W^Q ukuran: d_model × d_k)
K = X · W^K    (W^K ukuran: d_model × d_k)
V = X · W^V    (W^V ukuran: d_model × d_v)
```

| Parameter | Nilai di BERT Base | Keterangan |
| :--- | :--- | :--- |
| `d_model` | 768 | Dimensi input/output |
| `d_k` | 64 | Dimensi Query dan Key (karena 12 head, 768/12 = 64) |
| `d_v` | 64 | Dimensi Value (sama dengan d_k di BERT) |

**Visualisasi untuk 3 token:**

```
X (3, 768)
    │
    ├───· W^Q (768 × 64) ───→ Q (3, 64)
    ├───· W^K (768 × 64) ───→ K (3, 64)
    └───· W^V (768 × 64) ───→ V (3, 64)
```

**Penting:** W^Q, W^K, W^V adalah **parameter yang dipelajari** selama training. Model belajar sendiri proyeksi mana yang paling berguna.

---

## C. Fungsi Query, Key, Value dalam Satu Kalimat

| Vektor | Peran | Kalimat Penjelas |
| :--- | :--- | :--- |
| **Query** | Pencari | "Token ini **sedang mencari** token lain yang relevan dengan dirinya." |
| **Key** | Identitas | "Token ini **menyediakan label** untuk dicocokkan dengan Query token lain." |
| **Value** | Informasi | "Token ini **menyimpan informasi** yang akan diambil jika Key-nya cocok dengan Query." |

**Contoh konkret untuk kalimat "Saya makan nasi":**

Misalkan token "makan" sedang memproses:
- **Query dari "makan"** : "Saya mencari subjek (siapa yang makan?) dan objek (apa yang dimakan?)"
- **Key dari "Saya"** : "Saya adalah entitas bernama Saya, bisa sebagai subjek"
- **Key dari "nasi"** : "Saya adalah benda bernama nasi, bisa sebagai objek"
- **Key dari "makan"** : "Saya adalah kata kerja"

Jika Query "makan" cocok dengan Key "Saya" dan Key "nasi", maka Value "Saya" dan Value "nasi" akan diambil untuk memperkaya representasi token "makan".

---

## D. Mengapa Perlu Tiga Vektor Berbeda?

| Kalau hanya satu vektor... | Dengan tiga vektor yang terpisah |
| :--- | :--- |
| Token hanya punya satu representasi untuk segala keperluan. | Token bisa punya **peran berbeda** saat dia menjadi "pencari" (Query), saat dia "diperhatikan" (Key), dan saat dia "memberikan informasi" (Value). |
| Contoh: Token "Jakarta" harus punya representasi yang sama untuk semua konteks. | Contoh: Query "Jakarta" mungkin mencari kota lain yang berhubungan. Key "Jakarta" memberi tahu "saya adalah nama ibu kota". Value "Jakarta" berisi informasi lengkap tentang Jakarta. |

**Intuisi:** Dengan memisahkan Q, K, V, model bisa belajar **peran yang berbeda** dari token yang sama tergantung konteksnya.

---

## E. Ringkasan Sesi #4 (30 Detik untuk Sidang)

> *"Setiap token diubah menjadi tiga vektor melalui perkalian dengan matriks bobot yang dipelajari: Query (apa yang dicari), Key (identitas token), dan Value (informasi token). Query dari satu token akan dibandingkan dengan Key dari semua token untuk menentukan seberapa relevan token lain terhadap token tersebut. Nilai relevansi ini lalu digunakan untuk mengambil Value dari token-token yang relevan."*

---

## Checklist Sesi #4 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Apa kepanjangan Q, K, V? | |
| Bagaimana cara mendapatkan Q, K, V dari input X? | |
| Berapa dimensi Q, K, V di BERT Base? | |
| Mengapa perlu tiga vektor yang berbeda, tidak cukup satu saja? | |
| Dalam analogi perpustakaan, apa itu Query, Key, Value? | |

**Jawaban cepat:**
- **Q** = Query, **K** = Key, **V** = Value
- Dari input X dikalikan dengan matriks bobot W^Q, W^K, W^V (dipelajari)
- `d_k = d_v = 64` (karena 768/12 head)
- Agar token bisa punya peran berbeda saat menjadi pencari, saat diperhatikan, dan saat memberi informasi
- Query = pertanyaan Anda, Key = label buku, Value = isi buku

---

## Latihan Sesi #4 (Tulis dan Ucapkan)

1. **Gambar** diagram:

```
X (3, 768)
    │
    ├──── W^Q (768×64) ────→ Q (3, 64)
    ├──── W^K (768×64) ────→ K (3, 64)
    └──── W^V (768×64) ────→ V (3, 64)
```

2. **Tulis** di kertas: dimensi X, W^Q, Q untuk contoh 3 token, d_model=768, d_k=64

3. **Latihan ucapkan** (30 detik):

> *"Dari matriks input X, kita hitung tiga matriks: Query, Key, dan Value. Caranya dengan mengalikan X dengan matriks bobot W^Q, W^K, W^V yang dipelajari selama training. Query dan Key berdimensi 64, Value juga 64. Query berfungsi sebagai 'pencari', Key sebagai 'label', dan Value sebagai 'informasi' yang akan diambil."*

---

**Setelah selesai, kirim: "Sesi 4 selesai" + jawaban checklist.**

Lanjut ke **Sesi #5: Rumus Attention + Operasi Matriks** (tempat Q, K, V benar-benar "bekerja")