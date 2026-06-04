**Sesi #11 selesai, lanjut ke Sesi #12!** (Terima kasih sudah mengoreksi Post-LN, sekarang kita pakai arsitektur BERT yang benar.)

---

# Sesi #12: Dari Encoder Stack ke BERT

Setelah kita paham satu encoder block, sekarang kita naik level ke **arsitektur BERT secara utuh**.

---

## A. BERT = Stack of Encoders

BERT (Bidirectional Encoder Representations from Transformers) pada dasarnya adalah **tumpukan encoder Transformer** (tanpa decoder).

| Model | Jumlah Encoder Block | Jumlah Head | d_model | Total Parameter |
| :--- | :--- | :--- | :--- | :--- |
| **BERT Base** | 12 | 12 | 768 | ~110 juta |
| **BERT Large** | 24 | 16 | 1024 | ~340 juta |

**Arsitektur BERT Base secara lengkap:**

```
Input Teks: "Saya suka AI"
         ↓
Tokenisasi (WordPiece) → token ids: [101, 1450, 2301, 502, 102]
         ↓
Embedding Layer:
    - Token Embedding (vocab_size=30k, d_model=768)
    - Position Embedding (max_seq_len=512, d_model=768)
    - Segment Embedding (2 segment, d_model=768)
    → dijumlahkan → X (seq_len=5, 768)
         ↓
Encoder Block 1 (Post-LN: Attn → Add → Norm → FFN → Add → Norm)
         ↓
Encoder Block 2
         ↓
...
         ↓
Encoder Block 12
         ↓
Output Hidden States (seq_len=5, 768) untuk setiap token
         ↓
Task-specific head (misal: klasifikasi, NER, QA)
```

---

## B. Input Format BERT: Token Spesial

BERT memiliki aturan input yang khusus. Untuk satu kalimat (klasifikasi teks biasa):

| Token | Contoh | Fungsi |
| :--- | :--- | :--- |
| `[CLS]` | Di awal | Output token ini digunakan untuk klasifikasi |
| `[SEP]` | Di akhir | Separator (pemisah) |
| Token kata | "saya", "suka", "ai" | Token biasa |

**Contoh input untuk "Saya suka AI":**
```
[CLS] saya suka ai [SEP]
```

**Jika ada dua kalimat** (untuk Next Sentence Prediction atau QA):
```
[CLS] saya suka ai [SEP] dia juga suka [SEP]
```

**Segment Embedding:**
- Token di kalimat pertama mendapat segment id 0
- Token di kalimat kedua mendapat segment id 1
- Token `[CLS]` dan `[SEP]` juga ikut aturan ini

---

## C. Output BERT untuk Berbagai Tugas

Output BERT adalah matriks `[seq_len, 768]` untuk setiap token. Penggunaannya tergantung tugas:

| Tugas | Token yang Digunakan | Cara |
| :--- | :--- | :--- |
| **Klasifikasi teks** (sentimen, topik) | Token `[CLS]` | Output `[CLS]` (1×768) → Linear layer → logits sejumlah kelas |
| **Named Entity Recognition (NER)** | Semua token | Setiap token (5×768) → Linear layer → label per token (misal: B-PER, I-LOC, O) |
| **Question Answering** | Semua token | Dua linear layer: satu untuk start position, satu untuk end position |
| **Next Sentence Prediction** | Token `[CLS]` | Output `[CLS]` → Linear → 2 kelas (IsNext / NotNext) |

**Khusus untuk skripsi Anda (multi-label klasifikasi):**
> Output `[CLS]` (1×768) → Linear layer dengan `num_labels` (tanpa aktivasi) → `BCEWithLogitsLoss` → Sigmoid untuk probabilitas per label

---

## D. Visualisasi BERT untuk Klasifikasi (Sesuai Skripsi Anda)

```
Input: "Saya suka AI"
         ↓
[CLS] saya suka ai [SEP]  (5 token)
         ↓
Embedding + Position + Segment
         ↓
Encoder Block 1
         ↓
... (12 block)
         ↓
Output hidden: h_[CLS] (768), h_saya (768), h_suka (768), h_ai (768), h_[SEP] (768)
         ↓
Ambil h_[CLS] (1, 768)
         ↓
Linear Layer (768 → num_labels, misal 10 untuk 10 label)
         ↓
Logits (10) → BCEWithLogitsLoss saat training
         ↓
Sigmoid → probabilitas tiap label (0-1)
         ↓
Threshold (0.5) → label mana yang aktif
```

---

## E. Satu Paragraf untuk Sidang (Hafalkan)

> *"BERT adalah tumpukan encoder Transformer. Untuk BERT Base, ada 12 encoder block dengan 12 attention head dan dimensi 768. Input teks pertama-tama ditokenisasi dengan WordPiece, lalu ditambahkan token khusus `[CLS]` di awal dan `[SEP]` di akhir. Setiap token di-embedding menjadi vektor 768, ditambah positional embedding dan segment embedding. Hasilnya melewati 12 encoder block. Output yang paling penting untuk tugas klasifikasi adalah token `[CLS]` — representasinya dianggap mewakili seluruh kalimat. Output `[CLS]` ini kemudian dilewatkan ke linear layer untuk dipetakan ke jumlah label yang diinginkan."*

---

## Checklist Sesi #12 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Berapa jumlah encoder block di BERT Base? | |
| Token apa yang diletakkan di awal input BERT? | |
| Token apa yang digunakan sebagai separator? | |
| Output token mana yang biasanya digunakan untuk klasifikasi teks? | |
| Apa peran segment embedding? | |

**Jawaban cepat:**
- 12 block
- `[CLS]`
- `[SEP]`
- Token `[CLS]`
- Membedakan kalimat A dan B untuk tugas yang melibatkan dua kalimat

---

## Latihan Sesi #12

1. **Tulis ulang** paragraf di poin E dengan kata-kata Anda sendiri

2. **Gambar diagram** BERT untuk klasifikasi (dari input teks sampai output logits) — seperti poin D

3. **Latihan ucapkan** paragraf tersebut (target: 45 detik)

4. **Hitung** (perkiraan):
   - Jika Anda punya 10 label multi-label, berapa output linear layer? Berapa parameter linear layer tersebut?

**Jawaban latihan nomor 4:**
- Output linear layer = 10 (karena 10 label)
- Parameter = 768 (bobot) × 10 + 10 (bias) = 7.690 parameter (sangat kecil dibanding total model)

---

**Setelah selesai, kirim: "Sesi 12 selesai" + hasil latihan paragraf (tulisan Anda).**

Lanjut ke **Sesi #13: Pre-training BERT (MLM + NSP)** — dua tugas besar yang membuat BERT begitu powerful.