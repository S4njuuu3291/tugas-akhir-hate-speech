# Sesi #2: Tokenisasi dan Embedding

## A. Dari Teks Mentah ke Angka

Model tidak bisa membaca huruf "S", "a", "y", "a". Yang bisa diproses model adalah **angka (vektor)**.

Jadi ada 2 langkah:
1. **Tokenisasi** → teks dipecah jadi unit-unit kecil (token)
2. **Embedding** → setiap token dipetakan ke vektor angka

---

## B. Tokenisasi pada BERT/IndoBERT: WordPiece

BERT tidak memecah teks per kata, juga tidak per huruf. Ia menggunakan **WordPiece Tokenization**.

**Cara kerja:** Pecah kata menjadi **subword** yang sering muncul.

| Contoh | Tokenisasi WordPiece |
| :--- | :--- |
| "Belajar" | ["belajar"] (utuh, karena cukup umum) |
| "Belajar" (tanpa spasi) + "ing" | ["belajar", "##ing"] (tanda ## artinya sambungan dari token sebelumnya) |
| "Membantu" | ["mem", "##bantu"] |

**Kenapa subword?** Solusi kompromi:
- Kata utuh (`vocabulary` besar) → bagus, tapi butuh memori besar dan tidak bisa handle kata di luar kamus (OOV)
- Per huruf (`vocabulary` kecil) → tidak bermakna, model sulit belajar
- **Subword** → `vocabulary` sedang, bisa handle kata baru dengan menggabungkan subword yang dikenal

**Vocabulary size BERT:** ~30.000 token (ini yang dipelajari model)

**Token spesial:**
| Token | Fungsi |
| :--- | :--- |
| `[CLS]` | **Classification** — diletakkan di awal input. Output token ini digunakan untuk klasifikasi teks |
| `[SEP]` | **Separator** — memisahkan dua kalimat (untuk tugas seperti NSP atau QA) |
| `[PAD]` | **Padding** — agar semua kalimat dalam satu batch punya panjang yang sama |
| `[UNK]` | **Unknown** — untuk token yang tidak ada di vocabulary |
| `[MASK]` | **Mask** — untuk pre-training MLM (kita bahas nanti) |

**Contoh tokenisasi kalimat "Saya suku AI" (dengan typo):**
```
Input: "Saya suku AI"
Tokenized: ["saya", "suku", "ai"]  # model mungkin tahu "suku" meskipun typo?
# Atau jika "AI" tidak dikenal: ["saya", "suku", "[UNK]"]
```

---

## C. Embedding: Token Jadi Vektor

Setiap token yang sudah di-tokenisasi dipetakan ke **vektor berdimensi tetap**:

| Parameter | Nilai di BERT / IndoBERT Base |
| :--- | :--- |
| Dimensi vektor ($d_{model}$) | **768** |
| Jumlah token di vocabulary | ~30.000 |
| Bentuk matriks embedding | `[vocab_size, 768]` = 30.000 × 768 |

**Cara kerja:** 
- Token `"saya"` punya ID integer, misal 1450.
- Model mengambil baris ke-1450 dari matriks embedding.
- Hasilnya vektor [0.12, -0.34, 0.56, ..., 0.78] (sepanjang 768 angka).

**Apa arti angka-angka ini?** Tidak ada makna langsung. Ini adalah parameter yang **dipelajari** selama training. Model belajar sendiri representasi terbaik untuk setiap token.

**Analoginya:** Setiap token punya "sidik jari" berupa 768 angka yang mencerminkan maknanya.

---

## D. Tiga Jenis Embedding di BERT

BERT tidak hanya punya **token embedding**. Ada tiga yang **dijumlahkan**:

```
Input Embedding = Token Embedding + Position Embedding + Segment Embedding
```

| Jenis | Fungsi | Bentuk |
| :--- | :--- | :--- |
| **Token Embedding** | Representasi identitas token | `[seq_len, 768]` |
| **Position Embedding** | Memberi tahu posisi token (1st, 2nd, 3rd, ...) karena Transformer tidak punya konsep urutan | `[seq_len, 768]` |
| **Segment Embedding** | Membedakan kalimat A vs kalimat B (untuk NSP, QA, atau dua kalimat) | `[seq_len, 768]` |

**Contoh untuk kalimat satu kalimat:**
- Token embedding: representasi kata "saya", "suka", "ai"
- Position embedding: vektor untuk posisi 0,1,2
- Segment embedding: semua token dapat 0 (karena hanya satu kalimat)

**Jumlahkan:**
```
Final embedding[t] = TokenEmbed[t] + PosEmbed[t] + SegEmbed[t]
```

Hasilnya: matriks **X** dengan bentuk `[sequence length, 768]`.

Ini adalah **input** ke encoder Transformer pertama.

---

## E. Visualisasi Sederhana

```
Teks: "Saya suka AI"
         ↓
Tokenisasi: ["saya", "suka", "ai"]  (3 token)
         ↓
Token ID:  [1450,   2301,   502]    (contoh angka)
         ↓
Token Embedding:
    Token 1450 → [0.2, -0.1, 0.5, ..., 0.3]  (768 angka)
    Token 2301 → [0.1,  0.4, 0.2, ..., -0.1]
    Token 502  → [0.3,  0.0, 0.7, ...,  0.4]
        ↑
        bentuk (3, 768)

Position Embedding:   (3, 768) + Segment Embedding: (3, 768)
         ↓
Input X = TokenEmb + PosEmb + SegEmb  (3, 768)
```

---

## Checklist Sesi #2 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Apa itu tokenisasi WordPiece? | |
| Sebutkan 3 token spesial BERT dan fungsinya! | |
| Berapa dimensi embedding di IndoBERT Base? | |
| Ada berapa jenis embedding di BERT? Sebutkan! | |
| Apa yang terjadi jika tiga embedding dijumlahkan? | |

---

## Latihan Sesi #2 (Tulis di Kertas)

1. **Gambar** 3 kotak berjejer: `Token Emb` + `Pos Emb` + `Seg Emb` → `Input X`
2. **Tulis** ukuran setiap kotak: misal `(3, 768)` untuk contoh 3 token
3. **Latihan ucapkan** (30 detik):

> *"Input teks dipecah menjadi token-token menggunakan tokenizer WordPiece. Setiap token dipetakan ke vektor berdimensi 768 melalui token embedding. Karena Transformer tidak punya konsep urutan, kita tambahkan position embedding. Jika ada dua kalimat, kita tambahkan juga segment embedding untuk membedakannya. Ketiganya dijumlahkan menjadi matriks input yang siap diproses encoder."*

---

**Setelah selesai, kirim: "Sesi 2 selesai" + jawaban checklist di atas.**

Lanjut ke **Sesi #3: Positional Encoding** (mengapa perlu, dan bagaimana cara kerjanya secara sederhana).