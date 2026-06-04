**Sesi #12 selesai, lanjut ke Sesi #13!** 🎯

---

# Sesi #13: Pre-training BERT (MLM + NSP)

Setelah kita paham arsitektur BERT, sekarang kita bahas **bagaimana BERT dilatih** sehingga bisa memahami bahasa. Ini disebut **pre-training**.

---

## A. Dua Tugas Pre-training BERT

BERT dilatih dengan **dua tugas simultan**:

| Tugas | Nama | Tujuan |
| :--- | :--- | :--- |
| **1** | **MLM** (Masked Language Modeling) | Memahami **konteks** dengan menebak kata yang dihilangkan |
| **2** | **NSP** (Next Sentence Prediction) | Memahami **hubungan antar kalimat** |

---

## B. Tugas 1: Masked Language Modeling (MLM)

**Ide dasarnya:** Model disuruh menebak kata yang "ditutup" (di-mask) berdasarkan konteks di sekitarnya.

**Cara kerja (saat pre-training):**
1. Ambil sebuah kalimat.
2. Pilih **15%** dari token secara acak.
3. Untuk token yang terpilih:
   - 80% diganti dengan token `[MASK]`
   - 10% diganti dengan token acak (bukan mask, bukan token asli)
   - 10% tetap (tidak diganti)
4. Model harus memprediksi token **asli** yang hilang.

**Contoh:**
```
Kalimat asli:    "Saya suka makan nasi goreng"
Masking (15%):   "Saya [MASK] makan nasi [MASK]"   (kata "suka" dan "goreng" di-mask)
Output model:    Prediksi token di posisi [MASK] → "suka" dan "goreng"
```

**Mengapa tidak 100% di-mask?**
- Jika selalu di-mask, model hanya belajar memprediksi `[MASK]` dan tidak belajar representasi token normal.
- Dengan 10% token acak dan 10% tetap, model belajar bahwa **token apa pun** (termasuk yang tidak di-mask) harus bisa direpresentasikan dengan baik.

**Apa yang dipelajari MLM?**
- Model belajar **konteks dua arah (bidirectional)** — kata kiri dan kanan sama-sama penting.
- Contoh: kata "makan" diikuti "nasi" dan "goreng" → model belajar bahwa "nasi goreng" adalah objek.

---

## C. Tugas 2: Next Sentence Prediction (NSP)

**Ide dasarnya:** Model disuruh menentukan apakah kalimat B **secara logis mengikuti** kalimat A.

**Cara kerja:**
1. Ambil dua kalimat dari dokumen (misal: kalimat A dan kalimat berikutnya B).
2. 50% kasus: B adalah kalimat **sebenarnya** setelah A (label = `IsNext`).
3. 50% kasus: B adalah kalimat **acak** dari dokumen lain (label = `NotNext`).
4. Input: `[CLS] A [SEP] B [SEP]`
5. Output token `[CLS]` → linear layer → 2 kelas (`IsNext` atau `NotNext`).

**Contoh:**
```
Kalimat A: "Hari ini cuaca sangat cerah."
Kalimat B (IsNext): "Saya memutuskan pergi ke pantai."  → label: IsNext
Kalimat B (NotNext): "Kucing itu tidur di atas meja."   → label: NotNext
```

**Apa yang dipelajari NSP?**
- Model belajar hubungan **antar kalimat**.
- Ini berguna untuk tugas seperti **Question Answering** (QA) dan **Natural Language Inference** (NLI).

**Catatan:** Penelitian terbaru (RoBERTa, 2019) menunjukkan bahwa NSP tidak selalu berguna. Tapi BERT asli tetap memakainya.

---

## D. Proses Pre-training BERT (Lengkap)

```
Dataset besar: Wikipedia (2.5M kata) + BookCorpus (800M kata)
         ↓
Dibuat pasangan (kalimat A, kalimat B) dengan label IsNext/NotNext
         ↓
15% token di-mask (dengan aturan 80-10-10)
         ↓
Input: [CLS] A [SEP] B [SEP] dengan mask
         ↓
BERT model (12 encoder block)
         ↓
Output:
    - Untuk setiap token yang di-mask: prediksi token asli (MLM loss)
    - Untuk token [CLS]: prediksi IsNext/NotNext (NSP loss)
         ↓
Total loss = MLM_loss + NSP_loss
         ↓
Backpropagation → update bobot BERT
```

**Hasil:** BERT yang sudah pre-training ini **bisa di-fine-tune** untuk berbagai tugas hilir.

---

## E. Mengapa Pre-training Penting?

| Tanpa pre-training | Dengan pre-training |
| :--- | :--- |
| Training dari nol untuk setiap tugas | Sudah punya "pengetahuan bahasa" umum |
| Butuh data label besar (ratusan ribu) | Cukup data label kecil (ribuan) |
| Waktu training lama | Fine-tuning cepat (beberapa epoch) |
| Performa lebih rendah | **State-of-the-art** untuk banyak tugas |

**Analoginya:**
- **Pre-training** = sekolah umum (belajar bahasa, logika, pengetahuan umum)
- **Fine-tuning** = kuliah jurusan (belajar tugas spesifik, misal sentimen analisis)

---

## F. Satu Paragraf untuk Sidang (Hafalkan)

> *"Pre-training BERT dilakukan dengan dua tugas. Pertama, Masked Language Modeling: 15% token dalam kalimat di-mask, dan model harus memprediksi token asli berdasarkan konteks kiri dan kanan. Ini mengajarkan model memahami hubungan antar kata dalam satu kalimat. Kedua, Next Sentence Prediction: model diberi dua kalimat dan harus menentukan apakah kalimat kedua adalah lanjutan logis dari kalimat pertama. Ini mengajarkan model memahami hubungan antar kalimat. Kedua tugas dilatih secara simultan pada dataset besar seperti Wikipedia dan BookCorpus. Hasilnya adalah model BERT yang sudah memiliki pemahaman bahasa umum, siap di-fine-tune untuk tugas spesifik seperti klasifikasi teks."*

---

## Checklist Sesi #13 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Apa kepanjangan MLM? | |
| Berapa persen token yang di-mask saat pre-training? | |
| Apa aturan 80-10-10 dalam MLM? | |
| Apa tujuan NSP? | |
| Apa yang dipelajari model dari NSP? | |

**Jawaban cepat:**
- Masked Language Modeling
- 15%
- 80% jadi [MASK], 10% token acak, 10% tetap (tidak diubah)
- Mengajarkan hubungan antar kalimat (apakah kalimat B mengikuti kalimat A)
- Representasi `[CLS]` yang bagus untuk tugas yang melibatkan dua kalimat

---

## Latihan Sesi #13

1. **Tulis ulang** paragraf di poin F dengan kata-kata Anda sendiri

2. **Latihan ucapkan** (target: 55 detik — ini agak panjang, latihan sampai lancar)

3. **Simulasi tanya jawab:**
   - Dosen: "Mengapa tidak semua token yang di-mask? Kenapa ada yang diganti acak dan ada yang tetap?"
   - Jawab dengan kalimat Anda sendiri

---

**Setelah selesai, kirim: "Sesi 13 selesai" + hasil latihan paragraf.**

Lanjut ke **Sesi #14: Fine-tuning BERT untuk Tugas Hilir** (khususnya untuk multi-label klasifikasi seperti skripsi Anda).