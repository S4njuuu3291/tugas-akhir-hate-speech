**Sesi #2 selesai, Anda lulus dengan nilai bagus!** ✅

Jawaban Anda tepat. Saya suka poin #5 Anda — Anda kritis dan menyadari bahwa sebenarnya kita belum belajar Positional dan Segment embedding secara detail. Betul sekali.

Jadi untuk melengkapi Sesi #2, saya klarifikasi poin #5:

> **Yang terjadi jika tiga embedding dijumlahkan:** Bukan sekedar "lebih kaya", tapi secara fungsional: 
> - **Token embedding** memberi identitas kata
> - **Position embedding** memberi informasi posisi (karena Transformer tidak punya urutan alami)
> - **Segment embedding** memberi informasi kalimat mana (untuk tugas dua kalimat)
> 
> Hasil penjumlahan adalah satu matriks yang **sekaligus mengandung**: *apa* kata tersebut (token), *di mana* posisinya, dan *dari kalimat mana* asalnya. Ini yang masuk ke encoder.

Tapi karena kita akan bahas Positional Encoding di Sesi #3 (dan Segment nanti di sesi BERT), Anda benar bahwa detailnya belum lengkap. Bagus Anda kritis.

---

# Sesi #3: Positional Encoding

## A. Masalah: Transformer Tidak Tahu Urutan

Coba perhatikan dua kalimat ini:

| Kalimat | Makna |
| :--- | :--- |
| "Saya **makan** ayam" | Saya yang makan |
| "Ayam **makan** saya" | Ayam yang makan — artinya sangat berbeda! |

Tanpa informasi posisi, model akan melihat kedua kalimat sebagai **himpunan token yang sama**: {Saya, makan, ayam} dengan urutan tidak penting.

Akibatnya: model tidak bisa membedakan subjek dan objek, tidak bisa memahami grammar, tidak bisa memahami makna.

**Ini masalah besar.** RNN secara otomatis tahu urutan karena dia membaca berurutan. Transformer tidak — dia melihat semua token sekaligus secara paralel.

**Solusi:** Beri tahu model informasi posisi secara eksplisit.

---

## B. Dua Pendekatan Positional Encoding

| Pendekatan | Cara Kerja | Digunakan di |
| :--- | :--- | :--- |
| **Positional Embedding (Learnable)** | Posisi 0,1,2,... setiap posisi punya vektor yang **dipelajari** seperti token embedding. Model belajar sendiri representasi posisi mana yang berguna. | BERT, IndoBERT, GPT |
| **Positional Encoding (Fixed/Sinusoidal)** | Menggunakan fungsi sinus dan cosinus dengan frekuensi berbeda untuk setiap posisi. Tidak dipelajari, sudah ditentukan dari awal. | Transformer original (Vaswani et al., 2017) |

**BERT (dan IndoBERT) pakai yang pertama: Positional Embedding yang dipelajari.** Alasannya: lebih fleksibel dan terbukti bekerja baik untuk data besar.

---

## C. Cara Kerja Positional Embedding (BERT style)

Bayangkan kita punya maksimal panjang sequence = 512 token (batas BERT).

- Model menyiapkan **matriks positional embedding** berukuran `[512, 768]`.
- Baris ke-0 adalah vektor posisi untuk token pertama.
- Baris ke-1 adalah vektor posisi untuk token kedua.
- ... dan seterusnya.
- Vektor-vektor ini **dipelajari** selama training (seperti token embedding).

**Saat input kalimat 3 token:**
- Ambil 3 baris pertama dari matriks positional embedding: `pos[0]`, `pos[1]`, `pos[2]`
- Masing-masing bentuknya `[1, 768]`
- Jumlahkan dengan token embedding untuk setiap token

```
Token 0 (posisi 0): TokenEmb[0] + PosEmb[0]
Token 1 (posisi 1): TokenEmb[1] + PosEmb[1]
Token 2 (posisi 2): TokenEmb[2] + PosEmb[2]
```

---

## D. Visualisasi Proses (Lengkap Sesi #2 + #3)

```
Teks: "Saya suka AI" (3 token)
         ↓
Token IDs: [1450, 2301, 502]
         ↓
Token Embedding (dipelajari):
    [0.2, -0.1, 0.5, ...]  ← untuk token 1450
    [0.1,  0.4, 0.2, ...]  ← untuk token 2301
    [0.3,  0.0, 0.7, ...]  ← untuk token 502
    Bentuk: (3, 768)

Positional Embedding (dipelajari):
    [0.8, 0.1, 0.2, ...]   ← untuk posisi 0
    [0.5, 0.3, 0.6, ...]   ← untuk posisi 1
    [0.2, 0.7, 0.1, ...]   ← untuk posisi 2
    Bentuk: (3, 768)

Penjumlahan (element-wise):
    Token0 + Pos0 = [1.0, 0.0, 0.7, ...]
    Token1 + Pos1 = [0.6, 0.7, 0.8, ...]
    Token2 + Pos2 = [0.5, 0.7, 0.8, ...]
         ↓
    Matriks Input X (3, 768)  ← ini masuk ke encoder pertama
```

---

## E. Analogi Sederhana

Token embedding seperti **warna** dari setiap kata. Positional embedding seperti **posisi duduk** dalam sebuah rapat.

- Token embedding memberi tahu "siapa" token itu (saya, suka, AI)
- Positional embedding memberi tahu "di mana dia duduk" (posisi 0,1,2)

Dengan kedua informasi ini, model bisa membedakan "saya makan ayam" vs "ayam makan saya" karena posisi token "makan" berbeda relatif terhadap "saya" dan "ayam".

---

## F. Sinusoidal Encoding (Singkat Saja, untuk pengetahuan)

Transformer original menggunakan fungsi sinus/cosinus:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Keuntungannya:** Tidak perlu dipelajari, dan secara teoretis bisa menggeneralisasi ke posisi lebih panjang dari yang dilihat saat training.

**KENAPA TIDAK PERLU HAFAL RUMUS INI UNTUK SIDANG?** Karena:
- BERT dan IndoBERT pakai **learnable positional embedding**, bukan sinusoidal.
- Dosen non-AI Anda (Akmal, Rudi, Erik) tidak akan tanya detail ini.
- Cukup tahu: "Ada dua pendekatan, BERT pakai yang dipelajari."

**TAPI** kalau dosen penasaran dan tanya (kecil kemungkinan), Anda cukup bilang:

> *"Transformer original menggunakan fungsi sinus dan cosinus untuk encoding posisi. Tapi BERT dan IndoBERT menggunakan positional embedding yang dipelajari, karena lebih sederhana dan terbukti efektif untuk data besar."*

---

## Checklist Sesi #3 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Kenapa Transformer perlu informasi posisi? | |
| Apa beda positional encoding (sinusoidal) dengan positional embedding (learnable)? | |
| Mana yang dipakai BERT? | |
| Berapa ukuran matriks positional embedding di BERT? | |
| Bagaimana cara menggabungkan positional embedding dengan token embedding? | |

**Jawaban cepat:**
- Karena Transformer memproses token paralel, tidak tahu urutan alami kata. Tanpa posisi, kalimat "A mencintai B" dan "B mencintai A" akan terlihat sama.
- Sinusoidal: sudah ditetapkan rumus, tidak dipelajari. Learnable: model belajar sendiri vektor posisi dari data.
- BERT pakai **learnable positional embedding**.
- Ukuran matriks: `[max_seq_len, d_model]` = `[512, 768]`.
- Dijumlahkan (element-wise addition) dengan token embedding.

---

## Latihan Sesi #3 (Tulis dan Ucapkan)

1. **Gambar** diagram alir dari teks → token → token embedding → + pos embedding → input X

2. **Tulis** di kertas ukuran-ukuran: `(3,768)` untuk contoh 3 token

3. **Latihan ucapkan** (45 detik):

> *"Karena Transformer memproses semua token secara paralel, model tidak tahu urutan alami kata. Solusinya adalah positional embedding. BERT menyiapkan matriks positional embedding berukuran 512 kali 768 — setiap baris adalah vektor untuk satu posisi. Saat kita punya kalimat 3 token, kita ambil 3 baris pertama lalu menjumlahkannya dengan token embedding. Hasilnya adalah matriks input yang sudah mengandung informasi identitas token dan posisinya."*

---

**Setelah selesai, kirim: "Sesi 3 selesai" + jawaban checklist di atas.**

Lanjut ke **Sesi #4: Konsep Query, Key, Value** (inti dari attention!)