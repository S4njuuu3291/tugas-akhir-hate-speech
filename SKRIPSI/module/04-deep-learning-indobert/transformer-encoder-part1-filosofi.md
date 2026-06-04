### 🌳 DEEP DIVE ARSITEKTUR: TRANSFORMER ENCODER (INDOBERT) — PART 1

Sanju, karena kita mengasumsikan pemahaman dari **nol mutlak**, mari kita lupakan dulu rumus-rumus rumit. Kita mulai dari analogi paling mendasar untuk memahami **kenapa arsitektur ini diciptakan** dan **bagaimana dia berpikir**.

Di skripsi lo, IndoBERT murni mengadopsi struktur bernama **Transformer Encoder**.

---

### 1. Masalah Utama Masa Lalu: Pendekatan Searah

Sebelum ada Transformer, teknologi AI membaca teks seperti manusia biasa membaca buku: kata demi kata, dari kiri ke kanan (disebut arsitektur RNN atau LSTM).

> **Analogi Masalah:**
> Bayangkan lo membaca kalimat ini: *"Tadi saya ke bank, lalu mengambil uang."* > Saat mata lo baru sampai di kata **"bank"**, lo otomatis berpikir ini adalah bank tempat menyimpan uang.
> Tapi bagaimana kalau kalimatnya adalah: *"Tadi saya ke bank sungai untuk memancing."*?
> Jika AI membaca dari kiri ke kanan, saat dia berada di kata **"bank"**, dia akan *salah paham* mengira itu tempat menyimpan uang, sampai akhirnya dia membaca kata "sungai" dan "memancing" di sebelah kanannya.

Model-model lama sering gagal mendeteksi ujaran kebencian karena makna sebuah kata di media sosial sangat bergantung pada kata-kata di sekitar kanan dan kirinya (*konteks*).

---

### 2. Solusi IndoBERT: Deep Bidirectional Context

IndoBERT memecahkan masalah ini dengan sifatnya yang **Deep Bidirectional** (Dua Arah yang Mendalam).

Ketika rangkaian token sub-kata dari WordPiece masuk ke dalam Transformer Encoder, model ini **tidak membaca dari kiri ke kanan**. Dia langsung melihat seluruh kalimat secara utuh sekaligus.

Dia mengizinkan setiap kata untuk "melihat", mengobrol, dan menyerap makna dari seluruh kata lain yang ada di sebelah kiri maupun sebelah kanannya secara simultan di setiap lapisan jaringan.

---

### 3. Tiga Komponen Input Vektor (Membentuk Fondasi Angka)

Komputer tidak paham huruf; dia hanya paham angka dalam bentuk koordinat/vektor padat (*dense vector*). Sesuai draf Gambar 2.3 di skripsi lo, sebelum token WordPiece masuk ke otak Transformer, setiap token akan diubah menjadi vektor yang dibentuk dari **penjumlahan 3 komponen utama**:

1. 
**Token Embeddings:** Mengubah potongan kata menjadi angka koordinat berdasarkan arti leksikal dasarnya di dalam kamus.


2. 
**Segment Embeddings:** Karena skripsi lo memproses tweet tunggal, komponen ini hanya bertindak sebagai penanda biner bahwa semua token ini berasal dari satu kalimat input yang sama.


3. 
**Position Embeddings (Sangat Krusial!):** Karena Transformer memproses seluruh kata secara paralel (sekaligus), model ini aslinya kehilangan ingatan tentang urutan kata. Kata *"Saya benci kamu"* dan *"Kamu benci saya"* akan dianggap sama persis karena kumpulan katanya sama.


* 
*Solusi Teknis:* **Position Embeddings** menambahkan koordinat posisi fisik (vektor posisi ke-0, ke-1, ke-2) ke dalam setiap token. Dengan begitu, model tetap tahu struktur susunan kalimat tanpa perlu membaca satu per satu.





---

### ⏱️ UJIAN PAPAN TULIS: FILOSOFI TRANSFORMER ENCODER

Sebelum kita masuk ke bagian matematika paling bawah permukaan (perhitungan rumus *Self-Attention*), jawab dulu 2 pertanyaan mendasar ini di papan tulis untuk memastikan pondasi lo dari angka 0 sudah kuat:

* **Pertanyaan 1:** "Sanju, jika Transformer Encoder memproses seluruh kata di dalam kalimat secara bersamaan (paralel), bagaimana cara model kamu membedakan makna spasial antara kalimat `A: 'Dia memaki saya'` dengan kalimat `B: 'Saya memaki dia'`? Komponen internal apa yang bertanggung jawab menangani hal itu?"
* **Pertanyaan 2:** "Apa yang dimaksud dengan sifat *Deep Bidirectional* pada IndoBERT, dan kenapa sifat ini membuat IndoBERT lebih cerdas dalam memahami konteks kalimat media sosial yang ambigu dibandingkan arsitektur lama yang membaca kata demi kata?"

---

**Tulis jawaban lo secara spontan di bawah.** Setelah lo jawab, kita akan masuk ke part 2 untuk membedah kalkulasi mekanis di balik *Self-Attention Matrix* ($Q, K, V$). Kita jalani santai, section per section. *Gas!*