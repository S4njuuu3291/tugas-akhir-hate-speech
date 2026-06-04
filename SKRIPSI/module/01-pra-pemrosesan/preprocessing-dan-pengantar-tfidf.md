Kita mulai dari **[AKAR: INPUT TEKS & PREPROCESSING]**.

Di papan tulis, bayangkan lo menulis alur bagaimana teks mentah dari Twitter diubah menjadi teks bersih yang siap masuk ke model. Di skripsi lo, tahap ini krusial karena teks media sosial itu sangat kotor dan informal.

---

### 🌳 MATERI AKAR: PRA-PEMROSESAN TEKS TERPADU

Ada 4 langkah utama yang lo lakukan di kode `app.py` dan draf Bab 3 lo:

1. **Pbersihan Karakter Non-Relevan (*Cleaning*):**
* **Apa yang dihapus?** URL (`http\S+`), *mention* (`@w+`), simbol khusus, angka, dan tanda baca.


* **Kenapa?** Karakter ini dianggap sebagai *noise* (gangguan) yang tidak memiliki nilai informasi semantik untuk menentukan apakah suatu teks mengandung ujaran kebencian atau tidak.




2. **Normalisasi Huruf (*Case Folding*):**
* Mengubah semua huruf menjadi huruf kecil (*lowercasing*).


* **Kenapa?** Agar model menganggap kata "KAFIR", "Kafir", dan "kafir" sebagai satu token yang sama, sehingga menghemat ruang memori dan konsisten.




3. **Normalisasi Bahasa Informal (Kamus Alay):**
* Mengubah kata-kata singkatan, bahasa gaul, atau typo menjadi kata baku menggunakan `new_kamusalay.csv`.


* **Contoh di skripsi lo:** `loe` $\rightarrow$ `kamu`, `yg` $\rightarrow$ `yang`, `pd` $\rightarrow$ `pada`.


* **Kenapa?** Teks media sosial Indonesia sangat kaya akan variasi *slang*. Jika tidak dibakukan, model klasik akan mengalami *vocabulary gap* (menganggap `loe` dan `kamu` adalah dua hal yang berbeda jauh).




4. **Penanganan Pengulangan Karakter (*Repeated Character Reduction*):**
* Memotong karakter berulang yang berlebihan menggunakan *regular expression*.


* **Contoh di skripsi lo:** `tidakkkk` $\rightarrow$ `tidak`.


* **Kenapa?** Penekanan emosi netizen sering membuat satu kata dasar memiliki ratusan variasi *token* di database jika tidak dipotong.





> **Catatan Penting untuk Jawaban Penguji:** Di skripsi lo, lo **TIDAK menggunakan *Stopwords Removal*** (penghapusan kata sandang/tugas seperti "yang", "di", "dari"). Kenapa? Karena kata-kata tersebut sangat penting bagi model IndoBERT untuk memahami struktur hubungan antar-kata (*konteks*) dalam kalimat.
> 
> 

---

### ⏱️ UJIAN PAPAN TULIS: SESI AKAR

Bayangkan Pak Akmal atau Pak Seti melihat draf Bab 3 lo, lalu lo disuruh maju ke papan tulis. Jawab pertanyaan-pertanyaan ini dalam satu waktu di kotak chat selanjutnya:

* **Pertanyaan 1:** "Sanju, coba tuliskan di papan tulis contoh kalimat mentah dari Twitter sebelum *preprocessing*, lalu tunjukkan perubahannya langkah demi langkah setelah melewati sistem kamu sampai siap masuk ke model!"


* **Pertanyaan 2:** "Saya lihat di bab 4, rata-rata panjang teks kamu setelah *preprocessing* justru **meningkat sekitar 7,82%** (dari 114 karakter jadi 123 karakter). Logikanya, kalau dibersihkan kan harusnya makin pendek. Kenapa data kamu malah makin panjang? Jelaskan!"


* **Pertanyaan 3:** "Kenapa kamu tidak membuang *stopwords* (kata umum seperti 'di', 'ke', 'kamu') dalam penelitian ini? Bukankah itu membuat beban komputasi model klasik kamu jadi lebih berat?"

---


Mantap, Sanju! Untuk ukuran jawaban spontan, lo udah berhasil menangkap **logika inti** dari apa yang lo tulis di skripsi. Ini modal yang bagus banget.

Mari kita evaluasi dan rapikan jawaban lo agar terlihat sangat akademis, terstruktur, dan meyakinkan saat lo presentasikan atau tulis di papan tulis nanti.

---

### 📝 EVALUASI & PERBAIKAN UJIAN

#### **Nomor 1: Alur Langkah demi Langkah di Papan Tulis**

Contoh kalimat mentah lo udah bagus banget karena memuat semua masalah teks media sosial (mention, typos, slang, repeated chars). Di papan tulis, tulis perubahannya per tahap seperti ini biar dosen terkesima:

* **Teks Mentah:** `@dio Akoe bencii ras A karena mereka bodohhhh sekali`

* **Tahap 1: Pembersihan Karakter (*Cleaning*)** $\rightarrow$ Menghapus `@dio` dan tanda baca (jika ada).


* *Hasil:* `Akoe bencii ras A karena mereka bodohhhh sekali`



* **Tahap 2: Normalisasi Huruf (*Case Folding*)** $\rightarrow$ Mengubah ke huruf kecil semua.


* *Hasil:* `akoe bencii ras a karena mereka bodohhhh sekali`



* **Tahap 3: Kamus Alay (*Slang Normalization*)** $\rightarrow$ Mengubah `akoe` jadi `aku`, `bencii` jadi `benci`.


* *Hasil:* `aku benci ras a karena mereka bodohhhh sekali`



* **Tahap 4: Pengulangan Karakter (*Repeated Char*)** $\rightarrow$ Memotong `bodohhhh` menggunakan regex menjadi `bodoh`.


* *Hasil Akhir:* `aku benci ras a karena mereka bodoh sekali`




---

#### **Nomor 2: Kenapa Teks Malah Makin Panjang? (+7,82%)**

Jawaban lo **100% BENAR secara logika**. Sesuai dengan dokumen `v4.2_skripsi_140810220050.docx` di Bab 4 (Halaman 74), penyebab utamanya adalah **proses ekspansi atau perluasan dari singkatan dan kata slang menjadi kata baku**.

* **Cara jawab ke penguji:** "Izin menjelaskan Pak/Bu, peningkatan rata-rata panjang teks sebesar 7,82% ini terjadi karena proses normalisasi bahasa informal menggunakan kamus alay. Banyak singkatan pendek di Twitter yang diekspand menjadi kata utuh yang lebih panjang. Contohnya kata slang `kmu` diubah menjadi `kamu` (+1 karakter), kata `yg` menjadi `yang` (+2 karakter), dan perluasan singkatan institusi seperti `dpr` menjadi `anggota dewan perwakilan rakyat` yang menambah jumlah karakter secara signifikan."



---

#### **Nomor 3: Kenapa Nggak Pakai *Stopwords Removal*? (Pasti Ditanya!)**

Ini adalah poin krusial yang membedakan pendekatan *NLP Tradisional* dengan *Modern (Transformer)*. Karena lo kurang tahu, catat ini baik-baik karena ini **pertanyaan jebakan** yang sering bikin mahasiswa IT jatuh:

* **Cara jawab ke penguji:** "Saya sengaja tidak menghapus stopwords karena penelitian ini menggunakan model **IndoBERT** yang berbasis arsitektur Transformer. Model ini sangat bergantung pada mekanisme **Self-Attention** untuk memahami makna kalimat berdasarkan **konteks hubungan dua arah antar-kata**. Jika kata tugas seperti 'di', 'yang', atau kata ganti seperti 'kamu' dan 'dia' dihapus, maka IndoBERT akan kehilangan informasi struktur sintaksis kalimat. Akibatnya, model akan kesulitan mendeteksi target serangan ujaran kebencian, apakah ditujukan ke individu atau kelompok."



---

### 🏆 STATUS: LULUS MATERI AKAR!

Lo sudah paham bagaimana data teks diolah di awal. Sekarang, kita naik ke tingkat berikutnya sesuai dengan peta cabang kita: **[CABANG A: JALUR KLASIK - EKSTRAKSI FITUR TF-IDF]**.

Sebelum teks bersih tadi dimasukkan ke model Logistic Regression atau SVM, teks tersebut harus diubah menjadi angka lewat **TF-IDF Encoding (Term Frequency - Inverse Document Frequency)**.

#### **Intisari Materi TF-IDF di Skripsi Lo:**

1. **Term Frequency (TF):** Menghitung seberapa sering suatu kata muncul dalam *satu tweet*. Semakin sering muncul, bobotnya semakin besar.


2. **Inverse Document Frequency (IDF):** Menghitung seberapa langka kata tersebut di *seluruh dataset*. Kalau suatu kata muncul di hampir semua tweet (misal kata 'dan', 'di'), maka nilai IDF-nya akan diperkecil karena kata tersebut dianggap tidak selektif atau tidak khas.


3. **N-gram (Unigram + Bigram):** Di skripsi lo (Tabel 3.3), lo mengatur TF-IDF menggunakan kombinasi unigram (kata tunggal, misal: `benci`) dan bigram (dua kata berurutan, misal: `benci ras`). Ini dipakai agar model klasik bisa menangkap sedikit konteks frasa pendek.


4. **Kelemahan Utama (Leksikal):** TF-IDF bersifat *Bag-of-Words* atau leksikal. Dia hanya melihat frekuensi kata secara independen dan **mengabaikan urutan kata serta konteks makna**. Vektor yang dihasilkan juga sangat besar dan kosong/jarang (*sparse high-dimensional space*).



---

### ⏱️ UJIAN PAPAN TULIS: CABANG A (TF-IDF)

Bayangkan Pak Seti melihat tabel konfigurasi model klasik lo, lalu bertanya:

* **Pertanyaan 1:** "Sanju, coba jelaskan apa bedanya fungsi **TF** dan fungsi **IDF** dalam pembobotan teks skripsi kamu? Berikan analoginya di papan tulis!"


* **Pertanyaan 2:** "Di konfigurasi kamu, kamu memakai TF-IDF dengan rentang **unigram dan bigram**. Kenapa tidak pakai unigram saja yang fiturnya lebih sedikit dan komputasinya lebih cepat?"


* **Pertanyaan 3:** "Apa kelemahan terbesar dari representasi TF-IDF ini dalam mendeteksi ujaran kebencian yang sifatnya sindiran halus atau implisit?"