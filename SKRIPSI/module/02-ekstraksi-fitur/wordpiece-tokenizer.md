### 🌳 CABANG B: JALUR MODERN — TOKENSASI WORDPIECE (INDOBERT)

Sekarang kita menyeberang ke sisi kanan pohon arsitektur lo: **Jalur Modern berbasis Deep Learning (IndoBERT)**. Langkah pertama yang dilewati oleh teks bersih sebelum masuk ke dalam lapisan *neural network* IndoBERT adalah **Tokenisasi WordPiece**.

Di dalam file `app.py` lo, proses ini dijalankan secara otomatis saat lo memuat `AutoTokenizer.from_pretrained`. Berikut adalah aspek teknis mekanis yang wajib lo kuasai:

#### **1. Mengapa Tidak Pakai Tokenisasi Kata Biasa? (Tantangan OOV)**

Model klasik menggunakan pemisahan kata berdasarkan spasi (tokenisasi leksikal). Masalahnya, bahasa media sosial Indonesia dipenuhi kata gaul, singkatan, dan typo (*vocab* tidak terbatas).
Jika kita menggunakan tokenisasi kata biasa, model akan sering menemukan kata-kata baru yang tidak ada di kamus data latih. Kondisi ini disebut **Out-of-Vocabulary (OOV)**. Model akan menjadi "buta" dan mengabaikan kata tersebut.

#### **2. Cara Kerja Mekanisme WordPiece (Subword Tokenization)**

Untuk mengatasi OOV, IndoBERT memecah kalimat bukan berdasarkan kata utuh, melainkan berdasarkan unit sub-kata (*subword tokens*).

* 
**Kosakata Dasar:** Tokenizer memiliki daftar kosakata dasar (*vocabulary bank*) berisi karakter tunggal (a-z) dan potongan suku kata yang sering muncul dalam 4 miliar kata data pra-latih Indo4B.


* 
**Simbol Penanda Suku Kata Lanjutan (`##`):** Jika sebuah kata terpaksa dipecah, potongan kata pertama akan ditulis biasa, sedangkan potongan kata kedua dan seterusnya akan diberi awalan simbol `##`. Simbol ini memberi tahu model bahwa potongan tersebut adalah sambungan dari kosakata sebelumnya.



#### **3. Simulasi Riil WordPiece di Skripsi Lo**

Mari kita ambil kata kunci dari analisis kesalahan di Bab 4 skripsi lo, yaitu kata bodi-shaming implisit: `"mematikan"` dan kata gaul `"goblokkkk"`.

* **Kata Baku/Sering Muncul:** `"goblok"` $\rightarrow$ terdaftar di *vocabulary*, langsung menjadi 1 token: `["goblok"]`.
* 
**Kata Berimbuhan/Terpecah:** `"mematikan"` $\rightarrow$ tidak terdaftar utuh, dipecah menjadi: `["me", "##matikan"]`.


* **Kata Typo Ekstrem:** `"goblokkkk"` $\rightarrow$ dipecah menjadi unit terkecilnya: `["goblok", "##kk", "##kk"]`.

#### **4. Implikasi Penting pada Perhitungan KernelSHAP Lo**

Ini adalah alasan kenapa kode di Lampiran 3 lo memiliki fungsi `aggregate_shap_to_words`.
Karena KernelSHAP menghitung kontribusi per elemen input (yaitu per *token* hasil WordPiece), kata `"mematikan"` akan memiliki dua nilai SHAP terpisah (misal `ϕ("me") = 0.1` dan `ϕ("##matikan") = 0.3`).
Agar pengguna aplikasi Streamlit lo tidak bingung melihat kata yang terpotong-potong, kode lo secara cerdas **menjumlahkan** nilai kedua sub-token tersebut (`0.1 + 0.3 = 0.4`) sehingga menghasilkan satu nilai atribusi utuh untuk kata `"mematikan"`.

---

### ⏱️ UJIAN PAPAN TULIS: SESI WORDPIECE TOKENIZER

Bayangkan Pak Seti menguji pemahaman lo tentang prapemrosesan komponen *deep learning* ini di papan tulis:

* **Pertanyaan 1:** "Sanju, coba simulasikan di papan tulis bagaimana WordPiece Tokenizer dari IndoBERT memotong kalimat ujaran kebencian berikut: `"kamu murtaddd"`! Tunjukkan mana yang menjadi token utama dan mana yang menggunakan simbol `##`!"
* **Pertanyaan 2:** "Apa dampak positif dari penggunaan pendekatan *Subword Tokenization* ini terhadap penanganan masalah *Out-of-Vocabulary* (OOV) pada data Twitter yang penuh kata *slang* dibandingkan dengan model klasifikasi klasik kamu?"

---

**Tulis jawaban lo secara terpisah untuk kedua pertanyaan di atas.** Setelah lo jawab, kita akan masuk ke section evaluasi khusus tanpa langsung lanjut ke materi berikutnya. *Gas, Ju!*

### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, lo hebat! Intuisi lo tentang bagaimana sub-kata yang sering muncul membantu model menjadi "lebih pede" itu secara konsep kecerdasan buatan sudah sangat tepat. Penempatan simbol `##` lo hampir benar, cuma ada sedikit pergeseran teknis posisi penulisan simbolnya di dalam arsitektur BERT.

Mari kita rapikan aspek teknis penulisan token dan istilah ilmiahnya agar jawaban lo terlihat sangat matang di papan tulis:

---

#### **Nomor 1: Simulasi Penulisan Token WordPiece di Papan Tulis**

Logika pemecahan lo sudah benar, yaitu memisahkan kata baku dengan kata yang mengalami deformasi/typo. Namun, di dalam standar algoritma WordPiece (BERT), **simbol `##` ditaruh di AWAL sub-kata lanjutan**, bukan di akhir.

* **Teks Mentah:** `"kamu murtaddd"`
* **Hasil Potongan Tokenizer di Papan Tulis:** 
$$\text{["kamu", "murtad", "##d", "##d"]}$$


* 
**Penjelasan Mekanis ke Penguji:** * `"kamu"`: Menjadi token utuh karena frekuensi kemunculannya tinggi pada data *pre-training* IndoBERT.


* `"murtad"`: Merupakan basis kata utama yang dikenali oleh sistem.
* `"##d"`, `"##d"`: Simbol `##` di depan huruf 'd' menandakan secara struktural bahwa token-token ini tidak berdiri sendiri, melainkan harus ditempelkan ke token sebelumnya (`"murtad"`) untuk merekonstruksi kata asli di tingkat visual.



---

#### **Nomor 2: Analisis Keunggulan Terhadap Tantangan OOV**

Jawaban lo sudah sangat bagus dalam membandingkan sifat leksikal TF-IDF dengan sifat sub-kata WordPiece. Mari kita formulasikan menggunakan terminologi ilmiah NLP untuk sidang:

* **Penajaman Istilah untuk Penguji:**
1. **Pada Model Klasik (TF-IDF):** Bersifat *Word-level tokenization*. Jika di data uji muncul kata `"murtaddd"` (pakai d tiga), model klasik akan menganggap ini sebagai satu kata baru yang berbeda total dari kata `"murtad"`. Jika kata typo ini tidak ada di kamus data latih, nilainya di matriks TF-IDF akan menjadi 0 atau memicu masalah **Out-of-Vocabulary (OOV)**, sehingga model menjadi "buta" terhadap sinyal ujaran kebencian tersebut.
2. 
**Pada IndoBERT (WordPiece):** Masalah OOV dapat dimitigasi karena kata seaneh apa pun akan dipecah menjadi unit sub-kata terkecil hingga mencapai tingkat karakter tunggal jika diperlukan. Karakter tunggal (a-z) sudah pasti terdaftar di kosakata dasar IndoBERT, sehingga model tetap mampu menghasilkan *vector embedding* yang valid dan tidak kehilangan makna semantik dari kata dasarnya (`"murtad"`).





---

### 🏆 STATUS: LULUS MATERI WORDPIECE TOKENIZER!

Sekarang lo sudah paham bagaimana teks diubah menjadi rangkaian token sub-kata yang aman dari masalah OOV. Langkah berikutnya, token-token ini akan dialirkan masuk ke dalam **Arsitektur Utama: Transformer Encoder (IndoBERT)**.

Kita tahan dulu di sini sesuai kesepakatan alur belajar kita.

Silakan berikan konfirmasi atau instruksi jika lo sudah mencatat perbedaan letak simbol `##` ini, dan kita siap masuk ke materi: **[Arsitektur Transformer Encoder & Konsep Self-Attention]**!