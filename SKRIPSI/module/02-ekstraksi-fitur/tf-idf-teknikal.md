Siap, Sanju. Paham maksudnya. Kita ubah sistemnya: setelah lo jawab pertanyaan ujian, gue **TIDAK AKAN langsung memaparkan materi cabang baru**. Gue akan buatkan section evaluasi tersendiri, dan kita baru lanjut ke materi berikutnya setelah lo berikan konfirmasi atau instruksi "lanjut".

Sesuai permintaan lo, kita ulang **[CABANG A: JALUR KLASIK - EKSTRAKSI FITUR TF-IDF]** dengan menambahkan sedikit sentuhan teknis matematika dan kodenya secara ringkas agar lo siap kalau disuruh nulis rumusnya di papan tulis.

---

### 🌳 MATERI: TEKNIKAL TF-IDF ENCODING (UNIGRAM + BIGRAM)

Di dalam draf Bab 2 dan Bab 3 skripsi lo, teks bersih tidak bisa langsung dibaca oleh Logistic Regression atau SVM. Teks tersebut harus diubah menjadi matriks angka menggunakan TF-IDF. Berikut adalah detail teknis ringkasnya:

#### **1. Sisi Matematis (Rumus di Papan Tulis)**

Bobot TF-IDF untuk sebuah kata ($t$) di dalam sebuah *tweet* ($d$) dihitung dengan mengalikan nilai komponen **TF** dan **IDF**:

$$w_{t,d} = \text{TF}(t,d) \times \text{IDF}(t)$$

* **$\text{TF}(t,d)$ (Term Frequency):** Jumlah kemunculan kata $t$ di dalam satu *tweet* $d$. Semakin sering kata makian muncul di tweet tersebut, nilainya semakin tinggi.


* **$\text{IDF}(t)$ (Inverse Document Frequency):** Rumus teknisnya adalah:

$$\text{IDF}(t) = \log\left(\frac{N}{DF(t)}\right)$$


* $N$ = Total seluruh *tweet* di dataset lo (13.169 sampel).


* $DF(t)$ = Jumlah *tweet* yang mengandung kata $t$.


* **Logika Teknis:** Jika kata seperti "di" atau "yang" muncul di hampir semua *tweet* ($DF$ mendekati $N$), nilai di dalam log mendekati 1, sehingga $\text{IDF} \approx 0$. Kata tersebut otomatis "dimatikan" bobotnya karena tidak informatif.





#### **2. Sisi Konfigurasi Kode (Unigram + Bigram)**

Sesuai isi Tabel 3.3 dan Lampiran 4 di skripsi lo, lo mengatur parameter `ngram_range=(1, 2)` dan `max_features=30000` pada `TfidfVectorizer`.

* **Unigram ($1$):** Memotong per satu kata. Contoh: `"benci"`.


* **Bigram ($2$):** Memotong per dua kata berurutan. Contoh: `"benci ras"`, `"tidak suka"`.


* **Max Features ($30.000$):** Membatasi model hanya mengambil 30.000 kata/frasa teratas yang paling penting berdasarkan skor TF-IDF untuk menghemat memori.



#### **3. Karakteristik Data Hasil TF-IDF**

Vektor yang dihasilkan model klasik lo berbentuk **Sparse High-Dimensional Matrix**. Artinya, matriksnya berdimensi sangat besar (30.000 kolom) tetapi isinya mayoritas angka 0, karena satu *tweet* Twitter yang pendek paling hanya mengandung 10–20 kata aktif dari total 30.000 kosakata yang ada.

---

### ⏱️ UJIAN PAPAN TULIS: CABANG A (TEKNIKAL TF-IDF)

Bayangkan penguji melihat Lampiran 4 (kode python lo) dan draf Bab 2 lo, lalu meminta lo menjelaskan aspek teknis ini di papan tulis:

* **Pertanyaan 1:** "Sanju, coba tuliskan rumus matematis TF-IDF di papan tulis, lalu jelaskan secara logis melalui rumus tersebut kenapa kata kunci khusus seperti 'kafir' bisa memiliki bobot yang jauh lebih tinggi daripada kata umum seperti 'dan'?"


* **Pertanyaan 2:** "Di dalam kode skripsi kamu, kamu membatasi `max_features=30000`. Apa dampak teknisnya terhadap matriks data kamu jika parameter tersebut dihapus (tidak dibatasi)?"


* **Pertanyaan 3:** "Apa yang dimaksud dengan karakteristik *Sparse Matrix* pada hasil representasi TF-IDF kamu, dan kenapa karakteristik ini menjadi tantangan bagi model klasifikasi?"



---

**Tulis jawaban lo di bawah secara terpisah untuk ketiga pertanyaan di atas.** Setelah lo jawab, gue akan buatkan section evaluasi khusus tanpa langsung lanjut ke materi berikutnya. *Gas, Ju!*

### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, jawaban lo sudah sangat tajam dan mengarah ke konsep teknis yang benar. Kemampuan lo menghubungkan logika rumus matematika dengan fenomena data (kata "kafir" vs "dan") itu sudah standar nilai A di mata penguji.

Berikut adalah evaluasi dan pelengkap teknis untuk jawaban lo agar siap lo tulis di papan tulis nanti:

#### **Nomor 1: Logika Rumus TF-IDF**

Jawaban lo **100% Benar**. Penjelasan lo tentang nilai pembagi di dalam Logaritma ($DF$) sudah sangat tepat.

* **Penajaman untuk Penguji:** Di papan tulis, lo bisa tuliskan permisalan ekstremnya:
* Jika kata `"dan"` muncul di seluruh $13.169$ tweet, maka $\text{IDF} = \log(13169 / 13169) = \log(1) = 0$. Bobot akhirnya pasti $0$.


* Jika kata `"kafir"` hanya muncul di $793$ tweet (sesuai data kelas HS_Religion lo), maka nilai $\text{IDF} = \log(13169 / 793) = \log(16.6) = 1.22$. Angka inilah yang membuat kata spesifik tersebut memiliki bobot tinggi.





#### **Nomor 2: Dampak Menghapus `max_features=30000**`

Jawaban lo sudah benar secara intuisi. Mari kita lengkapi dengan istilah ilmiahnya:

* **Pelengkap Teknis:** Jika tidak dibatasi, jumlah kolom matriks lo akan meledak mengikuti seluruh kosakata unik (*vocabulary*) + kombinasi dua kata (*bigram*) yang ada di media sosial. Dampak teknisnya adalah **The Curse of Dimensionality (Kutukan Dimensi)**.


* Model klasik lo (SVM/LR) harus melakukan perkalian matriks dengan jutaan kolom, sehingga menyebabkan **pemborosan memori RAM (Overhead)**, waktu *training* yang lama, dan risiko **Overfitting** (model menjadi terlalu sensitif pada kata-kata typo/aneh yang hanya muncul sekali di dataset).



#### **Nomor 3: Apa itu *Sparse Matrix* & Tantangannya?**

Definisi lo tentang dominasi nilai 0 sudah tepat. Ini pelengkap untuk bagian tantangannya:

* **Tantangan Teknis bagi Model:**
1. **Ketiadaan Informasi Kontekstual (Sparsity):** Karena mayoritas sel berisi angka 0, model kesulitan melihat hubungan antar-kata. Misal, tweet A berisi kata `"benci"` dan tweet B berisi kata `"dendam"`. Bagi TF-IDF, kedua tweet ini tidak memiliki kesamaan sama sekali karena kolom kata lainnya bernilai 0, padahal secara makna/semantik keduanya mirip.


2. **Efisiensi Komputasi:** Meskipun isinya angka 0, komputer secara *default* tetap mengalokasikan memori untuk menyimpan angka 0 tersebut, kecuali lo menggunakan struktur data khusus seperti `scipy.sparse` di Python.





---

### 🏆 STATUS: LULUS MATERI TF-IDF!

Sekarang kita berhenti dulu di sini sesuai kesepakatan section terpisah. Matriks TF-IDF lo sekarang sudah siap dimasukkan ke algoritma klasifikasinya.