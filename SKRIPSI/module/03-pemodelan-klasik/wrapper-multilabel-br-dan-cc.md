### 🌳 MATERI TEKNIKAL: WRAPPER MULTILABEL (PROBLEM TRANSFORMATION)

Sekarang kita masuk ke bagian yang menyatukan model-model individu tadi agar bisa memproses data *multilabel*. Seperti yang lo tahu, Logistic Regression dan SVM aslinya adalah model *single-label*. Di skripsi lo, untuk memaksa model klasik tersebut mendeteksi 6 kategori ujaran kebencian sekaligus, lo menerapkan pendekatan **Problem Transformation** menggunakan dua metode *wrapper* : **Binary Relevance (BR)** dan **Classifier Chains (CC)**.

Berikut adalah mekanisme kerja teknis dari kedua metode tersebut yang wajib lo kuasai untuk simulasi papan tulis:

---

#### **1. Binary Relevance (BR)**

Pendekatan ini adalah yang paling sederhana. Secara teknis, BR bekerja dengan cara **memecah masalah *multilabel* menjadi $q$ buah masalah klasifikasi biner yang sepenuhnya terpisah dan independen** (di skripsi lo, $q = 6$ sesuai jumlah label).

* 
**Mekanisme Kerja:** Sistem akan membuat dan melatih 6 model linear terpisah:


* Model 1 khusus mendeteksi `HS_Individual` vs Aman.


* Model 2 khusus mendeteksi `HS_Group` vs Aman ... sampai Model 6 untuk `HS_Gender`.




* 
**Proses Inferensi:** Ketika ada *tweet* masuk, ke-6 model ini akan bekerja secara paralel dan mengeluarkan prediksi probabilitasnya masing-masing tanpa memedulikan hasil dari model lainnya.


* 
**Kelemahan Terbesar (Kritik Penguji):** BR mengasumsikan bahwa setiap label bersifat **independen**. Padahal pada dunia nyata, ujaran kebencian sering kali tumpang tindih (misal, menyerang etnis tertentu biasanya otomatis menyerang kelompok juga). BR tidak mampu menangkap hubungan ketergantungan antar-label ini (*label dependency*).



---

#### **2. Classifier Chains (CC)**

Untuk mengatasi kelemahan BR, lo menguji metode kedua yaitu **Classifier Chains**. CC bekerja dengan cara **menyusun $q$ model biner tersebut ke dalam sebuah rantai (chain) berurutan**.

* 
**Mekanisme Kerja & Aliran Fitur:** Prediksi dari model sebelumnya di dalam rantai akan **dimasukkan kembali sebagai fitur tambahan** untuk model berikutnya.


* **Contoh Alur Rantai di Papan Tulis:**
Misalkan urutan rantai yang terbentuk adalah: $\text{Model}_1 (\text{Individual}) \rightarrow \text{Model}_2 (\text{Group}) \rightarrow \text{Model}_3 (\text{Religion})$.


1. 
$\text{Model}_1$ memprediksi label `HS_Individual` hanya berdasarkan fitur TF-IDF asli ($X$). Keluar hasil prediksi biner $\hat{y}_1$.


2. 
$\text{Model}_2$ memprediksi label `HS_Group`. Fitur inputnya sekarang **diperluas** menjadi $[X, \hat{y}_1]$. Model ini tahu apakah model pertama mendeteksi adanya serangan individu atau tidak.


3. 
$\text{Model}_3$ memprediksi label `HS_Religion`. Fitur inputnya meledak menjadi $[X, \hat{y}_1, \hat{y}_2]$.




* 
**Kelebihan:** CC mampu menangkap informasi ketergantungan antar-label secara eksplisit sehingga secara teori lebih informatif.


* 
**Kelemahan Terbesar (Fenomena *Error Propagation*):** CC sangat sensitif terhadap urutan rantai. Jika model di awal rantai salah membuat prediksi ($\hat{y}_1$ harusnya 0 tapi ditebak 1), maka **bias kesalahan tersebut akan merambat dan meracuni** keputusan model-model di sepanjang sisa rantai berikutnya.



---

### ⏱️ UJIAN PAPAN TULIS: SESI WRAPPER MULTILABEL

Berdasarkan analisis hasil eksperimen lo di Bab 4 (Tabel 4.3 dan 4.5), mari kita uji pemahaman konsep lo lewat skenario pertanyaan sidang:

* **Pertanyaan 1:** "Sanju, coba gambarkan diagram alur perbandingan di papan tulis antara cara kerja **Binary Relevance** dan **Classifier Chains** saat memproses sebuah input vektor TF-IDF yang sama!"
* 
**Pertanyaan 2:** "Di bab 4 halaman 83, hasil analisis korelasi Pearson menunjukkan bahwa ketergantungan antar-label pada dataset kamu tergolong rendah hingga sedang (korelasi tertinggi hanya 0.36 antara Group dan Religion). Berdasarkan karakteristik data tersebut, pendekatan mana (BR atau CC) yang secara logis tidak akan memberikan peningkatan performa yang signifikan? Jelaskan alasannya!"



---

**Tulis jawaban lo secara terpisah untuk kedua pertanyaan di atas.** Setelah lo jawab, kita akan masuk ke section evaluasi khusus sebelum kita mematahkan cabang terakhir di jalur klasik dan bersiap pindah ke dunia *deep learning* IndoBERT. *Gas, Ju!*

### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, jawaban lo luar biasa! Lo berhasil menyimpulkan mekanisme komparasi yang sangat abstrak ini menjadi logika alur data yang padat dan akurat. Jawaban lo untuk Pertanyaan 2 bahkan langsung menyentuh esensi temuan ilmiah yang lo tulis di Bab 3 skripsi lo.

Mari kita rapikan bahasanya agar siap disajikan di papan tulis dengan standar sidang yang tinggi:

---

#### **Nomor 1: Ilustrasi Alur Data BR vs CC**

Penjelasan lo sudah sangat tepat mengenai perbedaan perluasan fitur. Di papan tulis, lo bisa menggambarkan perbandingan input-outputnya seperti ini untuk meyakinkan penguji:

* **Binary Relevance (BR):**
* 
$\text{Input Model 1 hingga 6} \rightarrow \mathbf{X}_{\text{asli}}$ (Vektor TF-IDF).


* 
$\text{Output} \rightarrow [\hat{y}_1, \hat{y}_2, \hat{y}_3, \hat{y}_4, \hat{y}_5, \hat{y}_6]$ (Diproses secara paralel dan mandiri).




* **Classifier Chains (CC):**
* 
$\text{Model 1 (Individual)} \rightarrow \text{Input: } \mathbf{X}_{\text{asli}} \rightarrow \text{Output: } \hat{y}_1$.


* 
$\text{Model 2 (Group)} \rightarrow \text{Input: } [\mathbf{X}_{\text{asli}}, \hat{y}_1] \rightarrow \text{Output: } \hat{y}_2$.


* 
$\text{Model 3 (Religion)} \rightarrow \text{Input: } [\mathbf{X}_{\text{asli}}, \hat{y}_1, \hat{y}_2] \rightarrow \text{Output: } \hat{y}_3$.





---

#### **Nomor 2: Analisis Implikasi Karakteristik Data terhadap Performa CC**

Jawaban lo **100% Tepat dan Cerdas**. Ini adalah poin krusial yang lo sebutkan di Bab 3 halaman 49.

* 
**Cara penegasan jawaban ke penguji:** "Izin menjelaskan Pak/Bu, karena matriks korelasi Pearson menunjukkan bahwa mayoritas pasangan label pada dataset ini memiliki nilai korelasi mendekati nol (independen), maka pendekatan **Classifier Chains (CC)** secara logis tidak memberikan peningkatan performa yang signifikan dibandingkan **Binary Relevance (BR)**. CC baru akan sangat superior jika antar-label memiliki ketergantungan yang sangat kuat (misalnya, jika label A muncul, kemungkinan label B muncul adalah 90%). Karena data kita cenderung bersifat independen, informasi prediksi label sebelumnya yang dimasukkan ke model berikutnya hanya bertindak sebagai sinyal penguat yang lemah, bukan lompatan performa yang masif."



---

### 🏆 STATUS: LULUS TOTAL JALUR WRAPPER MULTILABEL!

Sekarang lo sudah resmi mematahkan seluruh cabang di **Jalur Klasik (Cabang A)**, mulai dari ekstraksi fitur TF-IDF, mekanisme matematis Logistic Regression dan SVM, kalibrasi probabilitas, hingga transformasi masalah multilabel.

Sesuai dengan kesepakatan alur belajar kita, kita berhenti dulu di sini sebelum melompat ke seberang pohon.

Silakan berikan konfirmasi atau instruksi jika lo sudah mantap dengan jalur klasik ini, dan kita siap untuk menyeberang ke **[CABANG B: JALUR MODERN - DEEP LEARNING INDOBERT]** dimulai dari bagaimana **WordPiece Tokenizer** bekerja!