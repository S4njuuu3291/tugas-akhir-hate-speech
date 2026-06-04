### 🌳 MATERI TEKNIKAL: SUPPORT VECTOR MACHINE & CALIBRATED SVM (JALUR KLASIK)

Model klasik kedua yang lo gunakan sebagai *baseline* kuat di skripsi lo adalah **Support Vector Machine (SVM)**, secara spesifik menggunakan varian `LinearSVC` dari Scikit-Learn. Namun, di sinilah letak keunikan teknis skripsi lo: lo menambahkan komponen bernama **CalibratedClassifierCV**.

Berikut adalah peta konsep teknis yang wajib lo kuasai:

#### **1. Cara Kerja Dasar SVM (Rumus Batas Keputusan)**

Berbeda dengan Logistic Regression yang mencari probabilitas, SVM adalah model geometri. Dia mencari sebuah **Hyperplane** (garis pemisah pada dimensi tinggi) yang memisahkan teks ujaran kebencian dengan teks netral.

SVM tidak hanya asal membuat garis pemisah, tetapi dia memaksimalkan nilai **Margin** (jarak terdekat antara garis pemisah dengan titik data terdekat yang disebut **Support Vectors**).

Secara matematis, fungsi keputusan dari linear SVM adalah:


$$f(x) = \mathbf{w}^T\mathbf{x} + b$$

* $\mathbf{x}$: Vektor bobot TF-IDF dari kalimat input.


* $\mathbf{w}$: Vektor bobot koefisien *hyperplane*.


* $b$: Nilai bias.


* **Kelemahan Output:** Output dari $f(x)$ ini adalah skor jarak geometris (bisa bernilai minus atau positif besar, bukan probabilitas antara 0 dan 1). Jika $f(x) > 0$, data diklasifikasikan sebagai positif, dan jika $f(x) < 0$ sebagai negatif.



#### **2. Kenapa Harus Pakai *Calibrated SVM*? (Masalah Krusial)**

Ini adalah makanan empuk buat penguji. Lo kan sedang membangun sistem klasifikasi **multilabel** dan menggunakan **KernelSHAP**. Kedua hal ini **wajib** menerima input berupa nilai **probabilitas** (angka 0 sampai 1).

Karena `LinearSVC` aslinya hanya mengeluarkan skor jarak mentah (*decision function*) dan tidak punya nilai probabilitas, lo membungkusnya dengan `CalibratedClassifierCV` menggunakan metode **Sigmoid** (juga dikenal sebagai **Platt Scaling**).

#### **3. Cara Kerja Platt Scaling (Sigmoid Calibration)**

Metode ini melatih sebuah fungsi Sigmoid tambahan di atas skor jarak mentah ($f(x)$) yang dihasilkan oleh SVM tadi:

$$P(y=1 \mid x) = \frac{1}{1 + e^{A \cdot f(x) + B}}$$

* $f(x)$: Skor jarak mentah dari SVM.


* $A, B$: Parameter skalar yang dipelajari model menggunakan data validasi terpisah (*hold-out validation split*) untuk mencegah kebocoran data (*data leakage*).


* **Hasil Akhir:** Skor jarak geometris tadi berhasil dikonversi menjadi nilai probabilitas yang valid antara 0 dan 1, sehingga SVM lo bisa dipakai untuk *threshold tuning* dan KernelSHAP.



---

### ⏱️ UJIAN PAPAN TULIS: SESI SUPPORT VECTOR MACHINE

Bayangkan Dr. Akmal melihat draf Bab 3 lo pada bagian konfigurasi SVM (Tabel 3.4), lalu beliau meminta lo maju ke papan tulis:

* **Pertanyaan 1:** "Sanju, coba gambarkan ilustrasi geometris di papan tulis mengenai bagaimana SVM memisahkan dua kelas data! Tunjukkan mana yang dimaksud *hyperplane*, *margin*, dan *support vectors*!"


* **Pertanyaan 2:** "Di draf kamu tertulis penggunaan `CalibratedClassifierCV` dengan metode Sigmoid pada SVM. Kenapa kamu harus repot-repot melakukan kalibrasi ini? Apa dampaknya pada sistem klasifikasi multilabel dan KernelSHAP kamu jika kalibrasi ini dihilangkan?"



---

**Tulis jawaban lo secara terpisah untuk kedua pertanyaan konsep di atas.** Jawab dengan gaya lo, nanti kita evaluasi di section khusus. *Gas, Ju!*

### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, pemahaman lo secara filosofis dan geometris tentang SVM sudah **sangat matang**. Kalimat *"SVM adalah model klasik berorientasi spasial"* itu keren banget dan sangat disukai penguji karena langsung menusuk ke karakteristik utama model ini.

Mari kita bahas rumus margin yang lo tanyakan, sekaligus mengevaluasi jawaban lo langkah demi langkah untuk persiapan sidang:

---

### 📏 Rumus Lebar Margin (Wajib Tahu!)

Di papan tulis, jika persamaan pembatas margin lo adalah $\mathbf{w}^T\mathbf{x} + b = 1$ dan $\mathbf{w}^T\mathbf{x} + b = -1$, maka **jarak geometris total (lebar margin)** antara kedua batas tersebut dirumuskan secara teknis sebagai:

$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

* $\|\mathbf{w}\|$ adalah *norm* (panjang) dari vektor bobot $\mathbf{w}$.
* **Logika Matematika:** Karena rumus margin adalah $\frac{2}{\|\mathbf{w}\|}$, maka untuk **memaksimalkan** nilai margin, kita harus **meminimalkan** nilai $\|\mathbf{w}\|$ (pembaginya harus sekecil mungkin). Itulah alasan kenapa di rumus optimasi SVM tadi tertulis fungsi $\min \frac{1}{2}\|\mathbf{w}\|^2$.

---

### 🔍 Koreksi Jawaban Ujian Lo

#### **Nomor 1: Interpretasi Jarak Titik Jauh vs Dekat**

Jawaban lo secara logika sudah benar, tapi ada **satu istilah yang harus diperbaiki** agar tidak dikritik penguji.

* **Koreksi Istilah:** Kata *"karena margin nya lebih besar"* kurang tepat, karena nilai **margin untuk satu model SVM itu konstan/tetap** (lebar jalurnya sama di sepanjang garis).
* **Jawaban yang Tepat untuk Sidang:** *"Untuk data yang posisinya sangat jauh di area positif, nilai skor mentah $f(x)$ atau nilai logits geometrisnya akan **jauh lebih besar** daripada data yang menempel di dekat hyperplane. Hal ini merepresentasikan **tingkat keyakinan (confidence) spasial** model yang lebih tinggi. Semakin jauh suatu titik dari bidang pembatas, semakin kecil kemungkinan model tersebut salah mengklasifikasikannya."*

#### **Nomor 2: Karakteristik Hyperplane & Fungsi *Slack Variable* ($\xi_i$)**

Jawaban lo sudah sangat bagus. Gue bantu pertegas dan lengkapi aspek teknisnya:

* **Apakah Hyperplane itu Garis Lurus?** **Ya, benar**, pada `LinearSVC` (SVM Linear), *hyperplane* berupa garis lurus jika datanya 2 dimensi, bidang datar jika 3 dimensi, dan *hyperplane* lurus multidimensi jika fiturnya ada 30.000 seperti skripsi lo.
* **Fungsi Teknis *Slack Variable* ($\xi_i$):** Nilai $\xi_i$ ini bertindak sebagai **nilai penalti kuantitatif** untuk setiap data yang melanggar batas margin.
* Jika data berada di zona aman (di luar jalur margin), maka $\xi_i = 0$.
* Jika data masuk ke dalam jalur margin atau menyeberang ke wilayah kelas lawan (*misclassification*), maka $\xi_i > 0$ (bernilai positif sebesar jarak pelanggarannya).
* **Fungsi di media sosial:** Variabel ini menangkap fakta bahwa teks Twitter lo sangat kotor. Tanpa adanya $\xi_i$, matematika SVM akan *crash* atau tidak menemukan solusi garis pemisah sama sekali karena datanya tumpang tindih (*non-linearly separable*).



---

### 🏆 STATUS: LULUS MEKANIKA DASAR SVM!

Sekarang lo sudah menguasai arsitektur mekanis SVM murni. Poin masalahnya masih sama: output dari perhitungan $\mathbf{w}^T\mathbf{x} + b$ tadi menghasilkan angka skalar spasial (misal $+0.65$ atau $-1.1$).

Silakan berikan konfirmasi atau instruksi jika lo sudah mencatat poin margin ini dan siap lanjut ke **[Bagaimana CalibratedClassifierCV mengubah skor spasial ini menjadi probabilitas]** sebelum kita pindah ke materi Wrapper Multilabel.

### 🌳 MATERI TEKNIKAL: PROSES KALIBRASI PROBABILITAS (PLATT SCALING)

Sekarang kita masuk ke bagian kedua dari SVM di skripsi lo, yaitu bagaimana cara mengubah skor jarak spasial mentah ($f(x)$) menjadi nilai probabilitas.

Di dalam file `app.py` lo, komponen ini dijalankan oleh fungsi `CalibratedClassifierCV(method='sigmoid')`. Di papan tulis, ini adalah jembatan penting yang menghubungkan **Cabang SVM** dengan **Mekanisme KernelSHAP**.

#### **1. Mengapa Perlu Kalibrasi? (Urgensi Akademik)**

Model `LinearSVC` bawaan Scikit-Learn bekerja dengan mencari batas pemisah spasial. Outputnya berupa nilai jarak geometris yang rentangnya dari minus tak hingga sampai positif tak hingga.

Kondisi ini memicu dua masalah besar di skripsi lo:

1. 
**Aturan Keputusan Multilabel:** Untuk menentukan apakah suatu label aktif, lo membutuhkan ambang batas keputusan (*threshold tuning*) yang seragam dan valid di skala probabilitas (0 sampai 1).


2. 
**KernelSHAP Requirement:** Algoritma KernelSHAP yang lo bangun secara mandiri bekerja dengan menghitung selisih nilai ekspektasi prediksi (probabilitas) saat sebuah kata ada vs absen. Jika inputnya berupa skor jarak spasial SVM yang tidak berwujud probabilitas, maka perhitungan aljabar matriks *Weighted Least Squares* (WLS) pada KernelSHAP lo akan menjadi tidak valid.



#### **2. Sisi Matematis: Rumus Platt Scaling**

Untuk mengatasi masalah tersebut, lo menerapkan metode **Platt Scaling** (aktivasi Sigmoid pada output fungsi keputusan SVM). Rumus teknisnya di papan tulis adalah:

$$P(y=1 \mid x) = \frac{1}{1 + e^{A \cdot f(x) + B}}$$

* 
$f(x)$: Skor jarak mentah (*decision function*) yang dikeluarkan oleh SVM murni.


* 
$A, B$: Parameter skalar (bobot dan bias baru) yang dipelajari secara khusus selama proses kalibrasi.


* 
$P(y=1 \mid x)$: Nilai probabilitas final yang berada tepat di rentang $0$ sampai $1$.



#### **3. Sisi Strategi Validasi: Mencegah Data Leakage**

Berdasarkan draf Tabel 3.4 lo, proses pencarian parameter $A$ dan $B$ ini diatur menggunakan strategi **Hold-out Split** (Data Validasi Terpisah).

* 
**Logika Teknisnya:** Parameter kalibrasi $A$ dan $B$ tidak boleh dilatih menggunakan Data Latih (*Training Set*) utama yang dipakai untuk mencari posisi *hyperplane* SVM. Jika dipaksakan menggunakan data yang sama, model kalibrasi akan menjadi terlalu percaya diri (*overfitting*).


* Oleh karena itu, lo memisahkan **10% Data Validasi** khusus untuk melatih fungsi kalibrasi ini, sehingga nilai probabilitas yang dihasilkan tetap objektif dan terhindar dari kebocoran data (*data leakage*).



---

### ⏱️ UJIAN PAPAN TULIS: KALIBRASI SVM

Jawab dua pertanyaan penguji ini di kotak chat selanjutnya untuk menutup materi Cabang SVM:

* **Pertanyaan 1:** "Sanju, jika output jarak mentah ($f(x)$) dari model SVM kamu untuk sebuah *tweet* adalah sebesar **$-5.0$** (sangat jauh di area negatif teks netral), jelaskan secara matematis menggunakan rumus Platt Scaling ke mana arah nilai probabilitasnya setelah dikalibrasi!"
* **Pertanyaan 2:** "Kenapa kamu menetapkan parameter data validasi terpisah (*hold-out split*) untuk proses kalibrasi SVM ini? Apa implikasinya terhadap validitas pengujian jika parameter $A$ dan $B$ kalibrasi tersebut langsung dicari menggunakan data training?"

---

**Tulis jawaban lo secara terpisah.** Setelah lo jawab, kita akan review di section evaluasi khusus sebelum masuk ke materi penyatuan multilabel (*Wrapper Multilabel*). *Gas, Ju!*

### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, insting aljabar lo tajam banget! Pemecahan matematika untuk rumus eksponensial di Pertanyaan 1 itu benar-benar runtut dan logis. Penguji kalau melihat lo bisa membedah arah matriks dan variabel seperti ini di papan tulis pasti langsung paham kalau lo bukan mahasiswa yang cuma asal copas *library*.

Mari kita rapikan bahasa akademisnya agar terlihat sempurna saat sidang nanti:

#### **Nomor 1: Analisis Nilai Jarak Mentah $-5.0$ pada Platt Scaling**

Analisis lo **100% Tepat**. Mari kita tuliskan pembuktiannya dalam bentuk notasi matematika papan tulis yang bersih:

* Secara umum, parameter $A$ pada Platt Scaling bernilai negatif ($A < 0$) karena fungsinya membalikkan skor jarak (skor minus harus jadi probabilitas kecil, skor plus jadi probabilitas besar).
* Jika $f(x) = -5.0$, maka kombinasi di dalam pangkat menjadi:

$$A \cdot f(x) = (\text{negatif}) \times (-5.0) = \text{Positif Besar}$$


* Masukkan ke dalam rumus:

$$P(y=1 \mid x) = \frac{1}{1 + e^{\text{Positif Besar}}}$$


* Seperti yang lo bilang, karena nilai eksponensial berpangkat positif besar itu nilainya meledak tinggi ($e^{\text{besar}} \rightarrow \infty$), maka nilai penyebutnya ($1 + \infty$) menjadi sangat besar.
* Hasil pembagian $1 / \text{angka sangat besar}$ akan **mendekati $0$**. Artinya, *tweet* tersebut memiliki probabilitas hampir $0\%$ mengandung ujaran kebencian pada label tersebut.

#### **Nomor 2: Urgensi Hold-out Split vs Data Leakage**

Jawaban lo sudah benar mengenai pencegahan *data leakage*. Ini penjelasan teknis untuk melengkapi bagian "kenapa tidak boleh pakai data train":

* **Penjelasan untuk Penguji:** *"Izin menjelaskan Pak/Bu, model SVM murni dilatih di atas data training untuk meminimalkan loss spasialnya, sehingga skor jarak $f(x)$ pada data training pasti akan cenderung sangat ideal dan terlalu percaya diri (overoptimistic). Jika kita mencari parameter kalibrasi $A$ dan $B$ menggunakan data training yang sama, maka fungsi Sigmoid kalibrasi akan ikut terdistorsi menjadi terlalu percaya diri (overfitting)."*
* Dengan menggunakan **10% Data Validasi terpisah (hold-out split)** , parameter $A$ dan $B$ dicari menggunakan data baru yang belum pernah dipakai untuk melatih *hyperplane* SVM. Hasilnya, pemetaan probabilitasnya menjadi lebih realistis, objektif, dan terhindar dari kebocoran data (*data leakage*).



---

### 🏆 STATUS: LULUS TOTAL JALUR MODEL INDIVIDU (SVM & LR)!

Sekarat kita kunci Cabang SVM dan Logistic Regression. Posisi lo sekarang sudah punya dua model klasik yang bisa mengeluarkan output berupa probabilitas valid (0 sampai 1).

Namun, ada satu masalah besar: Logistic Regression dan SVM itu aslinya adalah model **Single-Label Classification** (hanya bisa menebak 1 kelas/biner). Padahal skripsi lo bertema **Multilabel Classification** (1 tweet bisa mengandung label Individu, Agama, dan Kelompok sekaligus).

Sebelum masuk ke materi baru, berikan konfirmasi atau instruksi **"Lanjut Multilabel"** jika lo sudah siap untuk membedah bagaimana teknik **Binary Relevance (BR)** dan **Classifier Chains (CC)** memaksa model klasik lo menjadi bisa berpikir *multilabel*!