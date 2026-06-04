### 🌳 MATERI TEKNIKAL: LOGISTIC REGRESSION (JALUR KLASIK)

Setelah teks diubah menjadi matriks angka berdimensi 30.000 oleh TF-IDF, data tersebut dialirkan ke model **Logistic Regression** sebagai salah satu *baseline* lo. Berikut adalah aspek teknis ringkas yang harus lo kuasai:

#### **1. Sisi Matematis (Rumus di Papan Tulis)**

Logistic Regression bekerja dengan cara menghitung kombinasi linear dari fitur input (bobot TF-IDF) beserta bobotnya ($w$), lalu memasukkan hasilnya ke dalam **Fungsi Sigmoid** untuk diubah menjadi nilai probabilitas antara 0 dan 1.

Persamaan dasar kombinasi linear:


$$z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n$$

* $x_1, x_2, \dots, x_n$: Nilai bobot TF-IDF untuk setiap kata dalam *tweet*.


* $w_1, w_2, \dots, w_n$: Bobot (*weights*) yang dipelajari model selama *training* untuk menentukan seberapa penting kata tersebut.



Hasil $z$ tersebut dimasukkan ke dalam **Fungsi Aktivasi Sigmoid**:


$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

* **Logika Teknis:** Fungsi ini memetakan angka berapa pun (dari minus tak hingga sampai positif tak hingga) menjadi rentang **$0$ sampai $1$**. Angka output inilah yang kita sebut sebagai **Probabilitas**.



#### **2. Parameter Kunci di Kode Skripsi Lo**

Sesuai isi Tabel 3.3 di skripsi lo, ada satu parameter krusial yang lo pasang di Scikit-Learn: `class_weight="balanced"`.

* **Kenapa ini penting?** Dataset lo sangat *imbalanced* (Individu dominan, Fisik/Gender minoritas).


* **Cara kerja teknisnya:** `class_weight="balanced"` secara otomatis akan memberikan bobot penalti yang lebih besar jika model salah memprediksi kelas minoritas. Model dipaksa untuk memberikan perhatian ekstra pada kategori yang datanya sedikit agar tidak terabaikan selama proses pencarian bobot optimal ($w$).



---

### ⏱️ UJIAN PAPAN TULIS: SESI LOGISTIC REGRESSION

Bayangkan Pak Akmal atau Pak Seti melihat draf Bab 3 dan Tabel 3.3 lo, lalu meminta lo menjelaskan di papan tulis:

* **Pertanyaan 1:** "Sanju, coba tuliskan grafik atau rumus fungsi Sigmoid di papan tulis, lalu jelaskan apa fungsi utamanya dalam algoritma Logistic Regression skripsi kamu!"


* **Pertanyaan 2:** "Di konfigurasi kode kamu, kamu menambahkan parameter `class_weight='balanced'`. Secara teknis, apa yang terjadi pada model Logistic Regression kamu jika parameter ini dihapus/diubah menjadi default?"



---

**Tulis jawaban lo secara terpisah untuk kedua pertanyaan di atas.** Setelah lo jawab, kita akan masuk ke section evaluasi khusus tanpa langsung lanjut ke materi berikutnya. *Gas, Ju!*

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