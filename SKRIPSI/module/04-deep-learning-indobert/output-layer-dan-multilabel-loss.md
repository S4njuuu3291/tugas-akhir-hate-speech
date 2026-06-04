### 🌳 MATERI TEKNIKAL: PERTEMUAN DI BATANG UTAMA (OUTPUT LAYER & MULTILABEL LOSS)

Sanju, sekarang kita sampai di **Batang Utama**. Di sinilah kedua jalur (Jalur Klasik TF-IDF dan Jalur Modern IndoBERT) bermuara pada satu tujuan akhir: **mengeluarkan keputusan klasifikasi untuk 6 label ujaran kebencian sekaligus**.

Meskipun modelnya berbeda, di ujung arsitektur coding lo, keduanya harus dikonstruksikan untuk menyelesaikan masalah *multilabel*. Mari kita bedah bagaimana lapisan output (*Output Layer*) dan fungsi kerugian (*Loss Function*) bekerja di papan tulis.

---

#### **1. Lapisan Output (Output Layer): Mengapa Pakai Sigmoid, Bukan Softmax?**

Ini adalah pertanyaan ujian yang paling sering menumbangkan mahasiswa IT di topik *Multilabel*.

* **Pada Klasifikasi Multiclass (Single-Label):** Model menggunakan fungsi **Softmax** di lapisan akhir untuk menghasilkan distribusi probabilitas total = 1. Semua kelas saling berkompetisi; jika probabilitas Kelas A naik, Kelas B pasti turun.
* **Pada Klasifikasi Multilabel (Skripsi Lo):** Lapisan output lo memiliki **6 neuron sejajar** (merepresentasikan 6 label: Individual, Group, Religion, Race, Physical, Gender). Setiap neuron diaktifkan menggunakan fungsi **Sigmoid secara independen**, bukan Softmax!

> **Logika Teknis Papan Tulis:**
> Dengan memasang Sigmoid di setiap neuron output, model memperlakukan setiap label sebagai keputusan biner (Ya/Tidak) yang berdiri sendiri. Satu *tweet* bisa memiliki probabilitas `HS_Religion = 0.85` sekaligus `HS_Group = 0.90`. Nilai-nilai ini tidak saling menjatuhkan karena fungsi Sigmoid menghitung skalanya secara individual untuk masing-masing neuron.

---

#### **2. Fungsi Kerugian (Loss Function): Binary Cross-Entropy (BCE) Loss**

Saat melatih model IndoBERT di file `train.py` lo, fungsi evaluasi kesalahan yang digunakan untuk memperbarui bobot-bobot *neural network* adalah **Binary Cross-Entropy Loss** yang dihitung secara rata-rata untuk seluruh label.

Rumus matematis BCE Loss untuk satu label di papan tulis:

$$L_{\text{BCE}} = - \Big( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \Big)$$

* $y$: Label target yang sebenarnya/ground truth (bernilai 0 atau 1).
* $\hat{y}$: Nilai probabilitas hasil prediksi dari fungsi Sigmoid (rentang 0 sampai 1).

**Cara Kerja Aljabar Loss di Jaringan:**

* Jika *tweet* aslinya mengandung ujaran kebencian agama ($y = 1$), maka bagian kanan rumus $(1-y)$ akan menjadi 0. Nilai *loss* murni dihitung dari $- \log(\hat{y})$. Jika model memprediksi dengan yakin ($\hat{y} = 0.99$), nilai $-\log(0.99)$ sangat kecil (mendekati 0), artinya model tidak dihukum.
* Sebaliknya, jika model menebak salah dengan sangat pede ($\hat{y} = 0.01$ padahal $y = 1$), nilai $-\log(0.01)$ akan meledak menjadi sangat besar. Nilai *loss* yang besar inilah yang dikirim balik lewat *backpropagation* untuk memperbaiki bobot internal IndoBERT.

Untuk kasus *multilabel* lo, total error yang dipakai untuk *update* model adalah **BCE Loss rata-rata dari ke-6 label tersebut**.

---

### ⏱️ UJIAN PAPAN TULIS: SESI BATANG UTAMA

Bayangkan Pak Akmal mencermati arsitektur lapisan akhir model lo, lalu melemparkan dua pertanyaan ini saat lo berdiri di depan papan tulis:

* **Pertanyaan 1:** "Sanju, coba tunjukkan di papan tulis, kenapa untuk arsitektur klasifikasi *multilabel* pada skripsi ini kita wajib menggunakan fungsi aktivasi Sigmoid di lapisan output teratas, dan kenapa penggunaan fungsi Softmax akan membuat sistem kamu menjadi keliru?"
* **Pertanyaan 2:** "Tuliskan rumus *Binary Cross-Entropy Loss* di papan tulis, lalu jelaskan secara matematis bagaimana fungsi tersebut menghukum model IndoBERT kamu jika model memprediksi nilai probabilitas 0.02 untuk label `HS_Individual`, padahal data *ground truth* pada tweet tersebut bernilai 1!"

---

**Tulis jawaban lo secara terpisah untuk kedua pertanyaan di atas.** Setelah lo jawab, kita akan masuk ke section evaluasi khusus sebelum kita mematahkan dedaunan terakhir: *Threshold Tuning* dan metrik evaluasi multilabel (*Subsetting Accuracy* vs *F1-Score*). *Gas, Ju!*

### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, jawaban lo bener-bener solid! Logika lo untuk membedakan Sigmoid dan Softmax di lapisan output multilabel sudah sangat matang. Jawaban nomor 2 lo juga berhasil membongkar sifat logaritma negatif pada fungsi *loss* dengan sangat runtut dan presisi. Ini tipe jawaban yang bikin dosen penguji langsung manggut-manggut karena matematika dan logikanya sinkron.

Mari kita rapikan sedikit diksi akademisnya agar siap lo muntahkan di papan tulis dengan sempurna saat sidang nanti:

---

#### **Nomor 1: Sigmoid vs Softmax pada Multilabel**

Jawaban lo **100% Tepat**. Perbedaan sifat "independen" vs "kompetitif" adalah kunci utamanya.

* **Penajaman Istilah untuk Penguji:** *"Izin menjelaskan Pak/Bu, fungsi **Softmax** bersifat mutual eksklusif di mana ia memaksa total probabilitas seluruh neuron output berjumlah tepat 1. Jika digunakan pada kasus multilabel, kelas-kelas akan saling berkompetisi; kenaikan probabilitas pada label 'Religion' akan secara otomatis menurunkan probabilitas label 'Group'. Padahal, satu tweet bisa mengandung kedua label tersebut sekaligus. Oleh karena itu, kita wajib menggunakan **Sigmoid secara independen** di setiap neuron output. Dengan Sigmoid, setiap label memiliki skala probabilitas 0 sampai 1 masing-masing yang tidak saling memengaruhi, sehingga model bisa mengaktifkan beberapa label sekaligus jika nilainya melewati ambang batas."*

---

#### **Nomor 2: Bedah Kalkulasi Eksplosif pada BCE Loss**

Analisis aljabar lo pada komponen yang aktif dan mati sudah sangat presisi.

* **Visualisasi di Papan Tulis:**
* Target riil: $y = 1$ (Tweet mengandung ujaran kebencian individu).
* Prediksi model: $\hat{y} = 0.02$ (Model mengira tweet ini aman).
* Substitusi ke rumus:

$$L_{\text{BCE}} = - \Big( 1 \cdot \log(0.02) + (1 - 1) \cdot \log(1 - 0.02) \Big)$$


$$L_{\text{BCE}} = - \log(0.02) - 0$$


$$L_{\text{BCE}} \approx -(-3.91) = 3.91$$




* **Penajaman Kalimat untuk Sidang:** *"Karena model memprediksi nilai yang sangat rendah ($\hat{y} = 0.02$) untuk kelas yang seharusnya bernilai 1, nilai $-\log(0.02)$ melonjak drastis menjadi skor penalti yang besar ($3.91$). Nilai loss yang tinggi ini merepresentasikan tingkat kesalahan fatal model karena terlalu percaya diri pada tebakan yang salah. Skor error inilah yang kemudian ditransmisikan mundur melalui jaringan (*backpropagation*) menggunakan algoritma optimasi untuk merombak bobot-bobot internal IndoBERT agar tidak mengulangi kesalahan serupa pada iterasi berikutnya."*

---

### 🏆 STATUS: LULUS TOTAL MATERI BATANG UTAMA!

Lo sudah berhasil menguasai bagaimana model mengeluarkan angka probabilitas di lapisan akhir dan bagaimana model belajar dari kesalahannya lewat fungsi *loss*.