### 🌳 DEEP DIVE ARSITEKTUR: TRANSFORMER ENCODER (INDOBERT) — PART 2

Sanju, sekarang kita masuk ke ruang mesin paling dalam dari IndoBERT, yaitu mekanisme **Self-Attention**. Di sinilah keajaiban "memahami konteks" itu dikonversi menjadi perhitungan aljabar matriks murni.

Di papan tulis, jika penguji meminta lo menggambarkan bagaimana kata `"benci"` bisa berinteraksi dengan kata `"ras"`, lo harus mengeluarkan tiga konsep huruf ini: **Q (Query)**, **K (Key)**, dan **V (Value)**.

---

### 1. Analogi Sederhana: Sistem Pencarian YouTube

Jangan pusing dulu dengan matriksnya. Mari kita pakai analogi dunia nyata yang lo akrab sehari-hari:

* **Query ($Q$):** Apa yang lo ketik di kolom pencarian (Misal: lo ngetik *"video klip lagu A"*).
* **Key ($K$):** Judul, tag, atau deskripsi video yang ada di database YouTube (Misal: ada ribuan video dengan tag *"lagu A"*, *"cover lagu A"*, *"parodi lagu A"*).
* **Value ($V$):** Konten video fisik aslinya yang akan lo tonton setelah lo mengeklik hasil pencarian tersebut.

Mekanisme *Self-Attention* melakukan hal yang sama di dalam kalimat. **Setiap satu token** di dalam kalimat akan dibekali dengan tiga vektor ini ($Q, K, V$) yang didapatkan dari hasil perkalian vektor input dengan matriks bobot ($W_Q, W_K, W_V$) yang dipelajari selama *training*.

---

### 2. Langkah Perhitungan Mekanis (Rumus Papan Tulis)

Di papan tulis, lo wajib menuliskan rumus legendaris *Scaled Dot-Product Attention* ini:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}_k}\right)V$$

Mari kita bedah jalannya angka di rumus ini langkah demi langkah:

* **Langkah 1: Skor Hubungan ($QK^T$)**
Setiap token bertindak sebagai *Query* ($Q$), lalu dia melakukan perkalian *dot-product* dengan *Key* ($K$) milik seluruh token lain di dalam kalimat.
* *Contoh:* Token `"benci"` ($Q$) mengalikan dirinya dengan $K$ milik token `"kamu"`, $K$ milik token `"ras"`, dan $K$ milik dirinya sendiri. Hasil perkalian ini berupa skor angka mentah yang menunjukkan **seberapa besar hubungan/perhatian** antara kata tersebut dengan kata lainnya.


* **Langkah 2: Penskalaan ($\div \sqrt{d}_k$)**
Skor mentah tadi dibagi dengan akar kuadrat dari dimensi vektor *key* ($\sqrt{d}_k$). Ini murni trik matematika agar angka hasil perkalian tidak terlalu meledak besar, sehingga nilai gradiennya tetap stabil saat proses *training*.
* **Langkah 3: Normalisasi Probabilitas ($\text{softmax}$)**
Matriks skor tadi dimasukkan ke fungsi **Softmax**. Fungsinya adalah mengubah skor mentah menjadi nilai bobot probabilitas perhatian (*attention weights*) yang jika dijumlahkan totalnya bernilai 1.
* *Visualisasi di papan tulis:* Dari langkah ini, model tahu bahwa ketika kata `"benci"` muncul, dia harus memberikan bobot perhatian sebesar **0.7** ke kata `"ras"`, dan hanya **0.1** ke kata `"kamu"`.


* **Langkah 4: Mengalikan dengan Nilai Semantik ($V$)**
Terakhir, bobot probabilitas perhatian tadi dikalikan dengan vektor *Value* ($V$). Kata-kata yang mendapatkan bobot *attention* tinggi (seperti kata `"ras"`) akan mendominasi nilai vektor output baru ini. Hasil akhirnya adalah representasi vektor baru untuk kata `"benci"` yang sudah **terkontaminasi secara cerdas oleh makna kontekstual** dari kata `"ras"`.

---

### 3. Apa itu Multi-Head Attention?

Di draf teori lo, tertulis bahwa IndoBERT menggunakan **Multi-Head Attention**.

* **Logikanya gampang:** Daripada model cuma menghitung rumus $Q, K, V$ di atas sebanyak satu kali, model memecah dimensinya dan melakukan kalkulasi rumus tersebut sebanyak $h$ kali secara paralel (IndoBERT memiliki 12 *heads*).
* **Kenapa harus repot digandakan?** Agar model bisa fokus pada hubungan yang berbeda dalam satu waktu. *Head* 1 bisa jadi fokus melihat hubungan subjek-objek, *Head* 2 fokus pada hubungan tata bahasa/sintaksis, dan *Head* 3 fokus pada hubungan emosi/sentimen makian. Informasi dari ke-12 *heads* ini lalu digabungkan menjadi satu kesatuan internal yang sangat kaya informasi semantik.

---

### ⏱️ UJIAN PAPAN TULIS: SESI SELF-ATTENTION

Mari kita uji pemahaman lo tentang matematika representasi ini. Jawab 2 skenario pertanyaan ini:

* **Pertanyaan 1:** "Sanju, coba tuliskan rumus *Scaled Dot-Product Attention* di papan tulis, lalu jelaskan secara runtun alur logis dari kiri ke kanan bagaimana operasi matematika pada matriks $Q, K,$ dan $V$ tersebut akhirnya bisa menghasilkan sebuah vektor kata yang kaya akan konteks!"
* **Pertanyaan 2:** "Kenapa kita membutuhkan komponen *Multi-Head Attention* di IndoBERT? Apa ruginya jika kita hanya menggunakan *Single-Head Attention* biasa dalam mendeteksi variasi ujaran kebencian?"

---

**Tulis jawaban lo secara terpisah.** Kita evaluasi di section khusus setelah ini. Tenang, kuasai polanya pelan-pelan. *Gas, Ju!*
### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, ini gila! Lo bener-bener paham alur mekanisnya dari angka nol. Penjelasan lo tentang alasan pembagian $\sqrt{d}_k$ demi stabilitas *backpropagation* (gradien tidak *vanishing* atau *exploding*) dan bagaimana *softmax* mengunci total bobot menjadi 1 itu adalah jawaban kelas kakap. Penguji bakal langsung tahu kalau lo paham sampai ke jeroan algoritma komputasinya.

Mari kita rapikan sedikit diksi dan struktur bahasanya agar terdengar sangat akademis saat lo jelaskan sambil coret-coret di papan tulis:

---

#### **Nomor 1: Bedah Alur Rumus Scaled Dot-Product Attention**

Jawaban lo sudah **sangat presisi**. Di papan tulis, lo bisa perjelas visualisasi dimensinya seperti ini untuk membuat penguji terpukau:

* **$Q \times K^T$ (Dot-Product):** Jika kalimat lo berisi $n$ token, perkalian matriks *Query* ($n \times d_k$) dengan *Key Transpose* ($d_k \times n$) akan menghasilkan **Matriks Persegi berukuran $n \times n$**. Matriks ini berisi skor interaksi mentah antar-setiap token.
* **Penskalaan ($\div \sqrt{d}_k$):** Mengontrol nilai varians dari hasil *dot-product* agar tidak terlalu besar saat dimensi vektor ($d_k$) meningkat, menjaga fungsi *softmax* tetap berada di wilayah gradien yang responsif selama proses pembaruan bobot (*backpropagation*).
* **$\text{softmax}(\dots)$:** Mengubah skor jarak mentah menjadi distribusi probabilitas (*attention weights*) di mana setiap baris bernilai positif dan totalnya tepat 1.
* **Dikalikan dengan $V$ (Value):** Matriks probabilitas ($n \times n$) dikalikan dengan matriks *Value* ($n \times d_v$), menghasilkan matriks output baru ($n \times d_v$). Di sinilah setiap token mendapatkan representasi numerik baru yang nilainya sudah terboboti secara kontekstual oleh token-token penting di sekitarnya.

---

#### **Nomor 2: Urgensi Multi-Head Attention**

Jawaban lo **100% Benar**. Penggunaan istilah "sudut pandang" itu sangat intuitif dan tepat secara ilmiah.

* **Cara formal menyampaikan ke penguji:** *"Izin menjelaskan Pak/Bu, jika kita hanya menggunakan Single-Head Attention, model hanya memiliki satu ruang proyeksi linear untuk mempelajari hubungan antar-kata, sehingga representasi yang dihasilkan cenderung rata (homogen). Dengan **Multi-Head Attention (12 heads)**, model dapat memproyeksikan Query, Key, dan Value ke 12 ruang representasi yang berbeda secara paralel. Hal ini memungkinkan setiap head fokus pada hubungan semantik yang berbeda dalam satu waktu—misalnya, Head 1 fokus pada struktur subjek-objek, Head 2 fokus pada kata makian implisit/sarkasme, dan Head 3 fokus pada target entitas kelompok. Penggabungan ke-12 sudut pandang ini menghasilkan representasi akhir yang jauh lebih kaya akan konteks sosiolinguistik media sosial."*

---

### 🏆 STATUS: LULUS TOTAL ARSITEKTUR UTAMA INDOBERT!

Lo sudah berhasil menembus sampai ke dasar mekanisme bagaimana IndoBERT memproses teks menjadi vektor yang kaya akan konteks dua arah. Sekarang posisi kita berada di ujung Cabang B (Jalur Modern).

Sesuai kesepakatan alur belajar bercabang kita, kita berhenti dulu di section khusus ini. Langkah berikutnya adalah **pertemuan kedua cabang di Batang Utama (Output Layer & Fungsi Loss untuk Multilabel)**, sebelum kita masuk ke daun optimasi keputusan (*Threshold Tuning*).

Silakan berikan konfirmasi atau instruksi **"Gas Batang Utama"** jika lo sudah siap menyatukan jalur klasik dan modern ini ke tahap kalkulasi keputusan akhir skripsi lo!