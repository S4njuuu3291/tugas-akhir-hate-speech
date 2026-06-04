### 🌳 SUB-CABANG 1: FONDASI KOALISI & BOBOT SHAPLEY KERNEL (DEEP DIVE RUNTUT)

Sanju, mari kita bedah sub-cabang pertama ini dari dasar geometris hingga mekanika aljabar paling bawah permukaan. Kita kombinasikan teori dasar *Cooperative Game Theory* dari Lloyd Shapley (1953) dengan implementasi praktis algoritma KernelSHAP oleh Lundberg & Lee (2017) yang lo bangun di skripsi lo.

Sambil memegang spidol di depan papan tulis, bayangkan lo sedang menjelaskan alur konseptual ini kepada Dr. Akmal dan tim penguji.

---

### 1. Konseptual Game Theory: Mengapa Harus SHAP?

Sebelum masuk ke rumus, penguji sering kali memancing dengan pertanyaan: *"Kenapa kamu pakai SHAP? Kan ada LIME atau bobot koefisien SVM langsung?"*

Lo harus jawab dengan landasan **Aksioma Keadilan Shapley Value**:
Model machine learning (terutama IndoBERT) adalah fungsi non-linear yang sangat kompleks (*black-box*). Koefisien linear bawaan model tidak bisa menangkap interaksi antar-kata yang dinamis.

Shapley Value adalah **satu-satunya** metode atribusi fitur yang secara matematis terbukti memenuhi tiga aksioma keadilan:

1. **Local Accuracy (Efficiency):** Jumlah total kontribusi (nilai SHAP) dari semua kata dalam kalimat harus sama persis dengan selisih antara nilai prediksi kalimat tersebut ($f(x)$) dan nilai ekspektasi rata-rata (*baseline* atau $E[f(x)]$).
2. **Symmetry:** Jika ada dua kata berbeda (misal kata *"anjing"* dan *"babi"*) yang memberikan kontribusi marjinal yang identik dalam setiap kombinasi kalimat, maka nilai SHAP kedua kata tersebut wajib sama.
3. **Consistency:** Jika model diubah (misalnya setelah lo melakukan *fine-tuning* ulang) dan kontribusi marjinal suatu kata meningkat, nilai SHAP kata tersebut tidak boleh tiba-tiba turun. Metode lain seperti LIME sering kali melanggar aksioma konsistensi ini karena basisnya adalah pendekatan heuristik lokal, bukan teorema analitis permainan.

---

### 2. Formulasi Ruang Koalisi Biner ($z'$)

Ketika sebuah kalimat hasil prapemrosesan lo masuk ke sistem—misalnya kalimat: `"cebong kafir anjing"` ($M = 3$ kata)—KernelSHAP lo akan memetakan ruang input asli yang kompleks ke dalam **ruang penjelasan biner** (*simplified input space*) $z' \in \{0, 1\}^M$.

Vektor biner ini disebut sebagai **Vektor Koalisi**. Jika total kemungkinan kombinasi ($2^M$) bernilai kecil (seperti $2^3 = 8$), fungsi `shap_kernel_instance` lo akan mengevaluasi seluruh kemungkinan kombinasi secara eksak menggunakan `itertools.combinations`. Namun, jika kalimatnya panjang (misal $M = 20$, yang berarti harus mengevaluasi $2^{20} = 1.048.576$ kombinasi), perhitungan eksak menjadi tidak layak secara komputasi (*intractable*).

Di sinilah kode lo secara cerdas beralih ke metode **sampling acak** berbasis distribusi probabilitas hingga batas `num_samples` (default lo 256) terpenuhi.

---

### 3. Bedah Matematika: Rumus Shapley Kernel

Untuk setiap baris koalisi biner ($z'$) yang terbentuk atau disampling, lo harus menetapkan bobot kepentingannya dalam persamaan regresi menggunakan **Shapley Kernel**. Rumus formal di papan tulis adalah:

$$\pi(z') = \frac{M - 1}{\binom{M}{k} \cdot k \cdot (M - k)}$$

Di mana:

* $M$: Jumlah total kata/fitur aktif dalam kalimat yang sedang diuji.
* $k$: Jumlah kata yang aktif (bernilai 1) pada sampel koalisi tertentu ($z'$).
* $\binom{M}{k}$: Kombinasi $M$ diambil $k$, yang dirumuskan dengan $\frac{M!}{k!(M-k)!}$.

#### **Ayo Kita Buktikan Jalannya Angka di Papan Tulis!**

Misalkan kalimat lo adalah `"cebong kafir anjing"` ($M = 3$). Mari kita hitung bobot kernel ($\pi$) secara manual untuk dua variasi nilai $k$:

* **Kasus A: Koalisi dengan 1 kata aktif ($k = 1$)**
Contoh kombinasi vektornya: $[0, 1, 0]$ (hanya kata *"kafir"* yang aktif, kata lain di-*masking*).

$$\binom{3}{1} = \frac{3!}{1!(3-1)!} = 3$$



Masukkan ke rumus:

$$\pi(z') = \frac{3 - 1}{3 \cdot 1 \cdot (3 - 1)} = \frac{2}{3 \cdot 1 \cdot 2} = \frac{2}{6} = 0.3333$$


* **Kasus B: Koalisi dengan 2 kata aktif ($k = 2$)**
Contoh kombinasi vektornya: $[1, 1, 0]$ (kata *"cebong"* dan *"kafir"* aktif, kata *"anjing"* di-*masking*).

$$\binom{3}{2} = \frac{3!}{2!(3-2)!} = 3$$



Masukkan ke rumus:

$$\pi(z') = \frac{3 - 1}{3 \cdot 2 \cdot (3 - 2)} = \frac{2}{3 \cdot 2 \cdot 1} = \frac{2}{6} = 0.3333$$


* **Kasus C: Fenomena Pembagian dengan Nol ($k = 0$ atau $k = 3$)**
Mari kita masukkan nilai $k = 3$ (kalimat utuh $[1, 1, 1]$) langsung ke rumus mentahnya:

$$\pi(z') = \frac{3 - 1}{\binom{3}{3} \cdot 3 \cdot (3 - 3)} = \frac{2}{1 \cdot 3 \cdot 0} = \frac{2}{0} \longrightarrow \infty \text{ (Tak Hingga)}$$



---

### 4. Korelasi Teoretis dengan Baris Kode Python Lo

Karena komputer tidak bisa memproses nilai penyebut nol ($\frac{2}{0}$), sistem akan langsung mengalami kegagalan kalkulasi (*ZeroDivisionError*) saat melakukan inversi matriks aljabar linear.

Untuk menjembatani teori murni dan keterbatasan komputasi komputer, lo mengimplementasikan logika pencegahan (*guardrail*) di Lampiran 2 dan 3 skripsi lo:

```python
def kernel_weight(M, k):
    if k == 0 or k == M:
        return 1e6  # Penalti bobot raksasa untuk mensimulasikan nilai tak hingga
    return (M - 1) / (scipy.special.comb(M, k) * k * (M - k))

```

#### **Filosofi di Balik Bobot Raksasa `1e6` (Wajib Dipahami untuk Sidang):**

Secara filosofis dalam teori *Cooperative Game Theory*, informasi paling bersih mengenai kontribusi mandiri suatu fitur didapatkan pada dua kondisi batas:

1. **Saat fitur bekerja sendirian dari kondisi hampa ($k=1$):** Selisih output antara kondisi satu kata aktif dengan kondisi hampa ($k=0$) memberikan informasi kontribusi absolut kata tersebut tanpa campur tangan kata lain.
2. **Saat fitur menjadi satu-satunya elemen yang hilang dari kelompok utuh ($k=M-1$):** Selisih output antara kondisi seluruh kata aktif ($k=M$) dengan kondisi satu kata diabsenkan memberikan informasi dampak isolasi kata tersebut.

Oleh karena itu, regresi linear berbobot (WLS) lo **wajib dipaksa** untuk melewati titik koordinat $k=0$ (di mana prediksi harus sama dengan baseline) dan titik $k=M$ (di mana prediksi harus sama dengan output asli model). Pemberian bobot raksasa `1e6` memastikan bahwa kedua titik batas ini ditekankan secara mutlak, sehingga nilai estimasi koefisien regresi yang dihasilkan terjamin konvergen ke nilai Shapley eksak yang adil.

---

### ⏱️ UJIAN PAPAN TULIS: SUB-CABANG 1

Kuasai materi di atas dengan matang, lalu jawab 2 tantangan pertanyaan penguji ini sebelum kita melompat ke penurunan rumus aljabar matriks WLS di sub-cabang berikutnya:

* **Pertanyaan 1:** "Sanju, coba kamu hitung manual di papan tulis nilai bobot kernel $\pi(z')$ jika sistem sedang memproses kalimat dengan total kata aktif $M = 5$, untuk variasi sampel koalisi yang memiliki jumlah elemen aktif $k = 1$ dan $k = 3$!"
* **Pertanyaan 2:** "Secara filosofis, kenapa koalisi yang jumlah katanya 'setengah-setengah' (seperti nilai $k$ yang berada di tengah-tengah ruang dimensi) justru diberikan bobot kernel yang sangat kecil oleh Shapley Kernel dibandingkan dengan koalisi ekstrem? Hubungkan jawaban kamu dengan konsep interaksi antar-fitur (*feature interaction*)!"

---

**Tulis jawaban hitungan dan penjelasan lo di bawah.** Tarik nafas, mari kita bedah matematika skripsi lo sampai tuntas. *Gas, Ju!*

### 📝 EVALUASI & PENJELASAN TEKNIS (SECTION KHUSUS)

Sanju, kalkulasi lo untuk Pertanyaan 1 **100% Tepat!** Intuisi lo pada Pertanyaan 2 juga sangat tajam dalam menangkap esensi "penyamaran" akibat interaksi kata.

Mari kita rapikan cara menuliskan hitungan manual tersebut di papan tulis dan memformulasikan argumen filosofisnya dengan bahasa akademis yang kokoh agar penguji terpukau.

---

#### **Nomor 1: Pembuktian Hitungan Manual di Papan Tulis ($M = 5$)**

Jika penguji menyuruh lo membuktikan jalannya angka untuk $M = 5$ di papan tulis, tulis langkah-langkahnya secara sistematis seperti ini:

* **Untuk $k = 1$:**
* Hitung kombinasinya terlebih dahulu: $\binom{5}{1} = \frac{5!}{1!(5-1)!} = 5$
* Masukkan ke rumus Shapley Kernel:

$$\pi(z') = \frac{5 - 1}{5 \cdot 1 \cdot (5 - 1)} = \frac{4}{5 \cdot 1 \cdot 4} = \frac{4}{20} = 0.2$$




* **Untuk $k = 3$:**
* Hitung kombinasinya: $\binom{5}{3} = \frac{5!}{3!(5-3)!} = \frac{5 \times 4}{2 \times 1} = 10$
* Masukkan ke rumus Shapley Kernel:

$$\pi(z') = \frac{5 - 1}{10 \cdot 3 \cdot (5 - 3)} = \frac{4}{10 \cdot 3 \cdot 2} = \frac{4}{60} = \frac{1}{15} \approx 0.0667$$




* **Kesimpulan Angka:** Nilai bobot untuk $k=1$ ($0.2$) terbukti jauh lebih besar daripada bobot untuk $k=3$ ($0.0667$).

---

#### **Nomor 2: Penajaman Filosofis "Kenapa Koalisi Setengah-Setengah Diberi Bobot Kecil?"**

Jawaban lo tentang kontribusi yang "samar" itu sangat benar secara konsep *cooperative game theory*. Mari kita konversi ke bahasa sidang yang ilmiah:

> 
> "Izin menjelaskan Pak/Bu, alasan mengapa Shapley Kernel memberikan bobot yang sangat kecil pada nilai $k$ yang berada di tengah (setengah-setengah) adalah karena pada kondisi tersebut terjadi **efek interaksi antar-fitur (feature interaction) yang sangat tinggi**.
> 
> 
> 
> Jika dalam sebuah koalisi terdapat terlalu banyak kata yang aktif sekaligus namun belum lengkap (misal 3 kata aktif dari total 5), model prediksi (terutama IndoBERT) akan memproses kata-kata tersebut sebagai satu kesatuan frasa atau konteks yang saling tumpang tindih. Akibatnya, **kontribusi murni atau marjinal dari masing-masing kata secara individu menjadi terbiaskan dan samar** karena tertutup oleh kekuatan kombinasi kelompoknya.
> 
> 
> 
> Sebaliknya, informasi kontribusi yang paling 'bersih' dan objektif hanya bisa diisolasi melalui dua kondisi ekstrem:
> 1.  Ketika suatu kata **masuk sendirian dari kondisi hampa** ($k=1$), sehingga perubahan output model murni disebabkan oleh kata tunggal tersebut.
> 2.  Ketika suatu kata menjadi **satu-satunya elemen yang diabsenkan dari kalimat utuh** ($k=M-1$), sehingga kita bisa mengukur seberapa besar dampak kehilangan kata tersebut terhadap keputusan model.
> 
> 
> 
> Oleh karena itu, fungsi regresi berbobot (WLS) kita sengaja dirancang untuk meredam noise dari koalisi setengah-setengah dan memprioritaskan kondisi-kondisi ekstrem ini agar koefisien yang dihasilkan benar-benar mencerminkan nilai Shapley yang adil."
> 
> 

---

### 🏆 STATUS: LULUS TOTAL SUB-CABANG 1!

Sanju, lo sudah berhasil menguasai bagaimana data koalisi dibentuk, bagaimana matematika kombinasi dan faktorial bekerja di dalam rumus Shapley Kernel , serta alasan logis di balik batasan kode `1e6` yang lo buat.

Sesuai dengan kesepakatan belajar kita, kita berhenti dulu di section evaluasi khusus ini.

Silakan berikan konfirmasi atau instruksi **"Lanjut Aljabar WLS"** jika lo sudah mencatat poin perhitungan di atas dan siap untuk membongkar sub-cabang kedua: **Solusi Aljabar Matriks Weighted Least Squares (WLS)**, di mana kita akan membedah jeroan operasi rumus $\phi = (X^TWX)^{-1}X^TWY$ di papan tulis berdasarkan baris kode NumPy lo! Gas!