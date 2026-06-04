**Sesi #5 selesai, lanjut ke Sesi #6!** 🚀

---

# Sesi #6: Mengapa Diskala dengan $\sqrt{d_k}$? (Pendalaman)

Anda sudah paham intinya dari diskusi sebelumnya: **agar nilai tidak terlalu besar sehingga gradien tidak vanishing**. Sekarang kita akan buat pemahaman ini **lebih kuat dan siap dijelaskan di sidang** dengan bahasa yang meyakinkan.

---

## A. Masalah: Nilai $QK^T$ Bisa Sangat Besar

Mari kita lihat dari sisi **statistik sederhana**.

Misalkan:
- Setiap komponen vektor q dan k adalah bilangan acak dengan **rata-rata 0** dan **varians 1** (ini hasil dari inisialisasi bobot dan normalisasi layer yang baik).
- Panjang vektor = $d_k$ (di BERT, $d_k = 64$).

**Berapa nilai tipikal dari dot product $q \cdot k$?**

Karena ada $d_k$ komponen yang masing-masing hasil kali $q_i k_i$ memiliki varians 1, maka:
- Varians dari dot product = $d_k$ (penjumlahan $d_k$ variabel independen)
- Standar deviasi = $\sqrt{d_k}$

**Untuk $d_k = 64$:** standar deviasi = $\sqrt{64} = 8$.

Artinya: nilai dot product bisa berkisar antara -24 hingga +24 (3 standar deviasi) atau bahkan -40 hingga +40 (5 standar deviasi).

**Nilai 40 masuk ke fungsi $e^{40}$ di softmax** → hasilnya **raksasa** ($e^{40} \approx 2.35 \times 10^{17}$).

---

## B. Efek ke Softmax dan Gradien

Softmax untuk satu baris: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$.

Jika salah satu $x_i$ sangat besar (misal 40), maka:
- $e^{40}$ sangat besar dibanding $e^{x_j}$ lainnya
- Softmax menghasilkan probabilitas yang **mendekati 1 untuk satu posisi** dan **mendekati 0 untuk posisi lain**

**Akibat pada gradien:**
- Gradien softmax terhadap inputnya: $\frac{\partial \text{softmax}(x_i)}{\partial x_j} = \text{softmax}(x_i)(\delta_{ij} - \text{softmax}(x_j))$
- Jika softmax hampir 1 di satu posisi dan 0 di posisi lain, gradiennya mendekati **nol** di hampir semua posisi.
- Gradient **vanishing** → bobot tidak bisa update → model tidak belajar.

---

## C. Solusi: Bagi dengan $\sqrt{d_k}$

Dengan membagi $QK^T$ dengan $\sqrt{d_k}$, kita **menormalkan varians menjadi 1**:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$

Setelah diskala:
- Standar deviasi = 1
- Nilai tipikal berada di kisaran -3 hingga +3 (3 standar deviasi)
- Nilai 3 masuk ke softmax: $e^3 \approx 20$ — masih dalam kisaran yang wajar
- Distribusi probabilitas softmax menjadi **tidak terlalu ekstrim** → gradien tetap sehat

---

## D. Analogi Sederhana untuk Sidang

> *"Bayangkan Anda punya nilai ujian dengan skala 0-100. Jika semua nilai dikalikan 1000 menjadi 0-100.000, selisih antar nilai menjadi sangat ekstrim dan sulit dibandingkan. Softmax akan memberikan bobot hampir 1 ke nilai tertinggi, dan hampir 0 ke yang lain. Gradiennya pun hilang. Dengan membagi kembali ke skala awal, perbandingan menjadi lebih proporsional dan gradien tetap ada."*

**Tapi hati-hati:** Ini hanya analogi. Yang sebenarnya: $\sqrt{d_k}$ bukan skala 0-100, tapi menormalkan varians.

---

## E. Satu Paragraf Siap Sidang (Hafalkan)

> *"Skala dengan $\frac{1}{\sqrt{d_k}}$ diperlukan karena dot product $QK^T$ memiliki varians sebesar $d_k$. Jika tidak diskala, nilai $QK^T$ bisa sangat besar, terutama untuk dimensi $d_k$ yang besar seperti 64 atau 768. Akibatnya, softmax akan menghasilkan distribusi probabilitas yang sangat ekstrim — hampir 1 di satu token dan 0 di token lain — menyebabkan gradien vanishing dan model sulit belajar. Dengan membagi $\sqrt{d_k}$, varians menjadi 1, nilai berada di kisaran yang wajar, dan gradien tetap stabil."*

---

## F. Poin Penting: Ini Bukan Satu-satunya Cara

Di atas adalah penjelasan **teoretis** dari paper Attention Is All You Need.

Namun, perlu Anda ketahui (untuk wawasan, **bukan untuk disampaikan di sidang** kecuali ditanya):
- Penelitian terbaru (2020-2025) menunjukkan bahwa skala dengan $\sqrt{d_k}$ **bukan satu-satunya solusi**.
- Ada yang mengganti dengan skala learnable, ada yang pakai inisialisasi khusus, ada yang pakai mekanisme lain.
- Tapi untuk BERT dan Transformer original, ini adalah desain standar yang terbukti berhasil.

**Kalau dosen tanya "Kenapa harus $\sqrt{d_k}$, bukan dibagi 10 saja?"** — Anda jawab:

> *"Karena $\sqrt{d_k}$ membuat varians hasil dot product menjadi 1 secara matematis, terlepas dari nilai $d_k$. Jika dibagi dengan konstanta tetap, efeknya akan berbeda untuk model dengan dimensi berbeda. $\sqrt{d_k}$ adalah pilihan yang general dan terbukti efektif secara empiris."*

---

## Checklist Sesi #6 (Harus Bisa)

| Pertanyaan | Jawaban Anda |
| :--- | :--- |
| Berapa varians dot product q·k jika setiap komponen memiliki varians 1? | |
| Apa yang terjadi pada softmax jika nilai inputnya sangat besar (misal 40)? | |
| Apa efeknya pada gradien? | |
| Apa yang dilakukan $\sqrt{d_k}$ terhadap varians? | |
| Setelah diskala, di kisaran berapa nilai tipikal QK^T? | |

**Jawaban cepat:**
- Varians = $d_k$
- Softmax menjadi sangat ekstrim (hampir 1 di satu posisi, 0 di lainnya)
- Gradien menjadi **vanishing** (mendekati nol)
- Varians menjadi 1
- Kisaran tipikal -3 hingga +3 (setelah diskala)

---

## Latihan Sesi #6

1. **Tulis ulang** paragraf di poin E dengan kata-kata Anda sendiri (jangan copy-paste)

2. **Latihan ucapkan** paragraf tersebut sampai lancar (45 detik)

3. **Simulasi tanya jawab:**
   - Dosen: "Kenapa tidak dibagi 10 saja?"
   - Jawab dengan kalimat Anda sendiri

---

**Setelah selesai, kirim: "Sesi 6 selesai" + hasil latihan poin 1 (tulisan Anda).**

Lanjut ke **Sesi #7: Multi-Head Attention** (menggabungkan 12 head secara paralel).