# Sesi #1: Kenapa Perlu Transformer? (Keterbatasan RNN/LSTM)

## A. Masalah dengan RNN/LSTM (Model Sebelum Transformer)

Sebelum Transformer (2017), model terbaik untuk bahasa adalah **RNN/LSTM**. Cara kerjanya:

> RNN membaca teks secara **berurutan**, kata per kata. Saat membaca kata ke-t, ia menyimpan "memori" (hidden state) dari kata-kata sebelumnya.

**Analoginya:** Membaca kalimat sambil bergumam "saya... saya suka... saya suka AI...". Setiap langkah hanya tahu konteks dari langkah sebelumnya.

## B. Tiga Kelemahan Utama RNN

| Kelemahan | Penjelasan | Dampak |
| :--- | :--- | :--- |
| **1. Pemrosesan Berurutan (Sequential)** | Tidak bisa memproses kata ke-5 sebelum selesai memproses kata 1-4. | Training **lambat** karena tidak bisa paralel. GPU tidak bisa dimanfaatkan maksimal. |
| **2. Vanishing Gradient** | Gradien dari kata jauh akan semakin kecil saat backpropagation melalui banyak time step. | Model **sulit belajar** hubungan antara kata yang jaraknya jauh. Contoh: "10 tahun yang lalu di Jakarta... ... ... saya tinggal di **kota itu**" — kata "Jakarta" dan "kota itu" bisa sangat berjauhan. |
| **3. Long-Range Dependency** | Berhubungan dengan poin 2. RNN secara teori bisa mengingat informasi lama, tapi praktiknya sulit. | Untuk kalimat panjang (100+ token), performa menurun drastis. |

## C. Solusi Transformer

Transformer memperkenalkan **Self-Attention** yang:

| Keunggulan | Penjelasan |
| :--- | :--- |
| **Paralelisasi penuh** | Semua token diproses bersamaan dalam satu matriks. Tidak perlu menunggu token sebelumnya. Training jadi jauh lebih cepat. |
| **Jarak berapa pun hanya 1 step** | Token ke-1 dan token ke-100 bisa berinteraksi langsung dalam satu kali hitung $QK^T$. Tidak ada masalah long-range dependency. |
| **Gradient lebih lancar** | Karena jalur pendek dari output ke input (lewat residual connection), gradient vanishing berkurang drastis. |

## D. Analogi Sederhana

| RNN | Transformer |
| :--- | :--- |
| Seperti membaca buku dengan jari: setiap kata hanya tahu kata sebelumnya | Seperti melihat seluruh halaman buku sekaligus: semua kata bisa saling "melihat" |
| Kalau buku tebal 1000 halaman, butuh waktu lama untuk bolak-balik cari hubungan | Semua kata dalam 1 halaman bisa langsung terhubung |

## E. Satu Kalimat Kesimpulan untuk Sidang

> *"Sebelum Transformer, model seperti RNN memproses teks secara berurutan, yang menyebabkan training lambat dan sulit menangkap hubungan antar kata yang jaraknya jauh. Transformer memperkenalkan mekanisme self-attention yang memungkinkan semua token diproses secara paralel dan langsung berinteraksi satu sama lain, terlepas dari jaraknya."*

---

## Checklist Sesi #1 (Harus Bisa)

| Pertanyaan | Jawaban Anda (coba jawab sendiri) |
| :--- | :--- |
| Apa kelemahan utama RNN? | (Sebutkan minimal 2) |
| Kenapa Transformer lebih cepat trainingnya? | |
| Apa keunggulan Transformer untuk hubungan kata jarak jauh? | |

---

## Latihan Sesi #1

1. **Tulis di kertas:** 3 kelemahan RNN + 3 keunggulan Transformer
2. **Ucapkan dengan suara keras:** kalimat kesimpulan di atas sampai lancar (30 detik)

---

**Setelah selesai, kirim pesan: "Sesi 1 selesai" + jawaban checklist di atas (atau tanya jika ada yang kurang paham).**

Saya akan lanjut ke **Sesi #2: Tokenisasi dan Embedding**.

