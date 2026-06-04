**PENGEMBANGAN MODEL DETEKSI UJARAN KEBENCIAN _MULTILABEL_ PADA MEDIA SOSIAL BERBAHASA INDONESIA DENGAN PENDEKATAN _EXPLAINABLE AI_ (XAI)**

**SKRIPSI**

diajukan untuk menempuh ujian sarjana

pada Fakultas Matematika dan Ilmu Pengetahuan Alam  
Universitas Padjadjaran

SANJUKIN NDUBE PINEM

NPM 140810220050


UNIVERSITAS PADJADJARAN  
FAKULTAS MATEMATIKA DAN ILMU PENGETAHUAN ALAM  
PROGRAM STUDI TEKNIK INFORMATIKA  
SUMEDANG  
2026

ABSTRAK

Meningkatnya ujaran kebencian di media sosial Indonesia menuntut adanya sistem deteksi otomatis yang tidak hanya akurat, tetapi juga transparan dalam memberikan alasan logis di balik setiap keputusan prediksi melalui identifikasi kontribusi kata. Sebagian besar penelitian saat ini masih terbatas pada klasifikasi biner dan model yang bersifat _black-box_. Penelitian ini mengembangkan model deteksi ujaran kebencian _multilabel_ menggunakan IndoBERT untuk mengidentifikasi kategori individu, kelompok, agama, ras, fisik, dan gender secara simultan. Untuk mengatasi ketidakseimbangan data, dilakukan optimasi _threshold_ per label. Metode _Shapley Additive Explanations_ (SHAP) dengan varian KernelSHAP diimplementasikan secara mandiri guna memberikan penjelasan transparan mengenai kontribusi kata terhadap keputusan model.

Hasil eksperimen menunjukkan bahwa IndoBERT mencapai performa terbaik dengan _Macro_ _F1-score_ sebesar 0,69 (_Micro_ F1: 0,73, _Hamming Loss_: 0,0493), mengungguli model klasik (CC-SVM) yang memperoleh 0,61. Penyesuaian _threshold_ terbukti krusial dalam meningkatkan sensitivitas terhadap label minoritas seperti kategori fisik (+12,7%) dan gender (+6,5%). Analisis SHAP membuktikan bahwa IndoBERT mampu menangkap konteks semantik yang lebih dalam melalui mekanisme _self-attention_ yang memahami hubungan antar-token secara dua arah. Hal ini berbeda dengan model klasik yang bersifat leksikal karena hanya mengandalkan pembobotan frekuensi kata (TF-IDF) tanpa mempertimbangkan struktur dan makna kalimat secara utuh. Penelitian ini diintegrasikan ke dalam aplikasi _web_ berbasis Streamlit sebagai prototipe sistem deteksi yang transparan (menyediakan visualisasi alasan prediksi) serta dapat dipertanggungjawabkan (memungkinkan audit dan verifikasi keputusan model oleh pengguna).

**Kata Kunci**: Ujaran Kebencian, Klasifikasi _Multilabel_, IndoBERT, _Explainable AI_, KernelSHAP

_ABSTRACT_

_The rise of hate speech on Indonesian social media requires an automated detection system that is not only accurate, but also transparent in providing logical reasons for every prediction by identifying word contributions. Most current studies are still limited to binary classification and black-box models. This research develops a multi-label hate speech detection model using IndoBERT to identify categories of individual, group, religion, race, physical, and gender simultaneously. To handle data imbalance, per-label threshold optimization was applied. The Shapley Additive Explanations (SHAP) method, specifically KernelSHAP, was implemented independently to provide clear explanations of how words contribute to the model's decisions. Experimental results show that IndoBERT achieves the best performance with a Macro F1-score of 0.69 (Micro F1: 0.73, Hamming Loss: 0.0493), outperforming the classic model (CC-SVM) which achieved 0.61. Threshold adjustment proved crucial in increasing sensitivity for minority labels, with the most significant improvements observed in the physical category (+12.7%) and gender category (+6.5%). SHAP analysis demonstrates that IndoBERT can capture deeper semantic context through a self-attention mechanism that understands bidirectional relationships between tokens. This contrasts with classic lexical models that rely solely on word frequency weighting (TF-IDF) without considering the complete structure and meaning of sentences. This research is integrated into a Streamlit-based web application as a prototype of a transparent detection system (providing visualizations of prediction reasons) and an accountable one (allowing users to audit and verify model decisions)._

**_Keywords:_** _Hate Speech, Multilabel Classification,_ IndoBERT*, Explainable AI,* KernelSHAP

DAFTAR ISI

ABSTRAK v

_ABSTRACT_ vi

DAFTAR ISI vii

DAFTAR TABEL x

DAFTAR GAMBAR xi

DAFTAR LAMPIRAN xiii

BAB I PENDAHULUAN 1

1.1 Latar Belakang 1

1.2 Identifikasi Masalah 3

1.3 Batasan Masalah 4

1.4 Maksud dan Tujuan 5

1.5 Manfaat Penelitian 5

1.6 Metodologi Penelitian 6

1.7 Sistematika Penulisan 8

BAB II TINJAUAN PUSTAKA 10

2.1 Ujaran Kebencian 10

2.2 Klasifikasi _Multilabel_ 11

2.2.1 Konsep Dasar Klasifikasi _Multilabel_ 12

2.2.2 Pendekatan Problem Transformation 14

2.2.3 Representasi Label dan _Dataset_ _Multilabel_ 15

2.3 _Machine Learning_ Klasik untuk Deteksi Teks 17

2.3.1 Logistic Regression 17

2.3.2 Support Vector Machine 17

2.4 Pendekatan _Deep Learning_ untuk Deteksi Ujaran Kebencian 18

2.4.1 Transformer dan Self-attention 18

2.4.2 Mekanisme Self-attention dan Multi-Head Attention 20

2.4.3 Model BERT dan IndoBERT 23

2.5 Explainable Artificial Intelligence (XAI) 25

2.5.1 Shapley Value 26

2.5.2 Shapley Additive Explanations (SHAP) 28

2.5.3 Algoritma KernelSHAP 30

2.6 Evaluasi pada Klasifikasi _Multilabel_ 32

2.6.1 Precision, Recall, dan F1-score 33

2.6.2 Pendekatan Averaging: Micro, Macro, dan Weighted F1 34

2.6.3 Hamming Loss 36

2.7 Penelitian Terdahulu 37

BAB III ANALISIS DAN PERANCANGAN 40

3.1 Analisis Kebutuhan 40

3.1.1 Kebutuhan Data 40

3.1.2 Kebutuhan Perangkat Lunak 41

3.1.3 Kebutuhan Perangkat Keras 42

3.2 Analisis Data 43

3.2.1 Struktur dan Skema Label _Dataset_ 43

3.2.2 Karakteristik Distribusi Label 44

3.2.3 Karakteristik Teks 47

3.2.4 Implikasi Analisis Data terhadap Perancangan Sistem 49

3.3 Perancangan Sistem dan Model 49

3.3.1 Rencana Tahapan Penelitian 50

3.3.2 Pra-pemrosesan Teks 52

3.3.3 Representasi Fitur (_Encoding_) 54

3.3.4 Perancangan Model Klasifikasi 55

3.3.4.1 Logistic Regression (BR dan CC) 55

3.3.4.2 Support Vector Machine (BR dan CC) 56

3.3.4.3 Model IndoBERT _Multilabel_ 57

3.3.5 Perancangan _Threshold_ Tuning 58

3.3.6 Perancangan Mekanisme Interpretasi 59

3.4 Perancangan Antarmuka Pengguna 62

3.4.1 Alur Interaksi Pengguna 62

3.4.2 Rancangan Visualisasi Hasil dan Interpretasi SHAP 65

3.5 Perancangan Eksperimen 67

3.5.1 Pembagian Data 67

3.5.2 Model yang Diuji 68

3.5.3 Skenario Eksperimen 68

3.5.4 _Explainable AI_ dalam Eksperimen 69

3.6 Perancangan Evaluasi 70

BAB IV HASIL DAN PEMBAHASAN 71

4.1 Hasil _Preprocessing_ Data Teks 71

4.1.1 Contoh Perubahan Teks Sebelum dan Sesudah _Preprocessing_ 71

4.1.2 Dampak _Preprocessing_ terhadap Panjang Teks 73

4.1.3 Ringkasan Hasil _Preprocessing_ 74

4.2 Hasil Kinerja Model Klasifikasi _Multilabel_ 75

4.2.1 Hasil Model _Machine learning_ Klasik (_Threshold_ Default) 75

4.2.2 Hasil Model _Machine learning_ Klasik (_Threshold_ _Tuned)_ 77

4.2.3 Hasil Model IndoBERT 79

4.2.4 Analisis Performa Per Label 83

4.2.5 Analisis Kesalahan Model 85

4.3 Analisis Interpretabilitas Model 91

4.3.1 Analisis Studi Kasus: Ujaran Kebencian Eksplisit 92

4.3.2 Analisis Studi Kasus Ujaran Kebencian Kontekstual 98

4.4 Implementasi Aplikasi _Web_ 103

4.4.1 Lingkungan Pengembangan dan Teknologi 103

4.4.2 Antarmuka Pengguna dan Mekanisme Prediksi 104

4.4.3 Visualisasi Hasil Prediksi _Multilabel_ 105

4.4.4 Penjelasan Kontribusi Kata (SHAP) 106

BAB V KESIMPULAN DAN SARAN 109

5.1 Kesimpulan 109

5.2 Saran 111

DAFTAR PUSTAKA 112

LAMPIRAN 115

DAFTAR TABEL

[Tabel 2.1 Perbandingan Penelitian Terdahulu 38](#_Toc228954791)

[Tabel 3.1 Struktur _Dataset_ yang akan digunakan 43](#_Toc228954792)

[Tabel 3.2 Statistik Distribusi Label dan Rasio Ketidakseimbangan 45](#_Toc228954793)

[Tabel 3.3 Konfigurasi Model Logistic Regression 56](#_Toc228954794)

[Tabel 3.4 Konfigurasi Model SVM 56](#_Toc228954795)

[Tabel 3.5 Konfigurasi Model IndoBERT 57](#_Toc228954796)

[Tabel 4.1 Contoh Teks Sebelum dan Sesudah _Preprocessing_ 72](#_Toc228954797)

[Tabel 4.2 Statistik Panjang Teks Sebelum dan Sesudah _Preprocessing_ 74](#_Toc228954798)

[Tabel 4.3 Hasil Evaluasi Model Klasik dengan _Threshold_ Default (0,5) 75](#_Toc228954799)

[Tabel 4.4 _Threshold_ Optimal per Label pada Model Klasik 77](#_Toc228954800)

[Tabel 4.5 Perbandingan Kinerja Model Klasik Sebelum dan Sesudah _Threshold_ Tuning 78](#_Toc228954801)

[Tabel 4.6 Hasil Evaluasi IndoBERT dengan _Threshold_ Default 80](#_Toc228954802)

[Tabel 4.7 Hasil Evaluasi IndoBERT Setelah _Threshold_ Tuning 82](#_Toc228954803)

[Tabel 4.8 Perbandingan _F1-score_ Per Label pada Model Klasik dan IndoBERT 83](#_Toc228954804)

[Tabel 4.9 Contoh Kesalahan _False_ _Positive_ dan _False_ _Negative_ Per Label 87](#_Toc228954805)

[Tabel 4.10 Contoh Kalimat Uji (Setelah _Preprocessing_) 92](#_Toc228954806)

[Tabel 4.11 Hasil Prediksi _Multilabel_ pada Kalimat Uji 92](#_Toc228954807)

[Tabel 4.12 Contoh Kalimat Uji Kontekstual 98](#_Toc228954808)

[Tabel 4.13 Hasil Prediksi Kalimat Uji Kontekstual 98](#_Toc228954809)

DAFTAR GAMBAR

[Gambar 2.1 Arsitektur _Transformer_ (Vaswani et al., 2017) 19](#_Toc228954889)

[Gambar 2.2 Scaled Dot-Product Attention dan _Multi-Head Attention_ 22](#_Toc228954890)

[Gambar 2.3 Representasi Input pada BERT (Devlin et al., 2019) 24](#_Toc228954891)

[Gambar 3.1 Distribusi Frekuensi Label Ujaran Kebencian 44](#_Toc228954892)

[Gambar 3.2 Matriks Korelasi Pearson Antar Label _Dataset_ 46](#_Toc228954893)

[Gambar 3.3 Visualisasi _Word Cloud_ Kosakata Dominan pada _Dataset_ 48](#_Toc228954894)

[Gambar 3.4 Tahapan Penelitian 50](#_Toc228954895)

[Gambar 3.5 Diagram Alur Sistem 63](#_Toc228954896)

[Gambar 4.1 Distribusi Panjang Teks Sebelum dan Sesudah _Preprocessing_ 73](#_Toc228954897)

[Gambar 4.2 Heatmap Confusion Matrix model IndoBERT dengan _Threshold_ _Tuned_ 86](#_Toc228954898)

[Gambar 4.3 Visualisasi KernelSHAP CC-SVM untuk Label HS_Individual 93](#_Toc228954899)

[Gambar 4.4 Visualisasi KernelSHAP CC-SVM untuk Label HS_Religion 94](#_Toc228954900)

[Gambar 4.5 Visualisasi KernelSHAP IndoBERT untuk Label HS_Individual 95](#_Toc228954901)

[Gambar 4.6 Visualisasi KernelSHAP IndoBERT untuk Label HS_Religion 96](#_Toc228954902)

[Gambar 4.7 Visualisasi KernelSHAP CC-SVM untuk Label HS_Race 99](#_Toc228954903)

[Gambar 4.8 Visualisasi KernelSHAP CC-SVM untuk Label HS_Group 100](#_Toc228954904)

[Gambar 4.9 Visualisasi KernelSHAP IndoBERT untuk Label HS_Race 101](#_Toc228954905)

[Gambar 4.10 Visualisasi KernelSHAP IndoBERT untuk Label HS_Group 101](#_Toc228954906)

[Gambar 4.11 Tampilan Antarmuka Halaman Utama 105](#_Toc228954907)

[Gambar 4.12 Tabel Hasil Prediksi dan Status Deteksi pada Aplikasi 106](#_Toc228954908)

[Gambar 4.13 Visualisasi Text Highlighting dan Plot Batang SHAP Per Label 108](#_Toc228954909)

# DAFTAR LAMPIRAN

[Lampiran 1 Kode Algoritma KernelSHAP untuk Model Klasik 115](#_Toc220923041)

[Lampiran 2 Kode Algoritma KernelSHAP untuk Model IndoBERT 118](#_Toc220923042)

[Lampiran 3 Kode Implementasi Antarmuka _Web_ Menggunakan Streamlit 120](#_Toc220923043)

#

PENDAHULUAN

Bab ini memberikan gambaran umum penelitian yang mencakup latar belakang permasalahan deteksi ujaran kebencian multilabel di media sosial Indonesia, identifikasi dan batasan masalah yang diteliti, tujuan dan manfaat penelitian, metodologi yang digunakan, serta sistematika penulisan skripsi.

## Latar Belakang

Dalam satu dekade terakhir, penggunaan media sosial di Indonesia meningkat sangat pesat. Platform seperti Twitter, Instagram, dan Facebook menjadi wadah utama interaksi publik dan pertukaran informasi. Berdasarkan laporan dari We Are Social 2025, pengguna aktif media sosial di Indonesia pada awal tahun 2025 mencapai 143 juta orang, atau sekitar 50.2% dari jumlah populasi nasional (Meltwater, 2025). Ledakan penggunaan ini menghasilkan volume data teks yang sangat besar dan bersifat tidak terstruktur, yang sulit diolah secara manual.

Ujaran kebencian (_hate speech_) didefinisikan sebagai ekspresi yang menyerang atau merendahkan individu atau kelompok berdasarkan atribut tertentu seperti agama, etnis, ras, gender, atau orientasi tertentu (Fortuna & Nunes, 2018). Di Indonesia, fenomena ini meningkat terutama pada momen politik. Hasil pemantauan AJI dan Monash University Indonesia selama Pilkada 2024 menunjukkan bahwa 11,23% dari 185.083 percakapan mengandung ujaran kebencian. Peningkatan drastis ini memperlihatkan dampak _hate speech_ terhadap polarisasi sosial dan potensi konflik di dunia nyata.

Deteksi manual tidak lagi praktis mengingat volume teks yang masif, tingginya kecepatan percakapan, serta keragaman bahasa informal, _slang_, dan kesalahan ketik. Penilaian manusia yang bersifat subjektif juga berisiko menimbulkan inkonsistensi label. Kompleksitas ini membuat kebutuhan akan sistem otomatis berbasis kecerdasan buatan (AI) semakin penting untuk memastikan proses deteksi berlangsung cepat dan konsisten.

Berbagai pendekatan _machine learning_ dan _deep learning_ telah dikembangkan untuk mendeteksi ujaran kebencian. Namun, sebagian besar penelitian masih berfokus pada klasifikasi biner (_hate speech_ vs _non-hate speech_), sehingga belum mampu menangani kasus di mana satu teks dapat memiliki lebih dari satu kategori kebencian sekaligus. Mayoritas model juga bersifat _black-box_ yang sulit menjelaskan alasan di balik prediksi. Keterbatasan ini menunjukkan perlunya model _multilabel_ yang tidak hanya akurat, tetapi juga mampu memberikan penjelasan interpretatif yang dapat dipahami oleh pengguna.

Dalam pengembangan model _machine learning_, penggunaan _dataset_ standar (_benchmark_) sangat krusial untuk memastikan validitas pengukuran performa dan perbandingan dengan penelitian terdahulu. Oleh karena itu, penelitian ini menggunakan _Indonesian Abusive and Hate Speech Dataset_ (Ibrohim & Budi, 2019) sebagai rujukan utama. Meskipun dinamika topik media sosial terus berubah hingga tahun 2026, _dataset_ ini dipilih karena merupakan _dataset_ publik rujukan (_benchmark_) utama di Indonesia yang menyediakan anotasi _multilabel_ terstruktur dengan validasi ahli, sehingga memungkinkan evaluasi arsitektur model dilakukan secara objektif tanpa bias yang mungkin muncul dari _dataset_ baru yang belum teruji.

Penelitian ini bertujuan mengatasi keterbatasan tersebut dengan mengembangkan model deteksi ujaran kebencian _multilabel_ yang mampu mengidentifikasi lebih dari satu kategori kebencian secara bersamaan. Pendekatan yang digunakan meliputi beberapa metode _multilabel_ seperti _Binary Relevance_ dan _Classifier Chains_ (CC), disertai optimisasi _threshold_ per label untuk meningkatkan akurasi prediksi. Penelitian ini juga mengintegrasikan pendekatan _Explainable_ AI (XAI) menggunakan KernelSHAP untuk memberikan penjelasan kontribusi kata terhadap keputusan model. Model yang dihasilkan diharapkan tidak hanya akurat, tetapi juga transparan dan dapat dipahami oleh pengguna, serta diimplementasikan ke dalam aplikasi _web_.

## Identifikasi Masalah

Berdasarkan latar belakang yang telah dijelaskan sebelumnya, masalah yang akan dicari solusinya dalam penelitian ini adalah sebagai berikut:

Bagaimana membangun model deteksi ujaran kebencian _multilabel_ berbahasa Indonesia yang mampu mengidentifikasi lebih dari satu kategori kebencian dalam satu teks secara bersamaan?

Bagaimana pengaruh penerapan berbagai pendekatan _multilabel_, seperti _Binary Relevance_ dan _Classifier Chains_ (CC), serta _threshold_ _tuning_ per label, terhadap peningkatan performa model pada _dataset_ ujaran kebencian berbahasa Indonesia?

Bagaimana menerapkan metode _Explainable_ AI (XAI), khususnya KernelSHAP, untuk menjelaskan kontribusi token atau kata terhadap prediksi model _multilabel_ sehingga hasil prediksi dapat dipahami oleh pengguna?

Bagaimana mengevaluasi performa model _multilabel_ baik dari sisi akurasi prediksi (_Macro_/_Micro F1_, _Hamming Loss_) maupun kualitas interpretasi hasil prediksi berbasis SHAP?

## Batasan Masalah

Agar penelitian ini lebih terarah dan fokus, maka batasan masalah yang ditetapkan adalah sebagai berikut:

- Data yang digunakan adalah _Indonesian Abusive and Hate Speech Dataset_ (Ibrohim & Budi, 2019). Pemilihan _dataset_ ini didasarkan pada ketersediaan anotasi _multilabel_ yang baku untuk teks berbahasa Indonesia, yang memungkinkan fokus penelitian ditekankan pada pengujian efektivitas arsitektur model dan metode interpretasi (XAI).
- Penelitian difokuskan pada deteksi ujaran kebencian dengan pendekatan _multilabel classification_, sehingga model dapat mengenali berbagai kategori kebencian yang berbeda (misalnya agama, ras, dan gender).
- Model yang diteliti mencakup dua kelompok pendekatan, yaitu model _machine learning_ klasik (_Logistic Regression_ dan _Support Vector Machine_) sebagai _baseline_, serta model _deep learning_ _Transformer_/IndoBERT sebagai pembanding kinerja.
- Pendekatan _Explainable AI_ dibatasi pada metode SHAP, khususnya KernelSHAP, untuk memberikan interpretasi berbasis kontribusi fitur. Penelitian tidak membahas metode XAI lainnya seperti LIME, _Integrated Gradients_, Grad-CAM.
- Implementasi sistem dibatasi pada prototipe aplikasi _web_ untuk demonstrasi, tanpa mencakup aspek _deployment_.

## Maksud dan Tujuan

Penelitian ini bertujuan mengembangkan model deteksi _multilabel_ untuk ujaran kebencian pada teks berbahasa Indonesia di media sosial dengan menggunakan pendekatan _Explainable AI_ (XAI). Penerapan pendekatan ini bertujuan menghasilkan prediksi yang akurat sekaligus memberikan transparansi bagi pengguna mengenai bobot fitur yang paling memengaruhi hasil klasifikasi, sehingga dasar pengambilan keputusan model dapat dipahami secara jelas.

Tujuan yang ingin dicapai oleh penulis dari penelitian ini adalah:

- Membangun model deteksi ujaran kebencian yang mampu menangani lebih dari satu kategori secara bersamaan (_multilabel_) pada teks berbahasa Indonesia.
- Menerapkan pendekatan _Explainable AI_ untuk memberikan interpretasi terhadap hasil prediksi model deteksi ujaran kebencian _multilabel_.
- Mengevaluasi performa model baik dari segi akurasi maupun keterjelasan interpretasi.

## Manfaat Penelitian

Manfaat yang diharapkan dari penelitian ini adalah sebagai berikut.

- Memberikan kontribusi pada pengembangan metode klasifikasi _multilabel_ untuk teks berbahasa Indonesia, khususnya pada domain ujaran kebencian.
- Menyediakan analisis mengenai kinerja pendekatan _multilabel_ seperti _Binary Relevance_, _Classifier Chains_ (CC), serta _threshold_ _tuning_.
- Menghasilkan model deteksi ujaran kebencian _multilabel_ yang dapat digunakan sebagai alat bantu monitoring konten media sosial.
- Menyediakan sistem interpretasi model (XAI) yang membantu analis, moderator, maupun pemangku kepentingan memahami alasan di balik prediksi model.
- Menyediakan prototipe aplikasi _web_ interaktif yang dapat digunakan untuk menguji model secara langsung

## Metodologi Penelitian

- Pengumpulan Data. Data yang digunakan adalah _Indonesian Abusive and Hate Speech Dataset_ (Ibrohim & Budi, 2019) yang berisi teks media sosial dengan anotasi _multilabel_ mencakup kategori individu, kelompok, agama, ras, fisik, dan gender. _Dataset_ ini dipilih karena menyediakan struktur label _multilabel_ yang sesuai dengan tujuan penelitian.
- Pra-pemrosesan Teks. Tahap ini meliputi pembersihan teks, normalisasi karakter, penanganan _slang_ dan kata tidak baku, serta tokenisasi. Tujuan dari tahapan ini adalah memastikan konsistensi representasi teks sebelum proses pelatihan model.
- Representasi Fitur. Representasi teks dalam penelitian ini dilakukan menggunakan dua pendekatan utama. Untuk model _machine learning_ klasik digunakan TF-IDF n-grams, yang mengubah teks menjadi bobot numerik berdasarkan frekuensi kata dan kemunculan dalam dokumen lain. Representasi ini efektif menangkap kata maupun frasa pendek yang sering muncul dalam ujaran kebencian.
- Penelitian mengembangkan dan membandingkan dua kelompok model, yaitu _Machine learning_ Klasik, yang mana kelompok ini menggunakan _Logistic Regression_ dan _Support Vector Machine_. Untuk menangani _multilabel_ digunakan dua pendekatan, yaitu _Binary Relevance_ dan _Classifier Chains_ (CC). Selain itu, dilakukan _threshold_ _tuning_ per label untuk meningkatkan kualitas prediksi _multilabel_ yang sering dipengaruhi ketidakseimbangan data. Yang kedua adalah _Deep learning_, dimana kelompok ini menggunakan model berbasis _Transformer_ yaitu IndoBERT. Model ini dipilih sebagai representatif dari pendekatan _deep learning_ kontekstual karena kemampuan representasi dua arah (_bidirectional_) yang superior dalam menangkap nuansa semantik bahasa Indonesia, terutama pada teks informal media sosial. IndoBERT digunakan sebagai pembanding untuk melihat sejauh mana pendekatan representasi kontekstual mampu meningkatkan performa dibandingkan model klasik berbasis TF-IDF.
- Evaluasi model. Evaluasi menggunakan metrik _multilabel_, termasuk _Hamming Loss_, serta _Micro_ dan _Macro F1-score_. Proses validasi dilakukan menggunakan pembagian data train-test dan _validation_ _split_ untuk _threshold_ _tuning_.
- Implementasi _Explainable AI_. Metode KernelSHAP digunakan untuk memberikan interpretasi terhadap kontribusi token/kata pada setiap label. Tahap ini bertujuan meningkatkan transparansi dan interpretabilitas model _multilabel_.
- Pengembangan Aplikasi _Web_. Model terbaik diintegrasikan ke dalam prototipe aplikasi berbasis _web_ untuk memfasilitasi pengujian prediksi dan visualisasi SHAP secara interaktif.
- Analisis dan penarikan kesimpulan. Tahap ini meliputi analisis performa model, perbandingan antara pendekatan klasik dan _deep learning_, serta evaluasi kualitas interpretasi berbasis XAI.

## Sistematika Penulisan

Sistematika penulisan skripsi ini disusun sebagai berikut.

**BAB I PENDAHULUAN**

Bab ini berisi uraian mengenai latar belakang penelitian, identifikasi dan perumusan masalah, batasan masalah, tujuan penelitian, manfaat penelitian, metodologi penelitian, serta sistematika penulisan. Bab ini memberikan gambaran umum mengenai arah dan ruang lingkup penelitian.

**BAB II TINJAUAN PUSTAKA**

Bab ini memaparkan teori dan penelitian terdahulu yang relevan, meliputi konsep ujaran kebencian, klasifikasi _multilabel_, metode _Binary Relevance_ dan _Classifier Chains_ (CC), teknik representasi teks (TF-IDF dan _embedding_), model _machine learning_ dan _deep learning_ untuk pemrosesan bahasa alami, konsep _Explainable AI_ (XAI), serta metode SHAP.

**BAB III ANALISIS DAN PERANCANGAN**

Bab ini menjelaskan tahapan penelitian, mulai dari pengumpulan data, pra-pemrosesan teks, representasi fitur, pengembangan model _multilabel_, optimisasi _threshold_ per label, evaluasi model menggunakan metrik _multilabel_, implementasi KernelSHAP sebagai metode interpretasi, serta perancangan aplikasi _web_ sebagai antarmuka pengguna.

**BAB IV HASIL DAN PEMBAHASAN**

Bab ini menyajikan hasil eksperimen dan evaluasi model, perbandingan performa berbagai pendekatan (BR, CC, dan _Transformer_), hasil interpretasi menggunakan KernelSHAP untuk setiap label, serta implementasi dan pengujian aplikasi _web_. Pembahasan dilakukan untuk menganalisis temuan, kelebihan, dan kendala dari model yang dikembangkan.

**BAB V KESIMPULAN DAN SARAN**

Bab ini berisi kesimpulan utama dari penelitian serta saran untuk pengembangan lebih lanjut, baik dari sisi model _multilabel_, metode XAI, maupun implementasi sistem.

#
