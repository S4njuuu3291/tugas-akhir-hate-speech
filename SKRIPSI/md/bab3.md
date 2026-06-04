

ANALISIS DAN PERANCANGAN

## Analisis Kebutuhan

Analisis kebutuhan dilakukan untuk memastikan bahwa seluruh proses penelitian, mulai dari pengolahan data, pembangunan model klasifikasi _multilabel_, penerapan _Explainable AI_ (XAI), hingga integrasi ke dalam aplikasi _web_, dapat berjalan secara sistematis dan sesuai dengan tujuan penelitian. Analisis ini mencakup kebutuhan data, kebutuhan perangkat lunak, serta kebutuhan perangkat keras yang digunakan.

### Kebutuhan Data

Penelitian ini menggunakan data berupa teks media sosial berbahasa Indonesia yang mengandung ujaran kebencian (_hate speech_) dengan skema klasifikasi _multilabel_. _Dataset_ yang digunakan bersumber dari penelitian yang diperkenalkan oleh (Ibrohim & Budi, 2019), yang terdiri dari kumpulan cuitan (tweet) pengguna media sosial Twitter.

Setiap data teks dapat memiliki lebih dari satu label ujaran kebencian, sehingga permasalahan yang dihadapi termasuk ke dalam klasifikasi _multilabel_. Label yang digunakan dalam penelitian ini mencakup ujaran kebencian yang ditujukan kepada individu, kelompok, agama, ras, kondisi fisik, dan gender. Dengan karakteristik tersebut, data tidak dapat diperlakukan sebagai klasifikasi tunggal (_single-label classification_), melainkan memerlukan pendekatan khusus yang mampu menangani lebih dari satu label secara simultan.

Selain itu, _dataset_ memiliki karakteristik bahasa yang bersifat informal, tidak baku, serta mengandung singkatan, bahasa gaul (_slang_), dan variasi ejaan yang umum ditemukan pada media sosial. Distribusi label pada _dataset_ juga tidak seimbang (_imbalanced_), di mana beberapa kategori ujaran kebencian memiliki jumlah data yang jauh lebih sedikit dibandingkan kategori lainnya. Karakteristik ini mempengaruhi pemilihan metode pra-pemrosesan, representasi fitur, model klasifikasi, serta strategi evaluasi yang digunakan dalam penelitian.

### Kebutuhan Perangkat Lunak

Perangkat lunak yang digunakan dalam penelitian ini berfungsi untuk mendukung seluruh tahapan pengolahan data, pembangunan model _machine learning_ dan _deep learning_, evaluasi performa, penerapan _Explainable AI_, serta pengembangan aplikasi _web_. Adapun perangkat lunak yang digunakan adalah sebagai berikut:

- **Python,** digunakan sebagai bahasa pemrograman utama karena memiliki ekosistem pustaka yang luas dan mendukung pengolahan data, pembelajaran mesin, serta pengembangan aplikasi secara terpadu.
- **Scikit-learn**, digunakan untuk proses pra-pemrosesan data, ekstraksi fitur berbasis TF-IDF, pembangunan model _machine learning_ klasik seperti _Logistic Regression_ dan _Support Vector Machine_, serta evaluasi performa model _multilabel_.
- **TensorFlow,** digunakan sebagai _framework_ _deep learning_ untuk membangun dan melatih model _Transformer_ berbasis IndoBERT.
- **Hugging Face _Transformers_**, digunakan untuk memanfaatkan model pra-latih IndoBERT beserta tokenizer WordPiece yang mendukung pemrosesan bahasa alami berbahasa Indonesia.
- **Pandas dan NumPy**, digunakan untuk manipulasi data tabular, pengolahan label _multilabel_, serta operasi numerik selama proses eksperimen.
- **Matplotlib dan Seaborn**, digunakan untuk visualisasi data, distribusi label, serta hasil evaluasi model pada tahap analisis dan pembahasan.
- **Streamlit**, digunakan sebagai _framework_ untuk membangun aplikasi _web_ interaktif yang berfungsi sebagai antarmuka pengguna dalam menguji model serta menampilkan hasil interpretasi XAI.
- **Kaggle,** digunakan sebagai lingkungan komputasi berbasis cloud untuk menjalankan eksperimen, melatih model _deep learning_, serta memanfaatkan GPU tanpa perlu konfigurasi perangkat keras secara lokal.

### Kebutuhan Perangkat Keras

Kebutuhan perangkat keras ditetapkan agar proses penelitian, khususnya pelatihan model _deep learning_, dapat berjalan dengan lancar dan efisien. Perangkat keras yang digunakan dalam penelitian ini meliputi:

- Laptop/PC.

- Prosesor minimal Intel Core i5 atau setara.
- RAM minimal 8 GB.
- Penyimpanan SSD minimal 128 GB

- GPU yang tersedia melalui platform Kaggle untuk mempercepat proses pelatihan model _deep learning_.
- Koneksi internet yang stabil diperlukan untuk mengakses _dataset_, mengunduh model pra-latih, serta menjalankan eksperimen pada lingkungan _cloud-based_.

## Analisis Data

Analisis data pada penelitian ini dilakukan untuk memperoleh pemahaman awal mengenai struktur dan karakteristik _dataset_ yang digunakan, sehingga dapat menjadi dasar dalam perancangan metode klasifikasi _multilabel_ dan sistem pendukung yang dikembangkan. Analisis ini tidak bertujuan untuk mengevaluasi kinerja model, melainkan untuk mengidentifikasi aspek-aspek data yang berpengaruh terhadap pemilihan pendekatan pemodelan dan evaluasi.

### Struktur dan Skema Label _Dataset_

_Dataset_ yang digunakan terdiri dari data teks berbahasa Indonesia yang berasal dari media sosial Twitter dan telah dianotasi ke dalam beberapa kategori ujaran kebencian. Setiap data teks direpresentasikan oleh satu kolom teks utama dan sejumlah kolom label biner yang menunjukkan keberadaan kategori ujaran kebencian tertentu.

Skema anotasi yang digunakan bersifat _multilabel_, di mana satu teks dapat memiliki lebih dari satu label secara simultan. Dengan demikian, permasalahan yang dihadapi dalam penelitian ini tidak dapat diselesaikan menggunakan pendekatan klasifikasi tunggal, melainkan memerlukan metode yang mampu menangani prediksi beberapa label dalam satu proses klasifikasi.

Tabel 3.1 Struktur _Dataset_ yang akan digunakan

| **Kolom**       | **Deskripsi**                            |
| --------------- | ---------------------------------------- |
| _Tweet_         | Teks _tweet_ yang menjadi data utama     |
| _HS_Individual_ | _Hate speech_ yang ditujukan ke individu |
| _HS_Group_      | _Hate speech_ yang ditujukan ke kelompok |
| _HS_Religion_   | _Hate speech_ berdasarkan agama          |
| _HS_Race_       | _Hate speech_ berdasarkan ras            |
| _HS_Physical_   | _Hate speech_ berdasarkan kondisi fisik  |
| _HS_Gender_     | _Hate speech_ berdasarkan gender         |

### Karakteristik Distribusi Label

Analisis mendalam terhadap distribusi label dilakukan untuk memahami tingkat ketidakseimbangan (_class imbalance_) dan hubungan antar kategori yang dapat mempengaruhi kinerja model.


Gambar 3.1 Distribusi Frekuensi Label Ujaran Kebencian

Eksplorasi pada 13.169 data menemukan bahwa distribusi label sangat tidak seimbang. Berdasarkan Gambar 3.1, label HS*Individual mendominasi \_dataset* dengan 3.575 sampel, sedangkan label HS*Gender dan HS_Physical merupakan kelas minoritas dengan jumlah sampel di bawah 350. Statistik detil mengenai distribusi dan rasio ketidakseimbangan (\_Imbalance Ratio*) disajikan pada Tabel 3.2.

Tabel 3.2 Statistik Distribusi Label dan Rasio Ketidakseimbangan

| Nama Label    | Jumlah Data (Count) | Persentase (%) | Imbalance Ratio |
| ------------- | ------------------- | -------------- | --------------- |
| HS_Individual | 3575                | 27.15          | 1.00            |
| HS_Group      | 1986                | 15.08          | 1.80            |
| HS_Religion   | 793                 | 6.02           | 4.51            |
| HS_Race       | 566                 | 4.30           | 6.32            |
| HS_Physical   | 323                 | 2.45           | 11.07           |
| HS_Gender     | 306                 | 2.32           | 11.68           |

Tabel di atas menunjukkan tantangan signifikan pada label minoritas. Label HS*Gender memiliki Imbalance Ratio sebesar 11,68, yang berarti untuk setiap 11 data HS_Individual, hanya terdapat 1 data HS_Gender. Ketimpangan ekstrem ini menjadi dasar perlunya penerapan strategi \_Threshold* Tuning (Subbab 3.3.5) agar model tidak bias terhadap kelas mayoritas.

Selain frekuensi, karakteristik _multilabel_ diukur menggunakan metrik Label Cardinality dan Label Density.

- Label Cardinality tercatat sebesar 0.5732, yang menunjukkan rata-rata jumlah label aktif per dokumen sangat rendah. Hal ini dipengaruhi oleh banyaknya data bersih (tanpa label) yang mencapai 7.608 sampel (57,77%).
- Label Density sebesar 0.0955, mengindikasikan bahwa matriks label bersifat _sparse_ (jarang), di mana frekuensi kemunculan label relatif rendah dibandingkan total kemungkinan label yang ada.

Selanjutnya, untuk mengetahui ketergantungan antar kategori, dilakukan analisis korelasi Pearson pada seluruh _Dataset_. Hasil visualisasi _heatmap_ korelasi disajikan pada Gambar 3.2.


Gambar 3.2 Matriks Korelasi Pearson Antar Label _Dataset_

Hasil pada Gambar 3.2 menunjukkan bahwa secara umum, korelasi antar label tergolong rendah hingga sedang.

- Korelasi Positif: Nilai tertinggi ditemukan antara label HS_Group dengan HS_Religion (0.36) dan HS_Race (0.34). Hal ini mengindikasikan bahwa ujaran kebencian berbasis agama dan ras sering kali ditujukan kepada kelompok.
- Korelasi Negatif: Terdapat korelasi negatif lemah antara HS_Individual dan HS_Group (-0.26), yang menunjukkan pola bahwa serangan cenderung fokus pada satu target spesifik (individu atau kelompok), jarang keduanya sekaligus.
- Independensi: Sebagian besar pasangan label lainnya memiliki nilai korelasi mendekati nol.

Rendahnya tingkat korelasi ini memberikan wawasan awal bahwa pendekatan yang sangat bergantung pada ketergantungan label (seperti _Classifier Chains_) mungkin tidak akan memberikan peningkatan performa yang signifikan dibandingkan pendekatan transformasi masalah standar (_Binary Relevance_), mengingat label-label dalam _dataset_ ini cenderung bersifat independen.

### Karakteristik Teks

Analisis karakteristik teks dilakukan untuk memahami ragam bahasa dan topik dominan yang terdapat dalam _dataset_. Secara linguistik, _dataset_ ini didominasi oleh ragam bahasa informal khas media sosial Twitter, termasuk penggunaan singkatan, istilah _slang_, serta variasi ejaan tidak baku. Karakteristik ini menjadi pertimbangan penting dalam perancangan tahap pra-pemrosesan, terutama dalam menentukan strategi normalisasi kata tidak baku dan penanganan variasi ejaan yang umum ditemukan pada ekspresi ujaran kebencian di platform media sosial.

Untuk memetakan topik utama dalam ujaran kebencian, dilakukan visualisasi frekuensi kata menggunakan _Word Cloud_ yang disajikan pada Gambar 3.3.


Gambar 3.3 Visualisasi _Word Cloud_ Kosakata Dominan pada _Dataset_

Visualisasi pada Gambar 3.3 mengungkap bahwa _dataset_ ini memiliki konsentrasi kosakata yang tinggi pada topik polarisasi politik dan sentimen SARA. Hal ini terindikasi dari kemunculan kata-kata dominan sebagai berikut:

- Entitas Politik: Kata "indonesia", "presiden", "jokowi", dan "pemerintah" muncul dengan ukuran paling besar. Hal ini menandakan bahwa mayoritas ujaran kebencian dalam _dataset_ tidak berdiri sendiri, melainkan terbalut dalam narasi kritik politik atau serangan terhadap penyelenggara negara.
- Sentimen Ideologi dan SARA: Terdapat penggunaan masif kata-kata yang menyerang identitas kelompok tertentu. Kata "komunis", "asing", "cina", "kafir", dan "islam" terlihat sangat menonjol, mengonfirmasi bahwa isu ideologi dan agama menjadi bahan bakar utama dalam penyebaran ujaran kebencian.
- Diksi Merendahkan (Dehumanisasi): Selain topik besar, _dataset_ juga dipenuhi dengan kata-kata makian spesifik seperti "cebong", "kampret", "anjing", dan "babi". Kemunculan kata-kata ini mengindikasikan tingginya tingkat agresivitas dan toksisitas bahasa yang digunakan oleh pengguna.

### Implikasi Analisis Data terhadap Perancangan Sistem

Berdasarkan struktur _multilabel_, distribusi label yang tidak seimbang, serta karakteristik bahasa informal pada _Dataset_, penelitian ini dirancang menggunakan pendekatan klasifikasi _multilabel_ dengan kombinasi model _machine learning_ klasik dan model _deep learning_.

Model klasik digunakan sebagai _baseline_ untuk memberikan gambaran awal kemampuan klasifikasi, sementara model _deep learning_ berbasis _Transformer_ dipilih untuk menangkap konteks semantik yang lebih kompleks. Selain itu, kondisi data mendorong penggunaan metrik evaluasi _multilabel_ seperti _Micro F1_, _Macro F1_, dan _Hamming Loss_, yang lebih representatif dibandingkan metrik akurasi tunggal.

## Perancangan Sistem dan Model

Subbab ini menjelaskan perancangan sistem klasifikasi ujaran kebencian _multilabel_ yang dikembangkan dalam penelitian, mencakup alur pemrosesan data teks, pemodelan, interpretasi hasil menggunakan _Explainable AI_, serta integrasi sistem ke dalam aplikasi _web_. Perancangan difokuskan pada bagaimana setiap komponen sistem saling terhubung untuk menghasilkan prediksi _multilabel_ yang dapat diinterpretasikan.

### Rencana Tahapan Penelitian

Penelitian ini dirancang sebagai sebuah rangkaian proses ilmiah yang terstruktur untuk menjawab tantangan deteksi ujaran kebencian _multilabel_ yang akurat dan transparan. Rencana penelitian ini disusun dalam sebuah _pipeline_ yang mengintegrasikan pengolahan data mentah hingga menjadi sebuah sistem yang dapat dipertanggungjawabkan keputusannya. Seluruh tahapan penelitian ini secara sistematis diilustrasikan dalam Gambar 3.4 dan mencakup komponen utama sebagai berikut:


Gambar 3.4 Tahapan Penelitian

- Tahap Akuisisi dan Analisis Data Penelitian diawali dengan penggunaan _Indonesian Abusive and Hate Speech Dataset_ yang bersumber dari penelitian Ibrohim & Budi (2019). Pada tahap ini, dilakukan analisis mendalam terhadap karakteristik teks informal media sosial dan struktur label _multilabel_ guna menentukan strategi pemodelan yang tepat.
- Tahap Pra-pemrosesan Teks secara Terpadu Data mentah diproses melalui rangkaian mekanisme pembersihan (_cleaning_), normalisasi huruf, serta penanganan bahasa informal dan _slang_ . Tahapan ini direncanakan untuk memastikan teks memiliki representasi yang seragam sebelum memasuki unit pemodelan, sehingga dapat mereduksi tingkat _noise_ pada data.
- Tahap Pengembangan dan Komparasi Model Rencana pemodelan dilakukan dengan membagi jalur eksperimen menjadi dua kelompok utama:
  - Unit Baseline Klasik: Menggunakan ekstraksi fitur TF-IDF untuk melatih model _Logistic Regression_ dan SVM .
  - Unit Kontekstual _Deep learning_: Menggunakan model IndoBERT yang memanfaatkan arsitektur _Transformer Encoder_ untuk menangkap makna semantik dua arah.
- Tahap Optimasi Keputusan (_Threshold Tuning_) Setelah model dikembangkan, dilakukan mekanisme penyesuaian ambang batas (_threshold_) pada setiap label secara independen. Hal ini direncanakan sebagai solusi teknis untuk mengatasi ketidakseimbangan distribusi data pada kategori ujaran kebencian tertentu, seperti kategori fisik dan gender.
- Tahap Analisis Interpretabilitas (XAI) Untuk memberikan transparansi ilmiah, penelitian ini mengimplementasikan algoritma KernelSHAP dari dasar (_from scratch_). Tahapan ini bertujuan untuk membedah "kotak hitam" model dengan mengkalkulasi kontribusi marjinal setiap kata terhadap label prediksi yang dihasilkan.
- Tahap Implementasi Antarmuka dan Visualisasi Seluruh hasil penelitian diintegrasikan ke dalam sebuah antarmuka berbasis _web_ menggunakan _Streamlit_. Rencana akhir ini mencakup penyajian hasil prediksi _multilabel_ yang telah dioptimasi serta visualisasi penjelasan SHAP agar hasil deteksi dapat dipahami dan diaudit oleh pengguna secara langsung.

### Pra-pemrosesan Teks

Tahap pra-pemrosesan teks dilakukan untuk menyiapkan data teks agar berada dalam bentuk yang lebih konsisten dan representatif sebelum diproses oleh model klasifikasi _multilabel_. Pra-pemrosesan ini bertujuan untuk mengurangi _noise_ pada data, menyamakan variasi penulisan bahasa informal, serta memastikan bahwa informasi semantik yang relevan dapat dipelajari secara efektif oleh model. Pra-pemrosesan dirancang secara umum agar dapat digunakan pada seluruh model yang dikembangkan, dengan penyesuaian pada tahap representasi fitur sesuai dengan karakteristik masing-masing model.

Tahapan pra-pemrosesan teks yang diterapkan dalam penelitian ini meliputi:

- Pembersihan Karakter Non-Relevan

Teks dibersihkan dari elemen yang tidak berkontribusi langsung terhadap makna ujaran, seperti URL, mention pengguna, simbol khusus, angka, serta tanda baca yang tidak diperlukan. Langkah ini bertujuan untuk mengurangi _noise_ yang dapat mempengaruhi proses pembelajaran model.

- Normalisasi Huruf

Seluruh teks dikonversi ke huruf kecil (lowercasing) untuk menghindari perbedaan representasi kata yang memiliki makna sama namun berbeda dalam penggunaan huruf besar dan kecil.

- Normalisasi Bahasa Informal

Mengingat data berasal dari media sosial, teks mengandung banyak singkatan, bahasa gaul, dan variasi ejaan. Oleh karena itu, dilakukan normalisasi bahasa informal ke bentuk kata yang lebih baku menggunakan kamus normalisasi, sehingga makna kata dapat dipahami secara lebih konsisten oleh model.

- Penanganan Pengulangan Karakter

Pengulangan karakter yang berlebihan, yang umum digunakan untuk penekanan emosi dalam teks media sosial, dikurangi agar tidak menghasilkan token yang tidak bermakna secara linguistik.

- Tokenisasi

Teks yang telah dibersihkan kemudian diproses melalui tahap tokenisasi. Proses tokenisasi disesuaikan dengan jenis model yang digunakan, seperti tokenisasi berbasis kata untuk model klasik dan tokenisasi subword untuk model berbasis _Transformer_.

Langkah-langkah pra-pemrosesan ini dirancang untuk mempertahankan makna semantik utama teks, sekaligus mengurangi variasi penulisan yang tidak perlu.

### Representasi Fitur (_Encoding_)

Tahap representasi fitur bertujuan untuk mengubah teks hasil pra-pemrosesan ke dalam bentuk numerik agar dapat diproses oleh model klasifikasi _multilabel_. Pemilihan metode representasi fitur disesuaikan dengan karakteristik model yang digunakan, karena setiap model memiliki mekanisme berbeda dalam memanfaatkan informasi dari data teks.

Dalam penelitian ini, dua pendekatan representasi fitur digunakan, yaitu representasi berbasis bag-of-words menggunakan TF-IDF untuk model _machine learning_ klasik, serta representasi berbasis token subword untuk model _Transformer_ IndoBERT. Penjelasan untuk pendekatan representasi fitur yang digunakan adalah sebagai berikut:

- Representasi Fitur untuk Model _Machine learning_ Klasik (TF-IDF)

Teks direpresentasikan ke dalam bentuk vektor numerik menggunakan metode _Term Frequency-Inverse Document Frequency_ (TF-IDF). Metode ini menghitung bobot setiap kata berdasarkan tingkat kepentingannya dalam dokumen relatif terhadap seluruh korpus. TF-IDF digunakan karena efektivitasnya dalam menangani data teks media sosial yang berdimensi tinggi dan bersifat _sparse_, sehingga memudahkan model linear seperti _Logistic Regression_ dan SVM untuk mengenali pola kata yang relevan dengan ujaran kebencian.

- Representasi Fitur untuk Model _Transformer_ (IndoBERT)

Teks diproses menggunakan pendekatan representasi berbasis token _subword_ melalui mekanisme tokenisasi _WordPiece_. Metode ini memecah teks menjadi unit-unit kecil yang memungkinkan model IndoBERT menangani istilah tidak baku, singkatan, dan variasi ejaan yang umum di media sosial. Hasilnya adalah representasi fitur kontekstual, di mana setiap token memiliki makna yang dipengaruhi oleh hubungan semantik dengan token lain dalam kalimat, memberikan pemahaman konteks yang lebih dalam dibandingkan model tradisional.

Pemilihan dua pendekatan representasi fitur dalam penelitian ini didasarkan pada kebutuhan untuk membandingkan model dengan kompleksitas dan kemampuan yang berbeda. TF-IDF digunakan untuk memberikan _baseline_ yang sederhana dan efisien, sementara IndoBERT digunakan sebagai model utama untuk menangkap konteks bahasa yang lebih dalam.

### Perancangan Model Klasifikasi

Subbab ini menjelaskan perancangan model klasifikasi _multilabel_ yang digunakan dalam penelitian, yang terdiri dari model _machine learning_ klasik sebagai _baseline_ dan model berbasis _Transformer_ IndoBERT sebagai model utama.

#### Logistic Regression (BR dan CC)

Model _Logistic Regression_ digunakan sebagai _baseline_ karena memiliki arsitektur yang sederhana, stabil pada data berdimensi tinggi, serta secara alami menghasilkan keluaran probabilistik yang sesuai untuk klasifikasi _multilabel_ berbasis _threshold_. Pada penelitian ini, _Logistic Regression_ diimplementasikan menggunakan dua mekanisme _multilabel_, yaitu _Binary Relevance_ dan _Classifier Chain_.

Tabel 3.3 Konfigurasi Model Logistic Regression

| **Konfigurasi**        | **Nilai**                 |
| ---------------------- | ------------------------- |
| Representasi fitur     | TF-IDF (unigram + bigram) |
| Dimensi fitur maksimum | 30.000                    |
| max_iter               | 2500                      |
| class_weight           | "balanced"                |
| random_state           | 42                        |

#### Support Vector Machine (BR dan CC)

Model _Support Vector Machine_ (SVM) digunakan karena kemampuannya dalam menangani data teks berukuran besar dengan representasi TF-IDF yang bersifat sparse. Model dasar yang digunakan adalah LinearSVC. Karena LinearSVC tidak menghasilkan probabilitas secara langsung, digunakan CalibratedClassifierCV untuk melakukan kalibrasi probabilitas menggunakan fungsi sigmoid. Kalibrasi probabilitas dilakukan menggunakan pendekatan sigmoid. Proses validasi untuk kalibrasi ini memanfaatkan data validasi terpisah (hold-out) yang telah dialokasikan, guna mencegah kebocoran data (data leakage) dari data uji utama. Probabilitas ini diperlukan untuk mendukung evaluasi _multilabel_, _threshold_ _tuning_, serta interpretasi model. Seperti _Logistic Regression_, SVM dikembangkan menggunakan pendekatan _Binary Relevance_ dan _Classifier Chain_.

Tabel 3.4 Konfigurasi Model SVM

| **Konfigurasi**        | **Nilai**                            |
| ---------------------- | ------------------------------------ |
| Representasi fitur     | TF-IDF (unigram + bigram)            |
| Dimensi fitur maksimum | 30.000                               |
| Base model             | LinearSVC                            |
| max_iter               | 5000                                 |
| class_weight           | "balanced"                           |
| Kalibrasi probabilitas | CalibratedClassifierCV               |
| Metode kalibrasi       | Sigmoid                              |
| Strategi Validasi      | Hold-out Split                       |
| Wrapper _multilabel_   | OneVsRestClassifier, ClassifierChain |

#### Model IndoBERT _Multilabel_

Model utama yang digunakan dalam penelitian ini adalah IndoBERT, yaitu model _Transformer_ pra-latih yang dirancang khusus untuk bahasa Indonesia. IndoBERT digunakan untuk menangkap representasi kontekstual teks secara lebih mendalam dibandingkan model klasik berbasis TF-IDF.

Teks yang telah ditokenisasi menggunakan _WordPiece_ diproses oleh _encoder_ _Transformer_. Representasi token khusus \[CLS\] digunakan sebagai representasi keseluruhan teks dan diteruskan ke lapisan klasifikasi _multilabel_. Untuk menangani klasifikasi _multilabel_, lapisan keluaran menggunakan aktivasi sigmoid, sehingga setiap label diprediksi secara independen. Fungsi kerugian yang digunakan adalah Binary Cross-Entropy, yang sesuai untuk permasalahan klasifikasi _multilabel_.

Tabel 3.5 Konfigurasi Model IndoBERT

| **Konfigurasi**          | **Nilai**                      |
| ------------------------ | ------------------------------ |
| Model pra-latih          | indobenchmark/indobert-base-p1 |
| Jumlah label             | 6                              |
| _Problem_ type           | _Multi-label classification_   |
| Fungsi aktivasi _output_ | Sigmoid                        |
| Fungsi loss              | Binary Cross-Entropy           |
| Optimizer                | Adam                           |
| _Learning_ rate          | 2 × 10⁻⁵                       |
| Panjang maksimum token   | 256                            |
| Epoch                    | 5                              |

### Perancangan _Threshold_ Tuning

Pada klasifikasi _multilabel_, keluaran model berupa probabilitas untuk setiap label sehingga diperlukan aturan keputusan untuk mengonversi probabilitas tersebut menjadi label biner. Proses ini dilakukan menggunakan mekanisme \_threshold_ing.

Penggunaan _threshold_ default sebesar 0,5 tidak selalu optimal, khususnya pada _dataset_ dengan distribusi label yang tidak seimbang. Pada kondisi tersebut, _threshold_ tetap berpotensi menurunkan kemampuan model dalam mendeteksi label minoritas. Oleh karena itu, dalam penelitian ini dirancang mekanisme _threshold_ _tuning_ untuk menentukan nilai _threshold_ yang lebih sesuai pada masing-masing label.

_Threshold_ _tuning_ dirancang dengan cara menguji beberapa nilai _threshold_ dalam rentang tertentu untuk setiap label, kemudian memilih nilai yang memberikan keseimbangan terbaik antara _precision_ dan _recall_ berdasarkan metrik evaluasi _multilabel_.

### Perancangan Mekanisme Interpretasi

Proses estimasi nilai kontribusi fitur untuk setiap label target dilakukan melalui tahapan-tahapan berikut:

- 1. _Sampling_ Subset Fitur

Dari sejumlah fitur aktif (kata atau token), dilakukan _sampling_ untuk membentuk matriks biner . Setiap baris dalam matriks merepresentasikan satu kombinasi subset:

- : Fitur ke- dipertahankan dalam subset.
- Fitur ke- di-masking (dihapus atau diganti).

Contoh: Untuk kalimat "Saya benci dia", terdapat fitur aktif. Matriks _sampling_ mungkin berbentuk:

- Subset 1 (Semua aktif):
- Subset 2 (Hanya kata "benci"):
- Subset 3 (Kosong/_Baseline_):
  - Evaluasi Prediksi Model

Setiap baris subset dalam matriks dipetakan kembali ke format _input_ asli model untuk dievaluasi. Hasil prediksi dari seluruh subset _sampling_ ini dikumpulkan ke dalam vektor kolom :

|     |     | (3.1) |
| --- | --- | ----- |

Contoh: Jika probabilitas label "Kekerasan" pada teks asli adalah 0.85 dan pada teks kosong adalah 0.05, maka vektor berisi kumpulan nilai probabilitas tersebut.

- 1. Penghitungan Matriks Bobot (Shapley Kernel)

Agar regresi linier menghasilkan estimasi yang setara dengan nilai Shapley, setiap subset diberikan bobot menggunakan Shapley Kernel. Seluruh bobot disusun ke dalam sebuah matriks diagonal , di mana elemen diagonalnya dihitung dengan rumus:

|     |     | (3.2) |
| --- | --- | ----- |

Di mana adalah jumlah fitur yang aktif (bernilai 1) pada baris tersebut. Bobot ini memastikan bahwa subset yang sangat kecil (mendekati _baseline_) atau sangat besar (mendekati asli) mendapatkan perhatian lebih besar dalam regresi.

- 1. Estimasi Nilai SHAP dengan Notasi Matriks (WLS)

Nilai SHAP dihitung dengan menyelesaikan permasalahan _Weighted_ Least Squares (WLS). Secara matematis, vektor kontribusi diperoleh melalui operasi matriks berikut:

|     |     | (3.3) |
| --- | --- | ----- |

Keterangan:

- : Vektor berisi nilai kontribusi (SHAP values) tiap fitur.
- : Matriks biner subset fitur berukuran .
- : Matriks diagonal bobot kernel berukuran .
- : Vektor hasil prediksi model pada setiap subset berukuran .

Contoh: Operasi ini menghasilkan vektor = , yang berarti kata kedua ("benci") memberikan kontribusi terbesar (0.7) terhadap peningkatan probabilitas label.

- 1. Mekanisme Masking dan Agregasi

Sistem menerapkan teknik _masking_ yang berbeda berdasarkan karakteristik model:

- Model Klasik (LR & SVM): Masking dilakukan dengan mengubah nilai pada vektor fitur TF-IDF menjadi 0. Nilai yang dihasilkan langsung merepresentasikan kontribusi kata tersebut.
- Model IndoBERT: Masking dilakukan dengan mengganti token _input_ menjadi token khusus \[MASK\]. Karena satu kata dapat terpecah menjadi beberapa sub-tokens, nilai SHAP akhir sebuah kata dihitung dengan menjumlahkan seluruh nilai dari token-token penyusunnya:

|     |     | (3.4) |
| --- | --- | ----- |

Contoh Agregasi: Jika kata "mematikan" dipecah menjadi token \["me", "##matikan"\] dengan nilai masing-masing 0.1 dan 0.3, maka kontribusi total kata "mematikan" adalah 0.4.

- 1. Visualisasi Interpretasi

Hasil perhitungan dari vektor kemudian divisualisasikan untuk mempermudah analisis:

- Kontribusi Positif (): Kata-kata yang mendukung model dalam memprediksi suatu label.
- Kontribusi Negatif (): Kata-kata yang menurunkan probabilitas model terhadap suatu label.

## Perancangan Antarmuka Pengguna

Perancangan antarmuka pengguna merupakan tahap krusial untuk memastikan bahwa kompleksitas model klasifikasi multilabel dapat disajikan dalam bentuk yang intuitif dan mudah dipahami oleh pengguna. Fokus utama dari perancangan ini adalah untuk menjembatani interaksi antara pengguna dengan fungsionalitas sistem, sehingga proses input hingga visualisasi hasil deteksi berjalan secara efektif. Selain aspek estetika, perancangan ini juga mengutamakan kejelasan penyajian informasi hasil analisis sentimen dan deteksi ujaran kebencian. Hal ini dilakukan untuk memberikan transparansi kepada pengguna mengenai bagaimana model memberikan keputusan label pada setiap _instance_ data yang diuji. Berikut adalah rincian mengenai alur interaksi serta elemen visual yang diimplementasikan dalam aplikasi.

### Alur Interaksi Pengguna

Alur interaksi pengguna dalam aplikasi ini dirancang secara sistematis untuk memandu pengguna mulai dari input teks hingga memperoleh hasil prediksi dan interpretasi model. Secara visual, tahapan proses yang terjadi di dalam sistem saat pengguna berinteraksi digambarkan pada Gambar 3.4.


Gambar 3.5 Diagram Alur Sistem

Penjelasan mengenai mekanisme interaksi tersebut adalah sebagai berikut:

- Tahap Masukan Teks (_Input_): Interaksi dimulai ketika pengguna memasukkan teks media sosial berbahasa Indonesia ke dalam area teks (_text area_) yang disediakan pada antarmuka halaman utama.
- Konfigurasi Parameter Analisis: Sebelum memulai proses deteksi, pengguna melakukan konfigurasi melalui panel kontrol yang disediakan:
  - Pemilihan Arsitektur Model: Pengguna memilih jenis model yang akan digunakan melalui tombol radio, yaitu antara model _Machine learning_ Klasik (Logistic Regression/SVM) atau model _Deep learning_ (IndoBERT).
  - Pengaturan Sampel SHAP: Pengguna menentukan jumlah sampel (\$n\$) yang akan digunakan dalam algoritma KernelSHAP. Fitur ini memberikan fleksibilitas kepada pengguna untuk menyeimbangkan antara kecepatan komputasi (sampel rendah) dan presisi hasil interpretasi (sampel tinggi).
- Eksekusi Pra-pemrosesan _Real-time_: Setelah parameter dikonfigurasi, pengguna menekan tombol "Mulai Analisis". Sistem secara otomatis memproses teks masukan melalui tahap pembersihan (_cleaning_) dan normalisasi untuk memastikan data yang masuk ke model bebas dari _noise_.
- Proses Prediksi dan Penerapan _Threshold_: Sistem melakukan inferensi menggunakan model yang dipilih untuk menghasilkan skor probabilitas. Skor ini kemudian dibandingkan dengan nilai ambang batas (_threshold_) optimal per label untuk menghasilkan keputusan klasifikasi biner (Ya/Tidak) yang akurat.
- Kalkulasi Interpretasi (XAI): Berdasarkan jumlah sampel yang telah ditentukan pengguna pada tahap konfigurasi, sistem menjalankan algoritma KernelSHAP untuk menghitung nilai kontribusi fitur terhadap label prediksi yang aktif.
- Visualisasi Hasil (_Output_): Sistem menampilkan hasil akhir dalam dua bentuk visualisasi utama:
  - Tabel Prediksi: Menampilkan status deteksi untuk setiap kategori ujaran kebencian.
  - Visualisasi Interpretasi: Menyajikan teks dengan sorotan warna (_highlight_) dan grafik batang (_bar plot_) yang menjelaskan kontribusi positif atau negatif setiap kata terhadap keputusan model.

### Rancangan Visualisasi Hasil dan Interpretasi SHAP

Rancangan visualisasi pada antarmuka pengguna difokuskan untuk menyajikan hasil prediksi model secara komprehensif dan transparan. Komponen visualisasi ini dirancang untuk menjembatani kesenjangan antara keluaran numerik model (probabilitas dan nilai Shapley) dengan pemahaman kognitif pengguna. Berdasarkan spesifikasi perancangan aplikasi, terdapat tiga elemen visual utama yang dikembangkan:

1\. Tabel Prediksi _Multilabel_

Antarmuka menampilkan hasil klasifikasi dalam format tabel yang menyandingkan label ujaran kebencian dengan keputusan model. Berbeda dengan klasifikasi standar yang hanya menampilkan kelas argmax, tabel ini menyajikan dua informasi krusial secara simultan:

- Skor Probabilitas Mentah: Menunjukkan tingkat keyakinan model (0.0 hingga 1.0) terhadap keberadaan setiap kategori ujaran kebencian (Individu, Kelompok, Agama, Ras, Fisik, dan Gender).
- Status Keputusan Biner: Menampilkan status "Ya" atau "Tidak" yang diturunkan dari mekanisme _threshold tuning_. Keputusan ini memastikan bahwa label hanya dinyatakan aktif jika probabilitasnya melampaui ambang batas optimal yang telah ditentukan untuk kategori tersebut.

2\. Visualisasi Sorotan Teks (_Text Highlighting_)

Untuk memberikan intuisi langsung mengenai lokasi kata yang memicu deteksi ujaran kebencian, sistem menerapkan visualisasi berbasis teks berwarna (_text highlighting_). Setiap kata atau token dalam kalimat input diberi latar warna berdasarkan nilai Shapley yang dihasilkan:

- Sorotan Merah (Kontribusi Positif): Mengindikasikan kata-kata yang mendorong model untuk memprediksi adanya ujaran kebencian (menaikkan probabilitas).
- Sorotan Biru (Kontribusi Negatif): Mengindikasikan kata-kata yang bersifat netral atau meredam prediksi ujaran kebencian (menurunkan probabilitas).
- Intensitas Warna: Opasitas atau ketebalan warna dirancang berbanding lurus dengan nilai absolut Shapley, sehingga kata dengan pengaruh paling kuat akan terlihat lebih mencolok dibandingkan kata dengan pengaruh lemah.

3\. Grafik Peringkat Fitur (_Feature Importance Plot_)

Sistem menyajikan grafik batang (_bar plot_) interaktif yang dibangun menggunakan pustaka Matplotlib. Grafik ini mengurutkan kata-kata berdasarkan besaran kontribusinya () terhadap prediksi label tertentu. Visualisasi ini memungkinkan pengguna untuk melihat secara kuantitatif kata mana yang menjadi faktor determinan utama (top-k features) dalam pengambilan keputusan model, baik yang berkontribusi positif maupun negatif.

## Perancangan Eksperimen

Perancangan eksperimen pada penelitian ini bertujuan untuk mengevaluasi kinerja model klasifikasi ujaran kebencian _multilabel_ yang telah dirancang, serta menganalisis interpretabilitas prediksi menggunakan _Explainable AI_. Seluruh eksperimen dirancang secara konsisten agar setiap model diuji dalam kondisi yang setara dan hasil yang diperoleh dapat dibandingkan secara adil.

### Pembagian Data

_Dataset_ dibagi menjadi tiga bagian utama, yaitu data latih, data validasi, dan data uji. Pembagian data dilakukan dengan proporsi sebagai berikut:

- Data latih (_training_) : 70%
- Data validasi (_validation_) : 10%
- Data uji (_testing_) : 20%

Data latih digunakan untuk melatih seluruh model klasifikasi _multilabel_. Data validasi digunakan untuk proses penyesuaian _threshold_ pada keluaran probabilitas model. Data uji digunakan untuk mengevaluasi kinerja akhir model serta melakukan analisis _Explainable AI_. Pembagian data dilakukan dengan mempertahankan distribusi label _multilabel_ agar setiap subset data tetap merepresentasikan karakteristik _dataset_ secara keseluruhan.

### Model yang Diuji

Eksperimen dilakukan terhadap beberapa model klasifikasi _multilabel_ dengan tingkat kompleksitas yang berbeda, yaitu:

- Model _machine learning_ klasik

- _Logistic Regression_ dengan _Binary Relevance_
- _Logistic Regression_ dengan _Classifier Chain_
- Calibrated _Support Vector Machine_ dengan _Binary Relevance_
- Calibrated _Support Vector Machine_ dengan _Classifier Chain_

- Model _deep learning_ (IndoBERT _multilabel_)

### Skenario Eksperimen

Eksperimen dirancang dalam beberapa skenario utama sebagai berikut:

- Evaluasi model _baseline_

Model _Logistic Regression_ dan _Support Vector Machine_ dievaluasi sebagai _baseline_ untuk memperoleh gambaran performa klasifikasi _multilabel_ berbasis fitur statistik.

- Evaluasi model IndoBERT

Model IndoBERT dievaluasi untuk mengukur kemampuan model berbasis konteks dalam menangani variasi bahasa informal dan memahami makna ujaran kebencian yang kompleks.

- Evaluasi dengan _threshold_ _tuning_

_Threshold_ _tuning_ diterapkan pada seluruh model untuk menyesuaikan nilai ambang keputusan pada setiap label, sehingga diperoleh keseimbangan yang lebih baik antara _precision_ dan _recall_.

Setiap skenario dievaluasi menggunakan prosedur dan metrik yang sama agar hasil eksperimen dapat dibandingkan secara konsisten.

### _Explainable AI_ dalam Eksperimen

_Explainable AI_ diterapkan pada seluruh model yang diuji dalam penelitian ini. Penerapan _Explainable AI_ bertujuan untuk memberikan interpretasi terhadap prediksi model dan memahami faktor-faktor yang mempengaruhi keputusan klasifikasi.

Pada model berbasis TF-IDF, _Explainable AI_ digunakan untuk mengidentifikasi kata atau fitur yang paling berkontribusi terhadap prediksi label. Pada model IndoBERT, _Explainable AI_ digunakan untuk menganalisis kontribusi token dalam konteks kalimat terhadap keluaran _multilabel_. Dengan menerapkan _Explainable AI_ pada seluruh model, penelitian ini tidak hanya mengevaluasi performa klasifikasi, tetapi juga membandingkan karakteristik interpretabilitas antara model klasik dan model berbasis _Transformer_.

## Perancangan Evaluasi

Evaluasi dalam penelitian ini dirancang untuk mengukur kinerja model klasifikasi ujaran kebencian _multilabel_ secara komprehensif. Pengukuran kinerja model dilakukan menggunakan metrik evaluasi _multilabel_ yang umum digunakan, yaitu _Micro F1-score_, _Macro F1-score_, dan _Hamming Loss_. _Micro F1-score_ digunakan untuk merepresentasikan performa sistem secara agregat terhadap seluruh label, sedangkan _Macro F1-score_ digunakan untuk memberikan gambaran kinerja model pada masing-masing label dengan mempertimbangkan maupun tanpa mempertimbangkan distribusi data. _Hamming Loss_ digunakan untuk mengukur tingkat kesalahan prediksi label secara keseluruhan pada klasifikasi _multilabel_.

Seluruh metrik evaluasi diterapkan secara konsisten pada semua model yang diuji, baik model _machine learning_ klasik maupun model berbasis _Transformer_. Evaluasi dilakukan pada data uji setelah proses pelatihan model dan penentuan _threshold_ selesai dilakukan. Dengan pendekatan evaluasi ini, kinerja model dapat dibandingkan secara langsung dan objektif, serta memberikan gambaran menyeluruh mengenai kemampuan sistem dalam menangani permasalahan klasifikasi ujaran kebencian _multilabel_.

#
