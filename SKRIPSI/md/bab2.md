
TINJAUAN PUSTAKA

Bab ini memberikan pemaparan landasan teori dan konsep yang berhubungan dengan penelitian ini seperti ujaran kebencian, klasifikasi multilabel, _machine learning_ klasik, _deep learning_ berbasis _Transformer_ (IndoBERT), _Explainable_ AI (_Shapley Value_ dan KernelSHAP), metrik evaluasi multilabel, serta _positioning_ terhadap penelitian terdahulu.

## Ujaran Kebencian

_Hate speech_ adalah bentuk ekspresi yang bertujuan untuk menyerang individu atau kelompok berdasarkan karakteristik tertentu misalnya agama, etnis, ras, gender, ataupun orientasi seksual. Menurut (Fortuna & Nunes, 2018), _hate speech_ didefinisikan sebagai bahasa yang menyerang (_attacks_) atau merendahkan (_diminishes_), yang memicu kekerasan atau kebencian terhadap kelompok tertentu, berdasarkan karakteristik spesifik seperti penampilan fisik, agama, keturunan, asal negara atau etnis, orientasi seksual, atau identitas gender. Penyebaran _hate speech_ di media sosial juga berpotensi memperparah ketegangan sosial dan politik (Saputra & Sibaroni, 2025).

Di Indonesia sendiri, media sosial seperti Twitter/X adalah salah satu alat utama untuk penyebaran ujaran kebencian. (Ibrahim et al., 2022) menemukan bahwa ujaran kebencian di Twitter juga sering bercampur dengan bahasa kasar (_abusive_ _language_) yang bisa jadi tantangan dalam deteksi.

Secara hukum, fenomena ini diatur secara tegas dalam Undang-Undang Nomor 1 Tahun 2024 tentang Perubahan Kedua atas Undang-Undang Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik (UU ITE). Secara spesifik, Pasal 28 ayat (2) _Jo._ Pasal 45A ayat (2) UU ITE melarang setiap orang dengan sengaja menyebarkan informasi elektronik yang ditujukan untuk menimbulkan rasa kebencian atau permusuhan individu dan/atau kelompok masyarakat tertentu berdasarkan atas Suku, Agama, Ras, dan Antargolongan (SARA). Keterkaitan hukum ini menjadikan penelitian terkait ujaran kebencian di media sosial sangat relevan dengan upaya penegakan hukum siber di Indonesia.

Dalam konteks _Natural_ _Language_ _Processing_ (NLP), deteksi ujaran kebencian merupakan tugas klasifikasi teks yang menuntut model agar mampu memahami aspek semantik dan pragmatik bahasa alami, terutama di media sosial yang cenderung bersifat informal dan penuh variasi. Tingginya keragaman bentuk dan konteks ujaran kebencian di platform tersebut menyebabkan satu unggahan dapat memuat beberapa kategori sekaligus, sehingga pendekatan _multilabel_ _classification_ menjadi sangat penting untuk melakukan deteksi yang lebih akurat dan komprehensif (Ousidhoum et al., 2019). Penelitian ini mendukung penggunaan model multitask dan anotasi multi-aspek untuk meningkatkan efektivitas dalam pengenalan pola ujaran kebencian.

## Klasifikasi _Multilabel_

Sub-bab ini akan membahas mengenai konsep dasar klasifikasi _multilabel_ serta berbagai pendekatan yang digunakan untuk menangani permasalahan data dengan label majemuk. Pembahasan dimulai dari definisi formal hingga strategi transformasi masalah yang umum digunakan dalam pengembangan model.

### Konsep Dasar Klasifikasi _Multilabel_

Klasifikasi _multilabel_ merupakan pengembangan dari klasifikasi tradisional yang memungkinkan satu data memiliki lebih dari satu label secara bersamaan. Dalam pendekatan klasik, sebuah sistem klasifikasi hanya mengaitkan satu _instance_ ke satu label kelas tertentu (_single-label classification_). Misalnya, pada kasus analisis sentimen, sebuah teks hanya dikategorikan sebagai "positif" , "negatif", ataupun "netral". Namun, dalam banyak permasalahan dunia nyata, satu data sering kali relevan dengan lebih dari satu kategori sekaligus.

Menurut (Zhang & Zhou, 2014), pembelajaran _multilabel_ adalah proses pemetaan dari ruang fitur ke himpunan label berikut:

|     |     | (2.1) |
| --- | --- | ----- |

Keterangan:

- - - - 𝒴: Himpunan semua label yang mungkin - : Label ke-i dalam himpunan label - q: Jumlah total label dalam sistem klasifikasi

Setiap _instance_ memiliki himpunan label terkait . Tugas pembelajaran _multilabel_ adalah mempelajari fungsi berikut:

|     |     | (2.2) |
| --- | --- | ----- |

Keterangan:

- - - - : Fungsi hipotesis klasifikasi multilabel - : Ruang fitur input - : Himpunan kuasa (power set) dari 𝒴, yaitu semua kemungkinan subset label

Fungsi tersebut memprediksi himpunan label yang benar untuk _instance_ baru. Secara formal, himpunan data _multilabel_ dinyatakan sebagai:

|     |     | (2.3) |
| --- | --- | ----- |

Keterangan:

- : Dataset multilabel
- : Vektor fitur input untuk instance ke-i
- : Himpunan label yang terkait dengan instance xi
- : Jumlah total instance dalam dataset
- : Ruang fitur
- : Ruang label

Representasi tersebut menunjukkan bahwa setiap data dapat memiliki satu atau lebih label yang relevan dari himpunan label . Sebagai contoh, sebuah unggahan media sosial yang mengandung ujaran kebencian (_hate speech_) dapat termasuk ke dalam beberapa kategori sekaligus, seperti kebencian berbasis ras, agama, atau fisik. Sifat data yang tumpang tindih ini menuntut sistem deteksi agar mampu mengenali keberadaan beberapa label kebencian sekaligus pada satu _instance_ data.

Klasifikasi _multilabel_ semakin banyak digunakan dalam berbagai bidang aplikasi modern, seperti klasifikasi fungsi protein, pengkategorian musik, dan klasifikasi adegan semantik (Tsoumakas & Katakis, 2007). Tantangan utamanya terletak pada ketergantungan antarlabel, distribusi data yang tidak seimbang, serta kompleksitas evaluasi performa model.

Dalam konteks deteksi ujaran kebencian, pendekatan _multilabel_ menjadi penting karena satu kalimat bisa menyerang lebih dari satu aspek, seperti agama, ras, atau gender secara bersamaan. Penggunaan pendekatan ini menghasilkan sistem deteksi yang bekerja secara lebih realistis dan komprehensif dibandingkan klasifikasi biner sederhana.

### Pendekatan Problem Transformation

Dalam klasifikasi _multilabel_, model tidak hanya menentukan apakah suatu data termasuk dalam satu kelas tertentu. Klasifikasi _multilabel_ membutuhkan strategi khusus karena satu data dapat memiliki lebih dari satu label, sehingga tidak bisa diproses langsung dengan algoritma konvensional yang hanya menerima satu label _output_ (Zhang & Zhou, 2014). Sebagian besar algoritma _supervised_ _learning_ dirancang untuk _single-label_ sehingga perlu dilakukan _problem_ _transformation_ agar masalah _multilabel_ diubah menjadi bentuk yang kompatibel dengan model yang ada (Tsoumakas & Katakis, 2007). Transformasi ini bertujuan menghasilkan representasi baru yang tetap mempertahankan informasi label tetapi memungkinkan model untuk melakukan pembelajaran secara efektif. Beberapa contoh _problem_ _transformation_ untuk klasifikasi _multilabel_ yaitu:

_Binary Relevance_ (BR). _Binary Relevance_ memecah masalah _multilabel_ menjadi beberapa masalah klasifikasi biner, di mana setiap label dipelajari oleh satu model terpisah (Boutell et al., 2004). Pendekatan ini sederhana dan fleksibel sehingga mudah dipadukan dengan berbagai algoritma pembelajaran. Kelemahannya adalah BR tidak mampu menangkap hubungan antarlabel karena setiap label diproses secara independent (Madjarov et al., 2012). BR relevan digunakan pada _dataset_ dengan banyak label jarang dan kombinasi label yang sedikit, seperti _Indonesian Abusive and Hate Speech Dataset_.

_Classifier Chains_ (CC). _Classifier Chains_ memproses label secara berurutan, dan prediksi label sebelumnya ditambahkan sebagai fitur untuk memprediksi label berikutnya (Read et al., 2009). Pendekatan ini memberikan kemampuan untuk menangkap dependensi antarlabel sehingga lebih informatif dibanding BR. Kekurangannya adalah sensitivitas terhadap urutan rantai karena perubahan posisi label dapat menghasilkan performa berbeda.

_Label Powerset_ (LP). _Label Powerset_ mengubah setiap kombinasi label menjadi kelas baru, misalnya label A dan B digabung menjadi kelas AB (Gibaja & Ventura, 2015) . Metode ini dapat menangkap hubungan antarlabel secara eksplisit karena setiap kombinasi diperlakukan sebagai kategori tersendiri. Namun pendekatan ini memiliki kelemahan berupa ledakan jumlah kelas ketika kombinasi label banyak, sehingga tidak efisien untuk _dataset_ dengan distribusi tidak merata atau label jarang (Madjarov et al., 2012).

### Representasi Label dan _Dataset_ _Multilabel_

Dalam klasifikasi _multilabel_, Representasi label merupakan komponen sentral dalam klasifikasi _multilabel_ karena menentukan bagaimana model mempelajari hubungan antara fitur dan banyak kategori secara simultan. Setiap _instance_ direpresentasikan menggunakan vektor biner berdimensi k, di mana setiap elemen menunjukkan apakah sebuah label aktif (1) atau tidak aktif (0). Pendekatan ini merupakan standar dalam _multilabel_ _learning_ karena memungkinkan pemetaan langsung antara _instance_ dan kumpulan label yang menyertainya (Zhang & Zhou, 2014).

Pada konteks deteksi ujaran kebencian, satu teks dapat mengandung beberapa kategori kebencian sekaligus. Label seperti HS*Group, HS_Religion, dan HS_Physical dapat muncul bersamaan, sehingga teks direpresentasikan sebagai vektor biner yang memuat satuan nilai 1 untuk label yang relevan dan 0 untuk label yang tidak relevan. Representasi ini konsisten dengan praktik umum \_multilabel* _classification_ yang menggunakan _encoding_ biner independen untuk seluruh label (Tsoumakas & Katakis, 2007).

Secara matematis, vektor label untuk satu _instance_ ​ dapat dinyatakan sebagai:

|     |     | (2.4) |
| --- | --- | ----- |

Keterangan:

- - - - : Vektor label untuk instance ke-i - : Nilai biner (0 atau 1) yang menunjukkan keberadaan label ke-j pada instance ke-i - : Jumlah total label dalam sistem

Setiap elemen merepresentasikan apakah _instance_ ​ memiliki label ke-. Representasi ini memungkinkan model untuk melakukan prediksi dalam bentuk keluaran sigmoid multidimensi, yaitu menghasilkan probabilitas untuk setiap label secara terpisah.

## _Machine Learning_ Klasik untuk Deteksi Teks

Pada tahap awal, model _machine learning klasik_ digunakan sebagai _baseline_ untuk membandingkan performa dengan model berbasis _deep learning_ seperti IndoBERT. Model klasik relatif lebih sederhana, cepat dilatih, serta mudah diinterpretasikan menggunakan metode _Explainable AI (XAI)_ seperti SHAP. Penelitian ini menggunakan dua algoritma utama yang umum dan kuat untuk klasifikasi teks, yaitu _Logistic Regression_ dan _Support Vector Machine_ (SVM).

### Logistic Regression

_Logistic Regression_ adalah model klasifikasi yang mempelajari hubungan antara fitur dan label melalui fungsi logistik yang mengubah _output_ menjadi probabilitas (Jurafsky & Martin, 2025). _Logistic Regression_ banyak digunakan dalam klasifikasi teks karena proses latihnya cepat, strukturnya sederhana, dan performanya stabil pada representasi TF-IDF. Penelitian ini menggunakan _Logistic Regression_ sebagai _baseline_ sebelum membandingkannya dengan model yang lebih kompleks.

### Support Vector Machine

_Support Vector Machine_ bekerja dengan memetakan data ke ruang fitur berdimensi tinggi dan kemudian mencari _hyperplane_ yang memiliki margin paling besar agar batas klasifikasinya kuat (Cortes & Vapnik, 1995). Mekanisme ini membuat model hanya bergantung pada _support vectors_, yaitu titik data yang menentukan posisi _hyperplane_. SVM juga efisien untuk data berdimensi tinggi karena pemetaan ke ruang fitur besar tidak perlu dihitung secara eksplisit berkat penggunaan fungsi kernel yang menggantikan operasi di ruang tersebut (Cortes & Vapnik, 1995). Keterbatasannya adalah model ini tidak menangkap urutan atau konteks linguistik yang lebih dalam seperti pendekatan _neural_ modern. Penelitian ini menggunakan SVM sebagai _baseline_ karena stabil dan cenderung menghindari _overfitting_ pada batas keputusan yang kompleks.

## Pendekatan _Deep Learning_ untuk Deteksi Ujaran Kebencian

Pendekatan _deep learning_ telah menjadi standar baru dalam berbagai tugas _Natural Language Processing (NLP)_, termasuk deteksi ujaran kebencian. Berbeda dengan algoritma klasik yang mengandalkan representasi fitur statis seperti TF-IDF, model _deep learning_ mampu mempelajari representasi kata secara kontekstual dan menangkap hubungan semantik antar kata dalam suatu kalimat. Penelitian ini menggunakan arsitektur _Bidirectional_ _Encoder Representations_ from _Transformers_ (BERT), khususnya versi Bahasa Indonesia yaitu IndoBERT.

### Transformer dan Self-attention

Arsitektur _Transformer_ diperkenalkan oleh (Vaswani et al., 2017) sebagai model _sequence-to-sequence_ yang sepenuhnya mengandalkan mekanisme _attention_, tanpa menggunakan rekurensi (RNN/LSTM) atau konvolusi. Pendekatan ini memungkinkan paralelisasi komputasi yang signifikan dan kemampuan menangkap dependensi global antar input tanpa memandang jarak posisinya. Arsitektur _Transformer_ mencakup dua komponen utama berupa _Encoder_ dan _Decoder_. _Transformer_ menyusun _Encoder_ dan _Decoder_ dalam tumpukan lapisan identik sejumlah . Setiap lapisan memiliki parameter unik untuk mengekstraksi fitur teks secara mendalam. Arsitektur ini menerapkan mekanisme _Positional Encoding_ pada setiap representasi input. Mekanisme tersebut menjaga informasi urutan kata dalam kalimat tanpa bantuan unit rekurensi. Model _Transformer_ memproses seluruh urutan simbol secara paralel melalui fusi konteks global. Gambar 2.1 mengilustrasikan struktur komponen internal tersebut secara mendetail.


Gambar 2.1 Arsitektur _Transformer_ (Vaswani et al., 2017)

Komponen _Encoder_ bertugas memetakan urutan input simbol menjadi representasi vektor kontinyu _Encoder_ terdiri dari tumpukan lapisan identik yang memiliki dua sub-lapisan: _Multi-Head Self-attention_ dan _Position-wise Feed-Forward Networks_. Mekanisme ini memungkinkan model memahami konteks setiap kata terhadap seluruh kata lain dalam kalimat secara simultan.

Komponen _Decoder_ bertugas menerima representasi dari _Encoder_ untuk menghasilkan urutan output secara _auto-regressive_ (satu per satu). Meskipun strukturnya mirip _Encoder_, _Decoder_ menggunakan _Masked_ _Self-attention_ untuk mencegah posisi saat ini melihat posisi masa depan, memastikan prediksi hanya bergantung pada output sebelumnya.

Meskipun arsitektur aslinya melibatkan kedua komponen, model BERT (termasuk IndoBERT) mengadopsi struktur _Transformer_ _Encoder_ saja. (Devlin et al., 2019) menjelaskan bahwa penggunaan _Encoder-only_ memungkinkan pembelajaran representasi kontekstual dua arah (_deep bidirectional_) yang krusial untuk pemahaman bahasa. Berbeda dengan _Decoder_ yang bersifat searah (kiri-ke-kanan), _Encoder_ memungkinkan fusi konteks penuh dari kiri dan kanan, menjadikannya arsitektur yang ideal untuk tugas klasifikasi teks dan deteksi ujaran kebencian.

### Mekanisme Self-attention dan Multi-Head Attention

Inti dari keunggulan arsitektur _Transformer_ terletak pada mekanisme _Self-attention_, sebuah metode yang memungkinkan model untuk menimbang bobot relevansi antar token dalam satu sekuens input tanpa bergantung pada jarak temporal seperti pada RNN. Mekanisme ini memproyeksikan matriks input secara matematis. Proses tersebut menghasilkan tiga sub-ruang representasi berupa _Query_ (), _Key_ (), dan _Value_ ().

Proses proyeksi ini dilakukan melalui perkalian matriks input dengan tiga matriks bobot yang dapat dipelajari (_trainable weight matrices_) selama proses pelatihan, yaitu , , dan . Jika dimisalkan dimensi model adalah , maka persamaan proyeksi linear tersebut sebagai berikut.

|     |     | (2.5) |
| --- | --- | ----- |

Dimana adalah matriks hasil proyeksi yang digunakan untuk perhitungan atensi. (Vaswani et al., 2017) mendefinisikan perhitungan bobot perhatian menggunakan fungsi _Scaled Dot-Product Attention_. Fungsi ini menghitung kompatibilitas antara _query_ dan _key_ melalui operasi perkalian titik (_dot product_), yang kemudian dinormalisasi. Persamaan lengkapnya sebagai berikut**.**

|     |     | (2.6) |
| --- | --- | ----- |

Dalam persamaan di atas:

- : Perkalian matriks _Query_ dengan transpos matriks _Key_ menghasilkan matriks skor atensi mentah yang merepresentasikan derajat kesamaan antar kata.
- : Faktor skala (_scaling factor_) berupa akar dari dimensi key (). Penskalaan ini krusial untuk mencegah nilai hasil _dot product_ menjadi terlalu besar, yang dapat mendorong fungsi Softmax ke area dengan gradien yang sangat kecil (_vanishing gradients_), sehingga menghambat proses _backpropagation_ (Vaswani et al., 2017).
- Softmax: Mengubah skor menjadi distribusi probabilitas (bernilai 0 hingga 1) agar total bobot perhatian berjumlah 1.
- : Hasil akhir didapat dengan mengalikan bobot probabilitas tersebut dengan matriks _Value_, sehingga model fokus pada informasi yang relevan dan meredam informasi yang tidak relevan (noise).


Gambar 2.2 Scaled Dot-Product Attention dan _Multi-Head Attention_

(Vaswani et al., 2017)

Untuk meningkatkan kapasitas model dalam menangkap berbagai jenis hubungan semantik (misalnya hubungan sintaksis vs. hubungan kontekstual), _Transformer_ mengembangkan konsep tersebut menjadi _Multi-Head Attention_. Mekanisme ini membagi menjadi buah "kepala" (_heads_) yang bekerja secara paralel pada sub-ruang dimensi yang lebih kecil . Secara formal, _Multi-Head Attention_ didefinisikan dengan persamaan berikut.

|     |     | (2.7) |
| --- | --- | ----- |

Dimana setiap _head_ dihitung secara independen melalui persamaan

|     |     | (2.8) |
| --- | --- | ----- |

Output dari setiap _head_ kemudian digabungkan (_concatenated_) dan diproyeksikan kembali menggunakan matriks bobot linear . Pendekatan ini memungkinkan IndoBERT untuk mengekstraksi fitur dari berbagai representasi sub-ruang yang berbeda secara bersamaan. Dalam konteks deteksi ujaran kebencian _multilabel_, kemampuan ini sangat vital; satu _head_ mungkin fokus mengidentifikasi kata kunci kebencian (misalnya kata kasar), sementara _head_ lain fokus menangkap target kebencian (misalnya entitas agama atau ras) di posisi kalimat yang berbeda (Devlin et al., 2019; Vaswani et al., 2017).

### Model BERT dan IndoBERT

_Bidirectional Encoder Representations from Transformers_ (BERT) diperkenalkan oleh Devlin et al. (2019) sebagai inovasi model bahasa yang mengubah paradigma representasi teks dengan memanfaatkan arsitektur _Transformer Encoder_. Berbeda dengan model bahasa konvensional yang memproses teks secara searah (_unidirectional_) dari kiri ke kanan, atau metode yang menggabungkan representasi arah kiri dan kanan secara terpisah, BERT dirancang dengan sifat _deep bidirectional_. Karakteristik ini memungkinkan model untuk memahami konteks semantik suatu kata dengan mempertimbangkan seluruh kata yang berada di posisi kiri maupun kanannya secara simultan dalam setiap lapisan jaringan. Kemampuan ini menjadikan BERT sangat efektif dalam menangkap ambiguitas dan nuansa makna yang kompleks dalam kalimat (Devlin et al., 2019).

Kecerdasan representasi pada BERT dicapai melalui mekanisme pelatihan _pre-training_ yang melibatkan dua tugas utama secara bersamaan. Tugas pertama adalah _Masked Language Modeling_ (MLM), di mana sekitar 15% token dari input disembunyikan (_masked_) secara acak, dan model dipaksa untuk memprediksi token asli tersebut hanya berdasarkan konteks sekitarnya. Mekanisme ini mencegah model untuk sekadar "melihat" kata target, sehingga mendorong pembelajaran representasi kontekstual yang mendalam. Tugas kedua adalah _Next Sentence Prediction_ (NSP), yang melatih model untuk memprediksi apakah kalimat kedua adalah kelanjutan logis dari kalimat pertama. Kemampuan NSP ini sangat krusial untuk memahami hubungan antar kalimat dalam tugas tingkat paragraf. Fleksibilitas BERT juga didukung oleh representasi inputnya yang unik, di mana representasi akhir setiap token dibentuk dari penjumlahan tiga komponen vektor: _Token Embeddings_, _Segment Embeddings_, dan _Position Embeddings_ (Devlin et al., 2019).


Gambar 2.3 Representasi Input pada BERT (Devlin et al., 2019)

Mengadopsi arsitektur dasar BERT tersebut, (Wilie et al., 2020) mengembangkan IndoBERT sebagai model _pre-trained_ berbahasa Indonesia yang komprehensif. IndoBERT dilatih menggunakan _dataset_ berskala masif bernama Indo4B, yang terdiri dari sekitar 4 miliar kata (±23 GB data teks). Keunggulan utama _dataset_ ini adalah cakupannya yang luas, meliputi ragam bahasa formal dari berita dan artikel daring, serta ragam bahasa informal atau kolokial yang berasal dari media sosial seperti Twitter. Hal ini membedakan IndoBERT dari model bahasa umum yang sering kali hanya dilatih pada teks formal (Wilie et al., 2020).

Dalam konteks penelitian deteksi ujaran kebencian, penggunaan IndoBERT menawarkan keunggulan signifikan dibandingkan model BERT orisinal (Inggris) atau mBERT (_Multilingual BERT_). (Wilie et al., 2020) menunjukkan bahwa model monolingual seperti IndoBERT secara konsisten mengungguli model multilingual dalam tugas klasifikasi teks. Hal ini disebabkan oleh kemampuan IndoBERT yang lebih superior dalam menangkap semantik level sentimen dan nuansa bahasa gaul (_slang_) yang mendominasi ekspresi ujaran kebencian di media sosial Indonesia, aspek yang sering kali gagal ditangkap oleh model yang dilatih pada data multibahasa umum (Wilie et al., 2020).

## Explainable Artificial Intelligence (XAI)

Meskipun model _deep learning_ memiliki kemampuan prediksi yang tinggi, sifatnya sebagai sistem _black-box_ membuat proses pengambilan keputusannya sulit dijelaskan (Adadi & Berrada, 2018). Dalam konteks deteksi ujaran kebencian, aspek interpretabilitas menjadi krusial karena kesalahan klasifikasi dapat berdampak serius pada konsekuensi hukum, sosial, dan hak individu. Oleh karena itu, penelitian ini menerapkan pendekatan _Explainable_ _Artificial Intelligence_ (XAI) untuk menjamin transparansi dan akuntabilitas hasil klasifikasi _multilabel_. Melalui XAI, keputusan model dapat dijelaskan dengan mengidentifikasi fitur atau kata yang paling berpengaruh (Arrieta et al., 2019), sekaligus membantu memvalidasi pemahaman konteks dan meminimalisir bias model agar sistem dapat bekerja secara adil dan etis.

### Shapley Value

Konsep dasar metode interpretasi yang digunakan dalam penelitian ini berakar pada _Cooperative Game Theory_ (Teori Permainan Kooperatif), yang diperkenalkan oleh Lloyd Shapley pada tahun 1953. Dalam teori permainan klasik, tujuan utamanya adalah menentukan bagaimana mendistribusikan "_payout_" (total hasil atau keuntungan) secara adil kepada sekumpulan "pemain" yang bekerja sama dalam sebuah koalisi untuk mencapai hasil tersebut. (Lundberg & Lee, 2017) mengadopsi kerangka kerja matematis ini untuk memecahkan masalah interpretasi model _machine learning_ yang kompleks. Prediksi model dalam konteks ini diasumsikan sebagai sebuah permainan (_game_), di mana fitur-fitur input (seperti kata atau token dalam kalimat) bertindak sebagai pemain (_players_), dan nilai prediksi aktual model (setelah dikurangi rata-rata prediksi dasar) dianggap sebagai _payout_ yang harus didistribusikan (Lundberg & Lee, 2017).

Inti dari perhitungan _Shapley Value_ terletak pada konsep _Marginal Contribution_ atau kontribusi marjinal. Metode ini secara teknis tidak sekadar melihat pengaruh fitur secara isolasi, melainkan menghitung rata-rata kontribusi fitur tersebut di seluruh kemungkinan kombinasi himpunan bagian fitur lainnya. Karena keberadaan fitur lain dapat memengaruhi dampak fitur _i_ (interaksi antar-fitur), perhitungan ini dilakukan dengan melakukan permutasi terhadap semua kemungkinan urutan penambahan fitur ke dalam model. Nilai Shapley untuk sebuah fitur adalah rata-rata tertimbang dari selisih output model ketika fitur tersebut ada dibandingkan ketika fitur tersebut absen dalam setiap koalisi yang mungkin terbentuk (Lundberg & Lee, 2017).

Rumus matematis untuk menghitung nilai Shapley didefinisikan sebagai berikut:

|     |     | (2.9) |
| --- | --- | ----- |

Keterangan Variabel:

- : Nilai Shapley (kontribusi) untuk fitur .
- : Himpunan semua fitur input.
- : Subset fitur (koalisi) yang tidak menyertakan fitur .
- : Faktor pembobot yang merepresentasikan probabilitas permutasi koalisi tersebut.
- : Kontribusi marjinal fitur , yaitu selisih output model ketika fitur ada, dibandingkan ketika fitur tidak ada dalam koalisi .

Keunggulan utama _Shapley Value_ dibandingkan metode heuristik lainnya adalah landasan aksiomatisnya yang kuat. (Lundberg & Lee, 2017) menekankan bahwa _Shapley Value_ adalah satu-satunya metode atribusi fitur (_feature attribution_) yang memenuhi tiga properti atau aksioma keadilan secara sekaligus, yaitu: _Efficiency_ (total nilai atribusi fitur harus sama dengan selisih antara prediksi model dan prediksi rata-rata), _Symmetry_ (dua fitur yang memberikan kontribusi identik dalam semua koalisi harus memiliki nilai Shapley yang sama), dan _Additivity_ (jika model merupakan gabungan dari dua fungsi, maka nilai Shapley-nya adalah penjumlahan dari nilai Shapley masing-masing fungsi). Ketiga aksioma ini menjamin bahwa penjelasan yang dihasilkan konsisten secara matematis dan merefleksikan kontribusi fitur yang sebenarnya terhadap keputusan model (Lundberg & Lee, 2017).

### Shapley Additive Explanations (SHAP)

SHAP diperkenalkan oleh (Lundberg & Lee, 2017) sebagai sebuah kerangka kerja terpadu yang menghubungkan teori _Shapley Value_ dengan kelas metode interpretasi yang dikenal sebagai _Additive Feature Attribution Methods_. Kerangka kerja ini menjelaskan bahwa prediksi model tidak dipandang sebagai analisis terpisah, melainkan didefinisikan sebagai sebuah model tersendiri yang disebut _explanation model_ (). Tujuan utama dari model penjelasan ini adalah memberikan aproksimasi lokal terhadap model prediksi kompleks () menggunakan fungsi linear yang lebih sederhana dan mudah diinterpretasikan. SHAP bekerja dengan menjumlahkan efek atribusi dari setiap fitur untuk mendekati nilai prediksi aktual model asli (Lundberg & Lee, 2017).

SHAP menyederhanakan ruang input yang kompleks menjadi vektor fitur biner , di mana adalah jumlah fitur input yang disederhanakan. Dalam representasi ini, yang juga sering disebut sebagai vektor koalisi, nilai 1 mengindikasikan bahwa fitur tersebut "hadir" atau diamati, sedangkan nilai 0 mengindikasikan bahwa fitur tersebut "absen" atau tidak diketahui. Model penjelasan () kemudian didefinisikan sebagai kombinasi linear dari variabel biner tersebut. Persamaan matematis untuk model penjelasan aditif ini didefinisikan sebagai berikut.

|     |     | (2.10) |
| --- | --- | ------ |

Keterangan Variabel:

- : Model penjelasan (_explanation model_) yang mengaproksimasi prediksi model asli.
- : Variabel biner yang bernilai 1 jika fitur ke- hadir dalam koalisi, dan 0 jika absen.
- : Jumlah total fitur input yang disederhanakan.
- : Nilai Shapley (kontribusi) untuk fitur ke-.
- : Nilai bias, yaitu rata-rata prediksi model pada _dataset_ ().

Syarat fundamental dari model ini adalah ketika seluruh fitur hadir (vektor bernilai 1 semua), maka output dari model penjelasan harus sama persis dengan output prediksi model asli (Lundberg & Lee, 2017). Kontribusi teoretis paling signifikan dari Lundberg dan Lee adalah pembuktian bahwa SHAP merupakan satu-satunya metode atribusi fitur yang memenuhi tiga properti aksiomatis sekaligus, yang menjamin validitas penjelasannya. Properti pertama adalah _Local Accuracy_, yang mengharuskan model penjelasan untuk setidaknya menyamai output model asli pada input lokal tertentu. Properti kedua adalah _Missingness_, yang mensyaratkan bahwa fitur yang absen atau bernilai nol dalam representasi input tidak boleh memiliki nilai atribusi. Properti ketiga, dan yang paling krusial, adalah _Consistency_; properti ini menjamin bahwa jika suatu model diubah sedemikian rupa sehingga kontribusi marjinal suatu fitur meningkat atau tetap sama, maka nilai SHAP fitur tersebut tidak boleh berkurang. Kepatuhan terhadap ketiga properti ini membedakan SHAP dari metode interpretasi lain yang sering kali melanggar konsistensi (Lundberg & Lee, 2017).

### Algoritma KernelSHAP

Meskipun nilai Shapley menawarkan landasan teoretis yang solid untuk interpretasi model, perhitungan eksaknya menghadapi kendala komputasi yang signifikan. Dalam menghitung nilai Shapley secara presisi, diperlukan evaluasi terhadap seluruh kemungkinan kombinasi fitur (koalisi) yang ada. Dalam konteks teori kompleksitas, masalah ini memiliki kompleksitas waktu eksponensial , di mana adalah jumlah fitur input. Untuk model pemrosesan bahasa alami (NLP) dengan jumlah fitur (token kata) yang besar, perhitungan ini menjadi _NP-hard_ dan tidak layak dilakukan secara komputasi (_intractable_). Metode aproksimasi yang efisien namun tetap mempertahankan properti aksiomatis dari nilai Shapley diperlukan dalam hal ini (Lundberg & Lee, 2017).

Sebagai solusi atas permasalahan tersebut, Lundberg dan Lee memperkenalkan algoritma KernelSHAP. Metode ini merupakan pendekatan _model-agnostic_ yang menggabungkan efisiensi komputasi dari LIME (_Linear Interpretable Model-agnostic Explanations_) dengan landasan teoretis _Shapley Value_. Inti dari algoritma KernelSHAP adalah mengubah masalah penghitungan nilai Shapley menjadi masalah optimasi fungsi kerugian (_loss function_) menggunakan regresi linear berbobot (_weighted linear regression_). Dalam kerangka kerja ini, nilai Shapley diperoleh dengan meminimalkan fungsi kerugian berikut.

|     |     | (2.11) |
| --- | --- | ------ |

Keterangan Variabel:

- : Fungsi kerugian (_squared loss_) yang harus diminimalkan.
- : Prediksi model asli pada _instance_ .
- : Prediksi dari model penjelasan linear.
- : Bobot kernel Shapley untuk koalisi .

Komponen paling krusial yang membedakan KernelSHAP dari metode LIME standar adalah penggunaan fungsi pembobotan yang disebut Shapley Kernel (). Jika LIME menggunakan kernel heuristik, KernelSHAP menggunakan kernel yang diturunkan secara analitis untuk memastikan bahwa solusi regresi linear konvergen ke nilai Shapley yang sebenarnya. Secara matematis, bobot kernel untuk setiap koalisi dihitung menggunakan persamaan:

|     |     | (2.12) |
| --- | --- | ------ |

Logika di balik Shapley Kernel ini adalah memberikan bobot yang sangat tinggi pada koalisi yang melibatkan sedikit fitur (subset kecil) atau koalisi yang melibatkan hampir seluruh fitur (subset besar). Sebaliknya, koalisi dengan jumlah fitur "setengah-setengah" mendapatkan bobot yang sangat kecil. Distribusi bobot ini didasarkan pada intuisi bahwa pengaruh marjinal suatu fitur paling baik dipelajari ketika fitur tersebut bekerja secara isolasi atau ketika fitur tersebut menjadi satu-satunya yang absen dari himpunan lengkap (Lundberg & Lee, 2017).

Secara prosedural, algoritma KernelSHAP bekerja melalui langkah-langkah komputasi sebagai berikut.

- _Sampling_ Koalisi: Melakukan _sampling_ acak terhadap vektor fitur biner untuk membentuk _dataset_ sintetik yang merepresentasikan keberadaan atau ketidakhadiran fitur.
- Prediksi Model Asli: Menghitung output prediksi dari model asli

untuk setiap sampel koalisi tersebut guna mendapatkan target regresi.

- Pembobotan Kernel: Menghitung bobot untuk setiap sampel menggunakan rumus Shapley Kernel di atas.
- _Weighted Linear Regression_: Melakukan regresi linear berbobot pada _dataset_ tersebut. Koefisien yang dihasilkan dari model regresi ini secara otomatis merupakan estimasi dari nilai Shapley () untuk setiap fitur, yang memenuhi properti _local accuracy_ dan _consistency_ (Lundberg & Lee, 2017).

## Evaluasi pada Klasifikasi _Multilabel_

Evaluasi pada klasifikasi _multilabel_ memiliki karakteristik yang berbeda dibandingkan dengan klasifikasi _single-label_, karena satu _instance_ dapat memiliki lebih dari satu label secara bersamaan (Zhang & Zhou, 2014). Dalam konteks ini, setiap label memiliki nilai _True_ _Positive_ (TP), _False_ _Positive_ (FP), dan _False_ _Negative_ (FN) masing-masing, sehingga penilaian kinerja model tidak bisa hanya dilakukan secara tunggal. Pada tugas deteksi ujaran kebencian _multilabel_, evaluasi perlu dilakukan dari dua sisi: (1) per label, untuk menilai seberapa baik model mengenali masing-masing kategori kebencian seperti "agama", "ras", atau "gender"; dan (2) secara agregat, untuk mengukur performa keseluruhan model dalam menangani semua label secara bersamaan. Pendekatan evaluasi ini penting agar hasil penilaian tidak bias terhadap label yang dominan dan tetap mencerminkan kemampuan model dalam mendeteksi variasi bentuk ujaran kebencian yang kompleks di media sosial.

### Precision, Recall, dan F1-score

Pada klasifikasi _multilabel_, konsep evaluasi tetap menggunakan ukuran dasar yang sama seperti pada klasifikasi tunggal, yaitu _True_ _Positive_ (TP), _False_ _Positive_ (FP), dan _False_ _Negative_ (FN), tetapi dihitung per label. Setiap label seperti "agama", "ras", atau "gender" memiliki nilai TP, FP, dan FN sendiri tergantung pada prediksi model terhadap label tersebut.

Secara umum, tiga metrik utama yang digunakan adalah _precision_, _recall_, dan _F1-score_.

- _Precision_ mengukur seberapa tepat model dalam memprediksi label, yaitu proporsi prediksi positif yang benar. Nilai _precision_ dapat diperoleh melalui persamaan:

|     |     | (2.13) |
| --- | --- | ------ |

Keterangan:

- - - - TP (_True Positive_): Jumlah label yang diprediksi positif dan benar - FP (_False Positive_): Jumlah label yang diprediksi positif tetapi salah

- _Recall_ mengukur seberapa lengkap deteksi model, yaitu seberapa banyak label positif yang berhasil dikenali dengan benar. Nilai _recall_ dapat diperoleh melalui persamaan:

|     |     | (2.14) |
| --- | --- | ------ |

Keterangan:

- - - - TP (_True Positive_): Jumlah label yang diprediksi positif dan benar - FN (_False Negative_): Jumlah label yang seharusnya positif tetapi diprediksi negatif

- _F1-score_ merupakan rata-rata harmonik antara _precision_ dan _recall_, yang menggambarkan keseimbangan antara ketepatan dan kelengkapan prediksi. Nilai _F1-score_ dapat diperoleh melalui persamaan:

|     |     | (2.15) |
| --- | --- | ------ |

Keterangan:

- - - - P = _Precision_ - R = _Recall_

Dalam klasifikasi _multilabel_, ketiga metrik ini biasanya dihitung untuk setiap label secara terpisah, kemudian digabung menggunakan metode _averaging_ seperti _macro_ _averaging_ (rata-rata antar label) atau _micro_ _averaging_ (menggabungkan seluruh TP, FP, dan FN sebelum perhitungan). Pendekatan ini penting untuk memastikan evaluasi tidak bias terhadap label yang frekuensinya lebih tinggi, terutama pada _dataset_ ujaran kebencian yang cenderung tidak seimbang antar kategori.

### Pendekatan Averaging: Micro, Macro, dan Weighted F1

Dalam klasifikasi _multilabel_, setiap label memiliki nilai _precision_, _recall_, dan _F1-score_ sendiri. Untuk mendapatkan satu nilai kinerja keseluruhan yang mewakili seluruh label, digunakan pendekatan _averaging_. Pendekatan ini menggabungkan hasil evaluasi per label menjadi satu metrik agregat agar performa model dapat dibandingkan secara menyeluruh (Zhang & Zhou, 2014). Tiga metode _averaging_ yang paling umum adalah _micro_, _macro_, dan _weighted_ _averaging_.

- _Micro F1_

Pendekatan _micro_ _averaging_ menghitung nilai TP, FP, dan FN secara global, yaitu dengan menjumlahkan seluruh nilai dari semua label sebelum menghitung _precision_, _recall_, dan F1. Metode ini memberi bobot lebih besar pada label yang sering muncul, sehingga cocok digunakan untuk _dataset_ tidak seimbang (_imbalanced_) seperti ujaran kebencian di media sosial, di mana beberapa kategori (misalnya "agama") lebih dominan. Nilai _Micro F1_ didapatkan melalui persamaan:

|     |     | (2.16) |
| --- | --- | ------ |

Keterangan:

- - - - ​ = Total _True_ _Positive_ dari seluruh label - ​​ = Total _False_ _Positive_ dari seluruh label - ​​ = Total _False_ _Negative_ dari seluruh label

- _Macro F1_

Pendekatan _macro_ _averaging_ menghitung _precision_, _recall_, dan _F1-score_ untuk setiap label secara independen, lalu mengambil rata-rata tanpa memperhatikan jumlah _instance_ tiap label. _Macro F1_ cocok untuk menilai performa model pada label minoritas, karena semua label memiliki bobot yang sama. Pada _dataset multilabel_ _hate speech_ yang _imbalanced_, _Macro F1_ memberikan gambaran apakah model masih mampu mengenali kategori kebencian yang jarang muncul, seperti ujaran terkait gender atau etnis tertentu. Nilai _Macro F1_ didapatkan melalui persamaan:

|     |     | (2.17) |
| --- | --- | ------ |

Keterangan:

- ​ = Jumlah label
- ​​ = Total _False_ _Positive_ dari seluruh label

- _Weighted_ F1

Pendekatan _weighted_ _averaging_ mirip dengan _macro_, tetapi memberikan bobot pada setiap label berdasarkan jumlah _instance_\-nya. Metode ini menjaga keseimbangan antara _micro_ dan _macro_ karena memperhitungkan frekuensi label, tetapi tidak mengabaikan kontribusi label minoritas. _Weighted_ F1 cocok digunakan ketika ingin menilai performa agregat model secara proporsional tanpa bias yang terlalu besar terhadap label mayoritas. Nilai _Weighted_ F1 didapatkan melalui persamaan:

|     |     | (2.18) |
| --- | --- | ------ |

Keterangan:

- ​​ = Jumlah _instance_ untuk label ke-𝑖

### Hamming Loss

_Hamming Loss_ digunakan untuk mengukur frekuensi kesalahan model dalam memprediksi label pada klasifikasi multi-label, yakni menghitung rata-rata jumlah label yang salah diprediksi per contoh (Zhang & Zhou, 2014). Metrik ini menghitung proporsi label yang salah diprediksi, baik ketika model menandai label yang seharusnya negatif sebagai positif (_false_ _positive_, 0 → 1), maupun ketika gagal mengenali label yang seharusnya positif (_false_ _negative_, 1 → 0). Metrik ini sangat relevan untuk deteksi ujaran kebencian _multilabel_, karena sebagian besar data di media sosial tidak selalu mengandung semua kategori kebencian. Banyak unggahan hanya memiliki sedikit label aktif atau bahkan tidak memiliki label sama sekali. Dalam kondisi seperti ini, _Hamming Loss_ mampu menunjukkan seberapa sering model salah menandai suatu label, misalnya ketika unggahan netral dianggap mengandung ujaran kebencian, atau ketika model gagal mengenali ujaran kebencian terhadap kategori tertentu seperti "agama" atau "ras".

Karakteristik tersebut menjadikan _Hamming Loss_ sebagai indikator utama kesalahan model dalam penelitian ini karena kemampuannya memberikan gambaran akurasi prediksi yang seimbang di seluruh label dan _instance_. Secara sederhana, _Hamming Loss_ dapat dituliskan sebagai:

|     |     | (2.18) |
| --- | --- | ------ |

Keterangan:

- ​​ = Jumlah data
- \= Label sebenarnya untuk data ke-𝑖 pada label ke-𝑗
- \= Prediksi model untuk data ke-𝑖 pada label ke-𝑗

## Penelitian Terdahulu

Penelitian ini disusun berdasarkan tinjauan terhadap sejumlah literatur terdahulu yang relevan dengan topik deteksi ujaran kebencian (_hate speech_), klasifikasi _multilabel_, dan _Explainable_ _Artificial Intelligence_ (XAI). Kajian pustaka ini bertujuan untuk memetakan posisi penelitian (_state of the art_), mengidentifikasi celah penelitian (_research gap_), serta menentukan kontribusi kebaruan (_novelty_).

Beberapa penelitian kunci yang menjadi landasan utama dalam pengembangan metode pada skripsi ini dirangkum dalam Tabel 2.1.

Tabel 2.1 Perbandingan Penelitian Terdahulu

| **No** | **Peneliti (Tahun)**          | **Ringkasan Studi (Topik & Metode)**                                                                                                                                                                                                                                                                                                                                                                                                                                                         | **Temuan Utama & Perbedaan dengan Penelitian Ini**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------ | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | **Ibrohim & Budi (2019)**     | **Topik:** Deteksi ujaran kebencian dan bahasa kasar (_abusive_) secara _multilabel_ pada Twitter Indonesia dengan mengategorikan target, kategori, dan tingkat kebencian.<br><br>**Metode:** Menggunakan algoritma ML klasik (RFDT, SVM, Naive Bayes) dikombinasikan dengan teknik transformasi label (_Binary Relevance_, _Label Power-set_, _Classifier Chains_) serta ekstraksi fitur frekuensi kata, ortografi, dan leksikon.                                                           | **Hasil:** Algoritma RFDT dengan pendekatan _Label Power-set_ (LP) memberikan akurasi terbaik dan waktu komputasi yang efisien. Studi ini menjadi pionir dalam penyediaan _dataset_ publik untuk klasifikasi _multilabel_ ujaran kebencian.<br><br>**Perbedaan:** Penelitian ini beralih dari fitur statistik ke representasi kontekstual mendalam menggunakan IndoBERT, serta memperkenalkan optimasi keputusan melalui _Threshold_ Tuning dan transparansi model via KernelSHAP.                                                                                                                                    |
| 2      | **Ibrahim et al. (2022)**     | Eksplorasi aspek keterjelasan (explainability) pada sistem deteksi ujaran kebencian Twitter Indonesia guna meningkatkan kepercayaan pengguna. Metode penelitian melakukan klasifikasi ujaran kebencian dan bahasa kasar pada 13.169 tweet Indonesia menggunakan _machine learning_ (Logistic Regression, Naive Bayes, Random Forest, XGBoost) berbasis fitur TF-IDF. Metode LIME (Local Interpretable Model-Agnostic Explanations) diterapkan untuk menjelaskan interpretasi prediksi model. | **Hasil:** Model klasifikasi mencapai akurasi 83% dan _F1-score_ 0.79. Evaluasi LIME menunjukkan XGBoost memberikan penjelasan paling logis dalam mengidentifikasi kata kunci ujaran kebencian dibandingkan model lainnya yang sering mengalami salah identifikasi.<br><br>**Perbedaan:** Penelitian ini menggunakan KernelSHAP yang diimplementasikan secara manual (_from scratch_) untuk landasan teoritis yang lebih konsisten, serta memanfaatkan model IndoBERT yang memiliki kemampuan pemahaman konteks lebih unggul dibanding model _Machine learning_ konvensional yang digunakan pada penelitian tersebut. |
| 3      | **Saputra & Sibaroni (2025)** | **Topik:** Klasifikasi ujaran kebencian _multilabel_ pada diskursus politik di media sosial X dengan menekankan pada pengaruh panjang kalimat terhadap performa model.<br><br>**Metode:** Mengevaluasi berbagai variasi model berbasis BERT, termasuk arsitektur hibrida seperti BERT-CNN, BERT-LSTM, dan BERT-BiLSTM, untuk menangkap nuansa konteks pada teks dengan berbagai panjang.                                                                                                     | **Hasil:** BERT-BiLSTM memberikan akurasi terbaik pada teks panjang (82,00%) karena kemampuan menangkap konteks dua arah, sedangkan BERT-CNN lebih efektif mengekstraksi fitur pada teks pendek.<br><br>**Perbedaan:** Fokus penelitian ini bukan pada hibridisasi arsitektur, melainkan pada optimasi IndoBERT murni melalui penyesuaian ambang batas (_thresholding_) mandiri per label guna menangani ketidakseimbangan data, serta penggunaan interpretasi SHAP untuk analisis linguistik.                                                                                                                        |

Berdasarkan Tabel 2.1, terlihat bahwa meskipun penelitian mengenai ujaran kebencian berbahasa Indonesia telah banyak dilakukan, masih terdapat celah dalam penanganan masalah ketidakseimbangan kelas (_class imbalance_) secara spesifik pada kasus _multilabel_. Mayoritas penelitian sebelumnya berfokus pada eksplorasi arsitektur model (seperti BiLSTM atau Hybrid CNN), namun belum secara mendalam menerapkan strategi _Threshold_ Tuning adaptif untuk setiap label.

Implementasi _Explainable AI_ pada penelitian sebelumnya umumnya masih menggunakan pustaka instan (seperti _library_ LIME/SHAP). Penelitian ini mengisi celah tersebut dengan mengimplementasikan algoritma KernelSHAP secara dasar (_from scratch_) untuk memberikan transparansi model yang lebih mendalam, sekaligus membuktikan validitas model IndoBERT dalam menangkap konteks semantik ujaran kebencian.

#
