

HASIL DAN PEMBAHASAN

Bab ini menyajikan hasil implementasi dari keseluruhan perancangan sistem yang telah dijelaskan pada Bab III. Evaluasi dilakukan secara bertahap sesuai alur penelitian yang ditetapkan pada Gambar 3.4, meliputi hasil _Preprocessing_ data teks, evaluasi performa model klasifikasi _multilabel_, analisis interpretabilitas menggunakan _Explainable_ AI (XAI), dan implementasi sistem ke dalam aplikasi _web_. Setiap tahap evaluasi disertai dengan analisis mendalam untuk mengidentifikasi kekuatan dan keterbatasan pendekatan yang digunakan. Evaluasi dilakukan pada data uji yang terdiri dari 2.634 sampel (20% dari total 13.169 data), dengan pembagian 70% data latih (9.218 sampel) dan 10% data validasi (1.317 sampel) yang digunakan untuk _threshold_ tuning.

## Hasil _Preprocessing_ Data Teks

Subbab ini menyajikan hasil dari proses _Preprocessing_ data teks yang diterapkan sebelum tahap pelatihan dan pengujian model klasifikasi _multilabel_. Fokus pembahasan pada bagian ini adalah untuk menunjukkan perubahan nyata pada data teks serta dampak _Preprocessing_ terhadap karakteristik data yang digunakan dalam penelitian.

### Contoh Perubahan Teks Sebelum dan Sesudah _Preprocessing_

Untuk memastikan bahwa _Preprocessing_ bekerja secara efektif, dilakukan observasi langsung terhadap beberapa contoh data teks sebelum dan sesudah proses pembersihan dan normalisasi. Tabel berikut menunjukkan contoh hasil _Preprocessing_ pada lima data yang diambil secara acak dari _Dataset_.

Tabel 4.1 Contoh Teks Sebelum dan Sesudah _Preprocessing_

| **No** | **Teks Sebelum _Preprocessing_**                                                                                                                                                                                                                               | **Teks Setelah _Preprocessing_**                                                                                                                                                                                                                                          |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | USER dulu pakai aparat dgn mantra "anda anti pembangunan. anda komunis"'                                                                                                                                                                                       | pengguna dulu pakai aparat dengan mantra anda anti pembangunan anda komunis                                                                                                                                                                                               |
| 2      | USER USER USER USER USER Yaaelah chat aja loe urusin,, noh yg dah keliatan mesum anggota DPR pd mingkem,, yg nyata selingkuh 7th kalian pura2 budek,,, #bong200 gk nyampe otaknya...'                                                                          | pengguna pengguna pengguna pengguna pengguna yaelah chat saja kamu uruskan itu yang sudah kelihatan mesum anggota dewan perwakilan rakyat pada diam yang nyata selingkuh tahun kalian pura budek bong tidak sampai otaknya                                                |
| 3      | Prabowo cs teriak2 mnghujat Jokowi antek asing, Jokowi diam tak membalas. Dlm diam Jokowi mmbuktikan sbaliknya, dgn bukti setelah 50 thn Indonesia menguasai 51% saham PT Freeport Indonesia, Prabowo cs dlm diam ternyata menghrpkan dukungan AS dgn janji jk | prabowo rekan teriak menghujat jokowi antek asing jokowi diam tak membalas dalam diam jokowi membuktikan sebaliknya dengan bukti setelah tahun indonesia menguasai saham pt freepo indonesia prabowo rekan dalam diam ternyata mengharapkan dukungan as dengan janji jika |
| 4      | USER USER Ajarannya harusnya sama. Sama2 bersumber dari kitab Weda. Bisa jadi juga beda tafsir. Tapi secara umum 5 dasar kepercayaan sama : percaya brahman, atman, karma, reinkarnasi dan moksa. \\nKl ritual, hari raya, cara sembahyang kebany              | pengguna pengguna ajarannya harusnya sama sama bersumber dari kitab weda bisa jadi juga beda tafsir tapi secara umum dasar kepercayaan sama percaya brahman atman karma reinkarnasi dan moksa kalau ritual hari raya cara sembahyang kebanyakan                           |
| 5      | RT USER: Besok kalau Prabowo nyapres lagi kalau masih teriak-teriak NKRI harga mati jangan dipercaya ya                                                                                                                                                        | pengguna besok kalau prabowo mencalonkan presiden lagi kalau masih teriak teriak negara kesatuan republik indonesia harga mati jangan dipercaya ya                                                                                                                        |

Berdasarkan contoh-contoh tersebut, dapat dilihat bahwa _Preprocessing_ berhasil:

- Menghapus simbol, tanda baca, dan karakter non-alfabet,
- Menormalkan kata tidak baku dan _slang_
- Mengganti token anonim seperti USER menjadi bentuk yang konsisten,
- Mengurangi variasi ejaan akibat pengulangan karakter,
- Menghasilkan teks yang lebih bersih tanpa menghilangkan makna utama kalimat.

### Dampak _Preprocessing_ terhadap Panjang Teks

Selain observasi kualitatif, dampak _Preprocessing_ juga dianalisis secara kuantitatif melalui distribusi panjang teks sebelum dan sesudah _Preprocessing_. Panjang teks diukur berdasarkan jumlah karakter.


Gambar 4.1 Distribusi Panjang Teks Sebelum dan Sesudah _Preprocessing_

Berdasarkan grafik tersebut, terlihat bahwa distribusi panjang teks setelah _Preprocessing_ cenderung bergeser ke kanan dibandingkan dengan teks sebelum _Preprocessing_. Hal ini menunjukkan bahwa _Preprocessing_ tidak menghilangkan konten secara agresif, melainkan melakukan normalisasi dan standarisasi yang pada banyak kasus justru menambah panjang teks. Statistik deskriptif panjang teks sebelum dan sesudah _Preprocessing_ disajikan pada Tabel 4.2.

Tabel 4.2 Statistik Panjang Teks Sebelum dan Sesudah _Preprocessing_

| **Statistik** | **Sebelum** | **Sesudah** |
| ------------- | ----------- | ----------- |
| Jumlah data   | 13.169      | 13.169      |
| ---           | ---         | ---         |
| Rata-rata     | 114,20      | 123,13      |
| ---           | ---         | ---         |
| Median        | 100         | 109         |
| ---           | ---         | ---         |
| Minimum       | 4           | 4           |
| ---           | ---         | ---         |
| Maksimum      | 561         | 501         |
| ---           | ---         | ---         |

Terjadi peningkatan rata-rata panjang teks sebesar sekitar 7,82% setelah _Preprocessing_. Peningkatan ini disebabkan oleh proses normalisasi kata tidak baku dan singkatan menjadi bentuk yang lebih eksplisit, seperti penggantian kata _slang_, perluasan singkatan, serta standarisasi token anonim. Di sisi lain, nilai maksimum panjang teks mengalami penurunan, yang menunjukkan bahwa karakter, simbol, atau token tidak relevan pada teks yang sangat panjang berhasil dihapus tanpa menghilangkan makna utama kalimat.

### Ringkasan Hasil _Preprocessing_

Berdasarkan analisis kualitatif dan kuantitatif, _Preprocessing_ data teks dalam penelitian ini berhasil menghasilkan data yang lebih bersih, terstandarisasi, dan konsisten. Proses _Preprocessing_ tidak menyebabkan hilangnya data secara signifikan, yang dibuktikan dengan tidak ditemukannya teks kosong setelah _Preprocessing_.

Normalisasi dan standarisasi teks media sosial yang bersifat informal justru menghasilkan representasi teks yang lebih eksplisit dan mudah diproses oleh model klasifikasi. Dengan demikian, data teks yang digunakan pada tahap eksperimen telah berada dalam kondisi yang siap untuk proses ekstraksi fitur dan pelatihan model pada tahap selanjutnya.

## Hasil Kinerja Model Klasifikasi _Multilabel_

Bagian ini menyajikan hasil eksperimen dan evaluasi performa dari berbagai model yang telah dibangun, mulai dari model _machine learning_ klasik hingga model berbasis _Transformer_ (IndoBERT). Pengujian dilakukan secara bertahap untuk melihat efektivitas penggunaan ambang batas (_threshold_) standar dibandingkan dengan _threshold_ yang telah dioptimasi.

### Hasil Model _Machine learning_ Klasik (_Threshold_ Default)

Subbab ini menyajikan hasil evaluasi kinerja model _machine learning_ klasik menggunakan _threshold_ default sebesar 0,5 untuk seluruh label. Evaluasi ini bertujuan untuk memberikan gambaran performa _baseline_ sebelum dilakukan penyesuaian _threshold_ pada klasifikasi _multilabel_.

Tabel 4.3 Hasil Evaluasi Model Klasik dengan _Threshold_ Default (0,5)

| **Model**                | **Micro F1** | **Macro F1** | **Hamming Loss** |
| ------------------------ | ------------ | ------------ | ---------------- |
| BR _Logistic Regression_ | 0,6583       | 0,6005       | 0,0743           |
| ---                      | ---          | ---          | ---              |
| BR SVM                   | 0,6297       | 0,5636       | 0,0600           |
| ---                      | ---          | ---          | ---              |
| CC _Logistic Regression_ | 0,6210       | 0,5501       | 0,0836           |
| ---                      | ---          | ---          | ---              |
| CC SVM                   | 0,6570       | 0,6174       | 0,0584           |
| ---                      | ---          | ---          | ---              |

Berdasarkan hasil pengujian pada Tabel 4.3, model SVM dengan pendekatan CC memperoleh nilai _Macro_ _F1-score_ tertinggi sebesar 0,6174, diikuti oleh model BR _Logistic Regression_ sebesar 0,6005. Hal ini menunjukkan bahwa model berbasis SVM cenderung lebih stabil dalam menangani ketidakseimbangan distribusi label dibandingkan Logistic Regression. Secara teknis, keunggulan SVM ini disebabkan oleh kemampuannya dalam menemukan _hyperplane_ optimal yang memaksimalkan margin antar kelas, bahkan dalam ruang fitur berdimensi tinggi dan jarang (sparse high-dimensional space) yang dihasilkan oleh ekstraksi fitur TF-IDF.

Di sisi lain, CC Logistic Regression menunjukkan performa terendah dengan _Hamming Loss_ tertinggi sebesar 0,0836. Rendahnya performa pada pendekatan _Classifier Chains_ (CC) ini mengindikasikan adanya fenomena _error propagation_ (perambatan kesalahan). Pada _threshold_ default 0,5, jika model pertama dalam rantai gagal memprediksi label dengan tepat, kesalahan tersebut akan diteruskan sebagai fitur ke model berikutnya dalam rantai, sehingga memperburuk hasil prediksi label-label selanjutnya. Nilai _Hamming Loss_ yang lebih rendah pada model SVM membuktikan tingkat kesalahan prediksi label individual yang lebih sedikit secara rata-rata dibandingkan model Logistic Regression.

Perbedaan antara _Micro_ F1 dan _Macro_ F1 menunjukkan adanya pengaruh ketimpangan data (_class imbalance_). Nilai _Macro_ F1 yang secara konsisten lebih rendah dari _Micro_ F1 pada seluruh model menandakan bahwa pada _threshold_ 0,5, model klasik masih kesulitan mengenali kategori minoritas yang memiliki jumlah sampel sedikit. Hal ini dikarenakan probabilitas prediksi untuk label-label minoritas cenderung "tertekan" dan sulit menembus angka 0,5, sehingga banyak kasus ujaran kebencian spesifik yang tidak terdeteksi.

Secara umum, evaluasi ini menunjukkan bahwa penggunaan _threshold default_ 0,5 belum menghasilkan performa yang optimal dalam menyeimbangkan hasil antar-label. Temuan ini menjadi dasar bagi dilakukannya penyesuaian _threshold_ per label yang akan dibahas pada subbab selanjutnya.

### Hasil Model _Machine learning_ Klasik (_Threshold_ _Tuned)_

Subbab ini menyajikan hasil evaluasi model _machine learning_ klasik setelah dilakukan _threshold_ _tuning_ per label. Penyesuaian _threshold_ dilakukan untuk mengatasi keterbatasan penggunaan _threshold_ default (0,5) pada data _multilabel_ yang memiliki distribusi label tidak seimbang. _Threshold_ ditentukan secara independen untuk setiap label, dengan tujuan meningkatkan performa _Macro F1-score_ sebagai metrik utama evaluasi. _Threshold_ optimal per label bisa dilihat pada tabel berikut:

Tabel 4.4 _Threshold_ Optimal per Label pada Model Klasik

| **Model**                  | **_Threshold_ per Label**              |
| -------------------------- | -------------------------------------- |
| _BR_ _Logistic Regression_ | \[0,50, 0,60, 0,58, 0,72, 0,76, 0,62\] |
| ---                        | ---                                    |
| _BR_ Calibrated SVM        | \[0,34, 0,30, 0,30, 0,32, 0,30, 0,16\] |
| ---                        | ---                                    |
| CC _Logistic Regression_   | \[0,50, 0,58, 0,62, 0,76, 0,80, 0,66\] |
| ---                        | ---                                    |
| CC Calibrated SVM          | \[0,34, 0,56, 0,20, 0,42, 0,48, 0,18\] |
| ---                        | ---                                    |

Perbedaan nilai _threshold_ pada setiap label menunjukkan bahwa tiap kategori ujaran kebencian memiliki karakteristik probabilitas yang unik. Model SVM secara konsisten menghasilkan skor probabilitas yang cenderung lebih rendah, dengan rentang _threshold_ optimal antara 0,16 hingga 0,56, dibandingkan dengan Logistic Regression yang berada pada rentang 0,50 hingga 0,80. Rendahnya skor pada SVM ini mengindikasikan adanya pemusatan nilai (_score compression_) pada label minoritas. Oleh karena itu, penurunan ambang batas diperlukan untuk meningkatkan sensitivitas model dalam mendeteksi pola kebencian. Penggunaan satu _threshold_ global 0,5 terbukti tidak efektif karena tidak mampu mengakomodasi keberagaman karakteristik tiap label.

Tabel 4.5 Perbandingan Kinerja Model Klasik Sebelum dan Sesudah _Threshold_ Tuning

| **Model**              | **Hamming Loss** | **Micro F1** | **Macro F1** |
| ---------------------- | ---------------- | ------------ | ------------ |
| BR Logistic Regression | 0,0743           | 0,6583       | 0,6005       |
| BR LR (Tuned)          | 0,0655           | 0,6645       | 0,6065       |
| BR SVM                 | 0,0600           | 0,6297       | 0,5636       |
| BR SVM (Tuned)         | 0,0668           | 0,6615       | 0,6008       |
| CC Logistic Regression | 0,0837           | 0,6210       | 0,5501       |
| CC LR (Tuned)          | 0,0668           | 0,6578       | 0,5991       |
| CC SVM                 | 0,0584           | 0,6570       | 0,6174       |
| CC SVM (Tuned)         | 0,0611           | 0,6754       | 0,6241       |

Berdasarkan hasil evaluasi pada Tabel 4.5, penerapan _threshold tuning_ secara konsisten meningkatkan _Macro F1-score_ pada seluruh konfigurasi model. Peningkatan signifikan terlihat pada _BR_ SVM yang naik menjadi 0,6008. Meskipun terdapat sedikit _trade-off_ berupa kenaikan _Hamming Loss_ dari 0,0600 menjadi 0,0668, hal ini menunjukkan bahwa model kini lebih berani mengambil risiko untuk memprediksi label positif pada kategori langka demi mengejar keseimbangan performa (_Macro F1_).

Efektivitas penyesuaian ambang batas juga sangat menonjol pada pendekatan _Classifier Chains_ (CC). Jika pada subbab sebelumnya pendekatan CC terhambat oleh masalah _error propagation_, hasil _tuning_ pada CC Logistic Regression membuktikan bahwa kendala tersebut dapat dimitigasi. Lonjakan performa dari 0,5501 ke 0,5991 disertai penurunan _Hamming Loss_ yang signifikan menunjukkan bahwa dengan _threshold_ yang tepat, model di awal rantai (_chain_) dapat memberikan informasi fitur yang lebih akurat bagi model selanjutnya. Hal ini memungkinkan dependensi antar-label dimanfaatkan sebagai sinyal penguat, bukan sebagai pembawa bias kesalahan.

Hasil terbaik dicapai oleh CC Calibrated SVM yang memperoleh nilai _Macro F1-score_ sebesar 0,6241. Secara keseluruhan, temuan ini menegaskan bahwa _threshold tuning_ per label merupakan komponen krusial dalam menangani ketidakseimbangan distribusi label pada data teks media sosial. Konfigurasi CC Calibrated SVM terpilih sebagai model klasik terbaik karena kemampuannya dalam menyeimbangkan pengenalan pola leksikal dengan keterkaitan antar-label, yang selanjutnya akan digunakan sebagai pembanding utama bagi model berbasis _Transformer_

### Hasil Model IndoBERT

Subbab ini menyajikan hasil evaluasi model IndoBERT pada tugas klasifikasi ujaran kebencian _multilabel_. Evaluasi dilakukan menggunakan dua skenario, yaitu _threshold_ default dan _threshold_ hasil _tuning_ per label, untuk menilai pengaruh penyesuaian _threshold_ terhadap kinerja model. Evaluasi dilakukan pada data uji yang sama dengan model _machine learning_ klasik, sehingga hasil yang diperoleh dapat dibandingkan secara langsung.

Pada tahap awal, model IndoBERT dievaluasi menggunakan _threshold_ default (0.5) untuk seluruh label. Hasil evaluasi ditunjukkan pada Tabel 4.6.

Tabel 4.6 Hasil Evaluasi IndoBERT dengan _Threshold_ Default

| **Metrik**     | **Nilai** |
| -------------- | --------- |
| _Micro F1_     | 0.72      |
| ---            | ---       |
| _Macro F1_     | 0.65      |
| ---            | ---       |
| _Hamming Loss_ | 0.0496    |
| ---            | ---       |

Berdasarkan hasil pada Tabel 4.6, IndoBERT menunjukkan performa yang cukup stabil dengan nilai _Macro_ _F1-score_ sebesar 0,65. Pencapaian ini sangat signifikan karena nilai _baseline_ IndoBERT (0,65) secara langsung melampaui performa terbaik dari seluruh model klasik yang telah di-_tuning_ pada Subbab 4.2.2 (0,6174). Hal ini membuktikan superioritas representasi fitur berbasis vektor padat (_dense vector_) dan konteks. Berbeda dengan TF-IDF yang hanya mengandalkan bobot statistik kata secara individual, IndoBERT memanfaatkan mekanisme _self-attention_ untuk memahami hubungan semantik antar-kata, sehingga mampu mengenali ujaran kebencian meskipun tidak menggunakan kata makian yang eksplisit.

Nilai _Hamming Loss_ yang rendah (0,0496) menunjukkan bahwa kesalahan prediksi label individual relatif kecil. Namun demikian, hasil per label menunjukkan bahwa beberapa kategori dengan jumlah data lebih sedikit, seperti _HS_Physical_ dan _HS_Gender_, masih memiliki nilai skor yang rendah. Hal ini mengonfirmasi temuan pada model klasik bahwa penggunaan _threshold_ 0,5 terlalu tinggi bagi kategori minoritas. Pada distribusi data yang sangat tidak seimbang (_imbalanced_), probabilitas prediksi untuk label dengan jumlah sampel sedikit cenderung sulit mencapai angka 0,5. Akibatnya, banyak kasus ujaran kebencian yang "nyaris terdeteksi" justru diklasifikasikan sebagai teks aman (_False Negative_), yang berdampak pada rendahnya nilai _recall_ sistem.

Untuk mengatasi keterbatasan tersebut, dilakukan _threshold tuning_ secara independen. Adapun nilai _threshold_ optimal yang ditemukan untuk IndoBERT adalah \[0.16, 0.64, 0.18, 0.66, 0.12, 0.34\]. _Threshold_ tersebut secara berurutan adalah untuk label HS*Individual (0.16), HS_Group (0.64), HS_Religion (0.18), HS_Race (0.66), HS_Physical (0.12), dan HS_Gender (0.34). Pola \_threshold* optimal menunjukkan korelasi negatif yang kuat dengan Imbalance Ratio yang telah diidentifikasi pada Tabel 3.2. Label dengan ketidakseimbangan tertinggi seperti HS*Physical (Imbalance Ratio: 11,07) dan HS_Gender (Imbalance Ratio: 11,68) memerlukan \_threshold* yang paling rendah (0,12 dan 0,34) untuk mengkompensasi penenekanan probabilitas akibat minimnya sampel pelatihan. Sebaliknya, label dengan distribusi lebih seimbang seperti HS*Individual (Imbalance Ratio: 1,00) dapat menggunakan \_threshold* yang juga rendah (0,16) namun dengan margin kesalahan yang lebih kecil karena model memiliki representasi fitur yang le bih kuat dari eksposur data yang cukup. Label HS*Group dan HS_Race menariknya justru memerlukan \_threshold* yang tinggi (0,64 dan 0,66), yang mengindikasikan bahwa meskipun jumlah sampelnya tidak sebanyak HS*Individual, pola linguistik pada kedua kategori ini lebih mudah dikenali model, sehingga probabilitas prediksinya cenderung lebih tinggi dan memerlukan \_threshold* yang lebih ketat untuk menghindari _false_ _positive_.

Penurunan drastis pada label spesifik seperti HS*Physical (0,12) dan HS_Gender (0,34) menunjukkan bahwa model \_Transformer* sebenarnya menangkap sinyal kebencian pada kategori tersebut, namun skor keyakinannya "tertekan" akibat minimnya sampel pelatihan. Hasil evaluasi setelah _threshold_ _tuning_ ditunjukkan pada Tabel 4.7.

Tabel 4.7 Hasil Evaluasi IndoBERT Setelah _Threshold_ Tuning

| **Metrik**     | **Nilai** |
| -------------- | --------- |
| _Micro F1_     | 0.73      |
| ---            | ---       |
| _Macro F1_     | 0.69      |
| ---            | ---       |
| _Hamming Loss_ | 0.0493    |
| ---            | ---       |

Kenaikan _Macro F1-score_ dari 0,65 ke 0,69 membuktikan bahwa mengatur ulang batas keputusan (_threshold_) sangat efektif untuk mengatasi ketimpangan distribusi kelas pada model _Transformer_. Dengan menurunkan _threshold_, terutama pada label minoritas seperti _HS_Physical_ dan _HS_Gender_, model menjadi lebih sensitif dalam mendeteksi pola-pola kebencian yang jarang muncul dalam data latih. Peningkatan ini terutama disebabkan oleh naiknya nilai _recall_ tanpa mengorbankan _precision_ secara signifikan. Nilai _Hamming Loss_ yang tetap rendah dan stabil (0,0493) mengindikasikan bahwa peningkatan sensitivitas terhadap label minoritas tidak menyebabkan lonjakan kesalahan prediksi label secara keseluruhan (_False Positive_).

Secara keseluruhan, model IndoBERT menunjukkan kinerja yang paling unggul dibandingkan seluruh konfigurasi model _machine learning_ klasik, baik sebelum maupun sesudah _threshold tuning_. Nilai _Macro F1-score_ sebesar 0,69 merupakan pencapaian tertinggi dalam penelitian ini. Hasil ini menegaskan bahwa kombinasi antara arsitektur _deep learning_ yang mampu memahami konteks semantik dengan strategi optimasi batas keputusan (_threshold tuning_) merupakan pendekatan yang sangat baik untuk menangani kompleksitas klasifikasi ujaran kebencian _multilabel_ pada data media sosial yang tidak seimbang.

### Analisis Performa Per Label

Subbab ini menyajikan perbandingan mendalam terhadap performa keempat konfigurasi model yang diuji dalam penelitian, yaitu CC-SVM dengan _threshold_ default dan _Tuned_, serta IndoBERT dengan _threshold_ default dan _Tuned_. Tabel 4.8 menyajikan perbandingan nilai _F1-score_ untuk setiap kategori ujaran kebencian pada keempat konfigurasi model yang diuji. Perbandingan ini memberikan gambaran komprehensif mengenai kemampuan masing-masing model dalam mengenali pola ujaran kebencian pada kategori yang berbeda.

Tabel 4.8 Perbandingan _F1-score_ Per Label pada Model Klasik dan IndoBERT

| **Label**     | **CC-SVM (Default)** | **CC-SVM (_Tuned_)** | **IndoBERT (Default)** | **IndoBERT (_Tuned_)** | **Δ Tuning (IndoBERT)** |
| ------------- | -------------------- | -------------------- | ---------------------- | ---------------------- | ----------------------- |
| HS_Individual | 0.6858               | 0.7202               | 0.7587                 | 0.7634                 | +0.0047                 |
| ---           | ---                  | ---                  | ---                    | ---                    | ---                     |
| HS_Group      | 0.6273               | 0.6206               | 0.7074                 | 0.7099                 | +0.0025                 |
| ---           | ---                  | ---                  | ---                    | ---                    | ---                     |
| HS_Religion   | 0.6255               | 0.6513               | 0.7200                 | 0.7591                 | +0.0391                 |
| ---           | ---                  | ---                  | ---                    | ---                    | ---                     |
| HS_Race       | 0.6961               | 0.7019               | 0.7296                 | 0.7511                 | +0.0215                 |
| ---           | ---                  | ---                  | ---                    | ---                    | ---                     |
| HS_Physical   | 0.5243               | 0.5385               | 0.4419                 | 0.5688                 | +0.1269                 |
| ---           | ---                  | ---                  | ---                    | ---                    | ---                     |
| HS_Gender     | 0.4490               | 0.4259               | 0.5149                 | 0.5794                 | +0.0645                 |
| ---           | ---                  | ---                  | ---                    | ---                    | ---                     |

Berdasarkan Tabel 4.8, terlihat bahwa IndoBERT secara konsisten mengungguli model klasik CC-SVM pada hampir seluruh kategori. Pada _threshold_ default, IndoBERT sudah memberikan _F1-score_ yang lebih tinggi dibandingkan CC-SVM bahkan setelah dilakukan _threshold_ tuning, kecuali pada label HS*Physical. Peningkatan performa yang paling signifikan akibat \_threshold* tuning terlihat pada label minoritas. Label HS*Physical mengalami lonjakan \_F1-score* sebesar 0.1269 (dari 0.4419 ke 0.5688), yang merupakan peningkatan relatif sebesar 28.7%. Hal ini membuktikan bahwa penggunaan _threshold_ default 0.5 sangat merugikan label dengan jumlah sampel sedikit, karena probabilitas prediksi model pada kategori minoritas cenderung tertekan (compressed) akibat ketidakseimbangan data latih. Dengan menurunkan _threshold_ menjadi 0.12 khusus untuk HS_Physical, model menjadi lebih sensitif dalam mendeteksi ujaran kebencian terkait kondisi fisik seseorang.

Label HS*Gender menunjukkan peningkatan \_F1-score* sebesar 0.0645 melalui _threshold tuning_. Variasi linguistik yang tinggi menyebabkan peningkatan skor ini tidak melampaui HS*Physical. Penggunaan istilah ambigu seperti "banci" atau "maho" menyulitkan model dalam membedakan konteks serangan. Kondisi tersebut menghambat kenaikan nilai \_recall* meskipun sistem telah menurunkan ambang batas ke 0.34.

Sebaliknya, label mayoritas seperti HS*Individual dan HS_Group menunjukkan peningkatan yang lebih moderat (masing-masing +0.0047 dan +0.0025). Hal ini wajar karena kedua kategori ini sudah memiliki \_F1-score* baseline yang tinggi (>0.70), sehingga ruang perbaikan terbatas. Lebih lanjut, pada _threshold_ default 0.5, model sudah cukup sensitif mendeteksi pola ujaran kebencian individual dan kelompok karena frekuensi kemunculannya yang tinggi dalam data latih, membuat representasi fitur kedua kategori ini lebih kuat.

Label HS*Religion mengalami peningkatan moderat (+0.0391). Meskipun memiliki Imbalance Ratio cukup tinggi (4.51), label ini menunjukkan performa yang relatif stabil. Ujaran kebencian berbasis agama cenderung menggunakan kata kunci eksplisit seperti "kafir", "murtad", atau "sesat" yang memiliki asosiasi kuat dengan kategori ini. Konsistensi penggunaan kata kunci tersebut memudahkan model, baik klasik maupun \_Transformer*, untuk mengenali pola kebencian agama, sehingga penyesuaian _threshold_ memberikan dampak yang lebih terbatas dibandingkan kategori lain.

Secara keseluruhan, hasil analisis per label ini menegaskan tiga temuan kunci:

- Model berbasis _Transformer_ (IndoBERT) secara fundamental lebih unggul dalam memahami konteks ujaran kebencian dibandingkan model klasik
- _Threshold_ tuning adalah strategi krusial untuk mengatasi ketidakseimbangan data, dengan dampak terbesar pada label minoritas, dan
- Karakteristik linguistik spesifik setiap kategori (eksplisitas kata kunci, ambiguitas konteks, variasi istilah) mempengaruhi sejauh mana penyesuaian _threshold_ dapat meningkatkan performa.

### Analisis Kesalahan Model

Subbab ini melakukan analisis mendalam terhadap pola kesalahan yang dilakukan oleh model terbaik (IndoBERT dengan _threshold tuned_) untuk memahami keterbatasan sistem deteksi dan mengidentifikasi area yang memerlukan perbaikan. Analisis dilakukan melalui dua pendekatan yakni evaluasi kuantitatif menggunakan _confusion matrix_ per label, dan analisis kualitatif terhadap contoh kesalahan _false positive_ dan _false negative_ yang representatif. Kombinasi kedua pendekatan ini memungkinkan identifikasi tidak hanya seberapa sering model salah (aspek kuantitatif), tetapi juga mengapa model salah pada kasus-kasus tertentu (aspek kualitatif), sehingga memberikan wawasan yang lebih komprehensif untuk perbaikan model di masa mendatang.

_Confusion matrix_ memberikan gambaran detail mengenai distribusi kesalahan prediksi model untuk setiap label. Gambar 4.2 menyajikan _heatmap confusion matrix_ untuk keenam kategori ujaran kebencian pada model IndoBERT dengan _threshold tuned_.


Gambar 4.2 Heatmap Confusion Matrix model IndoBERT dengan _Threshold_ _Tuned_

Berdasarkan visualisasi _confusion matrix_, terlihat bahwa model memiliki performa yang sangat baik pada label mayoritas namun mengalami kesulitan pada label minoritas. Label HS*Individual memiliki \_True* _Positive_ (TP) tertinggi sebesar 563 dari 715 kasus positif aktual, menghasilkan recall 0.7874. Namun, model juga menghasilkan 197 kasus _False_ _Positive_ (FP), yang berarti terdapat hampir 200 teks netral yang keliru diklasifikasikan sebagai ujaran kebencian individual. Penurunan treshold menjadi 0.16 meningkatkan sensitivitas deteksi secara signifikan, tetapi juga menyebabkan penurunan nilai presisi model sebagai konsekuensi teknis.

Label minoritas menunjukkan pola kegagalan deteksi yang ekstrem. Model hanya mengidentifikasi 31 dari 65 kasus positif pada label HS*Physical. Penurunan ambang batas ke nilai 0.12 tidak memberikan dampak signifikan terhadap performa sistem. Kuantitas data latih yang terbatas menghambat kemampuan model dalam mempelajari variasi linguistik spesifik. Keterbatasan sebaran data tersebut menyebabkan representasi semantik pada ruang vektor IndoBERT menjadi kurang optimal. Arsitektur \_Transformer* memerlukan frekuensi kemunculan pola yang tinggi untuk mengenali diksi tidak langsung secara akurat.

Untuk memahami lebih mendalam jenis kesalahan yang dilakukan model, Tabel 4.9 menyajikan contoh representatif dari kasus _False_ _Positive_ (FP) dan _False_ _Negative_ (FN) untuk setiap label. Contoh dipilih berdasarkan dua kriteria yakni FP dengan probabilitas prediksi tertinggi (model sangat yakin namun salah), dan FN dengan probabilitas paling mendekati _threshold_ (near-miss cases).

Tabel 4.9 Contoh Kesalahan _False_ _Positive_ dan _False_ _Negative_ Per Label

| **Label**            | **Jenis** | **Prob.** | **Teks**                                                                                                          | **Analisis Kesalahan**                                                                                                                                                                           |
| -------------------- | --------- | --------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HS<br><br>Individual | FP        | 0.996     | "USER USER USER Mulut buaya dia mah"                                                                              | Frasa "mulut buaya" merupakan idiom untuk orang yang suka berbohong, bukan serangan personal. Model salah menginterpretasi sebagai ujaran individu karena keberadaan kata "dia".                 |
| HS<br><br>Individual | FN        | 0.141     | "USER USER USER USER USER dulu sahabat sekarang jadi bangsad"                                                     | Kata "bangsad" adalah makian personal eksplisit, namun probabilitas (0.141) sedikit di bawah _threshold_ (0.16). Konteks "dulu sahabat sekarang" mungkin membingungkan model.                    |
| HS<br><br>Group      | FP        | 0.990     | "Gak usah baca, gw dah tau kalo lou itu bani kopar kapir koplak..."                                               | Meskipun terdapat kata "bani" (kelompok), konteks menunjukkan ini adalah makian personal, bukan serangan kelompok spesifik. Kata "kapir" memicu deteksi kelompok.                                |
| HS<br><br>Group      | FN        | 0.600     | "USER USER USER USER USER Anjing polisi polisi anjing keparat"                                                    | Serangan eksplisit terhadap institusi polisi, namun probabilitas (0.600) di bawah _threshold_ tinggi (0.64). Model mungkin menganggap ini kritik kasar, bukan _hate speech_ kelompok.            |
| HS<br><br>Religion   | FP        | 0.971     | "Diluruskan agar tidak menyimpang dari ajaran Quran, tentu tidak boleh Menteri Agama Islam menyimpang dari Quran" | Teks ini merupakan kritik kebijakan agama yang faktual dan tidak mengandung kebencian. Kata "Quran" dan "Islam" memicu _false_ _positive_ meskipun konteks netral.                               |
| HS<br><br>Religion   | FN        | 0.141     | "USER Ahhh , ulama karbitan gomong gak pernah membawa kesejukan..."                                               | Istilah "ulama karbitan" (ulama palsu) adalah kritik keras terhadap figur agama, namun model menganggap ini sebagai kritik personal, bukan kebencian agama.                                      |
| HS<br><br>Race       | FP        | 0.992     | "Cap gome dihalaman masjid= tabligh akbar di halm gereja..."                                                      | Konteks membandingkan acara agama di tempat ibadah lain. Kata "gome" (_slang_ untuk etnis Tionghoa) muncul dalam konteks perbandingan, bukan ujaran kebencian.                                   |
| HS<br><br>Race       | FN        | 0.569     | "USER USER USER USER USER USER USER Cebong itu badut nya China..."                                                | Serangan politik dengan menyebut "China" dan "cebong" (pendukung petahana). Probabilitas (0.569) di bawah _threshold_ tinggi (0.66), model tidak menangkap dimensi rasial.                       |
| HS<br><br>Physical   | FP        | 0.753     | "...si Ruben idih banget dah, kemaren kemarwn blg ke Nikita gausa ladenin orang sarap..."                         | Kritik terhadap perilaku ("orang sarap" = orang gila), bukan kondisi fisik. Model keliru karena kata "orang" + kata kasar. _Threshold_ rendah (0.12) memperparah _false_ alarm.                  |
| HS<br><br>Physical   | FN        | 0.106     | "USER USER ASTAGA BOLOT BANGET DIA"                                                                               | "Bolot" adalah istilah untuk orang gemuk yang bersifat body-shaming eksplisit. Probabilitas sangat dekat _threshold_ (0.12), namun model tidak cukup yakin karena tidak ada kata pendukung lain. |
| HS<br><br>Gender     | FP        | 0.981     | "USER USER Si lucinta lunaa El si banci tp geulis da oplas"                                                       | Konteks membahas public figure transgender dengan nada netral ("tapi cantik karena oplas"). Kata "banci" muncul faktual, bukan sebagai hinaan.                                                   |
| HS<br><br>Gender     | FN        | 0.307     | "USER Indahnya cocot pks. Kalau ngomong .ketum aja kaya bencong gaya bicaranya..."                                | Penggunaan "bencong" untuk mengejek gaya bicara politisi. Model menganggap ini kritik politik, tidak mendeteksi dimensi ujaran gender karena target adalah figur politik, bukan individu LGBT.   |

Analisis terhadap kasus-kasus kesalahan di atas mengungkapkan beberapa pola sistematis:

Ambiguitas Konteks dan Kata Kunci: Model sulit membedakan penggunaan kata kunci antara konteks netral dan ujaran kebencian. Kata dominan seperti "China" atau "kafir" menjadi faktor penentu utama klasifikasi pada label berambang batas rendah. Mekanisme _self-attention_ belum sepenuhnya mampu mengatasi bias kata kunci dalam diskusi faktual.

Keterbatasan Deteksi Implisit: Model gagal mendeteksi ujaran kebencian tidak langsung melalui istilah metaforis. Kata seperti "bolot" atau "bencong" memiliki representasi semantik yang lemah dalam ruang vektor IndoBERT. Frekuensi kemunculan istilah tersebut yang rendah dalam data latih menghambat kemampuan generalisasi model.

Tumpang Tindih Antar Kategori: Kesalahan prediksi muncul akibat dominasi sinyal pada kategori tertentu dalam teks _multilabel_. Model cenderung memprioritaskan kategori dengan asosiasi kata paling kuat seperti istilah "bani" pada label HS_Group. Kondisi ini menyebabkan pengabaian dimensi kebencian lain yang relevan secara simultan.

Dampak Ambang Batas Ekstrem: Penggunaan ambang batas sangat rendah memicu pertukaran (_trade-off_) antara _recall_ dan _precision_. Ambang batas 0.12 pada HS_Physical meningkatkan temuan kasus positif namun menambah jumlah alarm palsu. Penurunan ambang batas ini mengakibatkan banyak prediksi tidak yakin masuk ke dalam kategori positif.

Secara keseluruhan, analisis kesalahan ini mengonfirmasi bahwa deteksi ujaran kebencian _multilabel_ pada media sosial adalah permasalahan yang sangat kompleks. Keterbatasan sistem bersumber dari ambiguitas bahasa manusia dan keterbatasan data latih. Penelitian selanjutnya memerlukan pengayaan data ambigu dan integrasi pengetahuan eksternal untuk meningkatkan akurasi konteks.

Analisis kuantitatif yang telah dilakukan pada subbab-subbab sebelumnya menunjukkan performa model dari sisi metrik agregat dan distribusi kesalahan. Untuk memahami secara mendalam mengapa model membuat keputusan tertentu terutama pada kasus kesalahan, diperlukan pendekatan interpretabilitas yang mampu membedah kontribusi setiap kata terhadap prediksi. Subbab berikut menyajikan analisis _Explainable_ AI (XAI) menggunakan KernelSHAP untuk membandingkan mekanisme pengambilan keputusan antara model klasik dan IndoBERT pada kasus-kasus representatif.

## Analisis Interpretabilitas Model

Subbab ini menyajikan analisis interpretabilitas terhadap mekanisme pengambilan keputusan model klasifikasi _multilabel_ menggunakan metode KernelSHAP. Analisis difokuskan pada penjelasan lokal (_local explanation_) untuk mengetahui kontribusi setiap kata (fitur) terhadap probabilitas prediksi pada sampel teks tertentu.

Berbeda dengan evaluasi performa pada subbab sebelumnya, analisis ini bertujuan untuk memberikan transparansi mengenai mengapa model memprediksi label tertentu. Pendekatan yang digunakan adalah studi kasus dengan membandingkan hasil interpretasi antara model klasik (CC-SVM) dan model _Transformer_ (IndoBERT) pada kategori ujaran kebencian eksplisit.

### Analisis Studi Kasus: Ujaran Kebencian Eksplisit

Deskripsi Kalimat Uji

Kalimat yang dipilih untuk studi kasus ini merupakan ujaran kebencian yang mengandung makian literal dan target yang jelas. Kalimat yang digunakan yakni berasal dari data tes yang sudah dibuat sebelumnya dan belum pernah dilihat oleh model.

Tabel 4.10 Contoh Kalimat Uji (Setelah _Preprocessing_)

| **Teks**                                                                                                                       | **Label**                     |
| ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| sok mau memaki agama orang saja bawa data padahal otak tidak sampai dari awal goblok kamu cebong kafir kebanyakan makan tai ah | HS_Individual dan HS_Religion |
| ---                                                                                                                            | ---                           |

Kalimat ini dipilih karena memiliki kompleksitas serangan yang mencakup serangan personal ("goblok", "kamu", "otak tidak sampai") dan serangan berbasis SARA ("agama", "kafir"), sehingga relevan untuk menguji label HS_Individual dan HS_Religion.

- Hasil Prediksi _Multilabel_

Sebelum dilakukan ekstraksi nilai SHAP, dilakukan pengujian prediksi untuk melihat kesepakatan model. Hasil prediksi probabilitas ditunjukkan pada Tabel 4.11.

Tabel 4.11 Hasil Prediksi _Multilabel_ pada Kalimat Uji

| **Model** | **HS_Individual** | **HS_Religion** |
| --------- | ----------------- | --------------- |
| CC-SVM    | 0,8759 (Aktif)    | 0,7868 (Aktif)  |
| ---       | ---               | ---             |
| IndoBERT  | 0,9515 (Aktif)    | 0,9838 (Aktif)  |
| ---       | ---               | ---             |

Berdasarkan tabel tersebut, terlihat bahwa kedua model sepakat dalam memprediksi dua label ujaran kebencian yang sama, meskipun nilai probabilitas yang dihasilkan oleh IndoBERT lebih tinggi dibandingkan model klasik. Kedua model berhasil mengidentifikasi kedua label secara tepat (di atas ambang batas masing-masing). IndoBERT menunjukkan tingkat _confidence_ yang lebih tinggi dibandingkan CC-SVM, khususnya pada label HS_Religion (0,9838).

- Interpretasi KernelSHAP pada Model Klasik (CC-SVM)


Gambar 4.3 Visualisasi KernelSHAP CC-SVM untuk Label HS_Individual

Berdasarkan Gambar 4.3, fitur yang memberikan kontribusi positif terbesar untuk label HS*Individual adalah "sok" (0,1335), "cebong" (0,0955), dan "otak" (0,0679). Hal ini menunjukkan model SVM sangat bergantung pada keberadaan kata-kata kasar individual (\_unigram*) untuk menentukan kebencian terhadap individu.


Gambar 4.4 Visualisasi KernelSHAP CC-SVM untuk Label HS_Religion

Penjelasan untuk label HS_Religion yang tersaji pada Gambar 4.4 menunjukkan adanya dominasi fitur 'kafir' (0,3004) dan 'agama' (0,2363). Menariknya, kata "kafir" memiliki bobot yang jauh lebih tinggi daripada kata makian lain, membuktikan bahwa pada model klasik, kehadiran kata kunci (keywords) SARA adalah penentu utama label agama.

4\. Interpretasi SHAP pada Model IndoBERT

Analisis pada IndoBERT menggunakan _background dataset_ sebanyak 40 sampel netral untuk mendapatkan _baseline_ yang lebih representatif.


Gambar 4.5 Visualisasi KernelSHAP IndoBERT untuk Label HS_Individual

Hasil pada Gambar 4.5 sangat menarik, di mana kontribusi terbesar diberikan oleh kata "kamu" (0,5528). Hal ini berbeda drastis dengan model klasik. IndoBERT mampu menangkap bahwa kata ganti "kamu" dalam kalimat ini merupakan penunjuk target serangan individu. Kata pendukung kontekstual seperti "sok", "mau", dan "padahal" juga memberikan kontribusi signifikan, menunjukkan model memahami struktur serangan, bukan sekadar kata makian "goblok" (0,0638).


Gambar 4.6 Visualisasi KernelSHAP IndoBERT untuk Label HS_Religion

Pada Gambar 4.6, kata "kafir" (0,4858) menjadi fitur paling dominan. Penggunaan _baseline_ netral yang rendah (0,0010) memperlihatkan bagaimana satu kata eksplisit ini mampu melonjakkan probabilitas model secara ekstrem dari kondisi netral menuju prediksi 0,9838.

5\. Pembahasan Perbandingan Mekanisme Model

Perbandingan hasil SHAP antara kedua model memberikan beberapa temuan kunci:

- Leksikal vs Kontekstual: CC-SVM (Gambar 4.2) sangat dipengaruhi oleh kata makian eksplisit seperti 'goblok' dan 'cebong'. Ketergantungan ini muncul karena representasi TF-IDF pada model klasik hanya menitikberatkan pada bobot statistik kata tersebut secara individual. Sebaliknya, IndoBERT (Gambar 4.4) lebih menekankan pada subjek serangan ('kamu') dan kata-kata kontekstual ('padahal', 'sok'). Kemampuan ini dimungkinkan oleh arsitektur _Transformer_ yang mampu menangkap peran kata ganti dan kata pendukung dalam membangun narasi ujaran kebencian secara utuh.
- Transparansi _Multilabel_: KernelSHAP berhasil membedakan kata yang relevan untuk setiap label. Contohnya, kata "kafir" memiliki nilai SHAP negatif atau sangat rendah pada label HS_Individual (IndoBERT), namun menjadi fitur utama pada HS_Religion. Ini membuktikan model mampu melakukan disosiasi fitur antar label meskipun berada dalam satu kalimat yang sama.

Secara keseluruhan, implementasi KernelSHAP secara manual ini berhasil membuktikan bahwa IndoBERT tidak hanya "menghafal" kata makian, tetapi memahami peran sintaksis kata (seperti kata ganti "kamu") dalam membangun narasi ujaran kebencian.

### Analisis Studi Kasus Ujaran Kebencian Kontekstual

1\. Deskripsi Kalimat Uji

Kasus kedua menguji kemampuan model dalam mendeteksi ujaran kebencian yang tidak menggunakan kata makian kasar secara eksplisit, melainkan menggunakan narasi dehumanisasi dan sentimen penolakan kelompok. Detail kalimat dapat dilihat pada Tabel 4.10.

Tabel 4.12 Contoh Kalimat Uji Kontekstual

| **Teks**                                                                  | **Label**            |
| ------------------------------------------------------------------------- | -------------------- |
| cina tidak punya perikemanusiaan makanya aku tidak suka cina di indonesia | HS_Group dan HS_Race |
| ---                                                                       | ---                  |

2\. Hasil Prediksi _Multilabel_

Hasil probabilitas prediksi untuk kedua model ditunjukkan pada Tabel 4.15:

Tabel 4.13 Hasil Prediksi Kalimat Uji Kontekstual

| **Model** | **HS_Group**   | **HS_Race**    |
| --------- | -------------- | -------------- |
| CC-SVM    | 0,5890 (Aktif) | 0,9993 (Aktif) |
| ---       | ---            | ---            |
| IndoBERT  | 0,9972 (Aktif) | 0,9956 (Aktif) |
| ---       | ---            | ---            |

Kedua model berhasil mengidentifikasi label kebencian dengan nilai probabilitas yang sangat tinggi. Hal ini menarik untuk dibedah menggunakan KernelSHAP guna melihat apakah keduanya memiliki alasan yang sama dalam mengambil keputusan tersebut.

3\. Interpretasi SHAP pada Model Klasik (CC-SVM)

Pada model CC-SVM, interpretasi menunjukkan ketergantungan yang sangat tinggi pada kata kunci (_keyword_). Pola ini konsisten dengan karakteristik representasi TF-IDF yang bersifat leksikal, di mana bobot fitur ditentukan semata-mata oleh frekuensi kemunculan kata tanpa mempertimbangkan konteks semantik sekitarnya. Keterbatasan ini juga dipengaruhi oleh proses _feature extraction_ di mana kata "perikemanusiaan" tidak terdaftar dalam kosakata model akibat batasan frekuensi dokumen minimum (min*df=3) pada saat \_training*. Hal ini menyebabkan model klasik mengalami _vocabulary gap_ dan menjadi "buta" terhadap kata-kata formal yang jarang muncul dalam _dataset_. Analisis visual berikut menunjukkan bagaimana mekanisme ini bekerja pada setiap label yang diprediksi.


Gambar 4.7 Visualisasi KernelSHAP CC-SVM untuk Label HS_Race

Berdasarkan Gambar 4.7, kata "cina" (0,8832) memberikan kontribusi yang sangat dominan untuk Label HS_Race. Model klasik hampir sepenuhnya mengandalkan kemunculan kata benda etnis tersebut untuk mengklasifikasikan teks sebagai kebencian ras.


Gambar 4.8 Visualisasi KernelSHAP CC-SVM untuk Label HS_Group

Pola serupa terlihat pada Gambar 4.8, di mana kata "cina" kembali menjadi fitur terkuat (0,5272). Hal ini mengonfirmasi bahwa CC-SVM bersifat leksikal dimana model tidak memahami makna "tidak punya perikemanusiaan" karena fitur tersebut tidak tersedia dalam ruang fiturnya, melainkan hanya mendeteksi bahwa kata "cina" sering muncul pada kategori HS dalam data latih.

4\. Interpretasi SHAP pada Model IndoBERT

Analisis IndoBERT menunjukkan mekanisme yang lebih kompleks dan cerdas secara semantik.


Gambar 4.9 Visualisasi KernelSHAP IndoBERT untuk Label HS_Race

Label HS_Race: Berdasarkan Gambar 4.9, meskipun kata "cina" tetap menjadi pendorong utama, IndoBERT memberikan bobot yang signifikan pada kata "perikemanusiaan" (0,1686) dan kata negasi "tidak" (0,0871).


Gambar 4.10 Visualisasi KernelSHAP IndoBERT untuk Label HS_Group

Temuan paling signifikan terlihat pada Gambar 4.10. IndoBERT menempatkan fitur "perikemanusiaan" (0,3886) sebagai kontributor tertinggi, bahkan melampaui kata "cina" (0,3607). Ini membuktikan bahwa IndoBERT memahami bahwa inti dari kebencian kelompok pada kalimat ini adalah narasi dehumanisasi.

5\. Pembahasan dan Temuan Utama

Perbandingan mekanistik antara model klasik dan model _Transformer_ pada Kasus 2 ini menghasilkan temuan penting yakni terkait kemampuan semantik IndoBERT dalam mengatasi kendala _Out-of-Vocabulary_ (OOV). Berbeda dengan CC-SVM yang bersifat _keyword-heavy_ karena mengandalkan fitur leksikal, IndoBERT mampu mendistribusikan nilai kontribusi secara proporsional. Hal ini terlihat pada kemampuan model dalam mendeteksi narasi dehumanisasi, seperti pada frasa 'tidak punya perikemanusiaan'. IndoBERT memberikan bobot tinggi pada kata 'perikemanusiaan' (0,3886) karena memahami makna semantiknya dalam konteks serangan, bukan sekadar melihatnya sebagai kata netral. Sebaliknya, model klasik gagal menangkap sinyal ini bukan hanya karena bobot yang rendah, melainkan karena ketiadaan kata tersebut dalam _vocabulary_ akibat aturan ambang batas frekuensi (min_df). Temuan ini membuktikan bahwa IndoBERT memiliki generalisasi yang jauh lebih baik terhadap ujaran kebencian non-eksplisit dibandingkan model klasik yang bergantung pada kemunculan kata kunci makian.

## Implementasi Aplikasi _Web_

Subbab ini menyajikan penerapan dari keseluruhan penelitian ke dalam sebuah prototipe aplikasi _web_ interaktif. Berbeda dengan pengujian manual pada subbab sebelumnya, implementasi ini bertujuan untuk membuktikan bahwa model deteksi ujaran kebencian _multilabel_ dan metode _Explainable AI_ (XAI) dapat bekerja secara terpadu dalam sistem _real-time_. Aplikasi ini berfungsi sebagai alat bantu bagi pengguna untuk menguji teks, melihat hasil prediksi berdasarkan _threshold tuning_ yang telah dioptimasi, serta memahami alasan di balik keputusan model melalui visualisasi SHAP.

### Lingkungan Pengembangan dan Teknologi

Implementasi aplikasi _web_ ini dikembangkan menggunakan bahasa pemrograman Python dengan kerangka kerja (_framework_) Streamlit. Pemilihan Streamlit didasarkan pada kemampuannya untuk mengintegrasikan model pembelajaran mesin kompleks dan pustaka visualisasi data ke dalam antarmuka _web_ secara efisien. Berdasarkan arsitektur yang telah dibangun, komponen teknologi utama yang digunakan meliputi:

- TensorFlow & Hugging Face _Transformers_: Digunakan untuk memuat model IndoBERT yang telah melalui proses fine-_tuning_ beserta tokenizer _WordPiece_\-nya.
- Scikit-learn & Pickle: Digunakan untuk memuat model _Machine learning_ Klasik (_Logistic Regression_ dan SVM) yang disimpan dalam format artifact (.pkl), termasuk vektor fitur TF-IDF.
- Custom KernelSHAP: Fungsi komputasi yang dikembangkan secara mandiri menggunakan NumPy. Fungsi ini bertanggung jawab atas operasi aljabar linear untuk menghitung kontribusi fitur berdasarkan teori Shapley, menggantikan ketergantungan pada _black-box_ _library_ eksternal.
- Matplotlib: Digunakan sebagai mesin perender grafik batang (bar plot) SHAP yang menampilkan peringkat kontribusi kata terhadap probabilitas label.

### Antarmuka Pengguna dan Mekanisme Prediksi

Antarmuka aplikasi dirancang untuk memudahkan interaksi antara pengguna dengan model yang kompleks. Pada bagian ini, pengguna diberikan kendali penuh melalui beberapa komponen kontrol utama yang terintegrasi.

- Input Teks dan Kontrol Model: Pengguna memasukkan kalimat melalui text area. Melalui bilah samping atau tombol radio, pengguna dapat menentukan arsitektur model yang digunakan (IndoBERT atau Klasik). Berdasarkan kode app.py, aplikasi juga menyediakan _input_ angka untuk menentukan jumlah sampel _KernelSHAP_ (default 256), yang memberikan fleksibilitas antara kecepatan komputasi dan presisi hasil interpretasi.
- Proses Pra-pemrosesan _real-time_: Saat tombol "Mulai Analisis" ditekan, fungsi clean_for_shap dijalankan untuk melakukan normalisasi teks secara otomatis.


Gambar 4.11 Tampilan Antarmuka Halaman Utama

- Output Prediksi Berbasis _Threshold_: Setelah proses inferensi, aplikasi tidak hanya menampilkan probabilitas mentah, tetapi langsung melakukan klasifikasi biner menggunakan nilai ambang batas (_threshold_) yang telah dioptimasi pada tahap evaluasi sebelumnya. Untuk IndoBERT, aplikasi menerapkan BERT\_\_THRESHOLD_S yang spesifik untuk setiap label guna memastikan akurasi deteksi yang maksimal.

### Visualisasi Hasil Prediksi _Multilabel_

Setelah sistem melakukan proses inferensi, aplikasi menyajikan hasil klasifikasi dalam bentuk tabel yang informatif. Bagian ini dirancang agar pengguna dapat melihat transparansi skor yang dihasilkan model sebelum dikonversi menjadi keputusan akhir melalui nilai ambang batas (_threshold_).

- Indikator Status Keamanan: Aplikasi menggunakan komponen st.error untuk memberikan peringatan visual jika teks terdeteksi mengandung ujaran kebencian, lengkap dengan kategori label yang aktif. Sebaliknya, st.success digunakan untuk memberikan konfirmasi jika teks dinyatakan aman.
- Tabel Probabilitas dan _Threshold_: Berdasarkan logika kode, tabel hasil menampilkan empat kolom utama: Label, Probabilitas (%), _Threshold_, dan Hasil. Penggunaan kolom _threshold_ yang berbeda-beda untuk setiap label menunjukkan penerapan optimasi yang telah dibahas pada tahap eksperimen sebelumnya. Hal ini memberikan pemahaman kepada pengguna bahwa setiap kategori kebencian memiliki standar sensitivitas yang berbeda tergantung pada distribusi datanya.


Gambar 4.12 Tabel Hasil Prediksi dan Status Deteksi pada Aplikasi

### Penjelasan Kontribusi Kata (SHAP)

Bagian ini merupakan fitur utama aplikasi yang mengimplementasikan metode XAI untuk memberikan transparansi pada tingkat kata atau token.

- Mekanisme _Highlight_ Kontekstual

Dengan menggunakan fungsi shap_text_highlight, aplikasi memberikan pewarnaan latar belakang pada setiap kata dalam kalimat. Warna merah menunjukkan kontribusi positif yang memperkuat prediksi label tertentu, sementara warna biru menunjukkan kontribusi negatif. Intensitas warna diatur secara dinamis melalui normalisasi nilai SHAP, sehingga kata-kata "pemicu" utama akan terlihat lebih mencolok bagi pengguna.

- Integrasi Tab Per Label

Mengingat sifat model yang _multilabel_, aplikasi menyediakan fitur st.tabs yang memisahkan penjelasan SHAP untuk setiap kategori (Individu, Kelompok, Agama, dsb.). Hal ini memungkinkan pengguna melakukan analisis mendalam secara terpisah, misalnya melihat mengapa sebuah teks dianggap menghina "Agama" namun dianggap aman bagi "Gender".


Gambar 4.13 Visualisasi Text Highlighting dan Plot Batang SHAP Per Label

Implementasi kedua fitur visualisasi ini dapat dilihat pada Gambar 4.13 yang menampilkan tampilan aktual antarmuka aplikasi dengan contoh hasil deteksi dan penjelasan SHAP untuk setiap kategori ujaran kebencian. Melalui visualisasi ini, pengguna tidak hanya menerima hasil klasifikasi biner (Ya/Tidak), tetapi juga memahami kontribusi spesifik setiap kata terhadap keputusan model, sehingga sistem menjadi lebih transparan dan dapat dipertanggungjawabkan (_accountable_). Transparansi ini menjadi krusial terutama dalam konteks moderasi konten, di mana keputusan otomatis harus dapat diaudit dan dijelaskan kepada pengguna yang terkena dampak.

#
