# Laporan audit kesesuaian proyek Wayground

Tanggal pemeriksaan: 29 Oktober 2026 (Asia/Jakarta).

## 1. Kesimpulan utama

**Proyek belum konsisten secara menyeluruh dari dataset dan tujuh eksperimen sampai model terpasang dan visualisasi web.** Fondasi inference sudah berjalan, tetapi versi kode, dataset, hasil eksperimen, model biner, dokumentasi, dan gambar bercampur. Karena itu, belum tepat menyebut seluruh implementasi sebagai satu konfigurasi final yang sudah tervalidasi.

Audit ini tidak mengubah script, dataset, konfigurasi, model, maupun gambar penelitian. File laporan ini merupakan keluaran baru. Pengujian backend memakai SQLite di memori; build frontend disimpan di direktori visualisasi Codex, di luar source frontend. Workspace memiliki sejumlah perubahan Git yang tidak disentuh dalam audit.

Pemeriksaan mencakup pembacaan alur dan implementasi, perhitungan ulang kualitas dataset dan metrik, pembacaan langsung joblib, pengujian API dengan Flask test client, pengujian terpilih yang tersedia, build frontend, dan inspeksi dua gambar utama. **Tujuh eksperimen diaudit dari script dan hasil tersimpannya, bukan dijalankan ulang seluruh grid.** Tidak dilakukan scraping jaringan, pelatihan final ulang, pemilihan konfigurasi baru, atau evaluasi holdout baru untuk tuning. Sampel API kecil hanya menguji fungsi, bukan mengukur akurasi penelitian. Audit tampilan browser interaktif dan setiap PNG satu per satu belum dilakukan.

## 2. Peta kesesuaian

| Komponen | Penilaian | Bukti utama |
|---|---|---|
| Scraping dan batas periode | Belum sesuai artefak | Kode membatasi sampai 1 Juni 2026, tetapi data tersimpan sampai 29 Juni 2026 |
| Kebersihan dataset v2 | Cukup baik secara struktur | 3.082 baris; ID unik; content unik; teks olahan tidak kosong; tidak ada konflik label pada grup teks identik |
| Kualitas label | Masih terbatas | 293 baris berlabel pakar; mayoritas label berasal dari lexicon; pengujian lexicon menunjukkan kelemahan kelas netral |
| Preprocessing kode backend dan shared | Sesuai pada sampel | Hasil wrapper backend sama dengan shared untuk 120 pemeriksaan sampel |
| Preprocessing terhadap model lama | Tidak selaras | 11/60 teks versi lama berubah ketika diproses kode sekarang |
| Split berkelompok v2 | Sesuai pada file tersimpan | 2.470 development + 612 holdout; overlap processed_text = 0 |
| Split model lama | Bermasalah | 52 processed_text identik melintasi development dan holdout |
| Eksperimen 3–7 v2 | Ada dan dapat ditelusuri | Hasil tersimpan memakai dataset 3.082 baris, tetapi rantainya berbeda dari final_config |
| Eksperimen 1–2 v2 | Belum lengkap | Tidak ada hasil tersendiri di output_v2/experiments yang diperiksa |
| Konfigurasi final dan joblib | Tidak sesuai | Kode: 2.000 fitur/C=2/OVR; joblib: batas 3.000 fitur/C=1,5/OVO |
| Backend inference | Berfungsi, identitas model belum konsisten | Prediksi berhasil; /config menyebut konfigurasi baru, inference memakai biner lama |
| Evaluasi API | Berfungsi pada pengujian kecil | Dedup sampai evaluasi dan pengambilan hasil mengembalikan HTTP 200 |
| Frontend | Build lulus, fungsi hasil belum lengkap | Aspek masih placeholder; tab evaluasi tidak mengambil hasil evaluasi |
| Visualisasi offline | Sebagian tidak sesuai | PNG confusion matrix nol; JSON metrik berbeda versi; learning curve masih memakai split per baris |
| Visualisasi web | Belum lengkap | Tidak ditemukan penyajian confusion matrix, PCA/t-SNE, hyperplane, atau learning curve pada halaman aktif |

## 3. Alur pipeline dan titik putus

Alur yang dirancang adalah scraping → penyaringan periode/bahasa/dedup → preprocessing → pelabelan lexicon → integrasi pakar → deteksi aspek → split → eksperimen → locking → training final → evaluasi → inference web.

`pipeline/run_pipeline.py` memang memiliki tahap-tahap tersebut, tetapi Stage 6 **hanya mengumpulkan hasil eksperimen**, bukan menjalankan tujuh eksperimen. `scripts/experiments/run_experiment_final.py` mendaftarkan eksperimen 3–7; eksperimen 1–2 harus dikelola terpisah. Dengan demikian, menjalankan satu runner tidak otomatis membuktikan seluruh tujuh eksperimen sudah selesai pada dataset yang sama.

Stage 6 mencari nama seperti `experiment_3_tfidf_results_v4.json`, `experiment_4_svm_results_v3.json`, dan `experiment_7_regularization_gap_constrained_results.json`, sedangkan hasil yang tersedia memakai nama tanpa suffix versi dan `experiment_7_regularization_gap_constrained.json`. Eksperimen 2 sekarang menyimpan `experiment_2_threshold.json`, sedangkan pengumpul mencari `experiment_2_threshold_results.json`. Akibatnya checkpoint bisa melaporkan hasil tidak ditemukan atau menggunakan arsip yang berbeda.

Path default `shared/constants.py` masih menuju `data/pipeline_output`, `output/experiments`, dan `models/final`. Keberadaan folder `pipeline_output_v2` dan `output_v2` tidak otomatis membuat backend memakai keduanya. Override environment tersedia, tetapi tidak menyelesaikan ketidakcocokan biner dengan konfigurasi secara otomatis.

## 4. Kualitas dataset dan pelabelan

### 4.1 Hasil perhitungan langsung

File dibaca dengan deteksi delimiter dan encoding UTF-8-SIG. Sebagian file berekstensi CSV menggunakan pemisah yang tidak bisa diasumsikan koma.

| Tahap/file | Jumlah baris | Catatan |
|---|---:|---|
| data/01_semua_data.csv | 21.988 | Data mentah tersimpan mencakup 2018–2026; belum seluruhnya periode penelitian |
| data/02_data_akhir_dedup.csv | 4.425 | Ada 1.187 duplikat content menurut perbandingan persis |
| data/02_data_akhir.csv | 3.119 | reviewId dan content unik; tidak kosong |
| data/pipeline_output/stage1_scraped.csv | 3.119 | Jumlah sama dengan keluaran scraping final |
| data/pipeline_output/stage2_preprocessed_new.csv | 3.060 | Tidak ada processed_text kosong; 231 duplikat tambahan setelah preprocessing |
| data/pipeline_output/stage3_final_labeled.csv | 3.060 | 288 label pakar; tanpa konflik label pada grup teks identik |
| data/pipeline_output_v2/stage3_final_labeled.csv | 3.082 | 293 label pakar; 299 duplikat tambahan processed_text; tanpa konflik label pada grup identik |

Jumlah duplikat tambahan adalah `duplicated().sum()`, bukan jumlah semua baris yang tergabung dalam grup duplikat. Teks berbeda dapat menjadi identik setelah normalisasi/stemming; ini tidak harus dihapus, tetapi wajib dikelompokkan saat splitting dan CV.

| Label final | Versi lama | Versi v2 |
|---|---:|---:|
| Positif | 1.787 | 2.070 (67,16%) |
| Negatif | 705 | 635 (20,60%) |
| Netral | 568 | 377 (12,23%) |

Distribusi v2 masih tidak seimbang, dengan rasio kelas terbesar terhadap terkecil sekitar 5,49. Accuracy saja tidak cukup; Macro F1 dan recall kelas netral perlu menjadi pusat pelaporan. Rating v2 didominasi bintang 5, sebanyak 2.410/3.082. Rating merupakan metadata dan pembanding, bukan ground truth sentimen otomatis.

### 4.2 Ketidaksesuaian periode dan kontrak kolom

`pipeline/scraper.py` menetapkan periode 1 Januari 2024 sampai 1 Juni 2026 inklusif. Namun `data/02_data_akhir.csv` berakhir 29 Juni 2026 dan memiliki **38 baris di luar batas tersebut**. Dataset v2 memiliki **36 baris di luar batas**. Ini bukti ketidakselarasan kode dengan artefak, bukan bukti bahwa algoritma filter kode sekarang pasti salah; data bisa berasal dari eksekusi versi terdahulu.

Kontrak `shared/constants.py` menyebut kolom tanggal `at`, sedangkan file final yang diperiksa menggunakan `tanggal`. Konsumen harus memakai kontrak yang disepakati dan loader yang sesuai. Dokumentasi jumlah data 3.277/2.983 pada README juga tidak merepresentasikan snapshot yang diperiksa.

### 4.3 Kualitas anotasi

V2 berisi 293/3.082 (9,51%) label pakar dan 2.789 label lexicon menurut `output_v2/audit/expert_label_summary_v2.json`. Karena mayoritas ground truth bersifat otomatis, metrik SVM terutama mengukur kesesuaian terhadap skema label campuran tersebut, bukan langsung ketepatan sentimen manusia pada populasi umum.

Pemetaan pakar lama mencatat 288 kecocokan content persis, 1 prefix, 4 fuzzy, dan 7 tidak cocok. Lima pemetaan nonpersis perlu ditinjau manual. File anotasi tambahan mencatat 500 baris, 40 label pakar 1 terisi, dan 0 label pakar 2. Kappa antarpakar 1 dan 2 belum tersedia. Kappa 1,0 pada 40 baris pakar lama versus anotasi baru tidak cukup membuktikan reliabilitas dua anotator independen; identitas dan independensi anotasinya perlu dibuktikan.

`output/lexicon_audit/lexicon_v2_evaluation.json` mencatat pada 293 contoh: accuracy lexicon v2 82,25%, Macro F1 61,55%, dan recall netral 10,53% (2/19). Ini mengindikasikan sumber label netral masih lemah. Hasil tersebut adalah arsip evaluasi, bukan perhitungan ulang label seluruh dataset oleh audit ini.

Sebanyak 60 label pakar berada di holdout v2. Eksperimen 2 mencocokkan contoh pakar terhadap seluruh canonical dataset tanpa pembatas development. Jika kumpulan yang sama dipakai untuk memilih threshold atau menyusun lexicon, lalu juga dilaporkan sebagai evaluasi independen, ada risiko kontaminasi seleksi. **Belum dibuktikan seluruh holdout v2 benar-benar independen dari proses pengembangan label**; klaim dalam metadata belum cukup menggantikan provenance reviewId dan catatan sampling.

## 5. Audit tujuh eksperimen dan konfigurasi final

| Eksperimen | Hasil yang ditemukan | Penilaian kesesuaian |
|---|---|---|
| 1. Rasio split | Arsip lama: pemenang 90:10, CV Macro F1 0,82377; final tetap 80:20 | Harus dijelaskan sebagai sensitivitas dengan keputusan metodologis terpisah; bukan klaim bahwa 80:20 adalah pemenang arsip tersebut. Hasil v2 belum tersedia |
| 2. Threshold netral | Arsip lama: threshold 1,0, Macro F1 pakar 0,627185 pada 288 contoh | Tidak konsisten dengan final_config 0,0 dan shared.labeling 0,5. Script eksperimen memanggil shared.lexicon default v1; Stage 3 memakai shared.labeling dengan lexicon aktif v2. Hasil ini belum mengesahkan threshold v2 |
| 3. TF-IDF | V2: max_features=2.000, unigram, sublinear=True, min_df=1, max_df=0,9; CV 0,81006 ± 0,04613 | Sesuai parameter kode final. Pipeline dan grouped CV mencegah fit vocabulary pada validation fold. Tie 2.000–8.000 dijelaskan oleh vocabulary fold di bawah 2.000 |
| 4. SVM | V2 pemenang RBF C=2 gamma=1: CV 0,80089; linear C=2: 0,79905 | Kode final memilih linear, bukan pemenang mentah. Alasan simplifikasi ada, tetapi aturan keputusan perlu eksplisit. Metadata menyebut StratifiedKFold walaupun kode aktual menggunakan make_grouped_cv dan groups |
| 5. Imbalance | V2 pemenang SMOTE: 0,80508; none: 0,80089 | Pengujian dilakukan dengan RBF C=2. Kode final menetapkan none pada linear; perlu bukti pembanding yang sesuai konfigurasi final. Resampling di dalam CV pipeline merupakan desain yang benar |
| 6. OVO/OVR | V2 OVR 0,81445 ± 0,04278; OVO 0,80788 ± 0,04355 | Mendukung OVR pada RBF+SMOTE yang diuji; bukan skor CV konfigurasi final linear+none |
| 7. Regularisasi | V2 C=2: CV 0,81445, training 0,99688, gap 0,18244; alternatif C=0,1: CV 0,77173, gap 0,01155 | Eksperimen diagnostik pada RBF+SMOTE+OVR. Gap C=2 sekitar 18,24 poin persentase perlu dilaporkan. Tidak otomatis mengganti konfigurasi final |

Perbedaan SD pada eksperimen 6 dan 7 muncul meskipun daftar skor fold C=2 sama: 0,04278 versus 0,03827. Nilainya konsisten dengan perbedaan konvensi sample SD dan population SD. Konvensi harus disamakan atau diberi keterangan agar pembanding stabilitas tidak menyesatkan.

`configuration_comparison.json` mendukung adanya eksperimen tambahan pada development, tetapi arsip yang diperiksa belum memuat hasil pembanding SMOTE, meskipun kode sekarang sudah menambahkannya. Dengan demikian keputusan `imbalance=none` belum didukung pembanding SMOTE-versus-none pada linear dalam arsip itu.

Komentar final_config merujuk p McNemar RBF-versus-linear 0,064 sebagai alasan kesederhanaan. Namun tidak signifikan pada accuracy **tidak membuktikan ekuivalensi**, dan arsip yang sama mencatat CI selisih Macro F1 RBF terhadap linear sebesar [0,00504; 0,03757], positif seluruhnya. Ini perlu dibahas karena metrik seleksi proyek adalah Macro F1. Jangan menyimpulkan kedua model setara hanya dari p McNemar.

Arsip kalibrasi juga tidak konsisten: accuracy baseline 0,87571 dan calibrated 0,88623, tetapi McNemar mencatat b=c=0. Angka itu tidak dapat menggambarkan dua prediksi dengan jumlah benar yang berbeda pada sampel yang sama. Hasil kalibrasi perlu divalidasi ulang sebelum dipakai sebagai dasar keputusan. CalibratedClassifierCV internal cv=3 juga belum memakai grouped folds; ada risiko duplikat melintasi fold kalibrasi internal.

### Tiga konfigurasi yang harus dibedakan

| Sumber | TF-IDF | SVM | Balancing | Multiclass |
|---|---|---|---|---|
| Pemenang rantai eksperimen v2 3–6 | 2.000, unigram | RBF C=2 gamma=1 | SMOTE | OVR |
| shared/final_config.py sekarang | 2.000, unigram | Linear C=2 | none | OVR |
| Joblib terpasang models/final/final_pipeline.joblib | Batas 3.000, vocabulary aktual 2.344, unigram | Linear C=1,5 | none | OneVsOneClassifier(SVC) |

`models/final/metadata.json` merekam training 27 Oktober 2026, dataset 3.060, development 2.448 dan holdout 612. `final_model_metadata.json` lain bahkan merekam versi 18 Oktober dan dataset 3.065. Jangan memakai kedua metadata tersebut secara bergantian.

Panggilan langsung `verify_best_config_agrees()` **gagal** pada TF-IDF, SVM, dan multiclass. Ini menunjukkan proteksi training mendeteksi ketidakselarasan snapshot; model lama tetap dapat dimuat karena loader inference hanya memeriksa keberadaan kemampuan predict, bukan kesamaan parameter dengan konfigurasi yang ditampilkan API.

## 6. Split, kebocoran, dan evaluasi

| Split tersimpan | Development | Holdout | Overlap processed_text |
|---|---:|---:|---:|
| data/pipeline_output/stage5_split | 2.448 | 612 | **52** |
| data/pipeline_output/stage5_split_v2 | 2.449 | 611 | 0 |
| data/pipeline_output_v2/stage5_split_v2 | 2.470 | 612 | 0 |

Gabungan reviewId development dan holdout v2 sama dengan himpunan ID dataset v2. Group split adalah perbaikan yang tepat. Angka holdout sebaiknya tidak dipaksa selalu 612 karena pembagian grup dapat menghasilkan 611 atau ukuran lain yang mendekati target.

Perhitungan ulang `final_holdout_predictions.csv` menghasilkan accuracy **0,8790849673** dan Macro F1 **0,8386890442**. Pipeline joblib menghasilkan ulang **612/612 prediksi yang sama** pada processed_text arsip. Matriks aktual/prediksi dengan urutan negatif, netral, positif:

| Aktual / Prediksi | Negatif | Netral | Positif |
|---|---:|---:|---:|
| Negatif | 107 | 18 | 16 |
| Netral | 11 | 87 | 16 |
| Positif | 6 | 7 | 344 |

Hasil lama tersebut dapat direproduksi, tetapi tidak menghapus masalah overlap split dan drift preprocessing. Angka itu tidak boleh ditulis sebagai performa final dataset v2/config baru. Metadata `holdout_evaluated_once=true` adalah pernyataan proses; frekuensi historis pemakaian holdout tidak bisa dipastikan hanya dari field tersebut.

## 7. Preprocessing dan inference

Wrapper backend di `backend/services/preprocessing_service.py` mendelegasikan proses ke `shared/preprocessing.py`. Pengujian sampel acak tetap (random_state=42), masing-masing 60 ulasan per versi, memberi:

| Pemeriksaan | Lama | V2 |
|---|---:|---:|
| Teks tersimpan berbeda dari hasil preprocessing kode sekarang | 11/60 | 0/60 |
| Hasil shared berbeda dari wrapper backend | 0/60 | 0/60 |

Contohnya, kode sekarang menambahkan token `emoji_positif` yang tidak ada pada sebagian teks training lama, serta menormalisasi `jossssss` menjadi `jos`. Karena joblib tidak memuat preprocessing raw text sebagai langkah internal, versi preprocessing eksternal menjadi bagian penting identitas model. Kesamaan wrapper sekarang tidak menjamin kesamaan dengan proses yang digunakan saat melatih biner lama. Hasil sampel bukan jaminan seluruh 3.082 baris sudah identik.

Tes terpilih: `python -m pytest tests/test_preprocessing.py tests/test_split_leakage.py -q -p no:cacheprovider` menghasilkan **53 passed, 1 failed**. Kegagalan: `normalize_repeated_letters('bangett')` menghasilkan `bangett`, sedangkan tes mengharapkan `banget`. Ada pula assertion lama yang tidak memeriksa perilaku secara kuat, misalnya menerima kata stopword baik ada maupun tidak ada; jumlah tes lulus tidak otomatis berarti seluruh kualitas bahasa sudah tervalidasi.

Inference melalui service menggunakan pipeline pre-fitted dan tidak melakukan fit TF-IDF pada data upload. Ini sudah sesuai pemisahan training dan inference. Penanganan teks yang menjadi kosong setelah preprocessing masih perlu diperjelas: endpoint preprocess menyimpan string final tanpa menolak teks kosong, sedangkan prediksi hanya menyaring nilai None. Ini dapat membuat input tanpa fitur tetap menerima salah satu kelas berdasarkan model.

## 8. Backend dan frontend web

### 8.1 Backend yang berhasil diverifikasi

Flask test client dengan database SQLite di memori mengembalikan HTTP 200 untuk health, stats, model/stats, config, serta alur dedup → preprocess → predict → aspects → evaluate → GET evaluation pada tiga contoh sintetis yang dimasukkan langsung ke database memori. Jalur upload file belum diuji end-to-end. Database penelitian tidak digunakan untuk percobaan ini.

Endpoint evaluasi analisis memakai label aktual dan prediksi yang tersedia, labels eksplisit tiga kelas, dan zero_division=0. Ini tepat untuk evaluasi dataset upload berlabel. Endpoint model/stats mengembalikan metrik arsip lama, sedangkan /config mengembalikan konfigurasi baru. Health berstatus healthy tidak mendeteksi ketidakcocokan tersebut.

### 8.2 Status alur untuk upload tanpa label

`AnalysisDetail.jsx` menjalankan prediksi lalu aspek, dan hanya memanggil evaluasi jika ada label aktual. Endpoint aspek mengganti status menjadi `aspects_completed`. Frontend mengenali hasil tersedia terutama dari `predicted` atau `completed`, bukan status tersebut. Untuk data tanpa label, alur dapat berakhir di `aspects_completed` dan UI tetap menganggap prediksi belum tersedia atau mengizinkan pipeline diulang. Ini cacat integrasi status yang terlihat dari kode; skenario tersebut belum diuji dengan browser.

### 8.3 Halaman hasil belum menyajikan evaluasi dan aspek

`frontend/src/pages/AnalysisResults.jsx` hanya mengambil data prediksi. Tab aspek menampilkan teks “Fitur aspek dalam pengembangan”. Tab evaluasi menampilkan instruksi menjalankan evaluasi tanpa memanggil `getEvaluation()` atau merender metrik dan confusion matrix, bahkan ketika backend telah menyimpan evaluasi.

Filter tab `!t.condition || t.condition` selalu bernilai benar sehingga tab evaluasi tidak disembunyikan untuk dataset tanpa label. Halaman juga menginisialisasi activeTab='predictions' tanpa membaca suffix URL; route `/results/aspects` atau `/results/evaluation` tetap membuka tab prediksi. Data prediksi tidak diambil ulang berdasarkan perubahan id secara eksplisit, sehingga navigasi antaranalisis pada komponen yang sama berpotensi menyisakan hasil sebelumnya.

Route detail preprocessing dan compare memakai komponen AnalysisPipeline yang sama, tetapi komponen tidak merender data per tahap atau tabel perbandingan berdasarkan parameter stage. Tautan detail belum setara dengan implementasi detail tujuh tahap.

Dashboard mengambil `/stats`, yaitu statistik analisis upload di database. Itu sesuai sebagai dashboard aktivitas, tetapi bukan dashboard evaluasi model penelitian. Service frontend aktif belum memanggil `/model/stats` untuk menyajikan kualitas model final.

`npm run build` berhasil, dengan output 1.472 modul ditransformasi. Build sukses membuktikan bundling, bukan kelengkapan fitur atau ketepatan visualisasi.

## 9. Kesesuaian visualisasi

### 9.1 Confusion matrix PNG: tidak benar

Inspeksi langsung `output/visualizations/13_confusion_matrix.png` menunjukkan seluruh sembilan sel bernilai **0**, sedangkan CSV dan perhitungan ulang memiliki total **612** seperti matriks pada bagian 6. Gambar tersimpan itu tidak mewakili evaluasi model dan tidak layak dipakai dalam laporan hasil. Script sekarang sudah memiliki pemeriksaan kesamaan matriks; keberadaan pemeriksaan di kode tidak menjamin PNG lama sudah dibuat ulang dengan kode itu. Penyebab historis PNG nol belum dibuktikan.

### 9.2 Sumber metrik visualisasi bercampur

`output/visualizations/visualization_summary.json` mencatat accuracy 87,91% dan Macro F1 83,87%, tetapi `visualization_metrics.json` di folder yang sama mencatat accuracy 81,54% dan Macro F1 79,21%, dengan support kelas yang berbeda. Status PASS pada summary tidak cukup untuk mengesahkan semua gambar dalam folder. Penomoran wordcloud/aspek/learning curve ganda juga memperbesar risiko mengambil gambar usang.

### 9.3 PCA dan t-SNE

Report `final_explained/final_visualization_report.json` menyatakan PCA 2D hanya menjelaskan **5,81%** variansi, PCA 3D **7,83%**, dan t-SNE menggunakan 600 sampel setelah SVD 50 dimensi. Angka itu boleh dipakai sebagai visualisasi eksploratif dengan batasannya; kedekatan titik di 2D tidak langsung membuktikan kualitas pemisahan model pada ruang TF-IDF asli.

Script `visualisasi_final_pca_tsne_svm_2d.py` masih mengunci 3.060 total/2.448 development/612 holdout dan C=1,5, serta memakai train_test_split per baris. Ini tidak mengikuti split berkelompok v2. Klaim “holdout tidak digunakan” pada visualisasi lama mengacu pada split lamanya sendiri, bukan jaminan identik dengan canonical group holdout sekarang.

`shared/svm_visualization.py` melatih clone model baru pada koordinat PCA/t-SNE ketika menggambar boundary. Boundary tersebut merupakan model pada embedding, bukan hyperplane dari classifier final berdimensi tinggi. Default kelas menggunakan kapital awal, sedangkan output model lowercase; tanpa normalisasi dari pemanggil, scatter/matriks dapat kehilangan kelas. Beberapa ekspor seperti plot_pca_3d, plot_svm_hyperplane, dan generate_wordcloud di modul tersebut hanya stub yang mengembalikan string kosong. Modul ini juga tidak ditemukan terhubung sebagai visualisasi pada halaman web aktif.

### 9.4 Hyperplane dan margin 2D

Gambar `output/visualization_test/final_explained/05_svm_2d_hyperplane_margin_support_vector.png` secara eksplisit menyebut model biner khusus visualisasi. Pernyataan pembatas ini sudah tepat: grafik tersebut bukan model final tiga kelas.

Namun inspeksi gambar menunjukkan skala sumbu Y sampai sekitar -18.000, sehingga titik data berhimpit dekat nol. Report merekam ||w||≈0,000632, margin 2/||w||≈3.162,82, accuracy diagnostik 71,72%, dan Macro F1 41,76%. Accuracy ini sama dengan proporsi positif 1.430/1.994; bersama Macro F1 tersebut, hasil mengindikasikan classifier 2D praktis memilih kelas mayoritas. Rumus margin dapat benar secara matematis, tetapi model dan penyajiannya kurang membantu menjelaskan pemisahan sentimen.

Perbaikan yang diperlukan kemudian: zoom sebaran data, keterangan boundary di luar area data/inset, serta penegasan bahwa model biner dilatih ulang untuk ilustrasi. Jangan menampilkan margin besar sebagai bukti model final unggul.

### 9.5 Learning curve dan ROC

`scripts/learning_curve_analysis.py`, bagian learning curve pada `viz_final_results.py`, dan helper learning curve bersama masih memakai split/CV per baris. Ini belum konsisten dengan grouped evaluation; duplikat teks bisa menyeberangi fold. Jika diberi matriks TF-IDF yang sudah di-fit di luar CV, helper juga tidak menjamin vocabulary fold bebas leakage; pemanggil perlu memakai pipeline raw processed_text di dalam CV.

ROC di `viz_final_results.py` memakai decision_function. Skor margin boleh dipakai untuk kurva diskriminasi, tetapi bukan probabilitas terkalibrasi. Pemetaan kolom skor wajib mengikuti classes_ dan implementasi multiclass sebenarnya. Skrip memiliki fallback rekonstruksi holdout dengan train_test_split; fallback itu belum mengikuti group split. Audit ini belum mengesahkan setiap kurva ROC atau AUC tersimpan.

### 9.6 Visualisasi aspek dan arti implementasinya

Deteksi aspek backend memakai keyword dan kalimat dari ulasan asli, kemudian menempelkan **sentimen seluruh ulasan** pada setiap aspek yang ditemukan. Ini sesuai untuk distribusi sentimen ulasan yang menyebut aspek. Ini belum setara dengan model sentimen spesifik per aspek: ulasan “materinya bagus tetapi login buruk” dapat memberi label ulasan yang sama ke kedua aspek.

Jumlah penyebutan aspek dapat melebihi jumlah ulasan unik karena multi-aspek; denominator persentase harus dinyatakan. Tidak tepat mengartikan distribusi positif semata sebagai bukti kausal efektivitas pembelajaran. Halaman hasil web juga belum menampilkan analisis aspek tersebut.

## 10. Urutan tindak lanjut yang direkomendasikan

1. Tetapkan snapshot penelitian tunggal: periode, dataset, preprocessing, lexicon, threshold, reviewId split, dan path output. Simpan hash sumber/artefak agar provenance dapat dibuktikan.
2. Selesaikan perbedaan tanggal dan threshold sebelum menilai ulang model. Pisahkan data pengembangan label dari evaluasi independen; audit lima pemetaan fuzzy/prefix dan lengkapi anotasi netral serta anotator kedua.
3. Buat hasil eksperimen 1–2 yang konsisten dengan dataset/lexicon v2. Nyatakan keputusan 80:20 sebagai metodologi bila tidak mengikuti pemenang eksperimen sensitivitas.
4. Putuskan secara eksplisit apakah final mengikuti RBF+SMOTE+OVR atau linear+none+OVR. Untuk linear, lengkapi bukti pembanding balancing dan kinerja CV konfigurasi itu sendiri; gunakan Macro F1 sesuai metrik seleksi yang ditetapkan.
5. Selaraskan snapshot best_config dengan keputusan final, lalu latih model final hanya pada development yang benar. Regenerasi metadata dan evaluasi setelah protokol dikunci; jangan menggunakan hasil holdout untuk memilih konfigurasi.
6. Pastikan loader/API memvalidasi identitas model, config, dan dataset; sertakan versi preprocessing. Jangan membiarkan /config mendeskripsikan model yang berbeda dari yang digunakan predict.
7. Lengkapi frontend untuk hasil prediksi, evaluasi, confusion matrix, aspek, rincian preprocessing; perbaiki status tanpa label, pemilihan tab berdasarkan route, serta pemuatan ulang data per id.
8. Regenerasi visualisasi dari artefak final yang sama. Wajibkan total confusion matrix sama dengan jumlah prediksi, metrik konsisten, label kelas konsisten, dan grouped folds pada learning curve. Tinjau gambar, bukan hanya keberhasilan savefig.

## 11. Kriteria kelulusan audit berikutnya

Proyek dapat disebut sesuai ketika konfigurasi kode = snapshot JSON = parameter biner = identitas yang dikembalikan API; dataset dan periode lolos kontrak; split dan seluruh CV bebas overlap grup; hasil tujuh eksperimen serta keputusan penyimpangannya terdokumentasi; metrik dapat dihitung ulang dari prediksi; gambar sesuai angka sumber; dan frontend menampilkan hasil API yang benar pada data berlabel maupun tanpa label.

Audit saat ini menyimpulkan **backend dasar berjalan dan perbaikan dataset/split v2 menjanjikan, tetapi rantai finalisasi dan penyajian hasil belum selesai**. Tidak ada perubahan perbaikan yang diterapkan dalam pemeriksaan ini.
