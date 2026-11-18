# Tahap 4 — locking, model deployment, evaluasi dan identitas API

Tanggal: 29 Oktober 2026. Status: **bundle deployment baru selesai dan terverifikasi pada kode/backend test client**. Tidak dilakukan restart server yang mungkin sedang berjalan pada komputer pengguna.

## 1. Hasil utama

Konfigurasi hasil tahap 3 sekarang menjadi konfigurasi deployment di shared/final_config.py. Default path diarahkan secara bersamaan ke dataset `data/pipeline_output_v4`, hasil `output_v4`, dan model `models/final_v4`. Model lama di models/final tidak ditimpa.

| Komponen | Nilai deployment |
|---|---|
| Versi model | 20261029_223454 |
| Waktu training | 29 Oktober 2026, 22:34:54 WIB |
| Model canonical | models/final_v4/final_pipeline.joblib |
| Metadata | models/final_v4/metadata.json |
| TF-IDF | max_features=5.000; ngram_range=(1,3); sublinear_tf=False; min_df=1; max_df=0,9 |
| SVM | Linear, C=10, gamma=scale, class_weight=None |
| Multiclass | OneVsOneClassifier(SVC) |
| Resampling | none |
| Lexicon/threshold | v2 / 1,0 |
| Dataset | 3.078 baris; negatif 575, netral 556, positif 1.947 |
| Training | 2.474 development saja |
| Evaluasi diagnostik | 604 holdout, dengan provenance paparan historis |

Pilihan penelitian pada `selected_configuration.json` tidak diedit retroaktif: status pending deployment di file tersebut merupakan rekaman tahap 3. Promosi tahap 4 dicatat pada metadata model dan laporan ini. Artefak pemilihan tetap memiliki checksum yang sama.

## 2. Locking dan training

Perintah yang dijalankan dari root proyek:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts/lock_final_config.py
python -u scripts/train_final_model.py
```

Locking memverifikasi parameter terhadap hasil eksperimen 3–6, dataset/distribusi kelas, 288 label pakar, dan checksum seleksi/snapshot. Snapshot terkunci tersimpan pada `output_v4/experiments/best_config.json`. Field CV pada snapshot sekarang menggunakan hasil multiclass akhir eksperimen 6, dan implementasi OVO dicatat sebagai OneVsOneClassifier.

Training memakai split tersimpan berdasarkan reviewId dan menolak fallback recomputed. Pipeline dilatih hanya pada development. Pipeline yang disimpan adalah pipeline yang sama dengan yang dievaluasi; tidak dilakukan refit pada seluruh dataset setelah evaluasi.

Locking dan training menolak penimpaan artefak deployment yang sudah ada. Menjalankan ulang dua perintah di namespace yang sama memang akan berhenti. Rilis baru memerlukan direktori model/evaluasi baru dan kontrak seleksi yang sesuai; jangan menghapus hasil lama untuk memaksa retraining.

Baseline persiapan 0,5 tetap tersedia di research_config. Builder snapshot persiapan sekarang melewatkan baseline tersebut secara eksplisit, sehingga perubahan default deployment menjadi 1,0 tidak diam-diam mengubah resep data persiapan. Tidak ada rebuild atau overwrite snapshot v3/v4 pada tahap ini.

## 3. Hasil evaluasi — deskriptif, bukan independent gold test

Riwayat dari tahap 3 tetap dicatat: 527 dari 604 reviewId holdout sempat masuk training/CV dalam percobaan salah split yang kemudian dibatalkan. Konfigurasi terpilih berasal dari rangkaian yang diperbaiki dan mengecualikan canonical holdout, tetapi sejarah paparan itu tidak dapat dihapus.

Metadata dan API menggunakan scope **diagnostic_holdout_with_prior_exposure**, holdout_pristine=False, dan independent_gold_evaluation=False. Klaim untouched_holdout dan klaim historis holdout_evaluated_once tidak dipakai pada metadata baru. Field evaluation_passes_in_this_training_run=1 hanya menghitung pass evaluasi pada training ini.

| Metrik pada 604 baris | Hasil |
|---|---:|
| Accuracy | **84,27%** |
| Macro precision | 78,80% |
| Macro recall | 79,38% |
| Macro F1 | **79,02%** |
| Weighted F1 | 84,52% |
| Training accuracy, diagnostik | 99,39% |
| Training Macro F1, diagnostik | 99,04% |

Gap training–holdout Macro F1 sekitar 20,02 poin persentase. Hasil ini belum membuktikan masalah generalisasi atau kualitas label sudah terselesaikan. Tidak dilakukan tuning tambahan setelah melihat evaluasi ini.

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negatif | 0,807018 | 0,779661 | 0,793103 | 118 |
| Netral | 0,621212 | 0,689076 | 0,653386 | 119 |
| Positif | 0,935754 | 0,912807 | 0,924138 | 367 |

Confusion matrix menggunakan urutan kelas negatif, netral, positif. Baris adalah aktual dan kolom prediksi:

| Aktual / Prediksi | Negatif | Netral | Positif |
|---|---:|---:|---:|
| Negatif | 92 | 24 | 2 |
| Netral | 16 | 82 | 21 |
| Positif | 6 | 26 | 335 |

Total matriks = 604. Seluruh 604 prediksi dari model tersimpan cocok dengan `final_holdout_predictions.csv` ketika dihitung ulang.

### Evaluasi subset dari prediksi yang sama

| Subset | Baris | Accuracy | Macro F1, tiga kelas |
|---|---:|---:|---:|
| Pernah terpapar pada percobaan batal | 527 | 85,20% | 79,37% |
| Tidak masuk training pada percobaan batal tersebut | 77 | 77,92% | 75,99% |
| Berlabel pakar dalam holdout | 54 | 83,33% | 74,45% |

Subset 77 dan 54 tidak otomatis menjadi gold test independen: paparan terhadap pengembangan lexicon historis dan independensi anotator tetap belum terverifikasi. Ukuran subset juga kecil. Semua metrik ini dihitung dari satu vektor prediksi holdout; tidak ada pelatihan model berbeda untuk mengoptimalkan subset.

## 4. Bundle integrity dan identitas model

`shared/deployment_integrity.py` menjadi jalur verifikasi bersama. Sebelum inference, loader memeriksa:

- Metadata memiliki schema bundle deployment yang didukung dan status FINAL.
- Parameter metadata cocok dengan final_config dan threshold terpilih.
- Checksum pipeline, evaluasi, classification report, confusion matrix, prediksi, best_config, hasil seleksi, dataset, manifest dataset dan kedua split.
- Fingerprint kode preprocessing, konstanta, aturan threshold, kamus slang/normalisasi dan resource lexicon yang dipakai.
- Isi model benar-benar berupa TF-IDF yang fitted dan OneVsOneClassifier dengan SVC berparameter sesuai, serta urutan kelas yang benar.
- Versi model, konfigurasi dan identitas dataset dalam evaluasi sama dengan metadata.

Ini pemeriksaan konsistensi artefak lokal, bukan tanda tangan digital dari pihak eksternal. Cache objek lama tidak boleh menghindari pemeriksaan bundle pada perolehan model berikutnya.

Loader `backend/services/prediction_service.py` dan `pipeline/train_svm.py` memakai verifikator yang sama. Helper lama `_load_latest_verified()` pada ml_engine tidak lagi memilih pasangan pkl terbaru atau fallback pasangan yang belum diverifikasi; nama kompatibilitas itu sekarang mengarah ke deployment canonical.

## 5. Perbaikan API

| Endpoint/jalur | Perilaku setelah perbaikan |
|---|---|
| GET /api/v1/health | HTTP 200 hanya jika bundle terverifikasi; mengembalikan versi model dan provenance evaluasi; kegagalan menjadi 503 |
| GET /api/v1/config | Konfigurasi berasal dari kontrak yang cocok dengan model terpasang; versi dan status integrity disertakan |
| GET /api/v1/model/stats | Menggunakan evaluasi bundle yang sama; label sumbu matriks eksplisit; distribusi penelitian tidak ditimpa oleh riwayat upload |
| POST /api/v1/predict, /predict-single | Menggunakan model yang sama dan menyertakan model_version; teks kosong setelah preprocessing tidak diklasifikasikan |
| POST /api/v1/analyze | Sekarang inference SVM canonical ditambah deteksi aspek; tidak lagi mengganti prediksi model dengan hasil lexicon |
| POST /api/v1/label | Tetap merupakan pelabelan lexicon, terpisah dari inference SVM |
| Prediksi batch analisis | Memakai loader terverifikasi, menyertakan versi pada respons, dan menolak analisis tanpa teks hasil preprocessing yang valid |
| Evaluasi API | Macro metric memakai tiga kelas eksplisit agar definisinya konsisten dengan evaluasi offline; label aktual tidak dikenal ditolak |
| POST /api/v1/train | Tetap dinonaktifkan |

Statistik model menandai aspect_statistics_available=False: analisis aspek final belum dihitung ulang dalam tahap ini, sehingga placeholder tidak boleh dianggap sebagai bukti tidak ada aspek dalam dataset. Riwayat upload tetap tersedia melalui /stats.

Pengujian health, config dan model/stats semuanya mengembalikan versi **20261029_223454**. Simulasi checksum/metadata tidak cocok menghasilkan 503 pada endpoint inti tanpa menggunakan model pengganti diam-diam.

## 6. Verifikasi dan batas pemeriksaan

```powershell
python -m pytest tests/test_preprocessing.py tests/test_research_experiments.py tests/test_dataset_preparation.py tests/test_artifact_contracts.py tests/test_split_leakage.py tests/test_deployment_bundle.py -q -p no:cacheprovider
```

Hasil akhir: **103 passed, 0 failed**. Lima warning terdiri dari satu kasus tanggal invalid yang sengaja diuji dan empat pemanggilan Query.get() legacy SQLAlchemy. Warning legacy tersebut masih menjadi pekerjaan pemeliharaan; bukan kegagalan prediksi atau evaluasi.

Pengujian Flask memakai **SQLite di memori**, termasuk contoh riwayat dan batch analisis. Database pengguna tidak dipakai untuk percobaan. Pengujian tidak meluncurkan browser/server jaringan. Tidak semua script tes manual/legacy proyek dijalankan; angka di atas berlaku untuk enam modul pytest yang disebutkan.

Tes deployment mencakup perubahan parameter SVC, checksum model yang tidak cocok, fingerprint preprocessing berubah, klaim versi metadata salah, penolakan overwrite saat training ulang, health/config/statistik satu versi, konsistensi predict/analyze, penolakan input tidak valid, statistik penelitian yang stabil saat database memiliki riwayat, dan alur dedup–preprocess–predict–evaluate di memori.

Pemeriksaan tambahan pada sampel tetap 60 ulasan menunjukkan teks tersimpan = shared preprocessing = wrapper web, serta label otomatis cocok dengan threshold terpilih 1,0. Konfigurasi terpilih, best_config, metadata dan parameter model cocok. Checksum model lama dan dua dataset historis masih sama dengan tahap 1.

## 7. Struktur artefak dan dokumentasi

```text
models/
  README.md
  final/                       # model historis, tidak ditimpa
  final_v4/
    final_pipeline.joblib
    metadata.json
output_v4/experiments/
  selected_configuration.json  # keputusan tahap 3, tidak diedit
  best_config.json             # snapshot locking tahap 4
  final_model_evaluation.json
  final_classification_report.json
  final_confusion_matrix.csv
  final_holdout_predictions.csv
docs/remediation/
  TAHAP_04.md
  TAHAP_04_VERIFICATION.json
  TAHAP_04_TESTS.txt
  TAHAP_04_TESTS_INITIAL.txt
```

.gitignore diperbarui agar binary pipeline final_v4 dapat ikut dilacak, tanpa membuka semua file joblib lain. Log full test awal 102 kasus dipertahankan; log akhir memuat 103 kasus setelah penambahan tes identitas versi. Tidak ada pemindahan model historis karena script visualisasi lama masih perlu dimigrasikan pada tahap 6.

## 8. Menjalankan backend dan tahap berikutnya

Default kode sekarang menunjuk bundle final_v4. Jika backend lama masih berjalan, **restart proses backend** agar import konfigurasi dan model baru dimuat. Pemeriksaan tahap ini membuktikan perilaku proses/test client baru, bukan perubahan otomatis pada proses server yang sudah hidup.

Variabel WAYGROUND_OUTPUT_DIR, WAYGROUND_PIPELINE_OUTPUT_DIR dan WAYGROUND_FINAL_MODEL_DIR masih dapat meng-override default. Ketiga sumber harus cocok sebagai satu bundle. Memilih models/final lama saja tidak cukup untuk rollback karena metadata/schema dan preprocessing lama tidak cocok dengan kontrak baru.

**Tahap 5 berikutnya:** frontend dan alur pengguna—status tanpa label, pemilihan tab melalui URL, tampilan hasil evaluasi/aspek, rincian preprocessing, serta pencatatan versi model pada riwayat analisis. Histori analisis yang sudah tersimpan belum dimigrasikan/ditag versi model dalam tahap ini. Grafik offline/PCA/t-SNE/hyperplane dan regenerasi PNG tetap menjadi tahap 6.
