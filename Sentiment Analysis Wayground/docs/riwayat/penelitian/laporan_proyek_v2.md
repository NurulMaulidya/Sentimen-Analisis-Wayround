================================================================================
LAPORAN PROJECT LENGKAP AUDIT & DOKUMENTASI SISTEM
ANALISIS SENTIMEN WAYGROUND - SINGLE SOURCE OF TRUTH
Tanggal Audit: 15 Oktober 2026
================================================================================

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! INKONSISTENSI KRITIS DITEMUKAN !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

metadata_latest.json vs ACTUAL svm_model_latest.pkl:
| Parameter       | metadata_json | actual_model |
|----------------|--------------|--------------|
| kernel         | linear      | sigmoid     |
| C              | 10.0        | 100.0       |
| gamma          | scale       | 0.1         |

AKSI YANG DIPERLUKAN:
1. Perbarui metadata agar sesuai model aktual, ATAU
2. Latih ulang model dengan konfigurasi metadata

================================================================================
MASTER CONFIGURATION - MODEL AKTUAL (dari svm_model_latest.pkl)
================================================================================

SVM CONFIG:
| Parameter      | Nilai      |
|----------------|------------|
| Algorithm      | SVC        |
| Kernel         | sigmoid     |
| C              | 100.0     |
| Gamma          | 0.1     |
| Class Weight   | None       |

TF-IDF CONFIG (dari vectorizer_latest.pkl):
| Parameter        | Nilai        |
|------------------|--------------|
| max_features     | 3000        |
| ngram_range      | (1, 3)     |
| sublinear_tf     | False       |
| min_df           | 2           |
| max_df           | 0.9          |
| actual_features  | 640           |

================================================================================
EVALUASI MODEL (dari metadata_latest.json)
CATATAN: Metrik ini mungkin dari model dengan konfigurasi berbeda!
================================================================================

METRIK UJI:
| Metrik          | Nilai       |
|-----------------|-------------|
| Accuracy        | 93.37%     |
| Precision Macro | 88.32%     |
| Recall Macro    | 89.94%     |
| F1-Score Macro | 89.11%     |

CV METRICS (5-Fold):
| CV F1 Mean     | 84.47%     |
| CV F1 Std      | 1.53%     |

CONFUSION MATRIX:
              PREDIKSI
         Positif  Netral  Negatif
AKTUAL Positif    397      10       6
       Netral       5      80       7
       Negatif      4       7      72

================================================================================
PELABELAN SENTIMEN
================================================================================

LEXICON:
| Tipe     | Jumlah |
|----------|--------|
| Positive | 191    |
| Negative | 110    |
| Neutral  | TIDAK ADA |

CATATAN: File neutral_final.tsv yang direferensikan di constants.py
TIDAK ADA di folder data/lexicon/. Sistem menangani dengan try-except.

METHOD: InSet Lexicon + Ekspansi
- Base: InSet (Koto & Rahmaningtyas, 2017)
- Ekspansi: Slang dan istilah domain
- Negasi: tidak + bagus = tidak_bagus (skor dibalik)

================================================================================
DATASET
================================================================================

PENYUSUTAN DATA:
| Tahap                  | File                          | Baris  |
|------------------------|-------------------------------|--------|
| 1. Scraping            | 01_semua_data.csv           | 22.606 |
| 2. Filter Tanggal       | 02_data_akhir.csv           | 4.517  |
| 3. Deduplikasi          | 03_cleaned_data.csv         | 2.994  |
| 4. Preprocessing        | stage2_preprocessed.csv       | 2.936  |
| 5. Pelabelan            | stage3_labeled.csv          | 2.936  |

DISTRIBUSI KELAS:
| Kelas    | Train | Test  | Total | Persentase |
|---------|-------|-------|-------|------------|
| Positif  |  1649 |   413 |  2062 | 70.2%      |
| Netral   |   366 |    92 |   458 | 15.6%      |
| Negatif  |   333 |    83 |   416 | 14.2%      |

================================================================================
IDENTITAS PROJECT
================================================================================

| Parameter       | Nilai                              |
|-----------------|------------------------------------|
| Nama Aplikasi   | Wayground (Quizizz)               |
| Package ID      | com.quizizz_mobile                |
| Platform        | Google Play Store                  |
| Bahasa          | Indonesia                           |
| Periode Data    | 1 Jan 2024 - 30 Jun 2026          |
| Total Data      | 2.936 ulasan                      |

================================================================================
ATURAN DOKUMEN
================================================================================

1. Semua angka HARUS dapat dilacak ke file source/output
2. [DATA TIDAK DITEMUKAN - PERLU DILENGKAPI] jika info tidak tersedia
3. Konflik dicatat pada Daftar Ketidakkonsistensi
4. Ground truth = hasil pelabelan SVM (bukan validasi pakar)
5. Pakar = validator, BUKAN ground truth

================================================================================
DAFTAR KETIDAKKONSISTENSI
================================================================================

INKONSISTENSI 1: Model vs Metadata
- metadata_latest.json mencatat kernel=linear, C=10.0
- svm_model_latest.pkl aktual berisi kernel=sigmoid, C=100.0
- Status: BELUM DISELESAIKAN

INKONSISTENSI 2: Neutral Lexicon
- constants.py mereferensikan neutral_final.tsv
- File neutral_final.tsv TIDAK ADA
- Sistem menangani dengan try-except graceful
- Status: TIDAK ADA ERROR tapi lexicon netral tidak termuat

INKONSISTENSI 3: Experiment vs Best Config
- experiment_summary.json mencatat threshold=1.0, max_features=2000
- best_config.json mencatat threshold=0.0, max_features=3000
- Status: KONFLIK KONFIGURASI

================================================================================
HASIL ANALISIS ASPEK
================================================================================

4 ASPEK PEMBELAJARAN:
1. Kemudahan Akses - Login, koneksi, loading, antarmuka
2. Gamifikasi - Poin, leaderboard, reward, level, keseruan
3. Materi/Konten - Kuis, bank soal, kurikulum, pembahasan
4. Performa Teknis - Bug, crash, lag, force close

HASIL PER ASPEK:

================================================================================
VALIDASI PAKAR
================================================================================

IDENTITAS PAKAR:
- Nama: Anggik Budi Prasetiyo, S.Pd., M.Li., Gr.
- Jabatan: Guru Bahasa Indonesia
- Instansi: SMAN 3 Jember
- Keahlian: Linguistik dan Bahasa Indonesia
- Jumlah Sampel: 300 ulasan

HASIL VALIDASI LANGSUNG:
- 300/300 sampel (100%) dinyatakan SESUAI dalam form

ANALISIS KOMPARASI:
- Total sampel: 300
- Agreement sistem vs pakar: 211/300 (70.3%)

================================================================================
BACKEND & REST API
================================================================================

TEKNOLOGI: Flask REST API, SQLAlchemy, SQLite

FILE UTAMA:
- app.py - Entry point
- config.py - Konfigurasi database dan CORS
- models.py - Skema SQLAlchemy
- enhanced_api.py - Blueprint API modular
- ml_engine.py - Wrapper ML untuk Flask

ENDPOINT UTAMA:
| Method | Endpoint                 | Fungsi              |
|--------|--------------------------|---------------------|
| GET    | /api/v1/health            | Health check        |
| GET    | /api/v1/stats            | Statistik           |
| GET    | /api/v1/config           | Konfigurasi         |
| POST   | /api/v1/predict          | Prediksi sentimen   |
| POST   | /api/v1/preprocess       | Preprocessing       |
| POST   | /api/v1/label            | Pelabelan          |
| POST   | /api/v1/aspects          | Deteksi aspek      |
| POST   | /api/v1/train            | Training model      |

================================================================================
FINAL DECISION LOG
================================================================================

KEPUTUSAN UNTUK KONFLIK KONFIGURASI:

1. Model vs Metadata Inconsistency:
   - DITEMUKAN: metadata_latest.json tidak cocok dengan model aktual
   - AKSI: Retrain model ATAU perbarui metadata

2. Neutral Lexicon File Missing:
   - DITEMUKAN: neutral_final.tsv tidak ada
   - Sistem menangani dengan try-except graceful

3. Experiment vs Best Config Conflict:
   - DITEMUKAN: Konflik threshold dan max_features
   - AKSI: Pilih salah satu konfigurasi
