================================================================================
1. STATUS DAN ATURAN DOKUMEN
================================================================================

Dokumen ini adalah SINGLE SOURCE OF TRUTH untuk project ini.

ATURAN WAJIB:
1. Semua angka HARUS dapat dilacak ke file source/output
2. [DATA TIDAK DITEMUKAN - PERLU DILENGKAPI] jika info tidak tersedia
3. Konflik dicatat pada Daftar Ketidakkonsistensi
4. Ground truth = hasil pelabelan SVM (bukan validasi pakar)
5. Pakar = validator, BUKAN ground truth

================================================================================
2. IDENTITAS PROJECT
================================================================================

| Parameter       | Nilai                              |
|-----------------|------------------------------------|
| Nama Aplikasi   | Wayground (Quizizz)               |
| Package ID      | com.quizizz_mobile                |
| Platform        | Google Play Store                  |
| Bahasa          | Indonesia                           |
| Periode Data    | 1 Jan 2024 - 30 Jun 2026          |

================================================================================
5. DATASET
================================================================================

PENYUSUTAN DATA:
| Tahap                  | File                          | Baris  |
|------------------------|-------------------------------|--------|
| 1. Scraping Mentah     | 01_semua_data.csv           | 22.606 |
| 2. Filter Tanggal      | 02_data_akhir.csv           | 4.517  |
| 3. Deduplikasi         | 03_cleaned_data.csv         | 2.994  |
| 4. Preprocessing       | stage2_preprocessed.csv      | 2.936  |
| 5. Pelabelan           | stage3_labeled.csv           | 2.936  |

DISTRIBUSI KELAS (dari metadata_latest.json):
| Kelas    | Jumlah | Persentase |
|----------|--------|------------|
| Positif  | 2.062  | 70.23%     |
| Netral   | 458    | 15.60%     |
| Negatif  | 416    | 14.17%     |
| TOTAL    | 2.936  | 100%       |

================================================================================
12. PELABELAN SENTIMEN
================================================================================

LEXICON (dari file aktual):
- Positive: 191 kata
- Negative: 110 kata
- Neutral: [TIDAK ADA - neutral_final.tsv TIDAK DITEMUKAN]

CATATAN: File neutral_final.tsv yang direferensikan di constants.py
TIDAK ADA di folder data/lexicon/. Sistem menangani dengan try-except.
