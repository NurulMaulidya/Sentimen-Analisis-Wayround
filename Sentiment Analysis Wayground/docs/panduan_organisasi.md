# Peta proyek Wayground

Gunakan [README utama](../README.md) untuk membuka web. Gunakan [panduan terminal](../PANDUAN_PERINTAH_TERMINAL.md) saat menjalankan penelitian ulang.

## Dua alur utama

**Penelitian**

Data tersimpan → snapshot dan label → eksperimen development → konfigurasi terpilih → training dan evaluasi → model tersimpan.

**Web**

Unggahan CSV atau satu ulasan → preprocessing → TF-IDF dan SVM tersimpan → prediksi sentimen, aspek, dan riwayat.

| Ingin menelusuri | Titik masuk | Hasil atau lokasi |
|---|---|---|
| Menyiapkan dataset | [prepare_dataset_snapshot.py](../scripts/prepare_dataset_snapshot.py) | Snapshot v4 di [data/pipeline_output_v4](../data/pipeline_output_v4/). Run baru menulis ke **runs/**. |
| Menjalankan tujuh eksperimen | [run_research_experiments.py](../scripts/run_research_experiments.py) | [Konfigurasi terpilih](../output_v4/experiments/selected_configuration.json). Runner menyiapkan dependensi tiap eksperimen. |
| Mengunci dan melatih model | [lock_final_config.py](../scripts/lock_final_config.py) lalu [train_final_model.py](../scripts/train_final_model.py) | [Bundle model v4](../models/final_v4/) dan [evaluasi diagnostik](../output_v4/experiments/final_model_evaluation.json). |
| Membuka hasil penelitian | [Penelitian di web](../frontend/src/pages/Research.jsx) | [Grafik evaluasi](../output_v4/verified_visualizations/) dan [hasil aspek](../output_v4/experiments/aspect_analysis_summary.csv). |
| Mengikuti analisis unggahan | [Ringkasan sesi](../frontend/src/pages/AnalysisDetail.jsx) dan [API analisis](../backend/api/analysis.py) | Prediksi, analisis aspek, serta evaluasi bila ada label aktual. |
| Melengkapi skripsi | [Laporan dan lampiran](skripsi/README.md) | Dokumen Word dan bukti implementasi. |

## Lokasi berkas

| Lokasi | Isi |
|---|---|
| **frontend/** dan **backend/** | Tampilan React dan API Flask. |
| **shared/** dan **pipeline/** | Preprocessing, pelabelan, pemisahan data, scraping, serta deteksi aspek. |
| **scripts/** dan **tests/** | [Perintah penelitian menurut urutan](../scripts/README.md), verifikasi, dan pengujian. |
| **data/** dan **dataset/** | Ulasan, kamus, snapshot, anotasi, serta penyimpanan lokal aplikasi. |
| **models/** | [Model v4 dan kandidat karakter](../models/README.md). |
| **output_v4/** | Evaluasi, eksperimen dan visualisasi model penelitian. |
| **runs/** | Eksperimen baru, anotasi, replikasi, dan bukti tes lokal. |
| **docs/skripsi/** | Laporan akhir dan lampiran Word. |
| **docs/riwayat/** | Audit, panduan, dan laporan dari tahap terdahulu. |
| **archive/** | Snapshot salinan, artefak lama, dan visualisasi historis. |

Snapshot v3 dan output pemilihan ambangnya berada di **data/pipeline_output_v3/** dan **output_v3/**. Manifest v4 merujuk keduanya. Bundle v4 juga memverifikasi sejumlah artefak berdasarkan jalur dan SHA-256.

## Salinan proyek dan pemulihan

Salinan **Hasil Final** dan ZIP-nya berada di [arsip snapshot 10 November](../archive/2026-11-10_snapshot/). Lokasi baru setiap berkas yang dipindahkan tersedia di [manifest penataan](remediation/MANIFEST_PENATAAN_2026-11-10.json). Folder arsip dan run bersifat lokal.

Nama lama dan nama baru dokumen riwayat serta skrip pemeliharaan tercatat di [manifest pengelompokan](remediation/MANIFEST_PENGELOMPOKAN_2026-11-10.json).
