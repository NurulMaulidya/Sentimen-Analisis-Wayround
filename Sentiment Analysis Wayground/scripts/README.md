# Perintah penelitian Wayground

Jalankan dari root proyek. [Panduan terminal](../PANDUAN_PERINTAH_TERMINAL.md) berisi parameter, pengaturan environment, dan urutan lengkap yang dapat disalin.

## Alur utama

| Urutan | Berkas | Hasil |
|---|---|---|
| 1 | [prepare_dataset_snapshot.py](prepare_dataset_snapshot.py) | Dataset berlabel dan split berkelompok. |
| 2 | [run_research_experiments.py](run_research_experiments.py) | Menjalankan tujuh eksperimen dari folder `experiments/`. |
| 3 | [select_research_configuration.py](select_research_configuration.py) | Keputusan konfigurasi berdasarkan development. |
| 4 | [lock_final_config.py](lock_final_config.py) | Konfigurasi yang sudah diperiksa dan dikunci. |
| 5 | [train_final_model.py](train_final_model.py) | Model tersimpan dan evaluasi holdout diagnostik. |
| 6 | [visualize_verified_results.py](visualize_verified_results.py) dan [run_aspect_analysis.py](run_aspect_analysis.py) | Grafik evaluasi dan ringkasan aspek. |

## Jalur lain

- [project.ps1](project.ps1) menjalankan pengecekan model, tes, backend, frontend, build, dan persiapan eksperimen.
- [improve_sentiment.py](improve_sentiment.py) menyiapkan lembar anotasi dan membandingkan kandidat. Lihat [panduan perbaikan sentimen](../docs/PERBAIKAN_SENTIMEN.md).
- Skrip bernama `audit_*`, `evaluate_*`, `validate_*`, dan `verify_*` dipakai untuk memeriksa bagian tertentu. Jalankan sesuai kebutuhan setelah membaca tujuan skrip.
- [Skrip pemeliharaan lama](../archive/2026-11-10_penataan/scripts/) menyimpan utilitas satu kali untuk rekonstruksi konfigurasi dan metadata historis.
