# Model deployment dan arsip historis

| Direktori | Peran |
|---|---|
| final_v4/ | Deployment default terverifikasi tahap 4; pipeline dan metadata harus dipakai bersama |
| candidate_char/ | Kandidat TF-IDF kata dan karakter yang bisa dipilih di web. Evaluasi independen menunggu anotasi. |
| [Arsip model lama](../archive/2026-11-09_struktur/models/final/) | Artefak model historis untuk penelusuran |

Model aktif: `final_v4/final_pipeline.joblib`, versi **20261029_223454**. Konfigurasi: TF-IDF 5.000 fitur n-gram 1–3, linear SVM C=10, OVO, tanpa resampling. Preprocessing tetap dikerjakan oleh shared/preprocessing sebelum pipeline TF-IDF; fingerprint dan resource-nya diverifikasi saat loading.

Training kandidat menulis bundle ke subfolder `runs/` yang ditetapkan di [panduan terminal](../PANDUAN_PERINTAH_TERMINAL.md). Rincian evaluasi diagnostik dan riwayat paparan holdout tersedia pada [laporan tahap 4](../docs/remediation/TAHAP_04.md).
