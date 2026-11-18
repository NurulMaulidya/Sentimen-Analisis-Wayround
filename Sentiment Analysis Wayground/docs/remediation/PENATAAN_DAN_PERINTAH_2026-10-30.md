# Panduan eksekusi dan penataan file — 30 Oktober 2026

Permintaan: menyediakan urutan perintah terminal lengkap dan memilah berkas yang tidak digunakan. Hasil utama ada di `PANDUAN_PERINTAH_TERMINAL.md` pada root dan runner `scripts/project.ps1`.

## Entry point yang dipakai

| Kelompok | Entry point |
|---|---|
| Scraping opsional | `pipeline/scraper.py`, backup raw terlebih dahulu |
| Pipeline snapshot | `scripts/prepare_dataset_snapshot.py` |
| Tujuh eksperimen | `scripts/run_research_experiments.py`; urutan 2 → derive → 1 → 3–7 |
| Keputusan final | `scripts/select_research_configuration.py` |
| Lock dan training | `scripts/lock_final_config.py`, `scripts/train_final_model.py`; namespace baru, review kontrak/config dahulu |
| Pemeriksaan bundle | `scripts/project.ps1 -Task Check` |
| Visualisasi terverifikasi | `scripts/visualize_verified_results.py` |
| Test | `scripts/project.ps1 -Task Test`; browser terisolasi opsional sesuai panduan |
| Backend | `python -m flask --app backend.app run --host 127.0.0.1 --port 5001` |
| Frontend/build | `scripts/project.ps1 -Task Frontend` / `-Task Build` |

Runner tidak menjalankan scraping/training otomatis. Task Research membuat namespace baru, menghentikan proses bila subprocess gagal, dan berhenti pada konfigurasi terpilih untuk direview. Perintah manual replikasi model disediakan terpisah. Tidak ada eksperimen panjang atau training ulang yang dijalankan dalam pekerjaan penataan ini.

## File yang diarsipkan

| Sumber | Alasan |
|---|---|
| `frontend.zip` | Snapshot ZIP lama; tidak digunakan oleh import, npm atau runtime |
| `output.zip` | Snapshot ZIP output lama; output aktif tersedia sebagai direktori berversi |
| `pipeline.zip` | Snapshot ZIP pipeline lama; runtime memakai source Python |
| `tests/visualize_model.py` | Wrapper manual mengimpor modul yang tidak tersedia (`scripts.experiments.viz_final_results`); pengganti terverifikasi disediakan |

Keempat file dipindahkan utuh ke `archive/cleanup_2026-10-30/` dengan struktur relatif asal. Total 59.054.925 byte. Tidak ada penghapusan permanen. Target absolut diperiksa berada dalam workspace sebelum pemindahan. SHA-256 setelah pemindahan cocok dengan sumber. Manifest: `ARCHIVE_MANIFEST_2026-10-30.json`.

Pulihkan dengan menyalin satu file dari path `archive` dalam manifest ke path `source`, setelah memastikan tujuan tidak menimpa file baru; periksa hash terhadap manifest. `archive/` diabaikan Git: sertakan direktori tersebut dalam backup terpisah.

## File yang dipertahankan

- `models/final_v4`, `data/pipeline_output_v4`, dan `output_v4/experiments` adalah bundle aktif terikat hash. Database dan upload pengguna dipertahankan.
- Snapshot v3/v4, output eksperimen lama termasuk percobaan split yang dibatalkan, dataset pakar, kamus dan lexicon adalah bukti penelitian/provenance; tidak dianggap sampah hanya karena bukan default runtime.
- Model lama dan output v2/v3 tetap dipertahankan. Beberapa manifest mengacu path historis; memindahkan folder tanpa pemeriksaan provenance dapat membatalkan verifikasi.
- Skrip visualisasi lama `viz_final_results.py`, PCA/t-SNE, serta learning curve dipertahankan sebagai source historis yang masih disebut audit. Belum disahkan sebagai visualisasi final saat ini dan tidak menjadi perintah rekomendasi.
- `pipeline/run_pipeline.py` dan runner eksperimen lama tetap tersedia untuk referensi. Panduan baru mengarahkan pengguna ke snapshot terisolasi agar canonical aktif tidak ditulis ulang.
- Test diagnostik lama tidak dihapus. Suite acceptance saat ini disebut secara eksplisit agar hasil verifikasi tidak disamakan dengan klaim seluruh test historis lulus.

Pemilahan ini mengarsipkan file yang sudah terbukti tidak diperlukan untuk alur aktif. Ini belum menyatakan semua file historis di proyek aman dipindahkan; arsip lanjutan membutuhkan pemeriksaan referensi dan hash masing-masing.

## Visualisasi dan bukti verifikasi

Skrip baru memuat bundle melalui `load_verified_pipeline()`, membaca JSON evaluasi yang terikat metadata, dan menggunakan urutan kelas serta matrix tersimpan. Tidak melakukan fit, sampling, pemilihan parameter, atau fallback split. Output pada `output_v4/verified_visualizations/` berisi confusion matrix, metrik per kelas dan JSON provenance. Kedua PNG telah dibuka untuk memeriksa keterbacaan, label sumbu, angka, judul versi dan keterangan diagnostic holdout dengan prior exposure.

Nilai matrix: `[[92,24,2],[16,82,21],[6,26,335]]`, total 604. Versi `20261029_223454`, accuracy 0,842715 dan macro F1 0,790209. Angka ini diagnostik, bukan klaim evaluasi independen. Grafik ROC/PCA/t-SNE/learning curve lama belum diperbaiki melalui pekerjaan ini.

Verifikasi runner Check berhasil, Test menghasilkan **109 tes Python lulus** (satu warning fixture parsing tanggal tidak valid) dan **6 tes frontend lulus**. Log: `PERINTAH_2026-10-30_TESTS.txt`. Build juga diperiksa melalui runner. Task Research tidak dijalankan ulang karena menjalankan tujuh eksperimen penuh; alurnya mengikuti runner penelitian yang sudah diuji tahap 3–5. Instalasi fresh dependencies dan perintah replikasi training dalam panduan belum dieksekusi; environment yang sudah tersedia digunakan untuk verifikasi ini.

Hash model, metadata, keputusan konfigurasi dan artefak terikat diverifikasi setelah penataan; tidak ada perubahan pada model aktif atau dataset. Bukti mesin: `PERINTAH_2026-10-30_VERIFICATION.json`. Ini melengkapi dokumentasi dan dua visualisasi dasar; tahap 6 menyeluruh belum dinyatakan selesai.
