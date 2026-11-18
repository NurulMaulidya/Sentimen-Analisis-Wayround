# Tahap 1 — fondasi kontrak dan keterlacakan

Tanggal: 29 Oktober 2026. Status: selesai dalam lingkup tahap 1.

## Tujuan dan hasil

Menghindari pemakaian artefak eksperimen atau split yang salah versi sebelum perbaikan dataset dan eksperimen dimulai. Parameter penelitian, path default, hasil eksperimen, dan model aktif tidak diganti pada tahap ini.

| File | Perubahan | Dampak |
|---|---|---|
| shared/experiment_artifacts.py | Daftar nama JSON canonical eksperimen 1–7 | Checkpoint tidak lagi mencari suffix versi yang tidak dihasilkan writer saat ini |
| pipeline/run_pipeline.py | Stage 6 memakai daftar canonical, selalu membaca sumber terkini, memeriksa JSON object, mencatat hash, missing dan invalid | Checkpoint usang tidak menutupi hasil yang belum lengkap; `collected` hanya menyatakan file terkumpul, bukan validasi metodologi |
| scripts/experiments/experiment_2_threshold.py | Output menjadi experiment_2_threshold_results.json | Nama writer sesuai checkpoint; arsip hasil terdahulu tidak ditimpa melalui eksekusi eksperimen dalam tahap ini |
| scripts/experiments/experiment_4_svm.py | Membaca hasil TF-IDF canonical saja; metadata sumber sesuai path canonical dan CV menyebut StratifiedGroupKFold | Berkas bersuffix lama tidak diam-diam mengalahkan hasil canonical; hasil lama tidak diedit retroaktif |
| shared/final_config.py | Loader delimiter/BOM bersama; label dinormalisasi sebelum memeriksa distribusi; tolak label tak dikenal dan teks kosong | Label kapital awal tidak menghasilkan distribusi nol palsu; dataset lama tetap ditolak bila tidak cocok kontrak |
| shared/splitting.py | Loader dataset bersama dan verifikasi seluruh ID, keunikan ID, pemisahan ID, teks dan label | Cache subset, teks berubah, atau label berubah tidak digunakan; fallback grouped split meliputi seluruh baris |
| tests/test_artifact_contracts.py | Regresi dataset berubah, input BOM/semicolon, teks kosong dan checkpoint usang | Menjaga agar perbaikan tidak mengembalikan kehilangan data/leakage secara diam-diam |

Fallback menghitung split dalam memori dan menyertakan `source=recomputed` serta catatan cache. Fallback tidak menulis ulang split penelitian di disk. Split baru akibat perubahan data harus ditinjau dan disimpan sebagai bagian tahap dataset, sebelum eksperimen berikutnya dijalankan.

Stage 6 tetap merupakan pengumpul artefak. Parameter `force` dipertahankan untuk kompatibilitas pemanggil, tetapi sumber selalu diperiksa ulang. Status checkpoint tidak boleh dianggap sertifikat bahwa konfigurasi/model final sudah konsisten. Salinan lama di direktori checkpoint tidak dihapus; daftar sumber terkini dan status summary yang menjadi acuan.

## Perapian struktur pada tahap ini

Pencarian referensi log pada file Python dan Markdown tidak menemukan pemakaian empat log console historis. File berikut dipindahkan dari output_v2/ ke archive/remediation_2026-10-29/logs/:

- compare_configurations_log.txt
- experiment_run_log.txt
- experiment_run_log_tahap5_v2.txt
- experiment_run_log_tahap5_v2_r2.txt

Ukuran total: 116.769 byte. SHA-256 sebelum dan sesudah pemindahan sama untuk setiap file. Manifest tersedia pada `ARCHIVE_MANIFEST_2026-10-29.json`. Dataset, binary model, PNG, dan hasil eksperimen JSON/CSV tetap di tempatnya karena masih diperlukan untuk pembandingan dan migrasi berikutnya.

## Verifikasi yang dijalankan

Perintah regresi:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_artifact_contracts.py tests/test_split_leakage.py -q -p no:cacheprovider
```

Hasil: **15 passed, 0 failed**. Kasus meliputi baris tambahan yang dahulu bisa terbuang karena cache subset, perubahan processed_text yang bisa menyebabkan grup melintasi split, perubahan label, duplikat ID, perubahan urutan baris, kapitalisasi label, BOM/delimiter, teks kosong, dan checkpoint dengan sumber berubah/invalid.

Pemeriksaan langsung dataset dan checkpoint memakai direktori sementara; hasil lengkap tersimpan pada `TAHAP_01_VERIFICATION.json`:

| Pemeriksaan | Hasil |
|---|---|
| Dataset lama 3.060 vs kontrak 3.082 | Ditolak dengan pesan ukuran yang benar |
| Dataset v2 3.082, label kapital awal | Lulus kontrak ukuran, kelas, dan teks |
| Split berkelompok lama | 2.449/611, overlap grup 0, cache sesuai dataset |
| Split berkelompok v2 | 2.470/612, overlap grup 0, cache sesuai dataset |
| Checkpoint v2 | incomplete: eksperimen 1, eksperimen 2 dan best_config belum tersedia |
| Model aktif | Tidak dilatih ulang; checksum dicatat dalam JSON verifikasi |

Lulus kontrak v2 di sini hanya mencakup ukuran, label, dan teks. Ini **belum** mengesahkan periode penelitian, kualitas anotasi, independensi holdout, atau keputusan model final.

## Masalah terbuka dan langkah berikutnya

Default path masih menuju dataset/model lama. Mengubahnya sekarang akan mencampurkan inference lama dengan konfigurasi baru yang belum selesai diverifikasi. Tahap 2 perlu menyelesaikan versi dataset dan label sebelum promosi model.

Masih terbuka: periode yang berbeda antara artefak dan versi scraper; threshold 0/0,5/1; pengujian kualitas anotasi; hasil eksperimen 1–2 v2; keputusan linear versus RBF dan none versus SMOTE; mismatch model biner dengan final_config; normalisasi `bangett` yang gagal pada audit awal; status tanpa label dan halaman frontend belum lengkap; confusion matrix PNG nol serta visualisasi usang. Tes preprocessing penuh tidak dijalankan ulang pada tahap ini karena implementasi bahasa belum diubah; hasil 53 lulus/1 gagal dari audit awal tetap menjadi temuan terbuka.

Tahap berikutnya dimulai dari pembandingan versi scraper/kontrak tanggal dan pembentukan snapshot data yang dapat ditelusuri, tanpa mengubah data lama secara diam-diam.
