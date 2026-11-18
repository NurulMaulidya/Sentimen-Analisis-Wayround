# Tahap 2 — dataset, preprocessing dan provenance label

Tanggal: 29 Oktober 2026. Status: **selesai untuk persiapan dataset**, belum dipromosikan menjadi dataset/model aktif.

## 1. Keputusan periode dan kontrak data

Pada awal tahap ini, scraper di workspace sudah menggunakan 1 Januari 2024 sampai **30 Juni 2026**, berbeda dari versi 1 Juni yang diperiksa pada audit awal. Perubahan pengguna tersebut dipertahankan. `shared/research_config.py` kini menyimpan batas awal inklusif 2024-01-01 dan batas akhir eksklusif 2026-07-01.

`shared/data_contract.py` menyediakan normalisasi kolom dan pemeriksaan data. Input historis `tanggal` diterima dan diubah menjadi `at`. Jika keduanya ada tetapi tanggalnya berbeda, input ditolak. Kolom tanggal tidak valid, ID kosong/duplikat, content kosong, serta rating bukan bilangan bulat 1–5 juga ditolak.

Keluaran scraper sekarang mengikuti `FINAL_OUTPUT_COLUMNS`: reviewId, at, content, rating, appVersion. Tanggal disimpan dalam bentuk timestamp UTC. Untuk sumber historis yang hanya menyimpan tanggal tanpa jam, jam 00:00 merupakan representasi tanggal tersebut, bukan rekonstruksi waktu ulasan asli. Tidak ada scraping jaringan pada tahap ini.

Stage 1 pipeline memakai pemeriksaan periode dan mengekspor baris di luar periode bila ada. Cache Stage 1 yang masih memuat tanggal di luar periode akan ditolak dan perlu dibangun ulang. Cache/data lama tidak ditulis ulang otomatis pada tahap ini.

## 2. Pelabelan yang diselaraskan

Baseline persiapan threshold **0,5** dipusatkan di shared/research_config dan digunakan oleh shared.labeling serta shared.final_config. Baseline ini bukan hasil tuning atau keputusan final penelitian. Labeling method pada konfigurasi menyebut lexicon v2, sesuai jalur aktif.

Adapter pipeline.labeling dan labeler backend sekarang memakai lexicon aktif yang sama, bukan singleton v1 secara diam-diam. Fungsi low-level shared.lexicon tetap mendukung v1 sebagai pembanding historis; pemanggilan low-level tanpa lexicon eksplisit bukan jalur persiapan snapshot baru.

Eksperimen 2 sekarang memanggil shared.labeling dan hanya mencocokkan label pakar pada **development partition**. Distribusi seluruh dataset tetap boleh dihitung sebagai audit deskriptif; distribusi itu tidak menjadi kriteria seleksi. Metadata eksperimen mencatat partition, sumber split, lexicon dan baseline. Grid threshold belum dijalankan pada tahap ini.

## 3. Perbaikan preprocessing

- Ejaan penekanan `bangett` sekarang menjadi `banget` melalui pengecualian sempit; kata sah seperti error, allah, app dan saat dilindungi oleh tes.
- Emoji jempol turun dengan skin tone terang yang keliru tercatat positif diubah menjadi negatif.
- Pemetaan emoji mendahulukan rangkaian terpanjang agar variasi skin tone diproses sebagai satu unit.
- Versi persiapan dicatat sebagai `v3-stage2`, disertai hash kode, kamus normalisasi dan file lexicon dalam manifest.

Perubahan preprocessing dapat mengubah prediksi raw text pada model lama. Binary model tidak diganti, tetapi belum boleh menganggap kesetaraan preprocessing dengan training lama sudah selesai. Promosi model yang dilatih dengan versi ini menjadi pekerjaan tahap 4.

## 4. Snapshot baru dan struktur keluaran

Builder baru: `scripts/prepare_dataset_snapshot.py`. Keluaran disimpan pada **data/pipeline_output_v3/**. Direktori yang sudah ada ditolak; builder tidak menimpa snapshot, termasuk direktori parsial. Manifest ditulis terakhir sebagai penanda pembangunan selesai. Tidak ada penggantian default path aplikasi.

```text
data/pipeline_output_v3/
  dataset_manifest.json
  stage1_scraped.csv
  stage2_preprocessed_new.csv
  stage3_labeled.csv
  stage3_final_labeled.csv
  stage5_split_v2/
    development.csv
    holdout.csv
    split_metadata.json
  audit/
    excluded_period.csv
    duplicate_content.csv
    empty_processed.csv
    expert_mapping_candidates.csv
    annotation_review_queue.csv
```

Perintah yang dijalankan:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts/prepare_dataset_snapshot.py
```

Untuk pembangunan berikutnya, gunakan `--output data/nama_snapshot_baru`. Jangan menghapus atau mengubah snapshot v3 untuk memaksa rerun; buat versi baru jika kode, data, atau keputusan label berubah.

| Langkah | Jumlah |
|---|---:|
| Input data/02_data_akhir.csv | 3.119 |
| Di luar periode yang sekarang disepakati | 0 |
| Duplikat content setelah normalisasi teks | 4 |
| Setelah dedup content | 3.115 |
| Kosong setelah preprocessing | 37 |
| Dataset akhir | **3.078** |

Empat duplikat berbeda kapitalisasi/spasi pada data sumber sebelumnya lolos perbandingan content persis. Snapshot memilih representasi pertama menurut urutan tanggal dan reviewId yang stabil. Baris yang dipisahkan tetap disimpan di audit; tidak ada penghapusan permanen data sumber.

| Label final | Jumlah |
|---|---:|
| Positif | 2.065 |
| Negatif | 636 |
| Netral | 377 |

Snapshot mempunyai 2.776 grup processed_text unik, 122 grup duplikat yang mencakup 424 baris, dan **0 grup berlabel konflik**. Duplikat setelah preprocessing dipertahankan dan dipisahkan berdasarkan grup, bukan dipecah antarfold.

## 5. Anotasi, konflik dan independensi

Snapshot menerima **288 label pakar** melalui kecocokan content ternormalisasi yang persis, atau reviewId disertai content yang juga diverifikasi. Ada 0 label anotator kedua. Kolom expert_independence_verified tetap False: pembacaan file tidak dapat membuktikan bahwa anotator independen atau bahwa label belum pernah dipakai mengembangkan lexicon.

Sebanyak **12 kasus** masuk `audit/annotation_review_queue.csv`: 1 kandidat prefix, 4 fuzzy, dan 7 unmatched. Lima kandidat nonpersis yang sebelumnya dianggap valid tidak lagi otomatis menjadi ground truth pakar. Selama menunggu tinjauan, label baseline lexicon dipakai untuk baris terkait; label pakar tidak dibuat atau ditebak.

Worksheet anotasi tambahan tidak otomatis dihitung sebagai 40 anotasi independen baru karena berisi label dari baris yang juga sudah ada dalam anotasi lama. Snapshot tidak memakai kappa 1,0 sebagai klaim reliabilitas independen.

Jalur integrasi lama `pipeline/integrate_expert_labels.py` juga diperbaiki: loader delimiter bersama, ID wajib unik, prefix/fuzzy tidak diterapkan otomatis, content anotasi baru diverifikasi, dan konflik tidak diam-diam menimpa label. Duplikat ID teranotasi ditolak untuk adjudikasi.

Bug kappa diperbaiki dengan menangkap label lama **sebelum** pembaruan label baru. Tes regresi dengan dua contoh dan satu ketidaksetujuan menghasilkan observed agreement 0,5, bukan kesepakatan palsu 1,0. Angka kappa tetap harus dibaca sebagai perbandingan label pada sampel yang tersedia, bukan bukti independensi anotator. Hasil JSON lama tidak diedit retroaktif.

## 6. Kualitas label yang masih terbatas

Perbandingan baseline lexicon terhadap 288 label pakar yang dipakai kembali menghasilkan accuracy **82,64%**, Macro F1 **61,91%**, dan recall netral **10,53%** (2 dari 19 contoh netral). Ini pengukuran kesesuaian deskriptif pada anotasi historis, bukan evaluasi gold independen dan bukan dasar tuning dalam tahap ini.

Kualitas label netral belum dapat diperbaiki hanya dengan perubahan script. Tahap berikutnya dapat menguji threshold pada development dan melaporkan keterbatasan itu; anotasi baru atau adjudikasi manusia tidak boleh digantikan label buatan. Tidak ada klaim bahwa seluruh dataset sekarang telah tervalidasi pakar.

## 7. Split dan verifikasi

Split berkelompok menghasilkan **2.474 development dan 604 holdout**, mendekati 80:20. Ukuran tidak dipaksa menjadi 612 karena grup teks tidak boleh dipecah. Seluruh ID tercakup dan tidak ada overlap grup. Lima fold CV development juga diperiksa: overlap grup setiap fold = 0 dan semua kelas ada pada validation fold.

Karena histori pemakaian anotasi/lexicon tidak dapat direkonstruksi sepenuhnya, manifest menyatakan `holdout_independence_verified=false`. Split ini mengisolasi teks secara prospektif; jangan menyebutnya test set gold independen. Holdout tidak digunakan untuk memilih parameter pada tahap ini.

Verifikasi kode:

```powershell
python -m pytest tests/test_preprocessing.py tests/test_dataset_preparation.py tests/test_artifact_contracts.py tests/test_split_leakage.py -q -p no:cacheprovider
```

Hasil: **74 passed, 0 failed**. Ada satu warning pandas pada tes yang sengaja memasukkan tanggal tidak valid; nilai tersebut tetap ditolak. Tes meliputi batas tanggal, kontrak ID/rating, anotasi nonpersis dan konflik, perbaikan kappa, larangan overwrite, keseragaman labeler, serta seleksi threshold yang tidak menerima holdout.

Pada sampel tetap 60 ulasan v3: hasil tersimpan = preprocessing shared sekarang = wrapper web, tanpa perbedaan. Hash seluruh input, resource, kode dan output dalam manifest telah diverifikasi. Checksum dataset lama/v2 dan model aktif sama dengan catatan tahap 1. Syntax file yang diubah dan whitespace diff juga diperiksa.

Bukti rinci: `docs/remediation/TAHAP_02_VERIFICATION.json` dan `data/pipeline_output_v3/dataset_manifest.json`.

## 8. Batas tahap ini dan tahap berikutnya

Tidak dilakukan pelatihan SVM, pemilihan konfigurasi final, penggantian binary model, modifikasi frontend, atau regenerasi gambar evaluasi. Data lama/v2 dan hasil eksperimennya tetap dipertahankan sebagai bukti historis dan dependensi model lama. Tidak ada file sumber penelitian yang diarsipkan lagi pada tahap ini.

**Tahap 3 berikutnya:** jalankan tujuh eksperimen dalam namespace output baru dengan snapshot v3 dan split berkelompok; perbaiki eksperimen sensitivitas split dan dependency baseline; pilih threshold berdasarkan development, buat snapshot penerus jika berubah; bandingkan model menggunakan Macro F1 dan dokumentasikan keputusan final. Kontrak jumlah/distribusi di final_config masih untuk versi terdahulu dan baru boleh diperbarui saat keputusan dataset/model dikunci; menjalankan training final sekarang memang belum memenuhi kontraknya.
