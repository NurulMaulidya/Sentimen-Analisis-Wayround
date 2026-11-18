# Struktur proyek dan urutan perintah Wayground

Jalankan perintah dari root proyek di PowerShell. Pilih **Bagian 2 → 3 → 4 → 5** untuk mengulang penelitian sampai web, atau langsung **Bagian 5** untuk memakai model v4 yang sudah tersedia.

## 1. Peta proyek

```text
backend/                       Flask API, layanan inferensi dan penyimpanan
frontend/                      React/Vite, tes tampilan dan browser
shared/                        Preprocessing, pelabelan, split dan kontrak model
pipeline/                      Scraper, integrasi anotasi, deteksi aspek dan prediksi
scripts/                       Snapshot, eksperimen, training dan verifikasi
  experiments/                 Eksperimen 1–7 dan konfigurasi baseline
tests/                         Tes Python
data/                          Raw Google Play, kamus, SQLite, backup dan unggahan
  pipeline_output_v4/          Snapshot dataset dan split model aktif
dataset/                       TSV/XLSX anotasi dan sumber audit kamus
models/final_v4/               Bundle model aktif
models/candidate_char/         Kandidat eksperimental di web
output_v4/                     Evaluasi, log eksperimen dan visualisasi aktif
runs/                          Keluaran baru, replikasi dan pengujian lokal
docs/                          Metodologi, keputusan dan bukti verifikasi
  skripsi/                     Laporan akhir dan lampiran Word
archive/2026-11-09_struktur/   Artefak, skrip dan visualisasi historis yang dipindah
archive/2026-11-10_snapshot/   Salinan Hasil Final dan ZIP
```

Snapshot *data/pipeline_output_v3/* dan bukti threshold *output_v3/* tetap di lokasi asal untuk jejak sumber v4. Direktori aktif v4 juga tetap di lokasi asal karena path dan hash-nya terikat kontrak deployment. *pipeline/run_pipeline.py* dan pembantu aspek lamanya masih dipertahankan untuk kompatibilitas dan tes checkpoint historis. Untuk penelitian, gunakan snapshot dan runner pada Bagian 3.

[Peta proyek](docs/panduan_organisasi.md) menunjukkan alur aktif dan lokasi dokumen. [Manifest arsip lama](docs/remediation/ARCHIVE_MANIFEST_2026-11-09.json) dan [manifest penataan terbaru](docs/remediation/MANIFEST_PENATAAN_2026-11-10.json) mencatat lokasi baru dan hash. Untuk memulihkan berkas, gunakan path asal dalam manifest setelah memastikan tujuan kosong. Folder *archive/* dan *runs/* bersifat lokal dan diabaikan Git.

## 2. Persiapan lingkungan

```powershell
Set-Location -LiteralPath 'D:\ACER\Library\OneDrive\Documents\Tugas Akhir TIF\Sentiment Analysis Wayground'
$env:PYTHONUTF8 = '1'
node --version
npm.cmd --version
```

Jika environment web dan Node sudah tersedia, lewati instalasi. Untuk instalasi baru gunakan Python 3.14 untuk web, Python 3.10 untuk riset, dan Node 20 atau lebih baru. Jalankan tiap baris setelah baris sebelumnya berhasil.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
npm.cmd --prefix frontend ci

py -3.10 -m venv .venv310
.\.venv310\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv310\Scripts\python.exe -m pip check
```

Tetapkan interpreter untuk sesi PowerShell ini. Jika Python 3.10 yang sudah terpasang memuat versi inti yang sesuai, gunakan **$researchPy = (Get-Command python.exe).Source** sebagai alternatif, lalu jalankan pemeriksaan.

```powershell
$webPy = (Resolve-Path .venv/Scripts/python.exe).Path
$researchPy = (Resolve-Path .venv310/Scripts/python.exe).Path
& $researchPy scripts/check_environment.py
```

## 3. Data, snapshot, eksperimen dan seleksi

Untuk mengulang penelitian dengan input yang disimpan, gunakan **data/02_data_akhir.csv**. Jika perlu membangun ulang dari raw tanpa jaringan, jalankan blok opsional ini setelah membuat ID run pada Langkah 1. Untuk mengambil ulasan baru, jalankan scraper dengan **--output-dir "$run/raw"** saja. Ulasan dari scraping baru dapat menghasilkan dataset berbeda.

```powershell
New-Item -ItemType Directory -Path "$run/raw" -Force | Out-Null
Copy-Item -LiteralPath data/01_semua_data.csv -Destination "$run/raw/01_semua_data.csv"
& $researchPy pipeline/scraper.py --rebuild-from-raw --output-dir "$run/raw"
if ($LASTEXITCODE -ne 0) { throw 'Rebuild data gagal.' }
$source = "$run/raw/02_data_akhir.csv"
```

Jalankan langkah berikut satu per satu. Langkah 1 menetapkan variabel untuk semua langkah berikutnya. Untuk memakai rebuild atau scraping baru, jalankan blok opsional di atas setelah Langkah 1.

```powershell
# 1. Tentukan namespace baru dan sumber data.
$run = 'runs/riset_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
if (Test-Path -LiteralPath $run) { throw 'Pilih ID run baru.' }
$source = 'data/02_data_akhir.csv'

# 2. Buat snapshot persiapan dan split berkelompok.
& $researchPy scripts/prepare_dataset_snapshot.py --source $source --output "$run/preparation"
if ($LASTEXITCODE -ne 0) { throw 'Persiapan gagal.' }

# 3. Jalankan tujuh eksperimen.
& $researchPy scripts/run_research_experiments.py --preparation "$run/preparation" --selected "$run/selected" --output "$run/output"
if ($LASTEXITCODE -ne 0) { throw 'Eksperimen gagal. Baca log di folder research_run.' }

# 4. Catat konfigurasi terpilih.
& $researchPy scripts/select_research_configuration.py --output "$run/output"
if ($LASTEXITCODE -ne 0) { throw 'Seleksi konfigurasi gagal.' }
Get-Content "$run/output/experiments/selected_configuration.json"
```

Runner eksperimen menjalankan **2 → snapshot dengan threshold terpilih → 1 → 3 → 4 → 5 → 6 → 7**. Eksperimen 2 memilih threshold. Eksperimen berikutnya memeriksa sensitivitas split, TF-IDF, SVM, resampling, OVO/OVR dan regularisasi. Runner mengatur dependensi antar tahap, hash, log dan manifest. Jika run berstatus gagal dan snapshot *selected* sudah terbentuk, ulangi perintah langkah 3 dengan **--resume** dan path yang sama. Jika validasi resume menolak, gunakan ID baru.

Alternatif untuk langkah 1–4 dengan input tersimpan:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Research -RunId percobaan_01 -Python $researchPy
if ($LASTEXITCODE -ne 0) { throw 'Rangkaian riset gagal.' }
$run = 'runs/percobaan_01'
```

## 4. Lock, training, evaluasi dan laporan

Training mengikuti kontrak penelitian v4, termasuk konfigurasi terpilih, jumlah data, label, dan split. Untuk melatih dari **run baru yang memenuhi kontrak tersebut**, gunakan tiga variabel ini di terminal penelitian.

```powershell
$env:WAYGROUND_PIPELINE_OUTPUT_DIR = "$run/selected"
$env:WAYGROUND_OUTPUT_DIR = "$run/output"
$env:WAYGROUND_FINAL_MODEL_DIR = "$run/model"
```

Jika ingin **mereplikasi model v4 dengan dataset dan keputusan asli**, gunakan blok berikut sebagai alternatif pengaturan tiga variabel di atas. Bukti eksperimen yang disalin tetap merujuk artefak asli v4 untuk verifikasi. Pilih folder baru setiap kali.

```powershell
$run = 'runs/replika_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
if (Test-Path -LiteralPath $run) { throw 'Pilih ID run baru.' }
$env:WAYGROUND_PIPELINE_OUTPUT_DIR = 'data/pipeline_output_v4'
$env:WAYGROUND_OUTPUT_DIR = "$run/output"
$env:WAYGROUND_FINAL_MODEL_DIR = "$run/model"

New-Item -ItemType Directory -Path "$run/output/experiments" -Force | Out-Null
Copy-Item -LiteralPath output_v4/research_run -Destination "$run/output/research_run" -Recurse
Get-ChildItem -LiteralPath output_v4/experiments -File |
    Where-Object { $_.Name -like 'experiment_*' -or $_.Name -eq 'selected_configuration.json' } |
    Copy-Item -Destination "$run/output/experiments"
```

Jalankan urutan berikut dalam terminal yang sama. Jika hasil seleksi baru berbeda dari kontrak v4, berhenti di pemeriksaan lock dan tinjau perubahan metodologi sebelum menetapkan konfigurasi deployment baru. Holdout v4 memiliki paparan historis dan hasil replikasinya tetap diagnostik.

```powershell
# 5. Kunci dan verifikasi konfigurasi.
& $researchPy scripts/lock_final_config.py
if ($LASTEXITCODE -ne 0) { throw 'Lock gagal. Periksa konfigurasi dan kontrak dataset.' }

# 6. Latih model dan simpan evaluasi, prediksi holdout serta confusion matrix.
& $researchPy scripts/train_final_model.py
if ($LASTEXITCODE -ne 0) { throw 'Training gagal.' }

# 7. Muat dan verifikasi bundle yang baru dibuat.
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Check -Python $researchPy
if ($LASTEXITCODE -ne 0) { throw 'Verifikasi bundle gagal.' }

# 8. Buat grafik dan laporan aspek di folder run.
& $researchPy scripts/visualize_verified_results.py --output "$run/visualizations"
if ($LASTEXITCODE -ne 0) { throw 'Visualisasi gagal.' }
& $researchPy scripts/run_aspect_analysis.py --output-dir "$run/aspects"
if ($LASTEXITCODE -ne 0) { throw 'Analisis aspek gagal.' }
```

Bundle hasil training berada di **$run/model**, evaluasi di **$run/output/experiments**, dan grafik di **$run/visualizations**. Untuk memeriksa bundle yang baru dilatih di web, lanjutkan ke langkah Backend pada terminal yang masih memiliki ketiga variabel **WAYGROUND_***. Gunakan database dan folder upload terpisah untuk uji tersebut agar riwayat kandidat terpisah dari riwayat model aktif:

```powershell
$env:WAYGROUND_DATABASE_PATH = Join-Path (Get-Location) "$run/candidate.db"
$env:WAYGROUND_UPLOAD_DIR = Join-Path (Get-Location) "$run/uploads"
```

Untuk memakai kembali v4 aktif, buka terminal baru atau hapus variabel di atas sebelum menjalankan backend.

```powershell
'WAYGROUND_PIPELINE_OUTPUT_DIR','WAYGROUND_OUTPUT_DIR','WAYGROUND_FINAL_MODEL_DIR','WAYGROUND_DATABASE_PATH','WAYGROUND_UPLOAD_DIR' |
    ForEach-Object { Remove-Item "Env:$_" -ErrorAction SilentlyContinue }
```

## 5. Tes, build, backend dan frontend

Langkah 9 dan 10 dilakukan dari root dengan environment web. Tambahkan **-Full** pada tugas Test untuk menyertakan tes ekuivalensi preprocessing seluruh snapshot. Tes API memakai database sementara.

```powershell
$webPy = (Resolve-Path .venv/Scripts/python.exe).Path
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Test -Python $webPy
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Build
```

Untuk pemeriksaan konsistensi v4 tambahan, gunakan environment web untuk memeriksa bundle dan integrasi, atau interpreter riset dengan **--refit** untuk membandingkan hasil training ulang. Jalankan dari terminal dengan konfigurasi v4 aktif.

```powershell
$env:WAYGROUND_DATABASE_URI = 'sqlite:///:memory:'
$env:WAYGROUND_UPLOAD_DIR = 'runs/validasi/uploads'
& $webPy scripts/validate_pipeline_consistency.py
& $researchPy scripts/validate_pipeline_consistency.py --refit
Remove-Item Env:WAYGROUND_DATABASE_URI, Env:WAYGROUND_UPLOAD_DIR
```

**Langkah 11, terminal backend.** Untuk v4 aktif, gunakan interpreter web di bawah. Untuk kandidat, ganti baris pertama dengan **$backendPy = $researchPy** pada terminal yang telah disiapkan di Bagian 4. Biarkan server berjalan. Jika kedua model akan diperiksa, hentikan server pertama sebelum menjalankan server berikutnya.

```powershell
$backendPy = (Resolve-Path .venv/Scripts/python.exe).Path
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Check -Python $backendPy
if ($LASTEXITCODE -ne 0) { throw 'Verifikasi bundle gagal.' }
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Backend -Python $backendPy
```

**Langkah 12, terminal frontend.** Jalankan salah satu perintah. Keduanya meneruskan permintaan */api* ke backend lokal port 5001.

```powershell
# Mode pengembangan di http://127.0.0.1:5173
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Frontend

# Alternatif untuk memeriksa build di http://127.0.0.1:4173
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Preview
```

Periksa **Prediksi Satu Ulasan**, lalu **Analisis Data Uji**. Contoh CSV uji:

```csv
content,label
Aplikasi error tidak bisa login,negatif
Sangat membantu belajar dan mengerjakan soal,positif
Saya menggunakan aplikasi ini,netral
```

Di ringkasan sesi, pilih **Jalankan analisis lengkap**, lalu buka Prediksi, Aspek, dan Evaluasi. Kolom **label** opsional. Halaman **Penelitian & Evaluasi** menyajikan evaluasi model tersimpan. Hentikan server dengan Ctrl+C setelah selesai.

### Uji browser otomatis

Jalankan build terlebih dahulu. Gunakan dua terminal terpisah. Fixture memakai SQLite memory dan server lokal port 5075, sedangkan Chrome memakai port debug 9337.

```powershell
# Terminal fixture
.\.venv\Scripts\python.exe -u scripts/serve_stage5_verification.py --artifacts runs/browser
```

```powershell
# Terminal browser, setelah fixture siap
$env:STAGE5_ARTIFACTS = Join-Path (Get-Location) 'runs/browser'
npm.cmd --prefix frontend run test:e2e
```

Jika lokasi Chrome berbeda, isi **$env:STAGE5_BROWSER** dengan path executable. Bukti berupa screenshot dan *browser-report.json* berada di **runs/browser**.

