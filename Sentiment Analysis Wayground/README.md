# Analisis Sentimen Wayground

Proyek ini mengklasifikasikan ulasan Wayground menjadi **negatif, netral, atau positif** dan merangkum sentimen pada empat aspek. Web menggunakan model **v4** secara bawaan dan menyediakan kandidat **TF-IDF kata dan karakter** sebagai pilihan eksperimen.

## Mulai dari tujuan Anda

| Tujuan | Buka |
|---|---|
| Menjalankan atau mendemokan web | [Dua perintah di bawah](#menjalankan-web) |
| Memahami folder dan alur proyek | [Peta proyek](docs/panduan_organisasi.md) |
| Mengulang penelitian dari data sampai model | [Panduan perintah terminal](PANDUAN_PERINTAH_TERMINAL.md) |
| Memilih skrip yang perlu dijalankan | [Indeks skrip penelitian](scripts/README.md) |
| Membaca atau melengkapi skripsi | [Laporan dan lampiran Word](docs/skripsi/README.md) |
| Melanjutkan anotasi dan pengujian kandidat | [Panduan perbaikan sentimen](docs/PERBAIKAN_SENTIMEN.md) |

## Menjalankan web

Jalankan dari root proyek di dua terminal PowerShell. Untuk instalasi awal, ikuti [persiapan environment](PANDUAN_PERINTAH_TERMINAL.md#2-persiapan-lingkungan).

Terminal pertama menjalankan backend:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Backend -Python .venv/Scripts/python.exe
```

Terminal kedua menjalankan frontend:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Frontend
```

Buka [Wayground lokal](http://127.0.0.1:5173). **Dashboard** merangkum dataset penelitian atau unggahan terakhir. **Analisis Data Uji** memproses file, sedangkan **Prediksi Satu Ulasan** memproses satu teks. Model yang tersimpan dipakai langsung untuk inferensi.

## Hasil penelitian utama

Model v4 versi **20261029_223454** memakai TF-IDF kata 1–3 gram dan SVM linear C=10 dengan strategi OVO. Macro F1 pada development adalah **79,40%**. Pada holdout diagnostik, akurasi adalah **84,27%** dan macro F1 **79,02%**. Holdout memiliki paparan historis, dan sebagian besar labelnya berasal dari leksikon.

- [Evaluasi, confusion matrix, dan hasil per kelas](output_v4/experiments/final_model_evaluation.json)
- [Grafik evaluasi v4](output_v4/verified_visualizations/)
- [Distribusi sentimen per aspek](output_v4/experiments/aspect_analysis_summary.csv)
- [Identitas model v4 dan kandidat web](models/README.md)

Pemeriksaan bundle v4 tersedia melalui perintah berikut.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/project.ps1 -Task Check -Python .venv/Scripts/python.exe
```
