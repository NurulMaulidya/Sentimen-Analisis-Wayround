# Tahap 5 — Frontend dan siklus analisis

Tanggal: 29 Oktober 2026. Status: selesai untuk lingkup tahap ini. Model tetap versi `20261029_223454`; tidak dilakukan training ulang atau perubahan dataset penelitian.

## Hasil perbaikan

Halaman detail sekarang menjalankan alur deduplikasi → tujuh tahap preprocessing → prediksi SVM → aspek → evaluasi bila file memiliki label aktual. Dataset tanpa label selesai setelah aspek. Status antara evaluasi dan aspek tetap mempertahankan hasil prediksi yang tersedia; mengulang aspek tidak menurunkan status analisis yang sudah selesai.

Menjalankan ulang tahap hulu membatalkan hasil hilir yang sudah kedaluwarsa. Preprocessing membatalkan prediksi, aspek, dan evaluasi; prediksi ulang membatalkan aspek dan evaluasi. Prediksi mensyaratkan fingerprint preprocessing yang sesuai runtime aktif. Riwayat lama tanpa fingerprint harus menjalankan ulang preprocessing sebelum prediksi baru.

Identitas hasil disimpan pada tabel baru `analysis_executions`: versi model, hash pipeline, signature preprocessing, waktu prediksi, dan kesiapan aspek/evaluasi. Riwayat prediksi lama tanpa catatan ini ditampilkan sebagai **tidak diketahui**, bukan diberi versi model aktif secara otomatis. Penambahan tabel tidak mengubah tabel atau hasil lama. Pada startup pertama terhadap SQLite lama, aplikasi membuat backup konsisten melalui SQLite backup API ke direktori `backups/` di sebelah database, beserta metadata dan hash, sebelum membuat tabel baru. Pengujian migrasi memakai database sementara; database produksi belum dimigrasikan dalam pekerjaan ini.

## Kesesuaian tampilan

| Bagian | Perilaku yang diverifikasi |
|---|---|
| Detail dan riwayat | Versi hasil tersimpan, status tiap proses, jumlah data, analisis ulang, dan kegagalan API terlihat |
| Data awal/deduplikasi | Data API sebenarnya, pagination dan pilihan menyertakan duplikat |
| Preprocessing | Navigasi tujuh tahap dan hasil final; perbandingan dengan teks awal; pagination |
| Prediksi | Teks awal dan hasil preprocessing, prediksi, label aktual, kecocokan; filter/pencarian; pemberitahuan teks kosong |
| Evaluasi | Accuracy, macro precision/recall/F1, classification report dan confusion matrix berlabel aktual/prediksi; urutan negatif/netral/positif eksplisit |
| Tanpa label | Tab evaluasi disembunyikan; deep link menjelaskan bahwa evaluasi berlabel tidak tersedia |
| Aspek | Jumlah ulasan unik dibedakan dari jumlah penyebutan aspek; distribusi dan kalimat yang cocok ditampilkan |
| Navigasi | URL menentukan tab; respons dari permintaan lama diabaikan saat ID/tab berubah; pesan error menggantikan hasil lama |
| Ponsel | Sidebar awal mengecil dan mengikuti pergantian ukuran layar; tabel lebar dapat digeser secara horizontal |

Evaluasi pada halaman analisis membandingkan prediksi dengan label file yang diunggah. Angkanya **bukan** pengganti evaluasi penelitian tahap 4. Angka 100% pada screenshot fixture hanya berasal dari tiga contoh pengujian.

Sentimen aspek masih mengikuti sentimen ulasan utuh yang mengandung kata kunci aspek. Implementasi ini belum merupakan classifier sentimen terpisah per aspek. Satu ulasan bisa memuat beberapa aspek, sehingga total penyebutan dapat melebihi jumlah ulasan. Penjelasan ini ditampilkan pada halaman untuk menghindari interpretasi keliru. Visualisasi penelitian offline tetap menjadi lingkup tahap 6.

> **Koreksi 5 November 2026 (Tahap 6).** Paragraf di atas tidak lagi sesuai
> dengan kode. Backend sudah menghitung sentimen tiap aspek dari
> kalimat-kalimat aspek itu sendiri memakai artefak SVM canonical yang sama
> (`sentiment_scope: per_aspect_sentiment_from_aspect_sentences`). Yang
> berbasis kata kunci hanya deteksi aspek. Teks halaman dan dokumen ini
> sudah dikoreksi pada Tahap 6. Lihat `docs/remediation/TAHAP_06.md`.

Format upload frontend disamakan dengan backend: CSV/TSV/TXT. Contoh upload berlabel menggunakan kolom `content,label`, dengan label negatif/netral/positif. Hasil prediksi teks tunggal menampilkan versi dari respons model dan dibersihkan saat input berubah.

## Verifikasi

- Backend: **109 passed**, satu warning parsing tanggal pada fixture tanggal tidak valid. Perintah: `python -m pytest tests/test_artifact_contracts.py tests/test_dataset_preparation.py tests/test_research_experiments.py tests/test_deployment_bundle.py tests/test_analysis_lifecycle.py tests/test_split_leakage.py tests/test_preprocessing.py -q`. Log: `TAHAP_05_TESTS.txt`.
- Frontend: **6 passed** melalui `cd frontend` lalu `npm test`. Mencakup pemetaan URL/status, identitas legacy, nilai nol versus data kosong, matrix dan penyebut aspek.
- Build produksi: `npm run build` berhasil, 1475 modul; bundle akhir `index-Xpl2vV8m.js`.
- Browser Chrome headless: **9 skenario lulus**, tidak ada runtime exception. Menggunakan frontend hasil build, API nyata, model final_v4 nyata, SQLite memory, dan upload file nyata. Termasuk deep link evaluasi/aspek, perbandingan preprocessing, tanpa label, legacy, pergantian ID, pagination 25 baris, upload sampai evaluasi, tampilan ponsel dan respons analisis tidak ditemukan.
- Hash pipeline, metadata dan keputusan konfigurasi sama dengan bukti tahap 4. Detail: `TAHAP_05_VERIFICATION.json`.

Artefak browser lokal berada di `C:/Users/ACER/.codex/visualizations/2026/10/29/01a187fd-1294-7bb3-9df9-934e8163b25d/stage5-browser/`: `browser-report.json`, screenshot evaluasi desktop, perbandingan preprocessing dan aspek ponsel. `browser-error.json`, bila ada, mencatat percobaan awal; hasil akhir ada pada `browser-report.json`. Pengujian awal menemukan kendala akses sandbox, koreksi skrip uji dan fixture kolom label; pengujian akhir berhasil setelah perbaikan. Pemeriksaan screenshot menemukan sidebar menutupi konten ponsel; perilakunya diperbaiki dan browser diuji ulang.

Untuk mengulang: jalankan `python -u scripts/serve_stage5_verification.py --artifacts <direktori-absolut>` di terminal pertama. Di terminal kedua, pada `frontend/`, set environment `STAGE5_ARTIFACTS` ke direktori tersebut lalu `npm run test:e2e`. Memerlukan Chrome lokal pada path default Windows atau `STAGE5_BROWSER`. Server memakai port 5075; jangan gunakan database produksi untuk fixture ini.

## Struktur dan penerapan

Komponen tampilan bersama ada di `frontend/src/components/AnalysisViews.jsx`, utilitas UI di `frontend/src/utils/analysisUi.js`, pengelolaan invalidasi backend di `backend/services/analysis_state.py`, dan pengujian frontend di `frontend/tests/`. Context lama `PipelineContext.jsx` tidak memiliki import pemakai dan diarsipkan ke `archive/remediation_2026-10-29/stage5/frontend/`; manifest ukuran/hash/path/pemulihan ada di `TAHAP_05_ARCHIVE_MANIFEST.json`. Arsip lokal diabaikan Git, sehingga harus disertakan tersendiri dalam backup lengkap.

Restart backend untuk memuat API/schema baru; pada database SQLite lama backup otomatis dibuat sebelum penambahan tabel. Sajikan build frontend terbaru atau jalankan Vite dari sumber terbaru. Server verifikasi sementara tidak menggantikan server aplikasi utama.

Tahap berikutnya: menyelaraskan visualisasi penelitian dengan artefak final_v4 dan provenance evaluasi diagnostik, memperbarui panduan penggunaan, lalu mengarsipkan artefak lama setelah seluruh referensinya diperiksa.
