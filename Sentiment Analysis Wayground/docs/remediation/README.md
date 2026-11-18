# Perbaikan proyek secara bertahap

Perbaikan dimulai pada 29 Oktober 2026 berdasarkan [audit menyeluruh](../riwayat/audit/audit_menyeluruh_2026-10-29.md).

Setiap tahap mempunyai batas pekerjaan, bukti verifikasi, perubahan struktur, dan masalah terbuka. Status selesai suatu tahap tidak berarti seluruh proyek telah sesuai. Perubahan workspace yang sudah ada dipertahankan; tidak dilakukan reset Git.

| Tahap | Lingkup | Status |
|---|---|---|
| 1 | Kontrak artefak, validasi dataset/split, dokumentasi perbaikan, arsip log | Selesai; lihat TAHAP_01.md |
| 2 | Dataset: periode aktual, kontrak tanggal, preprocessing, threshold dan lexicon, provenance anotasi | Selesai persiapan; lihat TAHAP_02.md; snapshot v3 belum dipromosikan |
| 3 | Tujuh eksperimen dengan versi data dan grouped folds yang sama; keputusan final berdasarkan Macro F1 | Selesai; lihat TAHAP_03.md; percobaan salah split dibatalkan dan paparannya dicatat |
| 4 | Locking, training final, evaluasi dengan provenance paparan holdout, identitas model dan API; pisahkan model lama dari model pengganti | Selesai; lihat TAHAP_04.md; bundle final_v4 terverifikasi |
| 5 | Frontend: status alur, halaman detail, evaluasi, prediksi, aspek dan pengujian integrasi | Selesai; lihat TAHAP_05.md |
| 6 | Visualisasi final, dokumentasi penggunaan, arsip artefak lama setelah dependensi berpindah | Berikutnya |

## Struktur yang dipertahankan

| Direktori | Fungsi |
|---|---|
| shared/ | Kontrak, preprocessing, label, split, utilitas bersama |
| pipeline/ | Pengolahan dataset dan orchestration pipeline |
| scripts/experiments/ | Percobaan penelitian dan hasil seleksi pada development |
| scripts/ | Training final, evaluasi dan visualisasi offline |
| backend/ | API inference dan pengelolaan analisis |
| frontend/ | Antarmuka pengguna |
| tests/ | Pemeriksaan regresi dan integrasi |
| docs/remediation/ | Rencana bertahap, catatan perubahan, verifikasi dan manifest arsip |
| archive/remediation_2026-10-29/ | Berkas historis yang dipindahkan selama perbaikan |

Folder output dan dataset versi lama belum dipindahkan karena masih menjadi dependensi model aktif dan bukti eksperimen. Dokumen lama tetap menjadi catatan historis sampai panduan pengganti telah diverifikasi. Struktur akan dirapikan melalui pemindahan dependensi yang terdokumentasi, bukan penghapusan massal.

## Aturan arsip dan pemulihan

Manifest `ARCHIVE_MANIFEST_2026-10-29.json` mencatat path asal, path arsip, ukuran, SHA-256, dan alasan. Pemulihan dilakukan dengan menyalin file arsip ke path asal setelah memastikan tidak menimpa file baru. Hash sumber dan arsip harus sama dengan manifest.

`archive/` diabaikan oleh .gitignore proyek: berkas arsip tersedia secara lokal, tetapi tidak otomatis ikut commit/backup Git. Manifest dan dokumentasi berada dalam docs/ agar dapat dilacak. Untuk backup lengkap, sertakan folder archive/ secara terpisah. Tidak ada berkas yang dihapus permanen pada tahap 1.

## Panduan terminal dan penataan 30 Oktober 2026

Lihat [panduan perintah](../../PANDUAN_PERINTAH_TERMINAL.md) dan [catatan penataan](PENATAAN_DAN_PERINTAH_2026-10-30.md). Empat berkas lama diarsipkan dengan manifest. Dua plot diagnostik final terverifikasi tersedia; sisa penyelarasan visualisasi tahap 6 tetap terbuka.

## Implementasi alur web 30–31 Oktober 2026

[Perubahan struktur web, migrasi, dan verifikasi](WEB_FLOW_2026-10-30.md) menerapkan lima menu, navigasi per sesi, sumber upload lengkap, inspeksi TF-IDF, Labeling SVM, kartu/grafik/detail aspek, riwayat tunggal/snapshot dan evaluasi penelitian terpisah. Verifikasi akhir: 119 tes backend, 9 tes frontend, 14 skenario browser, build berhasil. Model final tetap sama.
