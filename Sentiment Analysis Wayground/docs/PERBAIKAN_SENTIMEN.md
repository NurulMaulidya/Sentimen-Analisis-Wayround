# Uji kandidat perbaikan sentimen Wayground

Gunakan PowerShell dari root proyek dengan interpreter riset yang sesuai `requirements-lock.txt`. Model web v4 menjadi pembanding tetap. Folder penelitian ini berisi model kandidat dan artefak eksperimen.

## 1. Siapkan lembar anotasi

Folder yang sudah disiapkan adalah `runs/perbaikan_sentimen_20261109`. Untuk memulai run lain, gunakan ID folder baru dan jalankan tahap `prepare`.

```powershell
$run = 'runs/perbaikan_sentimen_20261109'
# Hanya untuk run baru
# python scripts/improve_sentiment.py prepare --run $run
```

| Lembar | Yang diisi | Penggunaan |
|---|---|---|
| `anotasi_development.csv` | Kolom `label` dan, bila perlu, `catatan` | 150 ulasan development untuk memeriksa ulang label latih. Sampel diprioritaskan dari skor leksikon nol dan perbedaan dengan rating. |
| `anotasi_uji_baru.csv` | Kolom `label` dan kolom aspek jika aspek ikut diuji | 136 ulasan dari raw periode mulai 1 Juli 2026 yang lolos pemeriksaan duplikasi terhadap dataset v4 dan antarsesama ulasan baru. |

Nilai label adalah `positif`, `netral`, atau `negatif`. Gunakan `netral` untuk ulasan deskriptif tanpa polaritas jelas. Tandai kasus campuran atau ambigu di `catatan`, lalu sepakati cara menentukan kelas final sebelum mengisi label. Contohnya, jika pedoman memakai polaritas dominan, catat kedua opini dan labeli opini yang lebih ditekankan. Bila dua anotator tersedia, isi salinan masing-masing sebelum menyepakati satu label final. Jangan ubah kolom `reviewId` atau teks sumber. Simpan sebagai CSV UTF-8 dengan pemisah koma.

Untuk menguji aspek, isi empat kolom aspek pada ulasan yang dinilai. Gunakan tiga label sentimen untuk aspek yang disebut atau `absen` jika tidak disebut. Kosongkan seluruh kolom aspek pada ulasan yang belum dinilai. Label uji disusun dari teks asli tanpa melihat prediksi kandidat, kamus, atau rating.

## 2. Bandingkan kandidat di development

```powershell
python scripts/improve_sentiment.py benchmark --run $run --annotations "$run/anotasi_development.csv"
```

Eksperimen awal tanpa koreksi manusia tersedia di `benchmark_awal/hasil_cv.json`. Perintah dengan anotasi menulis hasil terpisah ke `benchmark_anotasi/`. Semua kandidat dinilai dengan label evaluasi development dan lima fold yang sama. Duplikat menurut kedua resep preprocessing dijaga dalam grup yang sama. TF-IDF dilatih di setiap fold.

Gunakan skor baseline dari laporan yang sama untuk membandingkan kandidat. Pembentukan grup pada eksperimen ini mencakup kedua resep preprocessing, sehingga skor baseline di sini dapat berbeda dari eksperimen resmi v4.

Urutan ablasinya adalah baseline, label manusia, normalisasi, revisi label leksikon, lalu TF-IDF kata dan karakter. Kandidat `karakter_saja` menambahkan fitur karakter pada baseline dengan teks dan label yang sama. Revisi leksikon hanya diterapkan pada label otomatis yang belum dikoreksi manusia. Perhatikan macro F1, F1 netral, dan confusion matrix. Lihat juga hasil pada subset beranotasi, yang sengaja memuat contoh sulit. Hasil development dengan label campuran bersifat diagnostik.

## 3. Evaluasi kandidat terpilih

Tentukan kandidat dari hasil development sebelum membuka hasil evaluasi akhir. Ganti nilai `$candidate` dengan salah satu dari `label_pakar`, `normalisasi`, `leksikon`, `karakter`, atau `karakter_saja`.

```powershell
$candidate = 'karakter_saja'
python scripts/improve_sentiment.py evaluate --run $run --annotations "$run/anotasi_uji_baru.csv" --candidate $candidate
```

Perintah memerlukan semua label sentimen uji terisi dan ketiga kelas terwakili. Hasilnya tersimpan di `evaluasi_uji_baru.json` dan membandingkan v4 serta kandidat pada ulasan yang sama. Bila aspek diisi, laporan juga membandingkan deteksi per kalimat dengan pemisahan klausa sambil memakai SVM v4 yang sama. Metriknya mencakup F1 deteksi, F1 gabungan aspek–sentimen, dan macro F1 sentimen pada aspek yang cocok. Simpan hasil ini sebagai evaluasi akhir dari run tersebut.

## Mencoba kandidat di web

Jalankan ulang backend dan frontend sesuai panduan terminal. Pada **Prediksi Satu Ulasan** atau ringkasan **Analisis Data Uji**, pilih **Kandidat · TF-IDF kata dan karakter** dari pilihan **Model analisis**. Pilihan juga tersedia pada langkah TF-IDF sebelum prediksi. Kandidat memakai preprocessing v4, SVM dengan fitur kata dan karakter, serta deteksi aspek per klausa. Versi dan hash model dicatat bersama hasil.

Model v4 dipilih secara bawaan. Halaman **Penelitian & Evaluasi** tetap membaca hasil penelitian v4. Evaluasi file unggahan mengikuti model yang dipilih untuk sesi tersebut. Artefak kandidat berasal dari `benchmark_awal/karakter_saja.joblib` dan tersedia di `models/candidate_char/` bersama metadata verifikasi.
