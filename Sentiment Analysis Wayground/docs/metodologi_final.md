# METODOLOGI PROJECT

## Analisis Sentimen Aplikasi Wayground

**Status:** Dokumen acuan konseptual. Selaraskan semua angka,
konfigurasi, dan nama artefak dengan kode serta hasil eksekusi final.

## 1. Tujuan

Dokumen ini menyatukan konsep pengumpulan data, preprocessing,
pelabelan, validasi pakar, eksperimen TF-IDF dan SVM, evaluasi, analisis
aspek, serta implementasi website. Tujuannya adalah menjaga konsistensi
antara dataset, kode, model final, backend, frontend, dan laporan TA.

Prinsip utama: hasil dan angka dalam laporan harus dapat ditelusuri ke
dataset, konfigurasi, log eksperimen, dan artefak evaluasi yang benar.
Placeholder dalam dokumen ini harus diisi berdasarkan hasil aktual,
bukan perkiraan.

## 2. Gambaran Umum Sistem

Project menganalisis sentimen ulasan aplikasi Wayground ke dalam kelas
positif, netral, dan negatif (jika ini sesuai dengan skema label final).
Alur konseptual penelitian:

**Pengumpulan Data → Validasi Data → Preprocessing → Analisis Aspek →
Labeling → Validasi Pakar → EDA → Pembagian Data → TF-IDF → SVM → Tuning
→ Evaluasi → Analisis Hasil → Dashboard**

Urutan teknis perlu mengikuti dependensi metode. Pembagian data harus
dilakukan sebelum tuning/evaluasi model; TF-IDF harus di-fit hanya pada
data training di setiap fold. Analisis aspek dapat dilakukan sebelum
pelabelan jika membantu proses anotasi, atau setelah prediksi jika
digunakan untuk analisis hasil. Laporan harus menjelaskan posisi yang
benar-benar digunakan.

### Fungsi utama website

1.  Menguji model terlatih menggunakan file data uji.
2.  Menampilkan prediksi untuk setiap ulasan.
3.  Menghitung dan menampilkan metrik evaluasi jika label aktual
    tersedia.
4.  Menganalisis distribusi sentimen pada empat aspek: Kemudahan Akses,
    Gamifikasi, Materi/Konten, dan Performa Teknis.
5.  Melakukan prediksi satu ulasan.
6.  Menyimpan dan menampilkan riwayat pengujian.

Website adalah media penerapan dan pengujian **model yang sudah
dilatih**, bukan tempat melatih ulang model pada setiap permintaan.

## 3. Pengumpulan dan Validasi Data

Catat sumber data, periode pengumpulan, metode pengambilan, kriteria
inklusi/eksklusi, dan format data mentah. Simpan data mentah tanpa
ditimpa agar setiap tahap dapat diaudit.

Validasi mencakup: - kolom ulasan tersedia; - nilai kosong dan teks yang
hanya berisi spasi; - duplikasi; - baris rusak atau tidak relevan; -
konsistensi format dan encoding; - aturan penanganan data yang tidak
memenuhi kriteria.

Dokumentasikan apakah duplikasi dihapus berdasarkan teks asli atau teks
yang telah dinormalisasi. Catat jumlah baris sebelum dan sesudah setiap
tahap.

  Tahap                              Jumlah baris Catatan
  ------------------------ ---------------------- -----------------------------
  Data mentah                `[isi hasil aktual]` Data yang dikumpulkan
  Setelah validasi           `[isi hasil aktual]` Sesuai aturan validasi
  Setelah deduplikasi        `[isi hasil aktual]` Jika diterapkan
  Dataset final berlabel     `[isi hasil aktual]` Sebelum split
  Training                   `[isi hasil aktual]` Data untuk pelatihan/tuning
  Holdout testing            `[isi hasil aktual]` Data evaluasi final

## 4. Preprocessing Teks

Preprocessing harus konsisten antara training dan inferensi. Urutan yang
mungkin digunakan, sesuai implementasi aktual:

1.  Pembersihan teks, seperti URL, mention, karakter yang tidak relevan,
    dan spasi berulang.
2.  Case folding.
3.  Tokenisasi.
4.  Normalisasi kata tidak baku menggunakan kamus yang terdokumentasi.
5.  Penanganan negasi.
6.  Stopword removal dengan mempertahankan kata negasi penting.
7.  Stemming jika digunakan dan sesuai dengan metode penelitian.

Tidak semua tahap wajib digunakan. Jangan mencantumkan operasi dalam
laporan jika kode tidak menjalankannya. Pertahankan kata seperti
"tidak", "bukan", "belum", dan "kurang" apabila diperlukan untuk menjaga
makna sentimen. Catat versi kamus normalisasi dan daftar stopword.

Gunakan satu modul preprocessing bersama untuk training, prediksi satu
ulasan, dan prediksi file. Uji teks kosong, negasi, slang, URL, emoji,
dan teks yang hanya berisi simbol.

## 5. Analisis Aspek

Empat aspek yang dianalisis: 1. **Kemudahan Akses:** login, akses akun,
navigasi, dan kemudahan penggunaan. 2. **Gamifikasi:** kuis interaktif,
poin, permainan, leaderboard, dan unsur kompetitif. 3.
**Materi/Konten:** kualitas materi, pertanyaan, konten pembelajaran, dan
relevansinya. 4. **Performa Teknis:** kecepatan, error, crash, kegagalan
memuat, dan stabilitas aplikasi.

Metode identifikasi aspek harus dinyatakan berdasarkan kode yang
sebenarnya, misalnya kamus kata kunci, aturan berbasis leksikon, anotasi
manual, atau model klasifikasi. Jangan mengklaim metode yang belum
diterapkan.

Tentukan apakah satu ulasan dapat memiliki lebih dari satu aspek. Jika
multi-aspek diperbolehkan, jumlah kemunculan aspek dapat lebih besar
daripada jumlah ulasan unik. Tetapkan cara menangani ulasan yang tidak
cocok dengan aspek mana pun. Sentimen dan aspek adalah dua informasi
berbeda: ulasan tentang Performa Teknis bisa positif, netral, atau
negatif.

## 6. Pelabelan Sentimen dan Validasi Pakar

Dokumentasikan metode pelabelan yang benar-benar digunakan. Jika
menggunakan leksikon atau aturan, jelaskan sumber leksikon, skor/aturan
keputusan, batas kelas, penanganan skor seri, dan kasus ambigu. Jika
anotasi manual digunakan, jelaskan pedoman anotasi.

Validasi pakar perlu mencatat: - peran atau kualifikasi pakar yang
relevan; - jumlah data yang diperiksa dan metode pemilihan sampel; -
definisi kelas dan petunjuk anotasi; - jumlah label yang sesuai/tidak
sesuai; - cara menyelesaikan perbedaan dan memperbaiki label.

Jika pakar memeriksa sebagian sampel saja, jangan menyatakan seluruh
dataset telah divalidasi pakar. Jelaskan cakupan validasi secara
eksplisit. Label aktual adalah acuan evaluasi; prediksi SVM tidak boleh
menggantikan label aktual. Data holdout tidak boleh digunakan untuk
menyesuaikan aturan label atau model setelah hasilnya dilihat.

## 7. EDA

EDA dapat mencakup distribusi kelas sentimen, proporsi kelas, panjang
ulasan, kata yang sering muncul, distribusi aspek, dan contoh teks untuk
setiap kelas. Visualisasi harus mencantumkan dataset/tahap yang
digunakan. Jangan menggunakan informasi dari holdout untuk mengambil
keputusan tuning yang dapat membuat evaluasi menjadi terlalu optimistis.

## 8. Pembagian Data dan Pencegahan Data Leakage

Tetapkan proporsi split, `random_state`, dan penggunaan stratifikasi.
Pisahkan: - **Data training/development:** untuk cross-validation dan
pemilihan parameter. - **Data holdout/testing:** hanya untuk evaluasi
akhir setelah konfigurasi dipilih.

Periksa risiko duplikasi atau ulasan sangat mirip tersebar pada training
dan testing. TF-IDF wajib di-fit hanya pada training fold. Gunakan
`Pipeline` scikit-learn yang mencakup TF-IDF dan SVM agar setiap fold
melakukan fitting hanya pada bagian training. Jangan melakukan
`fit_transform()` pada seluruh dataset sebelum split.

## 9. TF-IDF

TF-IDF mengubah teks menjadi fitur numerik. Parameter yang dapat diuji
meliputi `ngram_range`, `max_features`, `min_df`, `max_df`, dan
`sublinear_tf` jika digunakan. Nilai final harus diambil dari
konfigurasi yang benar-benar menghasilkan model final, bukan
menggabungkan nilai dari eksperimen berbeda.

Saat inferensi, gunakan transformer yang telah di-fit untuk melakukan
`transform()` terhadap teks baru. Jangan melakukan `fit()` atau
`fit_transform()` pada data uji. Simpan transformer bersama model atau
sebagai artefak terpisah dengan versi yang jelas.

## 10. Model SVM

SVM digunakan untuk mengklasifikasikan sentimen berdasarkan fitur
TF-IDF. Catat kelas estimator yang benar-benar digunakan (misalnya
`LinearSVC` atau `SVC`), parameter `C`, `class_weight` jika ada,
strategi multiclass bila diatur secara eksplisit, versi library, dan
konfigurasi lain yang relevan.

Jangan mencampur konfigurasi `SVC`, `LinearSVC`, `OneVsOneClassifier`,
atau strategi multiclass lain tanpa menjelaskan estimator final yang
digunakan. Nama estimator dan parameter harus sama pada kode training,
metadata, backend, dan laporan.

## 11. Eksperimen dan Tuning

Eksperimen dilakukan terpisah secara logis dari pipeline inferensi
produksi. Proses yang disarankan: 1. Tetapkan data development dan
holdout. 2. Buat baseline yang dapat diulang. 3. Bandingkan konfigurasi
TF-IDF. 4. Bandingkan parameter SVM. 5. Uji penanganan ketidakseimbangan
kelas jika relevan. 6. Gunakan cross-validation pada development set. 7.
Tentukan metrik utama sebelum membandingkan eksperimen. 8. Simpan
konfigurasi, skor per fold, skor rata-rata, dan log. 9. Pilih satu
konfigurasi final berdasarkan hasil eksperimen. 10. Evaluasi final pada
holdout setelah pemilihan konfigurasi selesai.

`GridSearchCV` dapat menguji kombinasi parameter secara sistematis. Jika
memakai `Pipeline`, parameter TF-IDF dan SVM dapat dituning tanpa
kebocoran informasi dari fold validasi.

Macro F1 dapat dipilih sebagai metrik utama saat kelas tidak seimbang
karena memberi bobot sama pada setiap kelas. Accuracy, precision,
recall, weighted F1, dan metrik per kelas dapat menjadi pelengkap.
Pilihan metrik harus konsisten di semua eksperimen.

## 12. Evaluasi Model

Evaluasi final membandingkan prediksi model dengan label aktual pada
holdout yang tidak digunakan untuk memilih model.

-   **Accuracy:** proporsi prediksi benar.
-   **Precision:** ketepatan prediksi suatu kelas.
-   **Recall:** proporsi data aktual suatu kelas yang berhasil dikenali.
-   **F1-score:** rata-rata harmonik precision dan recall.
-   **Confusion Matrix:** jumlah prediksi benar dan salah antara kelas
    aktual dan prediksi.

Sebutkan jenis averaging (`macro`, `weighted`, atau `micro`) untuk
precision, recall, dan F1-score. Sajikan metrik per kelas bila
memungkinkan.

Audit hasil evaluasi: support setiap kelas harus berjumlah sama dengan
total sampel evaluasi; jumlah confusion matrix harus sesuai;
classification report dan confusion matrix harus berasal dari pasangan
label/prediksi yang sama; hasil cross-validation tidak boleh dicampur
dengan holdout; artefak lama tidak boleh disalahartikan sebagai hasil
terbaru. Jangan menuliskan angka sebagai hasil final sebelum sumbernya
diverifikasi.

## 13. Artefak Model dan Struktur Project

Artefak final minimal mencakup model SVM, transformer TF-IDF yang cocok,
versi preprocessing/kamus, metadata, konfigurasi, informasi versi
dataset/library, dan laporan evaluasi.

Contoh struktur konseptual (sesuaikan dengan struktur aktual):

``` text
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── experiments/
│   ├── configs/
│   ├── logs/
│   └── results/
├── models/
│   └── final/
│       ├── model.joblib
│       ├── tfidf.joblib
│       └── metadata.json
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── aspect_analysis/
├── backend/
├── frontend/
├── tests/
├── reports/
└── metodologi.md
```

Struktur tersebut adalah rekomendasi, bukan klaim bahwa folder project
saat ini sudah sama. Tetapkan satu konfigurasi final sebagai sumber
kebenaran. Pastikan path penyimpanan model sama dengan path pemuatan
backend dan metadata tidak saling bertentangan. Simpan identitas
eksperimen, versi dataset, parameter TF-IDF/SVM, skor cross-validation,
skor holdout, dan versi library.

## 14. Backend

Backend bertanggung jawab untuk memuat model final, memvalidasi input,
menjalankan preprocessing dan prediksi, menghitung evaluasi, menjalankan
analisis aspek, dan menyimpan riwayat.

Endpoint yang dapat disediakan sesuai kebutuhan: - health check/model
readiness; - prediksi satu ulasan; - unggah dan uji file; - analisis
aspek; - daftar/detail riwayat; - unduh hasil prediksi.

Nama endpoint dan format respons mengikuti API yang benar-benar
diterapkan. Validasi file harus memeriksa ekstensi, ukuran, parsing,
kolom wajib, file kosong, nilai ulasan kosong, label tak dikenal, dan
jumlah baris yang diproses/dilewati. Tampilkan pesan error yang aman dan
informatif.

## 15. Pengujian Model di Website

Website menguji model terlatih, bukan melatih model baru. Alur file data
uji:

1.  Pengguna mengunggah file CSV/TSV yang didukung.
2.  Sistem memvalidasi file dan struktur kolom.
3.  Sistem membaca kolom ulasan serta label aktual jika tersedia.
4.  Ulasan menjalani preprocessing yang sama dengan training.
5.  Transformer TF-IDF final melakukan `transform()`.
6.  Model SVM final melakukan `predict()`.
7.  Jika label aktual tersedia, prediksi dibandingkan dengan label
    aktual.
8.  Sistem menghitung metrik evaluasi.
9.  Sistem menganalisis aspek dengan metode yang ditetapkan.
10. Website menampilkan hasil prediksi per baris, metrik, confusion
    matrix, dan distribusi aspek.
11. Sistem menyimpan riwayat sesuai kebijakan penyimpanan.

Jangan menjalankan `fit()`, `fit_transform()`, atau training ulang SVM
pada file uji. Jika label aktual tidak tersedia, sistem masih dapat
menampilkan prediksi dan distribusi sentimen, tetapi tidak dapat
menghitung metrik evaluasi yang memerlukan label pembanding.

Tabel hasil prediksi dapat memuat ID baris, teks ulasan, label aktual
(jika ada), label prediksi, status cocok/tidak cocok (jika label aktual
ada), dan aspek teridentifikasi. Tampilkan jumlah baris diunggah, valid,
diproses, dan dilewati.

Untuk prediksi satu ulasan, sistem memvalidasi teks, menjalankan
preprocessing, TF-IDF `transform()`, dan SVM `predict()`. Prediksi satu
ulasan tidak dapat dianggap sebagai evaluasi menyeluruh model.

## 16. Analisis Aspek di Website

Website menampilkan jumlah ulasan dan distribusi positif, netral, dan
negatif untuk: - Kemudahan Akses; - Gamifikasi; - Materi/Konten; -
Performa Teknis.

Visualisasi dapat berupa grafik batang bertumpuk atau tabel ringkasan.
Jelaskan denominator perhitungan. Pada analisis multi-aspek, satu ulasan
dapat masuk ke beberapa aspek, sehingga total kemunculan aspek dapat
melebihi jumlah ulasan unik. Jika aspek berbasis kata kunci, jelaskan
keterbatasannya dan lakukan pemeriksaan manual terhadap sampel.

## 17. Riwayat Pengujian

Riwayat dapat menyimpan ID dan waktu pengujian, nama/identitas file,
jumlah data, jumlah baris valid/gagal, versi model, metrik jika label
aktual tersedia, ringkasan distribusi sentimen/aspek, serta referensi
hasil prediksi.

Tentukan kebijakan retensi, akses, dan penghapusan. Hindari menyimpan
file mentah atau teks ulasan lebih lama daripada yang diperlukan. Untuk
aplikasi multi-pengguna, batasi riwayat sesuai hak akses.

## 18. Frontend

Halaman yang disarankan: 1. **Dashboard:** ringkasan dan navigasi fitur.
2. **Pengujian Data:** unggah file, validasi, pratinjau, dan mulai
pengujian. 3. **Hasil Evaluasi:** metrik, confusion matrix, dan
classification report. 4. **Hasil Prediksi:** tabel setiap ulasan, label
aktual, prediksi, dan status kecocokan. 5. **Analisis Aspek:**
distribusi sentimen pada empat aspek. 6. **Prediksi Satu Ulasan:** input
teks dan hasil prediksi. 7. **Riwayat Pengujian:** daftar dan detail
pengujian sebelumnya.

Tampilkan status loading, sukses, validasi, dan error. Hasil di frontend
harus berasal dari respons backend, bukan angka statis.

## 19. Pengujian Sistem

Lakukan unit test, integration test, dan pengujian fungsional.

### Preprocessing/model

-   teks normal, kosong, negasi, slang, URL, dan simbol;
-   model dan TF-IDF dapat dimuat;
-   jumlah fitur sesuai dengan transformer final;
-   label prediksi termasuk kelas yang didukung.

### File uji

-   CSV/TSV valid;
-   format tidak didukung;
-   file kosong/rusak;
-   kolom ulasan hilang;
-   label hilang atau tidak dikenal;
-   nilai ulasan kosong;
-   satu kelas saja;
-   duplikasi;
-   file melebihi batas.

### Evaluasi/integrasi

-   metrik berasal dari label aktual dan prediksi yang sama;
-   total confusion matrix sesuai dengan sampel evaluasi;
-   support kelas benar;
-   tidak ada metrik evaluasi palsu ketika label aktual tidak tersedia;
-   frontend konsisten dengan respons backend;
-   path model dan preprocessing benar;
-   riwayat tersimpan dan dapat dibuka kembali.

Jalankan pengujian pada lingkungan dengan seluruh dependensi proyek.
Catat test yang lulus, gagal, atau belum dijalankan.

## 20. Reproducibility dan Konsistensi Laporan

Catat versi dataset, waktu training, versi kode, konfigurasi split,
`random_state`, konfigurasi TF-IDF/SVM, strategi cross-validation, skor
eksperimen, hasil holdout, versi Python/library, dan path artefak final.

Dalam laporan: - **BAB III:** metodologi, sumber data, preprocessing,
labeling, validasi pakar, split, TF-IDF, SVM, tuning, evaluasi, analisis
aspek, dan rancangan sistem. - **BAB IV:** jumlah data aktual, EDA,
hasil eksperimen, konfigurasi terpilih, evaluasi holdout, analisis
aspek, dan hasil pengujian website. - **BAB V:** kesimpulan berdasarkan
hasil aktual, keterbatasan, dan saran pengembangan.

Checklist konsistensi: 1. jumlah data sama pada dataset, log, tabel, dan
pembahasan; 2. metode pelabelan sesuai kode dan validasi pakar; 3.
preprocessing laporan sama dengan preprocessing aktual; 4. parameter
TF-IDF/SVM sama dengan artefak final; 5. hasil tuning dapat ditelusuri
ke log; 6. metrik berasal dari dataset/model yang tepat; 7. confusion
matrix konsisten dengan classification report; 8. dashboard sesuai
respons backend; 9. istilah training, validasi, dan testing konsisten;
10. cakupan validasi pakar dan klaim kinerja tidak melebihi bukti.

## 21. Keputusan Final yang Harus Diisi

  Keputusan                      Nilai terverifikasi
  ------------------------------ --------------------------------------
  Sumber/periode data            `[isi sesuai penelitian]`
  Jumlah data final              `[verifikasi dataset final]`
  Skema label                    `[skema final]`
  Metode pelabelan               `[metode aktual]`
  Cakupan validasi pakar         `[jumlah dan sampling]`
  Urutan preprocessing           `[sesuai kode]`
  Metode identifikasi aspek      `[metode aktual]`
  Rasio split dan random state   `[konfigurasi final]`
  Parameter TF-IDF               `[konfigurasi final]`
  Estimator dan parameter SVM    `[konfigurasi final]`
  Strategi tuning dan scoring    `[konfigurasi final]`
  Metrik utama                   `[misalnya Macro F1 jika digunakan]`
  Hasil evaluasi holdout         `[hasil artefak final]`
  Path model dan TF-IDF          `[path aktual]`
  Format dan kolom file uji      `[aturan aktual]`
  Penyimpanan riwayat            `[mekanisme dan retensi]`

## 22. Kriteria Selesai

-   [ ] Log jumlah data pada setiap tahap tersedia.
-   [ ] Metode pelabelan dan validasi pakar terdokumentasi.
-   [ ] Split dan cross-validation mencegah data leakage.
-   [ ] Eksperimen dan konfigurasi dapat direproduksi.
-   [ ] Satu konfigurasi final telah dipilih.
-   [ ] Model dan TF-IDF final dapat dimuat tanpa mismatch.
-   [ ] Evaluasi final dapat dihitung ulang.
-   [ ] Prediksi satu ulasan berfungsi.
-   [ ] Unggah file menghasilkan prediksi per baris.
-   [ ] Evaluasi hanya dihitung jika label aktual tersedia.
-   [ ] Analisis aspek menggunakan metode terdokumentasi.
-   [ ] Riwayat tersimpan dan dapat ditinjau.
-   [ ] Frontend konsisten dengan backend.
-   [ ] Uji validasi file dan kasus error selesai.
-   [ ] Angka laporan cocok dengan artefak final.
-   [ ] Dependensi dan cara menjalankan aplikasi terdokumentasi.

## 23. Prinsip Akhir

`metodologi.md` adalah acuan proses dan aturan project, bukan pengganti
hasil eksekusi. Setelah audit kode selesai, isi seluruh placeholder
dengan nilai terverifikasi dan hapus metode atau opsi yang tidak
digunakan. Setiap perubahan pada data, preprocessing, label, split,
TF-IDF, SVM, atau analisis aspek harus diikuti pengujian yang relevan
dan pembaruan laporan.
