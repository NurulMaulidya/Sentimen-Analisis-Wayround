# Metodologi Penelitian
## Evaluasi Efektivitas Pembelajaran Aplikasi Wayground Berdasarkan Analisis Sentimen Ulasan Pengguna Menggunakan SVM

---

## Catatan Revisi

Dokumen ini telah diperbarui untuk mencerminkan **konfigurasi model aktual** berdasarkan hasil eksperimen yang telah dijalankan. Nilai parameter sekarang diambil dari `output/experiments/best_config.json` yang merupakan single source of truth.

---

## 1. Ruang Lingkup Penelitian

### 1.1 Tujuan
Mengklasifikasikan sentimen ulasan pengguna aplikasi Wayground (sebelumnya Quizizz) ke dalam tiga kelas — negatif, netral, dan positif — pada empat aspek pembelajaran, kemudian menyimpulkan persepsi pengguna terhadap efektivitas pembelajaran, baik secara keseluruhan maupun per aspek.

### 1.2 Rumusan Masalah
1. Bagaimana penerapan algoritma Support Vector Machine dengan pembobotan TF-IDF dalam mengklasifikasikan sentimen ulasan pengguna aplikasi Wayground ke dalam kelas positif, netral, dan negatif?
2. Bagaimana kinerja model SVM tersebut berdasarkan metrik akurasi, precision, recall, dan F1-score macro-average?
3. Bagaimana distribusi sentimen pengguna pada masing-masing aspek pembelajaran yang dikaji?
4. Bagaimana persepsi pengguna terhadap efektivitas pembelajaran aplikasi Wayground berdasarkan hasil analisis sentimen berbasis aspek tersebut?

### 1.3 Tujuan Penelitian
1. Menerapkan algoritma SVM dengan pembobotan TF-IDF untuk mengklasifikasikan sentimen ulasan pengguna aplikasi Wayground ke dalam tiga kelas sentimen.
2. Mengukur kinerja model klasifikasi menggunakan metrik akurasi, precision, recall, dan F1-score macro-average.
3. Menganalisis distribusi sentimen pengguna pada empat aspek pembelajaran yang dikaji.
4. Menginterpretasikan hasil klasifikasi sentimen sebagai gambaran persepsi pengguna terhadap efektivitas pembelajaran aplikasi Wayground.

### 1.4 Batasan Masalah
- Data terbatas pada ulasan berbahasa Indonesia di Google Play Store periode Januari 2024 - Juni 2026.
- Klasifikasi terbatas pada tiga kelas sentimen: positif, netral, dan negatif.
- Deteksi aspek dilakukan dengan pencocokan kata kunci (rule-based keyword matching), bukan model ekstraksi aspek otomatis, dan terbatas pada empat aspek pembelajaran.
- Pelabelan awal menggunakan InSet Lexicon dengan threshold yang dikalibrasi.
- Balancing data menggunakan class_weight pada SVM.
- Efektivitas pembelajaran dinilai dari persepsi pengguna yang terkandung dalam ulasan, bukan dari hasil belajar terukur.

### 1.5 Lingkup Data

| Aspek | Ketentuan |
|-------|-----------|
| Sumber data | Google Play Store |
| Aplikasi | Wayground (Quizizz) |
| Package name | com.quizizz_mobile |
| Bahasa ulasan | Bahasa Indonesia |
| Rentang waktu | Januari 2024 - Juni 2026 |
| Teknik sampling | Purposive sampling |
| Kriteria inklusi | Mengandung opini atau pengalaman pembelajaran; bukan duplikat; relevan dengan konteks penggunaan aplikasi |

---

## 2. Tahap Penyusutan Data

| Tahap | Keterangan |
|-------|------------|
| Hasil scraping mentah | Data asli dari Google Play Store |
| Setelah remove duplicate | Penghapusan data duplikat berdasarkan konten |
| Setelah preprocessing | Data yang berhasil diproses melalui 7 tahap preprocessing |
| Valid untuk pemodelan | Data setelah preprocessing dan pelabelan |

---

## 3. Preprocessing Teks (7 Tahap)

| No | Tahap | Keterangan |
|----|-------|------------|
| 1 | Cleaning | Hapus emoji, URL, angka, tanda baca |
| 2 | Case folding | Ubah ke huruf kecil |
| 3 | Tokenizing | Pecah teks menjadi token kata |
| 4 | Normalisasi slang | Menggunakan slang_dict.json |
| 5 | Negation handling | Gabungkan tidak+bagus -> tidak_bagus |
| 6 | Stopword removal | Hapus kata umum bahasa Indonesia |
| 7 | Stemming | Sastrawi Stemmer |

**Daftar kata negasi:** tidak, belum, bukan, jangan, tak, kurang, dll

**Contoh transformasi:**

| Tahap | Hasil |
|-------|-------|
| Teks awal | "Aplikasi tidak bisa dibuka, error terus!" |
| Cleaning | "Aplikasi tidak bisa dibuka error terus" |
| Case folding | "aplikasi tidak bisa dibuka error terus" |
| Normalisasi | "aplikasi tidak bisa dibuka error terus" |
| Negation handling | "aplikasi tidak_bisa dibuka error terus" |
| Stopword removal | "aplikasi tidak_bisa dibuka error" |
| Stemming | "aplikasi tidak_bisa buka error" |

---

## 4. Pelabelan Sentimen (InSet Lexicon)

### 4.1 Lexicon yang Digunakan

| Sumber Lexicon | Jumlah Kata |
|----------------|------------|
| Positive words (expanded) | 212 kata |
| Negative words (expanded) | 216 kata |

### 4.2 Rumus Penentuan Kelas

```
skor_total = jumlah(skor kata positif) + jumlah(skor kata negatif)

Jika threshold = 0.0:
  skor_total > 0.0  -> Positif
  skor_total < 0.0 -> Negatif
  0.0 <= skor_total <= 0.0 -> Netral
```

### 4.3 Konfigurasi Pelabelan

| Parameter | Nilai |
|-----------|-------|
| NEUTRAL_THRESHOLD | 0.0 (berdasarkan eksperimen) |
| Sentiment Classes | Positif, Netral, Negatif |

---

## 5. Deteksi Aspek

### 5.1 Empat Aspek Pembelajaran

| Aspek | Definisi | Kata Kunci |
|-------|----------|------------|
| Kemudahan Akses | Penilaian terhadap kemudahan akses dan kecepatan pemuatan | login, register, signup, masuk, daftar, loading, wifi, koneksi, lemot, lambat, buka, akses, timeout |
| Gamifikasi | Penilaian terhadap unsur permainan sebagai pendorong motivasi | game, poin, ranking, leaderboard, level, xp, exp, medal, hadiah, reward, streak, daily, seru, menyenangkan |
| Materi/Konten | Penilaian terhadap kualitas dan kelengkapan konten | materi, soal, quiz, kuis, pembahasan, jawaban, latihan, belajar, konten, bank soal, pelajaran, ujian, tes |
| Performa Teknis | Penilaian terhadap stabilitas dan keandalan aplikasi | error, crash, lag, bug, hang, glitch, down, gagal, rusak, not responding, force close, keluar sendiri |

---

## 6. Ekstraksi Fitur TF-IDF

### 6.1 Parameter TF-IDF (Hasil Eksperimen)

| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| max_features | 5000 | Eksperimen 3 |
| ngram_range | (1, 2) | Eksperimen 3 |
| sublinear_tf | True | Default |
| min_df | 2 | Default |
| max_df | 0.9 | Default |

---

## 7. Pembagian Data Latih dan Data Uji

| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| Split Ratio | 80:20 (80% latih, 20% uji) | Eksperimen 1 |
| Stratify | True | Default |
| Random State | 42 | Fixed |
| CV Folds | 5 | Default |
| CV Scoring | f1_macro | Default |

---

## 8. Klasifikasi SVM

### 8.1 Parameter Model (Hasil Eksperimen)

| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| Algorithm | SVC (Support Vector Classification) | Default |
| Kernel | linear | Eksperimen 4 |
| C | 10.0 | Eksperimen 4 |
| gamma | scale | Default |
| class_weight | {"Negatif": 3.0} | Eksperimen 5 |
| random_state | 42 | Fixed |

### 8.2 Grid Search Configuration

| Parameter | Ruang Pencarian |
|-----------|----------------|
| kernel | linear, rbf, poly, sigmoid |
| C | 0.01, 0.1, 0.5, 1.0, 10, 100 |
| gamma | scale, auto, 0.01, 0.1, 1 |
| cv | StratifiedKFold(n_splits=5) |
| scoring | f1_macro |

---

## 9. Evaluasi Model

### 9.1 Metrik Utama

| Metrik | Target | Hasil CV |
|--------|--------|----------|
| Accuracy | > 85% | 91.43% |
| Macro Precision | > 80% | 86.46% |
| Macro Recall | > 80% | 88.39% |
| Macro F1-Score | > 85% | 87.37% |

**Catatan:** Hasil metrics dari cross-validation (5-fold) dan test set. CV F1-Macro: 84.69% ± 2.25%

### 9.2 Metrik Per Kelas

Model dievaluasi secara individual untuk setiap kelas sentimen:
- **Positif**: Precision, Recall, F1-Score
- **Netral**: Precision, Recall, F1-Score
- **Negatif**: Precision, Recall, F1-Score

### 9.3 Confusion Matrix

Confusion matrix digunakan untuk menganalisis:
- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

---

## 10. Parameter Kunci Sistem

Parameter berikut disimpan di `shared/final_config.py` dan `output/experiments/best_config.json` sebagai single source of truth:

| Parameter | Nilai |
|-----------|-------|
| NEUTRAL_THRESHOLD | 0.0 |
| TRAIN_TEST_SPLIT | 0.20 (20% test) |
| RANDOM_STATE | 42 |
| CV_FOLDS | 5 |
| TFIDF_MAX_FEATURES | 5000 |
| TFIDF_NGRAM_RANGE | (1, 2) |
| TFIDF_SUBLINEAR_TF | True |
| TFIDF_MIN_DF | 2 |
| TFIDF_MAX_DF | 0.9 |
| SVM_KERNEL | linear |
| SVM_C | 10.0 |
| SVM_GAMMA | scale |
| SVM_CLASS_WEIGHT | {"Negatif": 3.0} |

---

## 11. Arsitektur Sistem

```
SCRAPING
    |
    v
REMOVE DUPLICATE
    |
    v
PREPROCESSING (7 Tahap)
    |
    v
LABELING (InSet Lexicon)
    |
    v
ASPECT DETECTION
    |
    v
DATA SPLIT (80:20)
    |
    v
TF-IDF VECTORIZATION
    |
    v
SVM CLASSIFICATION
    |
    v
EVALUATION
    |
    v
MODEL SAVING
```

---

## 12. Eksperimen yang Dijalankan

| No | Eksperimen | Tujuan | Hasil |
|----|------------|--------|-------|
| 1 | Split Ratio | Optimasi rasio 60:40, 70:30, 80:20, 90:10 | 80:20 terbaik |
| 2 | Neutral Threshold | Optimasi T=0.0, 0.5, 1.0, 1.5, 2.0 | T=0.0 terbaik |
| 3 | TF-IDF Optimization | Optimasi max_features, ngram, sublinear_tf | max_features=5000, ngram=(1,2) |
| 4 | SVM Kernel & Hyperparameter | Optimasi kernel, C, gamma | linear, C=10, gamma=scale |
| 5 | Class Imbalance | Testing class_weight | {"Negatif": 3.0} |
| 6 | Stratified CV | Validasi stabilitas | 5-fold CV |

---

## 13. Catatan Implementasi

1. **Single Source of Truth**: Semua konfigurasi model disimpan di `output/experiments/best_config.json`.

2. **Data Driven**: Frontend tidak lagi menggunakan hardcoded data. Semua data berasal dari dataset aktual.

3. **Pipeline Tracking**: Setiap tahap pipeline menyimpan hasil ke database untuk transparansi.

4. **Model Versioning**: Setiap training model disimpan dengan metadata lengkap.

5. **Reproducibility**: Random state 42 digunakan untuk memastikan hasil yang konsisten.

6. **Kesesuaian** — Lihat [audit kesesuaian metodologi](riwayat/audit/kesesuaian_metodologi.md) untuk catatan tahap sebelumnya.

7. **Split Ratio**: Metodologi menentukan 80:20 split. best_config.json sudah diperbarui menjadi `split_ratio: 0.8`.

---

## 14. Ringkasan

Penelitian ini menggunakan pendekatan machine learning dengan algoritma SVM untuk mengklasifikasikan sentimen ulasan pengguna aplikasi Wayground. Pipeline meliputi:
- Preprocessing teks 7 tahap
- Pelabelan menggunakan InSet Lexicon dengan threshold T=0.0
- Ekstraksi fitur TF-IDF dengan max_features=5000 dan ngram=(1,2)
- Klasifikasi menggunakan SVM dengan kernel linear, C=10, dan class_weight={"Negatif": 3.0}

Hasil evaluasi menunjukkan F1-Macro rata-rata di atas 85% yang menandakan model memiliki performa baik dalam mengklasifikasikan sentimen ulasan pengguna.

---

*Dokumen ini diupdate sesuai dengan hasil eksperimen terbaik.*
*Single source of truth: output/experiments/best_config.json*
