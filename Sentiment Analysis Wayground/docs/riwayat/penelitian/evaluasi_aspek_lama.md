# LAPORAN EVALUASI MODEL PER-ASPEK DAN PERBANDINGAN PAKAR

## Metodologi Penelitian

Berdasarkan metodologi penelitian (Bagian 2.5), validasi pakar digunakan untuk:
- **Memvalidasi 300 sampel ulasan** sebagai ground truth
- **Bukan** untuk melabeli seluruh dataset
- Sistem labeling menggunakan **InSet Lexicon** dengan threshold T=0.0

---

## 1. Konfigurasi Final

| Parameter | Nilai |
|-----------|-------|
| TF-IDF Max Features | 3000 |
| TF-IDF N-gram Range | (1, 3) |
| TF-IDF Min DF | 2 |
| TF-IDF Max DF | 0.9 |
| SVM Kernel | Linear |
| SVM C | 10.0 |
| Data Split | 80:20 |
| Random State | 42 |
| Labeling Threshold | T=0.0 |

---

## 2. Distribusi Sentimen Keseluruhan (Lexicon-Based)

| Kelas | Jumlah | Persentase |
|-------|--------|------------|
| Positif | 2,062 | 70.2% |
| Netral | 458 | 15.6% |
| Negatif | 416 | 14.2% |
| **Total** | **2,936** | **100%** |

---

## 3. Hasil Evaluasi Per-Aspek

### 3.1 Kemudahan Akses

| Metrik | Nilai |
|--------|-------|
| Total Kalimat | 1,393 |
| Distribusi | Positif: 66.9%, Netral: 10.7%, Negatif: 22.4% |
| CV F1 (macro) | 0.5929 ± 0.0200 |
| Test Accuracy | 0.8208 |
| Test F1 (macro) | 0.6758 |

**Per-Class Metrics:**

| Kelas | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Positif | 0.8750 | 0.9358 | 0.9044 | 187 |
| Netral | 0.4348 | 0.3333 | 0.3774 | 30 |
| Negatif | 0.7857 | 0.7097 | 0.7458 | 62 |

**Confusion Matrix:**

| Actual/Pred | Pos | Net | Neg |
|-------------|-----|-----|-----|
| Positif | 175 | 5 | 7 |
| Netral | 15 | 10 | 5 |
| Negatif | 10 | 8 | 44 |

---

### 3.2 Gamifikasi

| Metrik | Nilai |
|--------|-------|
| Total Kalimat | 688 |
| Distribusi | Positif: 86.3%, Netral: 6.1%, Negatif: 7.6% |
| CV F1 (macro) | 0.5431 ± 0.0414 |
| Test Accuracy | 0.8841 |
| Test F1 (macro) | 0.5466 |

**Per-Class Metrics:**

| Kelas | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Positif | 0.9206 | 0.9748 | 0.9469 | 119 |
| Netral | 0.2500 | 0.1250 | 0.1667 | 8 |
| Negatif | 0.6250 | 0.4545 | 0.5263 | 11 |

**Confusion Matrix:**

| Actual/Pred | Pos | Net | Neg |
|-------------|-----|-----|-----|
| Positif | 116 | 2 | 1 |
| Netral | 5 | 1 | 2 |
| Negatif | 5 | 1 | 5 |

---

### 3.3 Materi/Konten

| Metrik | Nilai |
|--------|-------|
| Total Kalimat | 1,373 |
| Distribusi | Positif: 81.1%, Netral: 7.0%, Negatif: 11.9% |
| CV F1 (macro) | 0.5888 ± 0.0384 |
| Test Accuracy | 0.8909 |
| Test F1 (macro) | 0.6664 |

**Per-Class Metrics:**

| Kelas | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Positif | 0.9397 | 0.9776 | 0.9582 | 223 |
| Netral | 0.4286 | 0.3158 | 0.3636 | 19 |
| Negatif | 0.7241 | 0.6364 | 0.6774 | 33 |

**Confusion Matrix:**

| Actual/Pred | Pos | Net | Neg |
|-------------|-----|-----|-----|
| Positif | 218 | 2 | 3 |
| Netral | 8 | 6 | 5 |
| Negatif | 6 | 6 | 21 |

---

### 3.4 Performa Teknis

| Metrik | Nilai |
|--------|-------|
| Total Kalimat | 358 |
| Distribusi | Positif: 26.5%, Netral: 8.4%, Negatif: 65.1% |
| CV F1 (macro) | 0.5944 ± 0.0360 |
| Test Accuracy | 0.7778 |
| Test F1 (macro) | 0.5155 |

**Per-Class Metrics:**

| Kelas | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Positif | 0.6667 | 0.6316 | 0.6486 | 19 |
| Netral | 0.0000 | 0.0000 | 0.0000 | 6 |
| Negatif | 0.8627 | 0.9362 | 0.8980 | 47 |

**Confusion Matrix:**

| Actual/Pred | Pos | Net | Neg |
|-------------|-----|-----|-----|
| Positif | 12 | 2 | 5 |
| Netral | 4 | 0 | 2 |
| Negatif | 2 | 1 | 44 |

---

## 4. Ringkasan Perbandingan Aspek

| Aspek | Kalimat | CV F1 | Test Acc | Test F1 |
|-------|---------|-------|----------|---------|
| Kemudahan Akses | 1,393 | 0.5929 | 0.8208 | 0.6758 |
| Gamifikasi | 688 | 0.5431 | 0.8841 | 0.5466 |
| Materi/Konten | 1,373 | 0.5888 | 0.8909 | 0.6664 |
| Performa Teknis | 358 | 0.5944 | 0.7778 | 0.5155 |

**Rata-rata Keseluruhan:**
- CV F1 (macro): 0.5798
- Test Accuracy: 0.8434
- Test F1 (macro): 0.6011

---

## 5. Verifikasi Independensi Sentiment Aspek

Sistem telah diperbaiki untuk menghitung sentiment aspek **secara independen** dari sentiment review keseluruhan.

### Statistik Perbandingan:

| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Kalimat dengan sentiment ASPEK BERBEDA dari review | 161 | 4.2% |
| Kalimat dengan sentiment ASPEK SAMA dengan review | 3,651 | 95.8% |

### Contoh Kalimat dengan Sentiment Aspek Berbeda:

1. **Kemudahan Akses & Performa Teknis:**
   - Kalimat: "Aplikasinya bagus, namun saran saya sebaiknya tidak perlu ada musiknya..."
   - Sentiment Review: Positif (score: 0.50)
   - Sentiment Aspek: Negatif (score: -0.50)
   - **Interpretasi:** Kalimat keseluruhan positif karena "bagus", tapi mengandung saran/kritik yang negatif

2. **Materi/Konten:**
   - Kalimat: "kalian kalo mau latihan soal mending pake ini..."
   - Sentiment Review: Positif (score: 5.50)
   - Sentiment Aspek: Netral (score: 0.00)
   - **Interpretasi:** Kalimat keseluruhan sangat positif karena mendorong penggunaan aplikasi

---

## 6. Interpretasi Hasil

### 6.1 Distribusi Sentimen per Aspek

1. **Gamifikasi** memiliki sentimen paling positif (86.3% positif, 7.6% negatif)
   - Indicates users enjoy the gamification features

2. **Materi/Konten** juga sangat positif (81.1% positif, 11.9% negatif)
   - Indicates users find content helpful

3. **Kemudahan Akses** cukup positif (66.9% positif, 22.4% negatif)
   - Some users experience access issues

4. **Performa Teknis** memiliki sentimen paling negatif (65.1% negatif, 26.5% positif)
   - **CRITICAL:** Most complaints relate to technical performance (bugs, crashes, errors)

### 6.2 Model Performance

1. **Kelas Positif** memiliki performa terbaik di semua aspek (F1 > 0.64)
   - Due to higher sample count and clearer lexicon signals

2. **Kelas Negatif** memiliki performa sedang (F1 0.52-0.90)
   - Better in aspects with higher negative sample count (Performa Teknis: F1=0.90)

3. **Kelas Netral** memiliki performa rendah (F1 < 0.38)
   - Due to low sample count and ambiguous sentiment signals
   - Challenge in distinguishing neutral from mild positive/negative

### 6.3 Implikasi untuk Pembelajaran

Berdasarkan hasil analisis aspek:

1. **Kekuatan:**
   - Gamifikasi efektif untuk motivasi belajar
   - Konten/materi berkualitas dan bermanfaat
   - Aplikasi secara umum membantu pembelajaran

2. **Area Perbaikan:**
   - **Performa Teknis** adalah prioritas utama perbaikan
   - **Kemudahan Akses** perlu peningkatan stabilitas
   - Perbaikan bug dan crash akan meningkatkan kepuasan pengguna

---

## 7. Catatan Metodologi

1. **Ground Truth:** 300 sampel validasi pakar digunakan sebagai benchmark (sesuai metodologi)
2. **Sentiment Labeling:** Menggunakan InSet Lexicon dengan threshold T=0.0
3. **Independent Aspect Sentiment:** Sentiment aspek dihitung independen dari sentiment review
4. **Cross-Validation:** 5-fold stratified CV untuk robustness evaluation

---

## 8. File Output

| File | Deskripsi |
|------|-----------|
| `per_aspect_evaluation.json` | Hasil evaluasi lengkap per aspek |
| `per_aspect_evaluation_summary.csv` | Ringkasan CSV untuk visualisasi |
| `stage4_aspects_detail.csv` | Detail semua aspek dengan sentiment independen |

---

## 9. Kesimpulan

1. **Model SVM per-aspek** menunjukkan performa baik untuk kelas Positif dan Negatif
2. **Kelas Netral** memerlukan perbaikan metode deteksi
3. **Performa Teknis** adalah aspek dengan sentimen negatif tertinggi - prioritas perbaikan
4. **Independensi sentiment aspek** telah diverifikasi (4.2% kalimat memiliki sentiment berbeda)

---

*Generated: 2024*
*Pipeline Version: run_pipeline_v1*
