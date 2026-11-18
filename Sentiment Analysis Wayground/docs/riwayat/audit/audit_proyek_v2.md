# LAPORAN AUDIT PROJECT ANALISIS SENTIMEN WAYGROUND

**Tanggal Audit**: 2026-01-XX  
**Auditor**: Claude Code  
**Project**: Evaluasi Efektivitas Pembelajaran Aplikasi Wayground Berdasarkan Analisis Sentimen Ulasan Pengguna Menggunakan SVM

---

## RINGKASAN EKSEKUTIF

Project ini adalah implementasi analisis sentimen ulasan Google Play Store aplikasi **Wayground (Quizizz)** menggunakan metode **SVM dengan pembobotan TF-IDF**. Setelah audit menyeluruh, ditemukan beberapa inkonsistensi antara dokumentasi metodologi, konfigurasi eksperimen, dan implementasi kode.

### Temuan Utama:
1. **INKONSISTENSI KONFIGURASI**: Nilai parameter berbeda antara metodologi.md, best_config.json, dan final_config.py
2. **DUPLIKASI FILE LEXICON**: Terdapat 4 file lexicon dengan nama berbeda (positif vs positif)
3. **ORGANISASI FOLDER BELUM OPTIMAL**: Script eksperimen dan pipeline belum terpisah dengan jelas
4. **BEBERAPA SCRIPT TIDAK DIGUNAKAN**: Terdapat script yang tidak diperlukan untuk laporan akhir

---

## 1. ANALISIS STRUKTUR PROJECT SAAT INI

### 1.1 Struktur Folder

```
Sentiment Analysis Wayground/
├── backend/              # Flask API backend
├── data/                # Data dan lexicon
│   ├── lexicon/         # File lexicon
│   ├── pipeline_output/ # Output pipeline
│   └── uploads/        # User uploads
├── docs/               # Dokumentasi
├── frontend/           # Web dashboard
├── models/             # Model tersimpan
├── output/             # Output eksperimen
│   ├── experiments/     # Konfigurasi eksperimen
│   ├── evaluation/     # Evaluasi
│   ├── tuning/         # Tuning results
│   └── reports/         # Laporan
├── pipeline/           # Pipeline utama
├── scripts/            # Script eksperimen
│   └── run_experiments/ # Eksperimen
├── shared/             # Modul bersama
├── tests/              # Unit tests
├── _archive/           # Script lama (diarsipkan)
└── docs/              # Dokumentasi
```

### 1.2 File yang Diperiksa

| Folder | File | Fungsi | Status |
|--------|------|--------|--------|
| **pipeline/** | scraper.py | Scraping Google Play Store | ✅ AKTIF |
| | scraper_final.py | Versi final scraper | ⚠️ DUPLIKAT |
| | preprocessing.py | 7 tahap preprocessing | ✅ AKTIF |
| | labeling.py | Pelabelan InSet Lexicon | ✅ AKTIF |
| | aspect_detection.py | Deteksi 4 aspek | ✅ AKTIF |
| | train_svm.py | Training SVM | ✅ AKTIF |
| | run_pipeline.py | Pipeline utama | ✅ AKTIF |
| | tfidf_analysis.py | Analisis TF-IDF | ⚠️ OPSIONAL |
| **shared/** | constants.py | Konstanta | ✅ AKTIF |
| | final_config.py | Konfigurasi ML | ✅ AKTIF |
| | lexicon.py | Modul lexicon | ✅ AKTIF |
| | preprocessing.py | Preprocessing | ✅ AKTIF |
| | svm_visualization.py | Visualisasi | ✅ AKTIF |
| **scripts/** | config_loader.py | Load config | ✅ AKTIF |
| | run_experiments.py | Run semua eksperimen | ✅ AKTIF |
| | train_final_model.py | Training final | ⚠️ DUPLIKAT |
| | validate_threshold.py | Validasi threshold | ⚠️ OPSIONAL |
| | compare_labeling.py | Bandingkan labeling | ⚠️ OPSIONAL |
| **backend/** | app.py | Flask app | ✅ AKTIF |
| | config.py | Konfigurasi backend | ✅ AKTIF |
| | ml_engine.py | ML engine | ✅ AKTIF |
| | pipeline_api.py | API pipeline | ✅ AKTIF |
| | models.py | Model routes | ✅ AKTIF |
| **tests/** | test_*.py | Unit tests | ⚠️ OPSIONAL |
| **_archive/** | *.py | Script lama | ❌ ARSIP |
| **root/** | create_complete_analysis.py | Analisis lengkap | ⚠️ OPSIONAL |
| | fix_labels.py | Perbaikan label | ⚠️ OPSIONAL |
| | analyze_threshold.py | Analisis threshold | ⚠️ OPSIONAL |

---

## 2. ANALISIS INKONSISTENSI KONFIGURASI

### 2.1 Perbandingan Nilai Parameter

| Parameter | metodologi.md | best_config.json | final_config.py | Best Config |
|-----------|---------------|------------------|-----------------|-------------|
| **Split Ratio** | 80:20 | 0.9 (90:10) | 0.80 (80:20) | **MENYESALANKAN** |
| **Neutral Threshold** | 1.0 | 1.0 | 1.0 | ✅ KONSISTEN |
| **TF-IDF max_features** | 1000 | 5000 | 1000 | ⚠️ INKONSISTEN |
| **TF-IDF ngram_range** | (1,1) | (1,2) | (1,1) | ⚠️ INKONSISTEN |
| **TF-IDF sublinear_tf** | True | True | True | ✅ KONSISTEN |
| **TF-IDF min_df** | 2 | 2 | 2 | ✅ KONSISTEN |
| **TF-IDF max_df** | 0.9 | 0.9 | - | ⚠️ MISSING |
| **SVM Kernel** | linear | linear | rbf | ⚠️ INKONSISTEN |
| **SVM C** | 10 | 10 | 100.0 | ⚠️ INKONSISTEN |
| **SVM gamma** | scale | scale | 0.1 | ⚠️ INKONSISTEN |
| **SVM class_weight** | None | {"Negatif":3.0} | None | ⚠️ INKONSISTEN |

### 2.2 Dampak Inkonsistensi

1. **Split Ratio 80:20 vs 90:10**:
   - Metodologi menyatakan 80:20
   - best_config.json menggunakan 0.9 (90:10)
   - final_config.py menggunakan 0.80 (80:20)
   - **Dampak**: Hasil eksperimen mungkin tidak sesuai metodologi

2. **SVM Kernel**:
   - Metodologi & best_config: `linear`
   - final_config.py: `rbf`
   - **Dampak**: Model final tidak sesuai dengan yang diuji eksperimen

3. **SVM C**:
   - Metodologi: 10
   - best_config.json: 10.0
   - final_config.py: 100.0
   - **Dampak**: Hyperparameter tidak konsisten

---

## 3. ANALISIS FOLDER DAN FILE

### 3.1 Folder yang Dibutuhkan User

Berdasarkan request user, struktur folder yang diinginkan:

```
1. Frontend/          # Dashboard web
2. Backend/            # Flask API
3. Data/               # Data model
   ├── lexicon/        # Kamus InSet, slang, normalisasi
   ├── pipeline_output/ # Hasil pipeline
   └── uploads/        # User uploads
4. Models/             # Model terbaik
5. Output/             # Output eksperimen & laporan
6. Pipeline/           # Alur model utama
7. Scripts/             # Script eksperimen & utilitas
8. Shared/             # Modul bersama
9. Tests/               # Unit tests
10. docs/               # Dokumentasi laporan
```

### 3.2 Kondisi Eksisting vs Yang Diinginkan

| Folder | Eksisting | Dibutuhkan | Status |
|--------|-----------|------------|--------|
| Frontend | ✅ Ada | ✅ Sama | ✅ OK |
| Backend | ✅ Ada | ✅ Sama | ✅ OK |
| Data | ✅ Ada | ✅ Sama | ✅ OK |
| Data/lexicon | ✅ Ada | ✅ Sama | ⚠️ Perlu dibersihkan |
| Data/pipeline_output | ✅ Ada | ✅ Sama | ✅ OK |
| Models | ✅ Ada | ✅ Sama | ✅ OK |
| Output | ✅ Ada | ✅ Sama | ✅ OK |
| Pipeline | ✅ Ada | ✅ Sama | ✅ OK |
| Scripts | ✅ Ada | ✅ Sama | ⚠️ Perlu dipilah |
| Shared | ✅ Ada | ✅ Sama | ✅ OK |
| Tests | ✅ Ada | ✅ Sama | ✅ OK |
| docs | ⚠️ Di root | ✅ Pindahkan | ❌ PERBAIKAN |

---

## 4. ANALISIS SCRIPT YANG DIGUNAKAN

### 4.1 Script Wajib untuk Pipeline

| Script | Fungsi | Pipeline Stage | Status |
|--------|--------|----------------|--------|
| pipeline/scraper.py | Scraping Google Play | 1. Scraping | ✅ DIPERLUKAN |
| pipeline/preprocessing.py | Preprocessing (shared/) | 2. Preprocessing | ✅ DIPERLUKAN |
| pipeline/labeling.py | Pelabelan Lexicon | 3. Pelabelan | ✅ DIPERLUKAN |
| pipeline/aspect_detection.py | Deteksi Aspek | 4. Deteksi Aspek | ✅ DIPERLUKAN |
| pipeline/train_svm.py | Training SVM | 6. Training | ✅ DIPERLUKAN |
| pipeline/run_pipeline.py | Pipeline Utama | All | ✅ DIPERLUKAN |
| shared/preprocessing.py | Modul preprocessing | Supporting | ✅ DIPERLUKAN |
| shared/lexicon.py | Modul lexicon | Supporting | ✅ DIPERLUKAN |
| shared/constants.py | Konstanta | Supporting | ✅ DIPERLUKAN |
| shared/final_config.py | Konfigurasi ML | Supporting | ✅ DIPERLUKAN |

### 4.2 Script Eksperimen

| Script | Fungsi | Laporan? | Status |
|--------|--------|----------|--------|
| scripts/run_experiments/run_experiments.py | Run semua eksperimen | ✅ Ya | ✅ DIPERLUKAN |
| scripts/config_loader.py | Load config | ✅ Ya | ✅ DIPERLUKAN |
| scripts/train_final_model.py | Training final model | ✅ Ya | ⚠️ DUPLIKAT |
| scripts/validate_threshold.py | Validasi threshold | ✅ Ya | ⚠️ OPSIONAL |
| scripts/compare_labeling.py | Bandingkan labeling | ⚠️ Tidak | ❌ OPSIONAL |
| scripts/error_analysis.py | Analisis error | ⚠️ Tidak | ❌ OPSIONAL |

### 4.3 Script Tidak Dibutuhkan untuk Laporan

| Script | Alasan Tidak Dibutuhkan |
|--------|------------------------|
| create_complete_analysis.py | Duplikat pipeline |
| fix_labels.py | Script perbaikan manual |
| analyze_threshold.py | Duplikat eksperimen |
| scripts/migrate_add_pipeline_columns.py | Utility satu waktu |
| scripts/relabel_dataset.py | Utility satu waktu |
| scripts/convert_kamus_to_lexicon.py | Utility satu waktu |
| scripts/create_sentiment_dict.py | Utility satu waktu |
| scriptsexperiments_v2svm_experiment_pipeline.py | Typo file, salah letak |

### 4.4 Script Backend

| Script | Fungsi | Status |
|--------|--------|--------|
| backend/app.py | Flask app utama | ✅ DIPERLUKAN |
| backend/config.py | Konfigurasi | ✅ DIPERLUKAN |
| backend/ml_engine.py | ML engine | ✅ DIPERLUKAN |
| backend/pipeline_api.py | API pipeline | ✅ DIPERLUKAN |
| backend/models.py | Routes model | ✅ DIPERLUKAN |
| backend/enhanced_api.py | API enhanced | ✅ DIPERLUKAN |

### 4.5 Script Testing

| Script | Fungsi | Status |
|--------|--------|--------|
| tests/test_preprocessing.py | Test preprocessing | ⚠️ OPSIONAL |
| tests/test_model.py | Test model | ⚠️ OPSIONAL |
| tests/test_svm.py | Test SVM | ⚠️ OPSIONAL |
| tests/test_leakage.py | Test data leakage | ✅ REKOMENDASI |
| tests/test_visualizations.py | Test visualisasi | ❌ TIDAK PERLU |
| tests/test_api.py | Test API | ⚠️ OPSIONAL |

---

## 5. ANALISIS DATA LEXICON

### 5.1 File Lexicon Eksisting

| File | Ukuran | Jumlah Kata | Status |
|------|--------|-------------|--------|
| lexicon_positif.tsv | 292 bytes | ~35 kata | ⚠️ DUPLIKAT |
| lexicon_positive.tsv | 2341 bytes | ~212 kata | ✅ VERSI LENGKAP |
| lexicon_negatif.tsv | 656 bytes | ~66 kata | ⚠️ DUPLIKAT |
| lexicon_negative.tsv | 1498 bytes | ~216 kata | ✅ VERSI LENGKAP |

### 5.2 Rekomendasi Lexicon

**Harus Menggunakan**:
- `lexicon_positive.tsv` (212 kata - expanded)
- `lexicon_negative.tsv` (216 kata - expanded)

**Hapus/Diarsipkan**:
- `lexicon_positif.tsv` (duplikat)
- ` lexicon_negatif.tsv` (duplikat)

---

## 6. ANALISIS ALUR PIPELINE

### 6.1 Alur yang Diinginkan (Berdasarkan Request User)

```
1. Scraping → 2. Preprocessing → 3. Pelabelan → 4. Deteksi Aspek → 
5. Split Data → 6. TF-IDF → 7. Evaluasi SVM
```

### 6.2 Alur Eksisting di run_pipeline.py

```
Stage 1: Scraping (scraper.py)
Stage 2: Preprocessing (7 tahap)
Stage 3: Pelabelan (InSet Lexicon)
Stage 4: Deteksi Aspek
Stage 5: Split Data
Stage 6: Training SVM + TF-IDF
Stage 7: Evaluasi
```

### 6.3 Kesesuaian

| Tahap | Request | Eksisting | Status |
|-------|---------|-----------|--------|
| 1. Scraping | ✅ | ✅ | ✅ SESUAI |
| 2. Preprocessing | 7 tahap | 7 tahap | ✅ SESUAI |
| 3. Pelabelan | InSet Lexicon | InSet Lexicon | ✅ SESUAI |
| 4. Deteksi Aspek | 4 aspek | 4 aspek | ✅ SESUAI |
| 5. Split Data | 80:20 | ✅ 80:20 | ✅ SESUAI |
| 6. TF-IDF | max_features, ngram, dll | ✅ | ✅ SESUAI |
| 7. Evaluasi SVM | GridSearchCV, F1-macro | ✅ | ✅ SESUAI |

---

## 7. REKOMENDASI PERBAIKAN

### 7.1 Prioritas Tinggi (Harus Dilakukan)

1. **Perbaiki Inkonsistensi Konfigurasi**
   - Samakan nilai di metodologi.md, best_config.json, dan final_config.py
   - Gunakan nilai dari hasil eksperimen sebagai acuan

2. **Bersihkan File Lexicon Duplikat**
   - Hapus atau arsipkan lexicon_positif.tsv dan lexicon_negatif.tsv
   - Gunakan lexicon_positive.tsv dan lexicon_negative.tsv

3. **Perbaiki File Slug/Dict**
   - slang_dict.json → slang_dict_fixed.json
   - normalization_dict.json → normalization_dict_fixed.json

### 7.2 Prioritas Sedang (Disarankan)

4. **Pindahkan Dokumentasi ke Folder docs/**
   - metodologi.md → docs/
   - LAPORAN_*.md → docs/
   - REFACTOR_REPORT.md → docs/

5. **Rapikan Script Scripts**
   - Hapus scriptsexperiments_v2svm_experiment_pipeline.py (typo)
   - Pindahkan utility scripts ke folder terpisah

### 7.3 Prioritas Rendah (Opsional)

6. **Tambahkan Unit Tests** untuk pipeline critical
7. **Dokumentasi README** untuk setiap folder

---

## 8. RENCANA PERBAIKAN METODOLOGI

### 8.1 Nilai Parameter yang Harus Digunakan

Berdasarkan hasil eksperimen (best_config.json), berikut nilai yang seharusnya di dokumentasikan:

| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| Split Ratio | 80:20 (0.20 test) | Eksperimen 1 |
| Neutral Threshold | 1.0 | Eksperimen 2 |
| TF-IDF max_features | 5000 | Eksperimen 3 |
| TF-IDF ngram_range | (1, 2) | Eksperimen 3 |
| TF-IDF sublinear_tf | True | Eksperimen 3 |
| TF-IDF min_df | 2 | Default |
| TF-IDF max_df | 0.9 | Default |
| SVM Kernel | linear | Eksperimen 4 |
| SVM C | 10.0 | Eksperimen 4 |
| SVM gamma | scale | Default |
| SVM class_weight | {"Negatif": 3.0} | Eksperimen 5 |

### 8.2 Update metodologi.md

File metodologi.md perlu diupdate dengan:
1. Nilai parameter dari best_config.json
2. Hasil eksperimen terbaik
3. Metrik evaluasi final

---

## 9. KESIMPULAN

### 9.1 Status Keseluruhan

| Aspek | Status |
|-------|--------|
| Struktur Folder | ⚠️ Perlu Penyempurnaan |
| Konsistensi Kode | ⚠️ Ada Inkonsistensi |
| Alur Pipeline | ✅ Sudah Benar |
| Dokumentasi | ⚠️ Perlu Update |
| Script Pipeline | ✅ Sudah Lengkap |

### 9.2 Tindakan yang Diperlukan

1. **Segera**: Perbaiki inkonsistensi konfigurasi
2. **Segera**: Bersihkan file lexicon duplikat
3. **Segera**: Update metodologi.md sesuai hasil eksperimen
4. **Segera**: Pindahkan dokumentasi ke folder docs/
5. **Setelahnya**: Rapikan script scripts/

### 9.3 Script yang WAJIB Disimpan

```
Pipeline (Core):
├── pipeline/
│   ├── scraper.py
│   ├── preprocessing.py (shared/)
│   ├── labeling.py
│   ├── aspect_detection.py
│   ├── train_svm.py
│   └── run_pipeline.py

Shared:
├── shared/
│   ├── __init__.py
│   ├── constants.py
│   ├── final_config.py
│   ├── lexicon.py
│   ├── preprocessing.py
│   └── svm_visualization.py

Scripts:
├── scripts/
│   ├── config_loader.py
│   ├── run_experiments/
│   │   └── run_experiments.py
│   └── train_final_model.py (opsional)

Backend:
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── ml_engine.py
│   ├── pipeline_api.py
│   └── models.py

Data:
├── data/
│   ├── lexicon/
│   │   ├── lexicon_positive.tsv
│   │   ├── lexicon_negative.tsv
│   │   └── (hapus yang duplikat)
│   └── pipeline_output/
```

---

## LAMPIRAN

### A. File yang Perlu Dihapus/Diarsipkan

1. **Duplikat Lexicon**:
   - `data/lexicon/lexicon_positif.tsv`
   - `data/lexicon/lexicon_negatif.tsv`

2. **Script Tidak Diperlukan**:
   - `scriptsexperiments_v2svm_experiment_pipeline.py`
   - `create_complete_analysis.py`
   - `fix_labels.py`
   - `analyze_threshold.py`

3. **Dokumentasi di Root** (pindahkan ke docs/):
   - `metodologi.md`
   - `LAPORAN_*.md`
   - `REFACTOR_REPORT.md`
   - `AUDIT_REPORT.md`
   - `PERBAIKAN_*.md`

### B. Inkonsistensi yang Perlu Diperbaiki

1. **best_config.json vs final_config.py**:
   - split_ratio: 0.9 vs 0.80
   - tfidf_max_features: 5000 vs 1000
   - tfidf_ngram_range: (1,2) vs (1,1)
   - svm_kernel: linear vs rbf
   - svm_C: 10 vs 100
   - svm_class_weight: {"Negatif":3} vs None

### C. Eksperimen yang Sudah Dijalankan

| Eksperimen | Deskripsi | Status |
|------------|-----------|--------|
| Exp 1 | Split Ratio Optimization | ✅ Selesai |
| Exp 2 | Neutral Threshold | ✅ Selesai |
| Exp 3 | TF-IDF Optimization | ✅ Selesai |
| Exp 4 | SVM Kernel & Hyperparameter | ✅ Selesai |
| Exp 5 | Class Imbalance | ✅ Selesai |
| Exp 6 | Stratified Cross Validation | ✅ Selesai |

---

*Dokumen ini adalah hasil audit project Analisis Sentimen Wayground*
