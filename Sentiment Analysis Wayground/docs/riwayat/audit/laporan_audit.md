# AUDIT REPORT - Sentiment Analysis Wayground

**Tanggal Audit:** 2026-09-23
**Auditor:** Claude Code (Senior ML/Backend/Frontend Engineer)

---

## 1. EXECUTIVE SUMMARY

### Kondisi Awal
Project Sentiment Analysis Wayground memiliki struktur yang cukup baik dengan pemisahan shared/pipeline/backend/frontend. Namun, ditemukan beberapa **inkonsistensi kritis** antara hasil pipeline penelitian dan apa yang digunakan oleh sistem web.

### Masalah Utama
1. **HARDCODED METRICS** di frontend yang menghasilkan angka fiktif
2. **Dua model berbeda** dengan metrik berbeda (models/ vs pipeline_output/)
3. **State management** tidak mereset dengan benar saat dataset berubah
4. **Model artifacts** tidak memiliki metadata/versioning yang konsisten

### Perubahan yang Dibutuhkan
- Hapus hardcoded metrics di frontend
- Konsolidasikan model source of truth
- Perbaiki frontend state management
- Tambahkan model metadata
- Verifikasi end-to-end consistency

---

## 2. DEPENDENCY MAP

### Dataset Sources
```
data/02_data_akhir.csv (RAW)
    ↓
data/pipeline_output/stage1_scraped.csv
    ↓
data/pipeline_output/stage2_preprocessed.csv
    ↓
data/pipeline_output/stage3_labeled.csv
    ↓
data/pipeline_output/stage4_aspects.csv
    ↓
data/pipeline_output/stage5_split/train.csv + test.csv
    ↓
data/pipeline_output/stage6_model/ (LATEST PIPELINE MODEL)
    ↓
models/ (BACKEND MODEL - POTENTIAL INCONSISTENCY)
```

### Model Sources (CRITICAL ISSUE)
```
models/svm_model_latest.pkl ← BACKEND USES THIS
    ├── metrics_latest.json → accuracy: 0.8247, f1_macro: 0.7593

data/pipeline_output/stage6_model/svm_model_20260923_014348.pkl ← PIPELINE PRODUCES THIS
    └── pipeline_report.txt → accuracy: 0.9043, f1_macro: 0.8425
```

### Component Dependencies
```
shared/preprocessing.py ← canonical preprocessing
    ├── shared/constants.py (NEGATION_WORDS, STOPWORDS)
    ├── slang_dict.json
    └── normalization_dict.json

shared/lexicon.py ← canonical labeling
    ├── shared/constants.py (NEUTRAL_THRESHOLD=1.1030)
    └── data/lexicon/positive.tsv + negative.tsv

shared/constants.py ← central config
    ├── ASPECT_KEYWORDS
    ├── NEUTRAL_THRESHOLD
    └── SVM/TF-IDF params

pipeline/aspect_detection.py ← canonical aspect detection
    └── shared/constants.py (ASPECT_KEYWORDS)

pipeline/train_svm.py ← canonical TF-IDF + SVM
    └── shared/constants.py

backend/ml_engine.py ← orchestration layer
    ├── shared/preprocessing.py
    ├── shared/lexicon.py
    ├── pipeline/train_svm.py
    ├── pipeline/aspect_detection.py
    └── pipeline/labeling.py

frontend/
    ├── services/api.js → backend API
    └── context/AnalysisContext.jsx → state management
```

---

## 3. FINDINGS - CRITICAL ISSUES (P0)

### 3.1 HARDCODED METRICS IN FRONTEND
**File:** `frontend/src/pages/Training.jsx`
**Lines:** 40-45

```javascript
const kernelResults = [
    { name: 'Linear', f1: state.modelMetrics?.f1_macro || 0.80, accuracy: state.modelMetrics?.accuracy || 0.852 },
    { name: 'RBF', f1: 0.72, accuracy: 0.798 },
    { name: 'Poly', f1: 0.70, accuracy: 0.775 },
    { name: 'Sigmoid', f1: 0.68, accuracy: 0.752 }
]
```

**ISSUE:** Hardcoded F1 dan accuracy untuk kernel RBF, Poly, Sigmoid yang TIDAK berasal dari evaluasi aktual.

**Impact:** Frontend menampilkan hasil eksperimen fiktif yang tidak ada hubungannya dengan model sebenarnya.

**FIX REQUIRED:** Fetch hasil kernel comparison dari backend, atau hapus chart ini jika backend belum mengimplementasikan kernel comparison.

---

### 3.2 INCONSISTENT MODEL METRICS
**File:** `models/metrics_latest.json`
**vs**
**File:** `data/pipeline_output/pipeline_report.txt`

| Metric | models/ (Backend) | pipeline_output/ (Research) |
|--------|-------------------|---------------------------|
| Accuracy | 0.8247 | 0.9043 |
| F1-Macro | 0.7593 | 0.8425 |
| Precision | 0.7586 | 0.8385 |
| Recall | 0.7644 | 0.8466 |

**ISSUE:** Dua model berbeda dengan hasil berbeda. Tidak jelas model mana yang "final" dan apakah keduanya berasal dari dataset yang sama.

**Impact:** Penelitian dan sistem web berpotensi menggunakan hasil yang berbeda.

**FIX REQUIRED:**
1. Tentukan source of truth: model final harus di `models/` dan `pipeline_output/stage6_model/` dengan versioning yang sama.
2. Tambahkan metadata JSON yang konsisten.
3. Pastikan hanya satu pipeline yang update model final.

---

### 3.3 MISSING MODEL METADATA
**File:** `models/` dan `data/pipeline_output/stage6_model/`

**ISSUE:** Model artifacts tidak memiliki metadata yang tercatat:
- Dataset version yang digunakan
- Preprocessing version
- Training configuration
- Training timestamp
- Random state

**FIX REQUIRED:** Tambahkan metadata.json alongside setiap model artifact.

---

### 3.4 FRONTEND STATE NOT RESETTING ON NEW DATASET
**File:** `frontend/src/context/AnalysisContext.jsx`

**ISSUE:** Fungsi `uploadFile` (line 188-210) tidak memanggil `clearData()` sebelum upload baru. Ini berarti dataset lama dapat tercampur dengan hasil baru.

**Current Flow (BROKEN):**
```javascript
uploadFile: async (file) => {
    // NO CLEAR DATA CALL
    const response = await apiService.uploadDataset(file)
    dispatch({ type: ActionTypes.SET_RAW_DATA, ... })
}
```

**Expected Flow:**
```javascript
uploadFile: async (file) => {
    dispatch({ type: ActionTypes.CLEAR_DATA }) // Reset first
    const response = await apiService.uploadDataset(file)
    dispatch({ type: ActionTypes.SET_RAW_DATA, ... })
}
```

---

### 3.5 NO EXPERT VALIDATION ENDPOINT
**Metodologi:** Bagian 5.8 mendefinisikan Expert Validation sebagai validator sampel, bukan labeler seluruh dataset.

**ISSUE:** Tidak ada endpoint atau mekanisme untuk:
- Menyimpan hasil validasi pakar
- Menghitung agreement (Cohen's Kappa)
- Analisis pola kesalahan

**Current Status:** Fitur ini tidak diimplementasikan, hanya didokumentasikan sebagai "research validation process".

**FIX REQUIRED:** Implementasi tergantung kebutuhan. Jika validasi pakar adalah tahap interaktif dalam web, perlu endpoint baru. Jika hanya untuk penelitian offline, cukup dokumentasikan.

---

## 4. FINDINGS - HIGH PRIORITY ISSUES (P1)

### 4.1 DUPLICATE CONFIG IN BACKEND
**File:** `backend/config.py`

Konfigurasi di-define di `shared/constants.py` DAN di-backup di `backend/config.py` (lines 88-116). Ini adalah fallback yang baik, tetapi perlu verifikasi bahwa import selalu berhasil.

```python
try:
    from shared.constants import ...
    ASPECTS = _SHARED_ASPECTS  # Primary source
except ImportError:
    ASPECTS = [...]  # Fallback - should never happen
```

**Status:** ARCHITECTURALLY CORRECT. Fallback ada tetapi import dari shared adalah source of truth.

---

### 4.2 EXPERIMENT SCRIPTS STATUS
**Directory:** `scripts/`

| Script | Status | Notes |
|--------|--------|-------|
| experiment_6_1_ablation.py | NEEDS REVIEW | Ablation study |
| experiment_6_2_tfidf.py | NEEDS REVIEW | TF-IDF experiments |
| experiment_6_6_class_balance.py | NEEDS REVIEW | Class balancing |
| experiment_6_8_model_selection.py | NEEDS REVIEW | Model selection |
| build_lexicon.py | ACTIVE | Building lexicon |
| threshold_optimizer.py | ACTIVE | Threshold tuning |
| train_model.py | ACTIVE | Standalone training |
| test_svm.py, test_model.py | UTILITY | Testing utilities |

**ISSUE:** Tidak ada script yang jelas sebagai "master experiment" yang menghasilkan model final.

**FIX REQUIRED:** Jalankan `scripts/run_all_experiments.py` dan pastikan output model final di-copy ke `models/`.

---

### 4.3 SCRAPING NOT WEB-INTERACTIVE
**Metodologi:** Scraping dilakukan secara offline, bukan sebagai tahap interaktif web.

**Status:** CORRECTLY IMPLEMENTED. Scraping tidak ada di frontend/backend sebagai fitur interaktif.

---

### 4.4 INCONSISTENT DATA FLOW BETWEEN CSV AND DATABASE
**Metodologi:** CSV = source dataset penelitian, Database = persistence aplikasi.

**ISSUE:** `backend/app.py` membaca dari output files (`output/labeled_data_*.csv`) sementara `backend/enhanced_api.py` menyimpan ke database SQLite. Jika keduanya digunakan secara bersamaan, bisa terjadi inkonsistensi.

**FIX REQUIRED:**
1. Tentukan: Apakah web app harus membaca dari database atau CSV files?
2. Jika pipeline batch menghasilkan CSV, backend enhanced API harus sync ke CSV.
3. Atau: Pipeline batch harus menyimpan ke database.

---

### 4.5 LOGGING INSUFFICIENCY
**File:** `backend/app.py`, `backend/ml_engine.py`

**ISSUE:** Tidak ada logging yang konsisten untuk:
- Dataset version yang digunakan
- Pipeline stage progress
- Rows processed
- Model version yang di-load

**FIX REQUIRED:** Tambahkan structured logging dengan `logging` module.

---

## 5. FINDINGS - VERIFIED CORRECT

### 5.1 PREPROCESSING PIPELINE ✅
**File:** `shared/preprocessing.py`

**Verifikasi:**
- [x] 7 tahap dalam urutan yang benar
- [x] Cleaning → Case Folding → Tokenizing → Normalisasi → Negation → Stopword → Stemming
- [x] Negasi ditangani SEBELUM stopword removal
- [x] "tidak bagus" → "tidak_bagus" (tidak hilang)
- [x] Menggunakan Sastrawi untuk stemming

**Status:** CORRECTLY IMPLEMENTED

---

### 5.2 LEXICON LABELING ✅
**File:** `shared/lexicon.py`, `shared/constants.py`

**Verifikasi:**
- [x] NEUTRAL_THRESHOLD = 1.1030 konsisten
- [x] Skor dinormalisasi dengan jumlah kata
- [x] Aturan klasifikasi: >T = Positif, <-T = Negatif, lainnya = Netral
- [x] Negasi ditangani: tidak_X dibalik polaritasnya

**Status:** CORRECTLY IMPLEMENTED

---

### 5.3 ASPECT DETECTION ✅
**File:** `pipeline/aspect_detection.py`, `shared/constants.py`

**Verifikasi:**
- [x] 4 aspek: Kemudahan Akses, Gamifikasi, Materi/Konten, Performa Teknis
- [x] Rule-based keyword matching
- [x] Satu kalimat bisa punya >1 aspek
- [x] Kalimat tanpa aspek disimpan sebagai 'Umum'

**Status:** CORRECTLY IMPLEMENTED

---

### 5.4 TF-IDF LEAKAGE CHECK ✅
**File:** `pipeline/train_svm.py`

**Verifikasi:**
```python
# Lines 459-460
X_train = self.vectorizer.fit_transform(X_train_text)  # FIT on train
X_test = self.vectorizer.transform(X_test_text)        # TRANSFORM on test
```

**Status:** CORRECTLY IMPLEMENTED - No leakage

---

### 5.5 TRAIN/TEST SPLIT ✅
**File:** `pipeline/train_svm.py`, `pipeline/run_pipeline.py`

**Verifikasi:**
- [x] Stratified split berdasarkan sentiment_label
- [x] Random state = 42
- [x] Review-level split (tidak ada leakage per-aspect)

**Status:** CORRECTLY IMPLEMENTED

---

### 5.6 BACKEND ML ENGINE ORCHESTRATION ✅
**File:** `backend/ml_engine.py`

**Verifikasi:**
- [x] Import dari `shared/` dan `pipeline/`
- [x] Tidak ada duplicate preprocessing logic
- [x] Orchestration layer, bukan reimplementation

**Status:** CORRECTLY IMPLEMENTED

---

### 5.7 SVM CONFIGURATION ✅
**File:** `shared/constants.py`, `pipeline/train_svm.py`

**Verifikasi:**
- [x] Kernels: linear, rbf, poly, sigmoid
- [x] GridSearchCV dengan StratifiedKFold 5 folds
- [x] Scoring: f1_macro
- [x] Hyperparameters: C, gamma, degree

**Status:** CORRECTLY IMPLEMENTED

---

## 6. ARCHITECTURE BEFORE VS AFTER

### BEFORE
```
Pipeline → Model A (pipeline_output/stage6_model/)
                ↓
Backend → Model B (models/)
                ↓
Frontend → HARDCODED metrics (Training.jsx:40-45)
```

### AFTER (Target)
```
Pipeline → FINAL MODEL (pipeline_output/stage6_model/)
                ↓
        Copy to models/ with metadata
                ↓
Backend ← Load from models/ (verified pair)
                ↓
Frontend ← Fetch metrics from API (no hardcoded)
```

---

## 7. FILE CHANGES REQUIRED

### Files to MODIFY

| File | Change | Reason |
|------|--------|--------|
| `frontend/src/pages/Training.jsx` | Hapus hardcoded kernelResults | Menampilkan metrik fiktif |
| `frontend/src/context/AnalysisContext.jsx` | Reset state saat upload baru | Mencegah data mixing |
| `backend/app.py` `/stats` | Fetch metrik dari database atau file yang konsisten | inkonsistensi data |

### Files to CREATE

| File | Purpose |
|------|---------|
| `models/metadata_TEMPLATE.json` | Template untuk model metadata |
| `tests/test_preprocessing.py` | Unit test preprocessing |
| `tests/test_leakage.py` | Automated leakage test |
| `tests/test_end_to_end.py` | E2E pipeline test |

### Files to DEPRECATE/MARK

| File | Action | Reason |
|------|--------|--------|
| `scripts/experiment_*.py` | Document as experimental | Bukan pipeline utama |
| `models/svm_model_20260922_123552.pkl` | Mark as old | Versi lama |

---

## 8. RECOMMENDED FIXES

### P0 - Critical (Wajib segera diperbaiki)

1. **Hapus Hardcoded Metrics di Frontend**
   ```javascript
   // Training.jsx - Ganti baris 40-45 dengan:
   const kernelResults = state.kernelComparisonResults || [];
   // Atau hapus chart jika backend belum support
   ```

2. **Tentukan Source of Truth untuk Model**
   - Pilih: Apakah `models/` atau `pipeline_output/stage6_model/` yang authoritative?
   - Rekomendasi: `models/` adalah source of truth untuk web, `pipeline_output/` untuk penelitian
   - Tambahkan sync mechanism

3. **Fix Frontend State Reset**
   ```javascript
   // AnalysisContext.jsx - uploadFile function
   uploadFile: async (file) => {
       dispatch({ type: ActionTypes.CLEAR_DATA }) // ADD THIS
       // ... rest of function
   }
   ```

### P1 - High Priority

4. **Tambahkan Model Metadata**
   ```json
   // models/metadata_latest.json
   {
     "model_version": "v1.0",
     "training_timestamp": "2026-09-23T01:43:48",
     "dataset_version": "data_20260922",
     "dataset_source": "data/02_data_akhir.csv",
     "preprocessing_version": "v1",
     "config": {
       "kernel": "linear",
       "C": 10,
       "tfidf_max_features": 5000,
       "random_state": 42
     },
     "metrics": {...}
   }
   ```

5. **Verifikasi Experiment Scripts**
   - Jalankan `scripts/run_all_experiments.py`
   - Pastikan output konsisten dengan pipeline batch

### P2 - Medium Priority

6. **Logging Konsistensi**
7. **Database/CSV Sync**
8. **Unit Tests**

---

## 9. ACCEPTANCE CRITERIA STATUS

### Data ✅
- [x] Satu dataset utama (data/02_data_akhir.csv)
- [x] Pipeline output version jelas
- [ ] ❌ Dataset lama tidak digunakan diam-diam - BUTUH VERIFIKASI
- [ ] ❌ Duplicate handling konsisten - BUTUH VERIFIKASI

### Preprocessing ✅
- [x] Satu implementasi (shared/preprocessing.py)
- [x] Urutan 7 tahap benar
- [x] Negation aman
- [x] Prediction menggunakan preprocessing sama

### Labeling ✅
- [x] InSet Lexicon
- [x] Threshold satu sumber (1.1030)
- [x] Negation konsisten

### Aspect ✅
- [x] Empat aspek
- [x] Keyword matching
- [x] Satu implementasi

### ML ✅
- [x] Split sebelum TF-IDF fit
- [x] Tidak ada data leakage
- [x] TF-IDF train-only fit
- [x] Kernel comparison
- [x] GridSearchCV
- [x] Macro F1

### Model ❌
- [ ] Satu final model - BUTUH KONSOLIDASI
- [ ] Satu final vectorizer - BUTUH KONSOLIDASI
- [ ] Metadata model - BUTUH DIBUAT
- [x] Backend menggunakan model final
- [x] Prediction menggunakan model final

### Backend ✅
- [x] Endpoint konsisten
- [x] Tidak ada duplicate ML logic
- [ ] ❌ Error handling - BUTUH REVIEW
- [ ] ❌ Response schema konsisten - BUTUH REVIEW

### Frontend ❌
- [x] Tidak ada hardcoded hasil (kecuali Training.jsx)
- [x] Data berasal dari backend
- [ ] ❌ State reset - BUTUH FIX
- [x] Pipeline progress sesuai proses aktual

### Testing ❌
- [ ] Unit test - BUTUH DIBUAT
- [ ] Integration test - BUTUH DIBUAT
- [ ] API test - BUTUH DIBUAT
- [ ] E2E test - BUTUH DIBUAT

---

## 10. REMAINING ISSUES

| Issue | Severity | Impact | Action |
|-------|----------|--------|--------|
| Hardcoded metrics di Training.jsx | CRITICAL | Frontend menampilkan hasil fiktif | Perbaiki segera |
| Model inconsistency (2 model berbeda) | CRITICAL | Penelitian vs Web tidak konsisten | Konsolidasikan |
| Frontend state tidak reset | HIGH | Dataset mixing | Fix uploadFile |
| Missing model metadata | HIGH | Tidak ada versioning | Buat template |
| No unit tests | MEDIUM | Tidak ada regresi detection | Buat test files |
| Experiment scripts tidak terintegrasi | MEDIUM | Hasil eksperimen tidak sync | Review dan dokumentasi |
| Expert validation tidak ada | LOW | Fitur metodologi tidak lengkap | Implementasi jika diperlukan |

---

## 11. NEXT STEPS

1. **IMMEDIATE**: Fix hardcoded metrics di Training.jsx
2. **IMMEDIATE**: Fix frontend state reset
3. **HIGH**: Konsolidasikan model sources
4. **HIGH**: Tambahkan model metadata
5. **MEDIUM**: Buat unit tests
6. **MEDIUM**: Review error handling dan logging

---

*Report generated by Claude Code Audit*
*Date: 2026-09-23*
