# IMPLEMENTATION PLAN - Sentiment Analysis Wayground

**Based on:** AUDIT_REPORT.md
**Date:** 2026-09-23

---

## PHASE 1: CRITICAL FIXES (P0)

### Issue 1: Hardcoded Metrics in Training.jsx
**File:** `frontend/src/pages/Training.jsx`
**Lines:** 40-45

**Current Behavior:**
```javascript
const kernelResults = [
    { name: 'Linear', f1: state.modelMetrics?.f1_macro || 0.80, accuracy: state.modelMetrics?.accuracy || 0.852 },
    { name: 'RBF', f1: 0.72, accuracy: 0.798 },
    { name: 'Poly', f1: 0.70, accuracy: 0.775 },
    { name: 'Sigmoid', f1: 0.68, accuracy: 0.752 }
]
```

**Expected Behavior:**
Chart hanya menampilkan data jika ada hasil kernel comparison aktual dari backend. Jika tidak ada, tampilkan placeholder.

**Change Required:**
1. Hapus array hardcoded
2. Fetch kernel comparison results dari backend API
3. Jika tidak tersedia, tampilkan "Jalankan eksperimen kernel comparison terlebih dahulu"

**Risk:** Low
**Test Required:** Frontend loads, chart displays correctly with actual data

---

### Issue 2: Frontend State Not Resetting
**File:** `frontend/src/context/AnalysisContext.jsx`
**Lines:** 188-210 (uploadFile function)

**Current Behavior:**
Upload file baru tidak mereset state sebelumnya.

**Expected Behavior:**
Upload file baru → CLEAR_DATA → set new raw data

**Change Required:**
```javascript
uploadFile: async (file) => {
    dispatch({ type: ActionTypes.CLEAR_DATA }) // ADD THIS LINE
    dispatch({ type: ActionTypes.SET_LOADING, payload: { key: 'upload', value: true } })
    // ... rest unchanged
}
```

**Risk:** Low
**Test Required:** Upload dataset A, process, then upload dataset B - verify A's data is cleared

---

### Issue 3: Model Source Consolidation
**Files:** `models/`, `data/pipeline_output/stage6_model/`

**Current Behavior:**
Dua lokasi model berbeda dengan metrik berbeda:
- `models/metrics_latest.json` → accuracy: 0.8247
- `pipeline_report.txt` → accuracy: 0.9043

**Expected Behavior:**
Satu model final di `models/` dengan metadata yang lengkap.

**Change Required:**
1. Tentukan model mana yang akan digunakan (rekomendasi: gunakan yang dari pipeline karena lebih baru)
2. Copy model ke `models/` dengan metadata
3. Update pipeline untuk sync ke `models/` setiap selesai training

**Risk:** Medium (terkait model loading)
**Test Required:**
- Backend loads model correctly
- Prediction results match pipeline output
- Metrics displayed in frontend match saved metadata

---

## PHASE 2: HIGH PRIORITY FIXES (P1)

### Issue 4: Model Metadata
**File:** `models/` directory

**Current Behavior:**
Model files tidak memiliki metadata.

**Expected Behavior:**
Setiap model memiliki `metadata.json` yang berisi:
- model_version
- training_timestamp
- dataset_version
- preprocessing_version
- config
- metrics

**Change Required:**
1. Buat template metadata
2. Update `pipeline/run_pipeline.py` untuk generate metadata
3. Update `backend/ml_engine.py` untuk load dan verify metadata

**Risk:** Low
**Test Required:** Model loading includes metadata verification

---

### Issue 5: Backend Stats Consistency
**File:** `backend/app.py` `/stats` endpoint

**Current Behavior:**
`/stats` membaca dari file yang berbeda (`output/summary_*.json`) sementara enhanced API menggunakan database.

**Expected Behavior:**
Semua stats berasal dari satu sumber yang konsisten.

**Change Required:**
1. Pilih source: database atau files
2. Update `/stats` endpoint untuk menggunakan source yang sama dengan enhanced API
3. Pastikan data sync antara pipeline output dan database

**Risk:** Medium
**Test Required:** Stats displayed match actual data source

---

### Issue 6: Pipeline-to-Models Sync
**File:** `pipeline/run_pipeline.py`, `pipeline/train_svm.py`

**Current Behavior:**
Pipeline menyimpan model di `data/pipeline_output/stage6_model/` tetapi backend membaca dari `models/`.

**Expected Behavior:**
Pipeline auto-syncs final model ke `models/` setiap selesai training.

**Change Required:**
1. Di `run_pipeline.py`, setelah stage 6 (training), copy model ke `models/`
2. Generate metadata.json di `models/`
3. Update `_latest` files

**Risk:** Medium
**Test Required:** Pipeline run → model appears in `models/` → backend loads correctly

---

## PHASE 3: MEDIUM PRIORITY FIXES (P2)

### Issue 7: Unit Tests
**Files:** `tests/` directory (new)

**Required Tests:**
```
tests/
├── test_preprocessing.py
│   ├── test_cleaning()
│   ├── test_case_folding()
│   ├── test_tokenizing()
│   ├── test_normalization()
│   ├── test_negation_handling()
│   ├── test_stopword_removal()
│   ├── test_stemming()
│   └── test_full_pipeline()
├── test_lexicon.py
│   ├── test_positive_words()
│   ├── test_negative_words()
│   ├── test_negation_reversal()
│   └── test_threshold_classification()
├── test_aspect_detection.py
│   ├── test_four_aspects()
│   └── test_keyword_matching()
├── test_leakage.py
│   ├── test_tfidf_no_leakage()
│   └── test_split_consistency()
└── test_end_to_end.py
    └── test_pipeline_consistency()
```

**Risk:** Low
**Test Required:** All tests pass

---

### Issue 8: Error Handling Enhancement
**File:** `backend/app.py`, `backend/ml_engine.py`

**Current Behavior:**
Error handling ada tapi tidak konsisten.

**Expected Behavior:**
1. Consistent error response format
2. Logging yang cukup untuk debugging
3. Graceful degradation

**Change Required:**
1. Standardize error response format
2. Add logging module
3. Add try/catch yang informative

**Risk:** Low
**Test Required:** Error cases return proper error messages

---

## FILE CHANGES SUMMARY

### Files to MODIFY

| File | Priority | Changes |
|------|----------|---------|
| `frontend/src/pages/Training.jsx` | P0 | Hapus hardcoded kernelResults |
| `frontend/src/context/AnalysisContext.jsx` | P0 | Add CLEAR_DATA before upload |
| `pipeline/run_pipeline.py` | P1 | Add model sync to models/ |
| `pipeline/train_svm.py` | P1 | Generate metadata.json |
| `backend/ml_engine.py` | P1 | Verify metadata on load |
| `backend/app.py` | P1 | Consistent stats source |

### Files to CREATE

| File | Priority | Purpose |
|------|----------|---------|
| `models/metadata_TEMPLATE.json` | P1 | Metadata template |
| `tests/test_preprocessing.py` | P2 | Unit tests |
| `tests/test_lexicon.py` | P2 | Unit tests |
| `tests/test_aspect_detection.py` | P2 | Unit tests |
| `tests/test_leakage.py` | P2 | Leakage tests |

### Files to DOCUMENT

| File | Action | Reason |
|------|--------|--------|
| `scripts/experiment_*.py` | Document as experimental | Bukan pipeline utama |
| `models/svm_model_20260922_123552.pkl` | Mark as v0.1 | Versi lama |

---

## IMPLEMENTATION ORDER

```
1. FIX: Frontend state reset (P0)
   ↓
2. FIX: Hardcoded metrics (P0)
   ↓
3. CREATE: Metadata template (P1)
   ↓
4. FIX: Pipeline model sync (P1)
   ↓
5. FIX: Backend metadata verification (P1)
   ↓
6. FIX: Stats consistency (P1)
   ↓
7. CREATE: Unit tests (P2)
   ↓
8. VERIFY: End-to-end consistency (All)
```

---

## VALIDATION CHECKLIST

After implementation, verify:

- [ ] Upload dataset A → process → upload dataset B → A's data is gone
- [ ] Training page shows ACTUAL kernel comparison results (not hardcoded)
- [ ] Model in `models/` matches model in `pipeline_output/stage6_model/`
- [ ] Metadata.json exists and is valid
- [ ] Backend loads model with metadata verification
- [ ] Prediction results from web match prediction results from pipeline
- [ ] Unit tests pass
- [ ] No hardcoded accuracy/F1/confusion_matrix anywhere in frontend

---

*Implementation Plan generated by Claude Code Audit*
*Date: 2026-09-23*
