# Tahap 6 — Audit akhir, gate integritas, dan konsistensi aspek

Tanggal: 5 November 2026. Status: selesai untuk lingkup tahap ini.
Model final **tidak** dilatih ulang: `model_version` tetap `20261029_223454`,
dataset canonical tetap 3078 baris, konfigurasi dan hasil eksperimen tidak
berubah.

## 1. Audit: apa yang ditemukan

Audit dijalankan pada state workspace setelah Tahap 5. Temuan dibagi menurut
dampak.

| # | Prioritas | Temuan | Bukti |
|---|---|---|---|
| F1 | Kritis | **Aplikasi mati total.** `load_verified_pipeline()` menolak bundle canonical, sehingga `/api/v1/health`, `/config`, `/stats`, seluruh alur upload→prediksi, dan 27 test gagal (19 failed + 8 error) | `ValueError: Preprocessing/resources differ from training` pada `shared/deployment_integrity.py` |
| F2 | Kritis | **Fingerprint terlalu kasar.** Gate meng-hash seluruh isi 13 sumber daya, termasuk `shared/constants.py` dan `shared/lexicon.py` yang tidak menentukan perilaku inferensi. Perubahan komentar/path default/refactor pelabelan tetap mematikan backend | `runtime_fingerprint()` versi lama |
| F3 | Tinggi | **`NameError: name 'LEXICON' is not defined`.** Refactor lazy-load menghapus global `LEXICON`, tetapi `classify_sentiment()`, `calculate_sentiment_score()`, dan `get_label_distribution()` masih mereferensikannya. Jalur pelabelan default ikut mati | `tests/test_model.py`, `tests/test_leakage.py` |
| F4 | Tinggi | **Default lexicon masih v1 (InSet).** Jalur tanpa argumen `lexicon=` memakai kamus v1, berbeda dari jalur canonical v2 | `shared/lexicon.py` |
| F5 | Tinggi | **Keterangan aspek bertentangan dengan kode.** README, `PANDUAN_PERINTAH_TERMINAL.md`, TAHAP_05, dan dokumen keputusan D4 menyatakan "sentimen aspek mengikuti sentimen ulasan utuh", padahal backend menghitung sentimen aspek dari kalimat aspeknya sendiri dengan SVM canonical | `backend/services/aspect_service.py:121` vs `README.md:47` |
| F6 | Sedang | **Prediksi satu ulasan tidak menghitung sentimen aspek.** Endpoint `/predict-single` mendeteksi aspek, tetapi `sentiment` pada `aspect_details` dibiarkan `None`, berbeda dari alur batch | `backend/api/analysis.py` (fungsi `predict_single_review`) |
| F7 | Sedang | **Implementasi lexicon kedua.** `scripts/shared/lexicon.py` (321 baris) adalah duplikat `shared/lexicon.py` (494 baris) dan tidak diimpor siapa pun | pencarian `scripts.shared` → 0 hasil |
| F8 | Sedang | **Kata kunci aspek kotor.** 54 duplikat dalam 4 daftar, satu karakter non-Indonesia (`卡`), ejaan non-Indonesia (`aksés`, `accès`), typo (`ceptable`), dan singkatan terlalu umum (`to`, `NR`) | `shared/constants.py` |
| F9 | Rendah | **Konstanta usang.** `NEUTRAL_LEXICON_PATH` menunjuk `neutral_final.tsv` yang tidak pernah ada; komentar ambang netral menyebut "Nilai final: 0.0" padahal canonical 1.0 | `shared/constants.py` |
| F10 | Rendah | **Environment aktif tidak sesuai deklarasi.** `.venv` berjalan pada Python 3.14.5 + numpy 2.5.3 + pandas 3.0.6 + pytest 9.1.1, sedangkan `requirements.txt` mensyaratkan numpy <2.0, pandas <3.0, pytest <9 dan `requirements-lock.txt` mengunci Python 3.10.6 | `pip list` vs `requirements*.txt` |

F1–F4 adalah satu rantai: refactor pada `shared/constants.py` dan
`shared/lexicon.py` setelah training (4 November) membuat gate menolak
bundle, sementara refactor yang sama Freeze `LEXICON` sehingga jalur
pelabelan error.

## 2. Perbaikan

### 2.1 Gate integritas: fingerprint semantik (F1, F2)

`shared/deployment_integrity.py` kini membedakan dua lingkup sumber daya:

| Lingkup | Isi | Sifat |
|---|---|---|
| `runtime_fingerprint()` → `inference_fingerprint` | `shared/preprocessing.py`, `shared/research_config.py`, `data/slang_dict.json`, `data/normalization_dict.json`, dan **nilai** `NEGATION_WORDS`, `ADDITIONAL_STOPWORDS`, `SENTIMENT_PROTECTED_WORDS`, `PREPROCESSING_STAGES` | Memakai gate integritas |
| `research_fingerprint()` | `shared/labeling.py`, `shared/lexicon.py`, dan 6 file lexicon TSV | Audit/re-verifikasi, bukan syarat inferensi |

Konsekuensinya:

- Perubahan yang **tidak** menyentuh perilaku inferensi (komentar, path
  default, refactor pelabelan) tidak lagi mematikan backend.
- Perubahan yang **menyentuh** perilaku preprocessing tetap terdeteksi:
  gate menolak memuat model.
- Jalur path memakai separator POSIX, sehingga fingerprint tidak lagi
  berbeda antar platform.

### 2.2 Re-binding fingerprint dengan bukti empiris

Perubahan `models/final_v4/metadata.json` pada tahap ini **hanya metadata**:
model, dataset, label, konfigurasi, split, dan seluruh hasil evaluasi tetap
byte-identical (hash-nya masih terikat pada `artifact_hashes`).

Metadata sekarang memuat:

- `inference_fingerprint` — fingerprint semantik yang berlaku (dikunci gate).
- `research_fingerprint` — jejak jalur riset.
- `fingerprint_rebind` — alasan, daftar berkas yang berubah, dan bukti
  ekuivalensi.
- `runtime_fingerprint_at_training` — fingerprint lama (skema whole-file)
  dipertahankan sebagai jejak training.

**Bukti ekuivalensi** (dihitung oleh `scripts/verify_deployment_bundle.py`):

| Yang dibandingkan | Hasil |
|---|---|
| `processed_text` canonical vs output preprocessing source saat ini | 3078 / 3078 identik |
| `sentiment_label_inset` canonical vs pelabelan lexicon v2 saat ini | 3078 / 3078 identik |

`processed_text` pada dataset canonical persis menjadi input TF-IDF saat
training, sehingga kesamaan seluruh baris adalah bukti langsung bahwa
perilaku inferensi tidak berubah.

Perintah verifikasi (tidak mengubah berkas):

```powershell
python scripts/verify_deployment_bundle.py
```

### 2.3 Pelabelan (F3, F4)

- `shared/lexicon.py`: `LEXICON` global diganti accessor
  `get_default_lexicon()`; `__getattr__` tetap menyediakan `LEXICON` untuk
  kompatibilitas, tetapi sekarang mengembalikan lexicon **aktif**.
- Jalur tanpa argumen `lexicon=` memakai lexicon **v2** (bukan v1/InSet),
  sesuai aturan canonical.
- Resolusi versi lexicon dipindah ke satu fungsi:
  `shared/constants.resolve_lexicon_version()`. `shared/labeling.py` dan
  `shared/lexicon.py` memakai fungsi yang sama, sehingga tidak ada dua
  implementasi yang bisa berbeda.
- Label canonical tidak berubah: pelabelan dataset tetap 3078/3078 identik.

### 2.4 Analisis aspek (F5, F6)

- `backend/api/analysis.py` (`/predict-single`) sekarang menghitung
  sentimen per aspek dengan pipeline canonical yang sama, dan mengembalikan
  `aspect_sentiment_scope`.
- `backend/services/aspect_service.py` tidak berubah: definisi aspek tetap
  "sentimen aspek dihitung dari kalimat-kalimat aspek itu sendiri".
- Teks frontend dan dokumen yang bertentangan dikoreksi:
  `frontend/src/components/AnalysisViews.jsx`,
  `frontend/src/pages/PredictNew.jsx`, `README.md`,
  `PANDUAN_PERINTAH_TERMINAL.md`, `docs/remediation/TAHAP_05.md`, dan
  koreksi pada `docs/DECISIONS/D4-KONFLIK-KONFIGURASI.md`.

### 2.5 Kebersihan kode (F7, F8, F9)

- `scripts/shared/lexicon.py` diarsipkan ke
  `archive/remediation_2026-11-05/stage6/scripts/lexicon.py`; manifest di
  `TAHAP_06_ARCHIVE_MANIFEST.json`. Tidak ada import yang rusak.
- `ASPECT_KEYWORDS` dinormalisasi: 4 daftar tanpa duplikat, total 474 entri
  unik. Entri yang dibuang: `卡`, `aksés`, `accès`, `ceptable`, `to`, `NR`.
  Verifikasi: semua entri yang dibuang tidak pernah cocok pada dataset
  canonical (0 kemunculan), kecuali `to` (7 kemunculan) yang memang bukan
  penanda materi. Pencocokan aspek berbasis "cocok atau tidak", sehingga
  duplikat tidak pernah memengaruhi hasil.
- `NEUTRAL_LEXICON_PATH` (file yang tidak pernah ada) diganti
  `NEUTRAL_LEXICON_DISABLED`; komentar ambang netral dikoreksi ke 1.0.
- `scripts/train_final_model.py` dan `scripts/validate_final_artifact.py`
  memakai kunci fingerprint baru agar training berikutnya konsisten.

## 3. Verifikasi

| Pemeriksaan | Perintah | Hasil |
|---|---|---|
| Backend + riset | `python -m pytest -q` | **155 passed**, 0 failed (sebelum perbaikan: 19 failed + 8 error) |
| Frontend unit | `cd frontend` → `npm test` | **9 passed** |
| Build frontend | `cd frontend` → `npm run build` | sukses, 1480 modul, bundle `index-oknRj10m.js` |
| Gate integritas | `python scripts/verify_deployment_bundle.py` | preprocessing 3078/3078, pelabelan 3078/3078, bundle termuat |
| Validasi artefak | `python scripts/validate_final_artifact.py` | Bundle canonical lolos pemeriksaan konfigurasi, hash, dan fingerprint |

Metrik model **tidak berubah** karena tidak ada training ulang:

| Angka | Nilai |
|---|---|
| Accuracy (holdout diagnostik) | 84,27% (509/604) |
| Macro F1 (holdout diagnostik) | 79,02% |
| Macro F1 (grouped CV development) | 79,40% |
| F1 Negatif / Netral / Positif | 79,31% / 65,34% / 92,41% |
| Jarak generalisasi (train F1 99,04% → OOF 78,82%) | 20,22 poin persentase |

## 4. Batasan dan pekerjaan yang belum selesai

1. **Holdout tetap bukan test independen.** 527 dari 604 baris pernah masuk
   training/CV pada percobaan yang dibatalkan. Angka holdout adalah
   diagnostik.
2. **Label didominasi lexicon.** 2790 dari 3078 baris (90,6%) berlabel
   lexicon v2; 288 baris koreksi pakar; `second_annotator_rows = 0`, jadi
   tidak ada Cohen's kappa.
3. **Overfitting masih ada.** Jarak train → OOF 20,22 poin persentase;
   Eksperimen 7 bersifat diagnostik dan tidak dipilih.
4. **Environment aktif ≠ environment terkunci.** `.venv` sekarang Python
   3.14.5 + numpy 2.5.3 + pandas 3.0.6 + pytest 9.1.1, di luar rentang
   `requirements.txt`. Test lolos, tetapi **training ulang di environment ini
   tidak mereproduksi angka canonical**. Untuk reproducibility penuh:
   `python -m venv .venv310` → `pip install -r requirements-lock.txt` →
   `.venv310\Scripts\python scripts\train_final_model.py`.
5. **Fingerprint lama tidak dapat direkonstruksi.** Source saat training
   (29 Oktober 22:34) tidak pernah di-commit dan tidak ada di arsip, jadi
   ekuivalensi dibuktikan secara empiris (reproduksi 3078/3078), bukan lewat
   pencocokan hash source.
6. **Visualisasi offline dan arsip artefak lama** (lingkup asli tahap 6)
   belum dirapikan. Dua plot terverifikasi tetap tersedia di
   `output_v4/verified_visualizations/`.
7. **Skrip eksperimen tidak dijalankan ulang** pada tahap ini. Eksperimen 1–7
   memakai konfigurasi canonical dan artefaknya tidak berubah; hanya
   artefak analisis aspek yang disegarkan setelah normalisasi kata kunci
   aspek.

## 5. Berkas yang berubah

| Berkas | Perubahan |
|---|---|
| `shared/lexicon.py` | Perbaikan `NameError LEXICON`; default lexicon v2; accessor `get_default_lexicon()` |
| `shared/labeling.py` | Pakai `constants.resolve_lexicon_version()`; hapus import `os` |
| `shared/constants.py` | Fingerprint-friendly: `resolve_lexicon_version()`; normalisasi `ASPECT_KEYWORDS`; hapus `NEUTRAL_LEXICON_PATH`; perbaiki komentar ambang netral |
| `shared/deployment_integrity.py` | Fingerprint semantik + `research_fingerprint()` + `effective_fingerprint()` + `legacy_runtime_fingerprint()` |
| `backend/api/analysis.py` | Sentimen aspek pada `/predict-single`; `effective_fingerprint()` |
| `backend/api/web_flow.py` | `effective_fingerprint()` |
| `backend/services/aspect_service.py` | Skope sentimen aspek (dokumentasi, tanpa perubahan perilaku) |
| `frontend/src/components/AnalysisViews.jsx` | Teks dan caption aspek sesuai cara kerja model |
| `frontend/src/pages/PredictNew.jsx` | Teks prediksi satu ulasan sesuai cara kerja model |
| `frontend/tests/analysis.test.mjs` | Assertion teks aspek diperbarui |
| `models/final_v4/metadata.json` | `inference_fingerprint`, `research_fingerprint`, `fingerprint_rebind`, `runtime_fingerprint_at_training` |
| `scripts/verify_deployment_bundle.py` | **Baru.** Verifikasi + re-binding fingerprint dengan bukti ekuivalensi |
| `scripts/train_final_model.py` | Menulis kunci fingerprint baru |
| `scripts/validate_final_artifact.py` | Memeriksa `inference_fingerprint` |
| `scripts/shared/lexicon.py` | Diarsipkan (duplikat, tidak terpakai) |
| `README.md`, `PANDUAN_PERINTAH_TERMINAL.md`, `docs/remediation/TAHAP_05.md`, `docs/DECISIONS/D4-KONFLIK-KONFIGURASI.md` | Koreksi keterangan aspek |
