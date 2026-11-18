# D4 — Keputusan Penguncian Konfigurasi Final

Status: **LOCKED** · Tanggal: 2026-11-02 · Sumber: audit artefak read-only

Dokumen ini mengunci canonical dataset, konfigurasi model, dan preprocessing.
Tujuannya satu: memastikan **riset → model → backend → frontend → laporan**
menggunakan sumber data, preprocessing, model, dan konfigurasi yang sama.

---

## 1. Konflik yang ditemukan

Spesifikasi brief yang diberikan dan kondisi repository **tidak sama**._sources
berbeda. Konflik ditelusuri ke artefak aslinya, bukan ditebak.

| # | Konflik | Nilai dalam brief | Nilai di artefak | Sumber artefak |
|---|---|---|---|---|
| L1 | Path dataset canonical | `data/pipeline_output/stage3_final_labeled.csv` | `data/pipeline_output_v4/stage3_final_labeled.csv` | `shared/final_config.py:42` |
| L2 | Total baris dataset | 3065 | 3078 | `selected_configuration.json` → `dataset.rows` |
| L3 | Development / Holdout | 2452 / 613 | 2474 / 604 | `selected_configuration.json` → `split` |
| L4 | `max_features` | 3000 | 5000 | `experiment_3_tfidf_results.json` → `best` |
| L5 | `ngram_range` | (1, 2) | (1, 3) | idem |
| L6 | `sublinear_tf` | True | False | idem |
| L7 | SVM `C` | 10 | 10 (v4) / 1.5 (v1) | `experiment_4_svm_results.json` |
| L8 | Path artifact model | `models/final/final_pipeline.joblib` | `models/final_v4/final_pipeline.joblib` | `shared/final_config.py:146` |
| L9 | Versi lexicon | InSet | v2 | `shared/constants.py:335` |
| L10 | Nama file metadata | `final_config.json` / `final_metrics.json` / `model_metadata.json` | `metadata.json` | `models/final_v4/` |
| L11 | Lokasi skrip validasi | `scripts/validate_*.py` | belum ada | — |

---

## 2. Dari mana angka "3065 / 2452 / 613" berasal

Angka itu **tidak hilang** — ia masih ada, di artefak tertua:

`models/final/final_model_metadata.json` → `model_version: 20261018_032439`
(18 Okt 2026), berisi `total_samples: 3065`, `development: 2452`, `holdout: 613`,
TF-IDF `3000 / (1,2) / sublinear_tf=true`, SVM `linear C=10`, dan metrik
`accuracy 0.8874 / f1_macro 0.8328`.

Artinya brief ini ditulis terhadap **state 18 Okt 2026**. Repo sudah bergerak
jauh sejak itu melalui tiga revisi berurutan:

```
18 Okt 2026  models/final/final_model_metadata.json   3065 / 2452 / 613
             3000 / (1,2) / sublinear=True / C=10
                    │
27 Okt 2026  models/final/metadata.json                3060 / 2448 / 612
             3000 / (1,1) / sublinear=True / C=1.5      ← dataset & config BERUBAH
                    │
             data/pipeline_output_v2/                   3082 baris
             data/pipeline_output_v3/                   3078 baris (sha 2af61747…)
                    │
29 Okt 2026  models/final_v4/  ← CANONICAL             3078 / 2474 / 604
             5000 / (1,3) / sublinear=False / C=10
```

Perubahan jumlah baris 3065 → 3060 → 3082 → 3078 berasal dari: filter periode
penelitian, deduplikasi konten, pelabelan ulang dengan lexicon v2, dan
integrasi label pakar (288 baris). `data/pipeline_output_v4/dataset_manifest.json`
mencatat jejaknya: `source 3119 → in_period_unique_content 3115 →
empty_processed 37 → duplicate_content 4 → final 3078`.

**Yang 3119 itu memang data scraping terbaru** — sesuai brief §3, ia diperlakukan
sebagai *raw input*, bukan dataset final. Konsisten.

---

## 3. Keputusan

### 3.1 Canonical dataset → `data/pipeline_output_v4/stage3_final_labeled.csv`

Alasannya, bukan karena "terbaru":

1. **Hanya chain v4 yang punya run eksperimen lengkap dan terverifikasi.**
   `output_v4/research_run/run_manifest.json` berstatus `complete`, dan
   `shared/deployment_integrity.py:verify_selection_contract()` memverifikasi
   sha256 **7 artefak eksperimen** + sha256 dataset + sha256 run manifest.
   Verifikasi ini **PASS** saat dijalankan.
2. **Chain v1 (`models/final/`) tidak konsisten dengan dirinya sendiri.**
   `final_model_metadata.json` (18 Okt) menyebut C=10 & 3065 baris, sedangkan
   `metadata.json` (27 Okt) menyebut C=1.5 & 3060 baris. Dua "final" di satu
   folder → tidak sah sebagai sumber kebenaran.
3. **Chain v1 tidak punya** `dataset_manifest.json`, `split_metadata.json`,
   maupun `selected_configuration.json` — tidak ada jejak audit.
4. Chain v3 punya 3078 baris **tetapi sha256 berbeda** (`2af61747…` vs
   `b25b6c29…`): isi berbeda dengan jumlah sama. Hanya v4 yang dikunci hash.

### 3.2 Konfigurasi final → hasil Eksperimen 3–6

| Komponen | Nilai | Pemenang |
|---|---|---|
| TF-IDF | `max_features=5000`, `ngram_range=(1,3)`, `sublinear_tf=False`, `min_df=1`, `max_df=0.90` | Exp 3, `cv_f1_mean 0.79299` |
| SVM | `kernel=linear`, `C=10.0`, `gamma=scale`, `class_weight=None` | Exp 4 |
| Multiclass | `One-vs-One` | Exp 6 |
| Imbalance | `none` | Exp 5 |
| Threshold netral | `1.0` | Exp 2, dipilih di development |

**Konfigurasi pada brief §6 sudah diuji dan kalah.** Di `experiment_3_tfidf_results.json`
kandidat `max_features=3000, ngram_range=(1,2), sublinear_tf=True, min_df=1, max_df=0.9`
mencatat `cv_f1_mean = 0.78260`, di bawah pemenang `0.79299`. Mengganti ke
konfigurasi brief justru **menurunkan** performa validasi. Sesuai aturan
"jangan mengganti konfigurasi hanya karena konfigurasi lain menghasilkan metrik
lebih tinggi" dan "jangan mengejar metrik yang lebih tinggi", konfigurasi
artefak **dipertahankan**.

### 3.3 Artifact → `models/final_v4/final_pipeline.joblib`

`models/final/` (v1) **diarsipkan**, tidak dihapus, karena
`final_model_metadata.json` di sana masih dipakai sebagai jejak provenance.
Backend, skrip training, dan API sudahwmengunjuk ke `final_v4` melalui
`shared/final_config.py` — tidak ada kode yang perlu diubah.

### 3.4 Preprocessing → `shared/preprocessing.py` (7 tahap, satu implementasi)

Urutan terkunci: Cleaning → Case Folding → Tokenizing → Normalization →
Negation Handling → Stopword Removal → Stemming.

**Berkas ini tidak boleh diubah** selama `runtime_fingerprint` masih mengunci
13 sumber daya. Perubahan satu karakter → `load_verified_pipeline()` menolak
memuat model → backend mati. Jika preprocessing harus berubah, seluruh rantai
harus diulang: eksperimen 3–7 → select → lock → train → evaluate.

### 3.5 Lexicon → v2 (bukan InSet)

`shared/constants.py` `ACTIVE_LEXICON = 'INSET'` adalah sisa usang dari
state lama dan sudah tidak dipakai di jalur mana pun (labeling lewat
`shared/labeling.py` → `get_active_lexicon()` → `DEFAULT_LEXICON_VERSION="v2"`).
Nilai itu harus dibersihkan menjadi v2 atau dihapus, **tapi tidak boleh
mengubah perilaku pelabelan** — memperbaiki komentar, bukan logika.

---

## 4. Batas validitas yang harus tertulis di BAB III & BAB IV

Hal-hal berikut **tidak boleh hilang** dari penulisan skripsi, karena sudah
tercatat di artefak:

1. **Holdout bukan test independen.** 527 dari 604 baris holdout canonical
   pernah masuk training/CV pada percobaan yang dibatalkan
   (`output_v4/research_run/invalidated_split_attempt/invalidation.json`).
   `holdout_independence_verified = false`. Angka holdout adalah **diagnostik**.
   Estimasi utama = grouped CV pada development (`f1_macro 0.79399`).
2. **Label bukan gold label.** 2790 dari 3078 baris (90,6%) label lexicon v2;
   288 baris koreksiexpert; `expert_independence_verified = false` untuk
   seluruh dataset; `second_annotator_rows = 0` → tidak ada Cohen's kappa.
   Metrik mengukur konsistensi model terhadap lexicon, bukan kebenaran.
3. **Eksperimen 7 adalah diagnostik, bukan selection.** `gap_pass = false`;
   gap train/CV = 19,64 poin persentase; `recommended_gap_compliant_candidate = null`.
   Model final berasal dari Eksperimen 3–6. Overfitting ini **dokumentasikan,
   jangan diperbaiki dengan menurunkan target**.
4. **Aspek pada aplikasi web bukan model per aspek.** Analisis aspek riset
   memakai SVM+TF-IDF terpisah per aspek; web memakai label tingkat ulasan
   (`sentiment_scope: review_level_sentiment_on_keyword_aspects`). Kedua hal
   ini harus dinyatakan terpisah di laporan.

   > **KOREKSI 5 November 2026 (Tahap 6).** Butir ini tidak lagi sesuai dengan
   > kode. Backend web (`backend/services/aspect_service.py`) menghitung
   > sentimen setiap aspek **dari kalimat-kalimat aspek itu sendiri** memakai
   > artefak SVM canonical yang sama; tidak ada model kedua. Yang berbasis
   > kata kunci hanya **deteksi** aspek (kalimat mana yang milik aspek mana).
   > Skope yang dikembalikan API adalah
   > `sentiment_scope: per_aspect_sentiment_from_aspect_sentences`.
   > Klaim "web memakai label tingkat ulasan" harus dihapus dari laporan dan
   > diganti dengan uraian di atas. Lihat `docs/remediation/TAHAP_06.md`.
5. **Pencarian eksperimen bersifat sequential**, bukan nested CV global
   (`config_source: sequential_experiment_winners_3_to_6`).

---

## 5. Artefak yang harus mengikat keputusan ini

| Artefak | Fungsi |
|---|---|
| `shared/final_config.py` | Sumber konfigurasi tunggal (dipakai model, backend, skrip) |
| `output_v4/experiments/selected_configuration.json` | Keputusan pemilihan eksperimen |
| `output_v4/experiments/best_config.json` | Snapshot konfigurasi terkunci |
| `models/final_v4/metadata.json` | Binding model → konfigurasi → dataset → evaluasi |
| `data/pipeline_output_v4/dataset_manifest.json` | Provenance dataset + hash keluaran |
| `data/pipeline_output_v4/stage5_split_v2/split_metadata.json` | Provenance split |
| `docs/DECISIONS/D4-KONFLIK-KONFIGURASI.md` | Dokumen ini |

Semua ikatan itu diverifikasi otomatis oleh:
- `scripts/validate_pipeline_consistency.py`
- `scripts/validate_final_artifact.py`
- `shared/deployment_integrity.py` (dipanggil setiap kali model dimuat)

---

## 6. Yang TIDAK berubah oleh keputusan ini

Tidak ada dataset, label, preprocessing, konfigurasi model, maupun hasil
eksperimen yang diubah. Keputusan ini hanya **mendokumentasikan apa yang
sudah benar-benar terkunci** dan **membuang ambiguitas** antara spesifikasi
brief lama dan state repository.
