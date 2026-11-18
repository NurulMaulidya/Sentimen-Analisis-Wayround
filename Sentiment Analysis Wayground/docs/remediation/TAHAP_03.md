# Tahap 3 — tujuh eksperimen dan seleksi konfigurasi penelitian

Tanggal: 29 Oktober 2026. Status: **tujuh eksperimen selesai; konfigurasi penelitian terpilih, belum dipromosikan ke deployment**.

## Protokol seleksi

Urutan eksekusi adalah **2 → snapshot penerus → 1 → 3 → 4 → 5 → 6 → 7**. Eksperimen threshold harus mendahului eksperimen model agar SVM mempelajari label yang telah dipilih dengan protokol development. Eksperimen 1 bersifat diagnostik dan tidak mengganti pembagian canonical setelah melihat hasilnya.

Namespace hasil baru adalah `output_v4/`. Dataset persiapan tetap `data/pipeline_output_v3`; dataset yang digunakan eksperimen model adalah `data/pipeline_output_v4`. Model deployment tidak dilatih atau diganti pada tahap ini.

Metrik seleksi adalah mean Macro F1 pada StratifiedGroupKFold 5 fold, grup = processed_text, random_state=42. Simpangan baku diseragamkan menggunakan **sample SD, ddof=1**. TF-IDF dan resampling berada di dalam pipeline setiap fold. Ties dipecahkan menurut SD dan kesederhanaan sebagaimana aturan masing-masing eksperimen; tidak ada perubahan pemenang berdasarkan metrik holdout.

Ini merupakan pencarian bertahap, bukan pencarian gabungan semua parameter atau estimasi generalisasi nested CV. Nilai CV yang dipakai untuk memilih model dapat optimistis akibat seleksi berulang. Mayoritas label dataset berasal dari lexicon, dan independensi anotasi/holdout terhadap pengembangan lexicon historis belum terbukti.

## Perbaikan kode sebelum run

| Komponen | Perbaikan |
|---|---|
| Experiment 1 | Baseline diagnostik eksplisit di scripts/experiments/baseline_config.json, tanpa dependency melingkar terhadap model final; sensitivitas rasio dan CV hanya di development dengan grup teks |
| Experiment 3 | Pertahankan reviewId saat loading; ukuran data/split dibaca dari manifest snapshot, bukan final_config model versi lama |
| Experiment 4 | C diperluas sampai 5 dan 10 agar baseline C=10 pada eksperimen TF-IDF tercakup; total 128 konfigurasi, bukan 96 |
| Experiment 5–7 | SMOTE menggunakan random_state=42 dan k_neighbors=3 secara konsisten |
| Experiment 3, 4, 7 | SD disamakan dengan eksperimen 5–6 menggunakan ddof=1 |
| shared/experiment_config_loader.py | Gunakan nama canonical dalam namespace output; verifikasi dependency eksperimen 5 dari struktur artifact_chain yang benar |
| shared/experiment_contract.py | Validasi hash dataset/split sebelum menerima ukuran snapshot |
| scripts/experiments/common.py | Loader delimiter/BOM bersama mempertahankan reviewId; snapshot berversi menolak fallback split yang dihitung ulang |
| scripts/derive_threshold_snapshot.py | Turunkan label dengan threshold terpilih tanpa memproses ulang teks atau membagi ulang ID |
| scripts/run_research_experiments.py | Entry point tujuh eksperimen; log terpisah, manifest status dan checksum dataset/split/dependency tiap eksperimen |
| scripts/select_research_configuration.py | Periksa kelengkapan tujuh eksperimen, identitas artefak dan dependency, lalu simpan pilihan penelitian tanpa mempromosikan model |

Baseline diagnostik eksperimen 1 adalah TF-IDF 2.000 unigram dan linear SVM C=1. Baseline eksperimen 3 adalah linear SVM C=10 untuk membandingkan konfigurasi TF-IDF. Keduanya ditetapkan sebelum melihat hasil, berperan berbeda, dan bukan model final. Eksperimen 4 kemudian mengoptimalkan SVM dengan TF-IDF pemenang eksperimen 3.

Pemenang eksperimen 3 dari 120 konfigurasi adalah max_features=5.000, ngram_range=(1,3), sublinear_tf=False, min_df=1, max_df=0,9. Mean grouped-CV Macro F1 pada baseline C=10 adalah **0,792991**, sample SD **0,018005**, vocabulary aktual development 5.000. Angka ini bukan hasil final model atau holdout.

## Threshold dan snapshot penerus

Eksperimen 2 memakai **234 ulasan pakar yang cocok di development** snapshot v3. Kandidat: 0; 0,1; 0,2; 0,3; 0,5; 1; 1,5; 2. Threshold **1,0** dipilih dengan Macro F1 0,636529, mengungguli baseline 0,5 dengan Macro F1 0,610650. Threshold 1,5 seri dengan 1,0; aturan memilih threshold lebih rendah. Accuracy pada threshold terpilih sekitar 81,62%, sehingga keputusan mengikuti Macro F1, bukan accuracy tertinggi.

Snapshot v4 menurunkan ulang label otomatis dari teks v3. Seluruh 288 label pakar tetap dipertahankan; 179 label final otomatis berubah. Tidak ada perubahan processed_text dan reviewId. Split diwarisi **persis** dari v3: 2.474 development dan 604 holdout. Grup yang sama tidak muncul di kedua bagian.

| Label final v4 | Jumlah |
|---|---:|
| Positif | 1.947 |
| Negatif | 575 |
| Netral | 556 |
| Total | 3.078 |

Manifest v4 mencatat hash manifest induk, hash hasil eksperimen threshold, kode derivasi dan seluruh artefak turunannya. File data v3 dan sumber anotasi tidak ditimpa. Perubahan proporsi kelas pada holdout merupakan akibat penerapan aturan label terpilih; metrik prediksi holdout tidak dihitung dan tidak dipakai memilih threshold.

## Arti eksperimen rasio split

Rasio 70:30, 80:20 dan 90:10 diuji pada **development** yang sudah terpisah dari canonical holdout. Bagian yang tidak dipakai dari development tidak dievaluasi sebagai test set. CV di subset training juga berkelompok.

Hasil ini menjawab sensitivitas terhadap jumlah contoh development yang tersedia; bukan perbandingan tiga test set independen. Split final sekitar 80:20 merupakan keputusan protokol sebelum seleksi model, dan tidak diganti dengan rasio diagnostik yang skornya tertinggi.

| Rasio nominal dalam development | Baris training aktual / 2.474 | Proporsi aktual | Mean CV Macro F1 | Sample SD |
|---|---:|---:|---:|---:|
| 70:30 | 1.680 | 67,91% | 0,722945 | 0,041259 |
| 80:20 | 1.963 | 79,35% | 0,736919 | 0,032005 |
| 90:10 | 2.226 | 89,98% | 0,744269 | 0,020595 |

Rasio di metadata adalah target nominal, bukan jaminan proporsi baris persis. Helper StratifiedGroupKFold menggunakan sekitar 1/test_size fold; untuk target 30% jumlah fold dibulatkan menjadi 3, sehingga subset yang diuji mendekati dua pertiga development. Angka aktual di atas harus ikut dilaporkan; hasil pertama tidak boleh diklaim sebagai pengukuran tepat 70% baris. Diagnostik ini tidak menentukan split canonical atau pemenang model.

## Menjalankan ulang

Run yang dijalankan pada tahap ini:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -u scripts/run_research_experiments.py
```

Runner menyimpan log di `output_v4/research_run/experiment_1.txt` sampai `experiment_7.txt`, serta `run_manifest.json`. Nomor urutan dan hasil parsing setiap artefak dicatat. Namespace run yang sudah memiliki manifest ditolak agar riwayat tidak ditimpa; gunakan `--output output_nama_baru` untuk eksekusi baru. Jika keputusan threshold berubah, gunakan pula `--selected data/nama_snapshot_baru`.

Run pertama berhenti sebelum menghitung skor eksperimen 4 karena validator search space masih berisi angka lama 96 setelah grid diperluas menjadi 128. Validator diperbaiki dan diberi tes regresi. Hasil eksperimen 1–3 tidak dibuang. Resume memverifikasi checksum script, hasil, dataset dan split dari tahap yang sudah selesai; manifest kegagalan disimpan di `research_run/run_manifest_attempt_1.json`. Log percobaan pertama tetap ada pada `experiment_4.txt`; percobaan berhasil dilanjutkan melalui `experiment_4_attempt_2.txt`.

Perintah pemulihan yang dijalankan:

```powershell
python -u scripts/run_research_experiments.py --resume
```

Resume hanya menerima run berstatus failed dengan artefak selesai yang tetap cocok checksum. Run selesai tidak boleh ditimpa melalui resume.

### Pembatalan percobaan akibat BOM dan paparan holdout

Pemeriksaan silang menemukan skor baseline none eksperimen 5 tidak mereproduksi skor konfigurasi identik eksperimen 4. Penyebabnya: loader bersama membaca BOM sebagai bagian nama `\ufeffreviewId`, sehingga ID tidak dikenali dan split dihitung ulang menjadi 2.467/611. Percobaan awal eksperimen 1, 5, 6 dan dependensi parsial 7 dibatalkan. File hasil, log dan manifest sebelumnya disimpan pada `output_v4/research_run/invalidated_split_attempt/`.

Loader bersama kemudian diarahkan ke shared.dataset_io. Snapshot berversi sekarang menolak fallback recomputed pada jalur model bersama, dan tes BOM ditambahkan. Eksperimen terdampak dijalankan ulang; hasil eksperimen 3–4 tetap dipakai karena loader masing-masing sudah mempertahankan reviewId dan menggunakan 2.474/604. Pemilihan threshold 2 diperiksa ulang lewat kesamaan himpunan reviewId: split yang sempat dihitung ulang pada snapshot persiapan v3 identik dengan split tersimpannya, sehingga hasil threshold tetap sah untuk partition development yang ditetapkan.

Rekonstruksi tanpa melatih model lagi menemukan **527 dari 604 reviewId holdout canonical sempat masuk training/CV pada percobaan yang dibatalkan**. Skor percobaan batal tidak dipakai dalam seleksi final. Namun paparan tersebut tidak dapat dihapus dari sejarah: holdout 604 baris tidak boleh diklaim pristine atau sepenuhnya belum pernah terlihat dalam pengembangan. Manifest mencatat daftar ID dan status `test_set_pristine=false`. Tersisa 77 baris canonical holdout yang tidak masuk training pada percobaan salah split itu, tetapi independensi dari pengembangan lexicon historis tetap belum terverifikasi.

Evaluasi tahap 4 harus menyebut keterbatasan ini secara eksplisit. Klaim generalisasi independen memerlukan data uji baru yang tidak terpapar pengembangan dan anotasi manusia yang memadai; angka dari holdout ini hanya boleh dilaporkan dengan provenance dan batasan tersebut.

Setelah seluruh eksperimen selesai, validasi dan pencatatan pilihan dilakukan dengan:

```powershell
python scripts/select_research_configuration.py --output output_v4
```

Keluaran adalah `output_v4/experiments/selected_configuration.json`. Ini keputusan penelitian **pending deployment**, bukan bukti bahwa aplikasi sudah memakai model tersebut. `shared/final_config.py`, snapshot best_config, binary model dan metadata deployment diselaraskan bersama pada tahap 4.

## Verifikasi kode

```powershell
python -m pytest tests/test_preprocessing.py tests/test_research_experiments.py tests/test_dataset_preparation.py tests/test_artifact_contracts.py tests/test_split_leakage.py -q -p no:cacheprovider
```

Hasil: **82 passed, 0 failed**. Satu warning pandas berasal dari tes yang sengaja memberi tanggal invalid; input itu tetap ditolak. Tes tambahan memeriksa bahwa perubahan split merusak validasi hash, eksperimen rasio tidak menerima contoh holdout, reviewId tidak hilang pada loader TF-IDF maupun loader bersama dengan BOM, parameter SMOTE tidak berubah antar-eksperimen, run yang belum lengkap tidak bisa dinyatakan final, jumlah grid SVM sesuai, dan resume ditolak bila hasil sebelumnya berubah.

Validasi akhir juga berhasil memeriksa checksum semua hasil dan script eksperimen, dependency antareksperimen, seluruh output snapshot, pemuatan split aktual lewat loader bersama (`stage5_files`, 2.474/604), dan overlap nol pada lima fold CV. ID split mencakup seluruh dataset. Eksperimen 7 pada C referensi mereproduksi mean Macro F1 eksperimen 6. Dataset lama dan binary model aktif memiliki checksum yang sama dengan catatan tahap 1.

Bukti rinci tersimpan di `docs/remediation/TAHAP_03_VERIFICATION.json`. Model deployment tidak dilatih ulang, dan prediksi holdout tidak dievaluasi sebagai final test pada tahap ini.

## Hasil tujuh eksperimen yang dipakai

Hasil pada direktori `invalidated_split_attempt` tidak termasuk tabel keputusan berikut.

| Eksperimen | Keputusan / hasil | Mean Macro F1 | Sample SD |
|---|---|---:|---:|
| 1 — sensitivitas jumlah development | Rasio nominal 90:10 tertinggi secara diagnostik; canonical split tetap | 0,744269 | 0,020595 |
| 2 — threshold | 1,0 pada 234 contoh pakar development | 0,636529 | Tidak memakai CV SVM |
| 3 — TF-IDF | 5.000 fitur, n-gram 1–3, sublinear=False, min_df=1 | 0,792991 | 0,018005 |
| 4 — SVM | Linear, C=10, gamma=scale | 0,792991 | 0,018005 |
| 5 — imbalance | none | 0,792991 | 0,018005 |
| 6 — multiclass | OVO | **0,793989** | **0,018218** |
| 7 — regularisasi diagnostik | C=10 tertinggi, tetapi gap besar; konfigurasi tidak diganti otomatis | 0,793989 | 0,018218 |

Pembanding eksperimen 5: class_weight_balanced 0,788685; random_over 0,785088; SMOTE 0,771971; random_under 0,741218. Karena none mendapat mean Macro F1 tertinggi pada dataset dan split yang sama, keputusan tanpa resampling sekarang memiliki bukti langsung yang konsisten. Tidak ada override berdasarkan McNemar atau kebutuhan visualisasi.

Pembanding eksperimen 6: OVR 0,779124 ± 0,019528. OVO dipilih sesuai aturan mean CV Macro F1; ini menggantikan keputusan sementara OVR dari rantai versi terdahulu, bukan mengganti binary aplikasi pada tahap ini.

### Konfigurasi penelitian terpilih

```json
{
  "labeling": {"lexicon_version": "v2", "neutral_threshold": 1.0},
  "model": {
    "tfidf": {
      "max_features": 5000,
      "ngram_range": [1, 3],
      "sublinear_tf": false,
      "min_df": 1,
      "max_df": 0.9
    },
    "svm": {"kernel": "linear", "C": 10.0, "gamma": "scale", "class_weight": null},
    "imbalance": {"method": "none"},
    "multiclass": {"strategy": "ovo"}
  }
}
```

Sumber keputusan lengkap adalah `output_v4/experiments/selected_configuration.json`, berstatus `selected_pending_deployment`. File itu memuat identitas dataset, hasil fold, versi library, hash artefak dan keterbatasan. max_features merupakan batas vocabulary; pada konfigurasi ini vocabulary aktual development mencapai 5.000. Gamma dicatat untuk kelengkapan konfigurasi dan tidak memengaruhi kernel linear.

### Pembacaan metrik dan gap

Mean Macro F1 lima fold = **79,40%**, sample SD sekitar **1,82 poin persentase**. Jika seluruh prediksi out-of-fold digabung, Macro F1 menjadi 79,65% dan accuracy 85,61%. Perbedaan mean fold dan metrik pooled wajar karena ukuran fold berbeda dan F1 bukan fungsi linear; keduanya diberi nama berbeda dalam hasil JSON.

| Kelas | Pooled OOF F1 | Recall | Support development |
|---|---:|---:|---:|
| Negatif | 0,804989 | 0,776805 | 457 |
| Netral | 0,661309 | 0,659039 | 437 |
| Positif | 0,923318 | 0,933544 | 1.580 |

Eksperimen 7 merekam training Macro F1 **0,990416** dan CV **0,793989**, sehingga gap **19,64 poin persentase**. Tidak ada kandidat yang memenuhi gap <3 poin sekaligus F1 setidaknya 90% dari skor terbaik. Statusnya `NO_GAP_COMPLIANT_CANDIDATE_WITH_COMPETITIVE_F1`. Gap tersebut belum terselesaikan dan tidak boleh disamarkan dengan memilih model yang tidak belajar hanya agar gap kecil.

C=10 berada di batas atas grid yang diuji; optimum di luar grid belum dibuktikan. Konfigurasi ini adalah pemenang pencarian bertahap yang dideklarasikan, bukan jaminan optimum global. Metrik CV tidak dibandingkan langsung dengan metrik holdout model lama karena versi data, aturan label, serta protokolnya berbeda.

## Batas tahap dan pekerjaan berikutnya

Tidak ada evaluasi prediksi holdout, pelatihan model deployment, pembaruan frontend, atau regenerasi gambar evaluasi pada tahap ini. Eksperimen 7 tetap diagnostik: gap kecil pada model yang underfit tidak otomatis mengalahkan Macro F1 model yang lebih baik. Kebutuhan adjudikasi 12 kasus anotasi dan anotator independen tetap terbuka.

Tahap 4 menyelaraskan konfigurasi deployment dengan keputusan penelitian ini, melatih artifact final pada development, mengevaluasi holdout sesuai protokol, dan memvalidasi identitas yang dilaporkan API. Hasil CV di sini tidak boleh ditampilkan sebagai hasil final holdout.
