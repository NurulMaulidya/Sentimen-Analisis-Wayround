# Environment Reproducibility — Wayground Sentiment Analysis

Tanggal penulisan: 5 November 2026. Berlaku untuk model final v4
(`model_version` `20261029_223454`).

Dokumen ini adalah rujukan yang disebut oleh `requirements-lock.txt`.
Isinya menjelaskan **environment mana yang mengulang hasil canonical**,
**mana yang tidak**, dan **bagaimana mengetahuinya** tanpa perlu menebak.

---

## 1. Dua environment, dua tanggung jawab

| | Environment terkunci | Environment pengembangan |
|---|---|---|
| Definisi | `requirements-lock.txt` | `requirements.txt` |
| Versi kunci | Python 3.10.6, numpy 1.26.2, pandas 2.1.4, scikit-learn 1.7.2 | rentang versi minimum + penjaga kompatibilitas |
| Dipakai untuk | **training** model final dan seluruh chain eksperimen | menjalankan backend, test, dan website |
| Status artefak canonical | **satu-satunya** yang dipakai untuk menghasilkan `models/final_v4/` | tidak menghasilkan artefak |
| Rilis | 29 Oktober 2026, 22:34 WIB (`metadata.json` → `training_timestamp`) | aktif di workspace ini |

Artefak canonical **tidak ditimpa** oleh environment pengembangan.
`scripts/train_final_model.py` menolak menimpa artefak yang sudah ada
(`FileExistsError`), dan `shared/deployment_integrity.py` memverifikasi
hash setiap artefak saat dimuat.

---

## 2. Apa yang terpengaruh dan apa yang tidak

### Tidak terpengaruh

Backend, seluruh test, dan website memakai artefak canonical **yang sudah
di-fit**. Model dipakai hanya untuk `transform()` (TF-IDF) dan `predict()`;
tidak ada `fit()` pada data pengguna maupun data uji. Karena itu:

- angka evaluasi penelitian (holdout diagnostik) tetap sama;
- `model_version` tetap `20261029_223454`;
- hash artefak dan gate integritas tetap lolos;
- seluruh 155 test tetap lulus.

### Terpengaruh

Training ulang. Selama training, TF-IDF di-`fit` pada data development,
sehingga hasil numeriknya bergantung pada versi numpy/pandas/scikit-learn.
Perbedaan versi pada kasus batas (mis. pengurutan stabil, pengurutan
`inf`/`nan`) dapat mengubah beberapa baris prediksi dan dengan demikian
menggeser metrik holdout.

**Karena itu, `scripts/train_final_model.py` sekarang punya gerbang
environment**: training DITOLAK bila numpy, pandas, atau scikit-learn tidak
cocok dengan versi terkunci, atau bila major.minor Python berbeda.

---

## 3. Cara memeriksa

```powershell
python scripts/check_environment.py
```

Keluarannya:

| Kode keluar | Arti |
|---|---|
| 0 | environment aktif == environment penelitian terkunci; training boleh |
| 1 | ada selisih; **training ditolak**, inference tetap boleh |

Contoh keluaran di workspace ini:

```
Pustaka         Terkunci      Aktif         Status
--------------------------------------------------------------------------------
python          3.10.6        3.14.5        TIDAK COCOK
numpy           1.26.2        2.5.3         TIDAK COCOK
pandas          2.1.4         3.0.6         TIDAK COCOK
scikit-learn    1.7.2         1.7.2         cocok
```

Scikit-learn sengaja dibandingkan **persis**, karena pustaka itu yang
membuat artefak `joblib` dan memuatkan struktur class-nya. Python hanya
 dibandingkan pada major.minor: perbedaan patch tidak mengubah hasil
numerik, sedangkan perbedaan minor bisa.

---

## 4. Cara memakai environment terkunci

```powershell
python -m venv .venv310
.venv310\Scripts\python -m pip install -r requirements-lock.txt
.venv310\Scripts\python -m pytest tests/ -q
```

Untuk **training ulang** (hanya bila memang diperlukan, dan bukan untuk
menimpa artefak canonical):

```powershell
.venv310\Scripts\python scripts\train_final_model.py
```

> Script menolak menimpa artefak canonical. Training ulang berarti memindahkan
> atau mengarsipkan `models/final_v4/` dan `output_v4/experiments/` terlebih
> dahulu, dan itu **perubahan terhadap artefak canonical yang memerlukan
> persetujuan** — bukan tindakan yang dilakukan otomatis oleh pipeline mana
> pun di repo ini.

Bila training di luar environment terkunci memang sengaja dilakukan,
set:

```powershell
$env:WAYGROUND_ALLOW_ENV_DRIFT = "1"
```

Gerbang akan dilewati, tetapi selisihnya **tetap tercatat** dan wajib
disebutkan pada laporan tahap tersebut.

---

## 5. Status pembuktian reproduktibilitas

> **Pernyataan jujur:** repo ini **tidak menyimpan artefak retrain
> terpisah**. Yang tersimpan hanya nilai canonical
> `holdout_f1_macro = 0.7902092778312039`.
> `requirements-lock.txt` mencantumkan bahwa retrain pada environment
> terkunci menghasilkan nilai yang identik; nilai itu dicatat dari
> Tahap 4 dan **belum diulang ulang di dalam workspace ini**, karena
> environment terkunci (Python 3.10.6) tidak terpasang di sini dan karena
> retrain akan menyentuh artefak canonical.

Arti praktisnya:

- Yang **terbukti langsung** di workspace ini: inference pada artefak
  canonical konsisten, preprocessing mereproduksi 3078/3078 baris
  `processed_text` dataset canonical, dan pelabelan lexicon mereproduksi
  3078/3078 label (`scripts/verify_deployment_bundle.py`).
- Yang **belum terverifikasi ulang di sini**: bahwa retrain pada
  environment terkunci menghasilkan metrik yang persis sama.

Untuk menutup celah tersebut, jalankan langkah berikut di lingkungan
yang memiliki Python 3.10.6, dan simpan keluarannya sebagai artefak
retrain terpisah:

```powershell
python -m venv .venv310
.venv310\Scripts\python -m pip install -r requirements-lock.txt
.venv310\Scripts\python -m pytest tests/ -q
```

lalu bandingkan `holdout_f1_macro` dari retrain dengan nilai canonical
`0.7902092778312039`. Hasil perbandingan itu yang sebaiknya menjadi
sumber angka pada laporan final.

---

## 6. Rujukan

- `requirements-lock.txt` — definisi environment terkunci.
- `requirements.txt` — environment pengembangan (backend, test, website).
- `shared/environment.py` — implementasi pemeriksaan drift.
- `scripts/check_environment.py` — laporan CLI.
- `scripts/verify_deployment_bundle.py` — bukti ekuivalensi preprocessing
  dan pelabelan terhadap artefak canonical.
- `docs/remediation/TAHAP_06.md` — temuan environment yang menjadi dasar
  dokumen ini.
