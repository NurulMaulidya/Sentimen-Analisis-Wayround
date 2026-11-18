# DATASET EKSPERIMEN FINAL WAYGROUND

Status: FROZEN_FOR_EXPERIMENTS

Dataset canonical:
`data/pipeline_output/stage3_final_labeled.csv`

Spesifikasi:
- Jumlah data: 3.065
- Kolom teks eksperimen: `processed_text`
- Kolom label final: `sentiment_label_final`
- Label awal: `sentiment_label_inset`
- Validasi pakar matched: 289
- Label sama: 223
- Label berubah: 66
- Expert unmatched: 11

Distribusi label final:
- Positif: 1.997 (65,15%)
- Netral: 625 (20,39%)
- Negatif: 443 (14,46%)

ATURAN FREEZE:
1. Jangan mengubah label_final.
2. Jangan menghapus/menambah data untuk menyeimbangkan kelas.
3. Jangan mengganti dataset di tengah Experiment 1-7.
4. Semua eksperimen menggunakan dataset ini.
5. Perubahan preprocessing/labeling setelah freeze harus membuat versi dataset baru, bukan menimpa dataset ini.
6. Final holdout hanya digunakan setelah konfigurasi model terkunci.

Catatan:
11 data pakar unmatched tetap tidak dimasukkan secara paksa karena tidak memiliki pasangan yang tervalidasi pada dataset Stage 3.
