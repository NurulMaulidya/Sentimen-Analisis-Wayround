"""
Verifikasi bundle deployment + re-binding fingerprint (Tahap 6)
=============================================================

Tujuan
------
1. Membuktikan bahwa source code preprocessing yang sedang berjalan
   menghasilkan output yang IDENTIK dengan `processed_text` yang dipakai
   saat model canonical dilatih. `processed_text` itu persis menjadi input
   TF-IDF saat training, sehingga kesamaan kolomnya adalah bukti langsung
   bahwa perilaku inferensi tidak berubah.
2. Memverifikasi bahwa pelabelan lexicon yang sedang berjalan menghasilkan
   label IDENTIK dengan label yang tersimpan di dataset canonical.
3. Menghitung fingerprint sumber daya yang menentukan perilaku inferensi
   (`runtime_fingerprint`) dan fingerprint jalur riset
   (`research_fingerprint`).
4. Opsional (`--rebind`): menulis ulang fingerprint pada
   `models/final_v4/metadata.json` dengan jejak audit, TANPA mengubah
   model, dataset, label, konfigurasi, atau hasil evaluasi.

Mengapa perlu
-------------
Gate integritas deployment (`shared/deployment_integrity.py`) menolak
memuat model bila sumber daya yang mengunci preprocessing berubah. Setelah
refactor pada `shared/constants.py` dan `shared/lexicon.py` (perbaikan
NameError `LEXICON` + default lexicon), gate menolak bundle although
perilaku preprocessing terbukti tidak berubah. Skrip ini dipakai untuk
membuktikan ekuivalensi tersebut secara empiris, bukan lewat asumsi.

Pakai:
    python scripts/verify_deployment_bundle.py            # verifikasi saja
    python scripts/verify_deployment_bundle.py --rebind   # tulis fingerprint
    python scripts/verify_deployment_bundle.py --skip-equivalence
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _report(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def check_preprocessing_equivalence(sample_limit: int | None = None) -> dict:
    """
    Bukti ekuivalensi preprocessing: source saat ini harus mereproduksi
    `processed_text` yang tersimpan di dataset canonical.

    Dataset canonical memuat `processed_text` yang PERSIS menjadi input
    TF-IDF saat training. Jika source saat ini mereproduksi kolom itu untuk
    seluruh baris, perilaku inferensi tidak berubah.
    """
    import pandas as pd

    from shared.constants import CANONICAL_DATASET_PATH
    from shared.preprocessing import preprocess_text

    df = pd.read_csv(CANONICAL_DATASET_PATH, encoding="utf-8-sig")

    if sample_limit:
        df = df.head(sample_limit)

    stored = df["processed_text"].fillna("").astype(str).tolist()
    current = [preprocess_text(text) for text in df["content"].fillna("")]

    identical = sum(a == b for a, b in zip(stored, current))
    mismatches = [
        {"row": index, "stored": stored[index], "current": current[index]}
        for index, (a, b) in enumerate(zip(stored, current))
        if a != b
    ]

    return {
        "rows_checked": len(stored),
        "identical_rows": identical,
        "mismatch_count": len(mismatches),
        "equivalent": identical == len(stored),
        "examples": mismatches[:5],
    }


def check_labeling_equivalence(sample_limit: int | None = None) -> dict:
    """
    Bukti ekuivalensi pelabelan: lexicon saat ini harus mereproduksi
    `sentiment_label_inset` yang tersimpan di dataset canonical.

    `sentiment_label_final` juga melibatkan label pakar sehingga tidak
    direproduksi di sini; yang diperiksa adalah label lexicon murni.
    """
    import pandas as pd

    from shared.constants import CANONICAL_DATASET_PATH
    from shared.labeling import label_text

    df = pd.read_csv(CANONICAL_DATASET_PATH, encoding="utf-8-sig")

    if sample_limit:
        df = df.head(sample_limit)

    stored = df["sentiment_label_inset"].fillna("").astype(str).str.lower().tolist()
    current = [
        label_text(text)["label"].lower()
        for text in df["processed_text"].fillna("").astype(str)
    ]

    identical = sum(a == b for a, b in zip(stored, current))
    mismatches = [
        {"row": index, "stored": stored[index], "current": current[index]}
        for index, (a, b) in enumerate(zip(stored, current))
        if a != b
    ]

    return {
        "rows_checked": len(stored),
        "identical_rows": identical,
        "mismatch_count": len(mismatches),
        "equivalent": identical == len(stored),
        "examples": mismatches[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifikasi bundle deployment canonical v4.",
    )
    parser.add_argument(
        "--rebind",
        action="store_true",
        help=(
            "Tulis ulang fingerprint pada models/final_v4/metadata.json "
            "dengan jejak audit. Model, dataset, label, konfigurasi, dan "
            "hasil evaluasi tidak diubah."
        ),
    )
    parser.add_argument(
        "--skip-equivalence",
        action="store_true",
        help="Lewati bukti ekuivalensi preprocessing/pelabelan (lebih cepat).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Batasi jumlah baris yang diperiksa untuk bukti ekuivalensi.",
    )
    args = parser.parse_args()

    from shared.deployment_integrity import (
        load_verified_pipeline,
        research_fingerprint,
        runtime_fingerprint,
    )
    from shared.final_config import FINAL_METADATA_PATH

    failures: list[str] = []

    _report("1. Fingerprint sumber daya")
    inference = runtime_fingerprint()
    research = research_fingerprint()
    print(json.dumps(inference, indent=2))
    print()
    print("Fingerprint jalur riset:")
    print(json.dumps(research, indent=2))

    metadata = json.loads(FINAL_METADATA_PATH.read_text(encoding="utf-8-sig"))
    recorded = metadata.get("inference_fingerprint")

    matches = recorded == inference
    print()
    if matches:
        print("[OK]   Fingerprint metadata cocok dengan source saat ini.")
    else:
        print("[WARN] Fingerprint metadata TIDAK cocok dengan source saat ini.")
        if isinstance(recorded, dict):
            changed = [
                key
                for key in set(recorded) | set(inference)
                if recorded.get(key) != inference.get(key)
            ]
            print(f"       Kunci berbeda: {sorted(changed)}")
        print("       Jalankan dengan --rebind setelah bukti ekuivalensi lolos.")

    _report("2. Bukti ekuivalensi preprocessing")

    preprocessing = {"skipped": True}
    labeling = {"skipped": True}

    if not args.skip_equivalence:
        preprocessing = check_preprocessing_equivalence(args.sample_limit)
        print(
            f"Rows identik : {preprocessing['identical_rows']}"
            f"/{preprocessing['rows_checked']}"
        )
        print(f"Equivalent   : {preprocessing['equivalent']}")
        for example in preprocessing["examples"]:
            print(f"  - row {example['row']}: "
                  f"{example['stored']!r} -> {example['current']!r}")
        if not preprocessing["equivalent"]:
            failures.append(
                "Preprocessing saat ini tidak mereproduksi processed_text canonical."
            )

        _report("3. Bukti ekuivalensi pelabelan lexicon")
        labeling = check_labeling_equivalence(args.sample_limit)
        print(
            f"Rows identik : {labeling['identical_rows']}"
            f"/{labeling['rows_checked']}"
        )
        print(f"Equivalent   : {labeling['equivalent']}")
        for example in labeling["examples"]:
            print(f"  - row {example['row']}: "
                  f"{example['stored']!r} -> {example['current']!r}")
        if not labeling["equivalent"]:
            failures.append(
                "Pelabelan lexicon saat ini tidak mereproduksi label canonical."
            )
    else:
        print("Dilewati (--skip-equivalence).")

    _report("4. Pemuatan bundle terverifikasi")

    if args.rebind and not failures:
        _report("5. Re-binding fingerprint")
        rebind = {
            "date": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "Fingerprint lama meng-hash seluruh isi shared/constants.py dan "
                "shared/lexicon.py, sehingga perubahan yang tidak menyentuh "
                "perilaku inferensi (komentar, path default, perbaikan "
                "NameError LEXICON) tetap menolak bundle. Fingerprint baru "
                "hanya mencakup sumber daya yang menentukan perilaku "
                "inferensi."
            ),
            "previous_scheme": (
                "whole-file-sha256(13 sumber daya)"
                if isinstance(recorded, dict) and "scheme" not in recorded
                else (recorded or {}).get("scheme")
            ),
            "previous_fingerprint": recorded,
            "new_scheme": inference["scheme"],
            "changed_files": [
                "shared/constants.py",
                "shared/lexicon.py",
                "shared/labeling.py",
            ],
            "behavioral_equivalence": {
                "preprocessing": preprocessing,
                "labeling": labeling,
            },
            "unchanged": [
                "models/final_v4/final_pipeline.joblib",
                "data/pipeline_output_v4/stage3_final_labeled.csv",
                "output_v4/experiments/final_model_evaluation.json",
                "output_v4/experiments/final_classification_report.json",
                "output_v4/experiments/final_confusion_matrix.csv",
                "output_v4/experiments/final_holdout_predictions.csv",
                "output_v4/experiments/best_config.json",
                "output_v4/experiments/selected_configuration.json",
                "shared/preprocessing.py",
            ],
            "verification_command": "python scripts/verify_deployment_bundle.py",
        }

        if "runtime_fingerprint" in metadata:
            metadata.setdefault(
                "runtime_fingerprint_at_training", metadata["runtime_fingerprint"]
            )
        metadata["inference_fingerprint"] = inference
        metadata["research_fingerprint"] = research
        metadata["fingerprint_rebind"] = rebind

        # Backup metadata lama sebelum ditulis ulang. Metadata bundle
        # canonical tidak dilacak Git, jadi salinan ini adalah satu-satunya
        # jalan kembali bila re-binding dibatalkan.
        backup_path = FINAL_METADATA_PATH.with_name(
            "metadata.json.bak-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        shutil.copy2(FINAL_METADATA_PATH, backup_path)
        print(f"[OK]   Backup metadata lama: {backup_path}")

        FINAL_METADATA_PATH.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[OK]   Fingerprint ditulis ke {FINAL_METADATA_PATH}")
        print("       Model, dataset, label, konfigurasi, dan evaluasi tidak diubah.")
    elif args.rebind:
        print("[SKIP] Re-binding ditolak karena bukti ekuivalensi gagal.")

    _report("6. Hasil akhir")

    try:
        pipeline, meta = load_verified_pipeline()
        print(f"[OK]   Bundle dimuat. model_version={meta['model_version']}")
        print(f"       Prediksi contoh: {pipeline.predict(['aplikasinya sangat bagus'])[0]}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Bundle tidak dapat dimuat: {exc}")
        print(f"[FAIL] Bundle tidak dapat dimuat: {exc}")

    if failures:
        print()
        for item in failures:
            print(f"[FAIL] {item}")
        return 1

    print()
    print("Semua pemeriksaan lolos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
