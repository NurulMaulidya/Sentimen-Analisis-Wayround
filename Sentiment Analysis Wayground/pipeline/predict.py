"""Prediksi menggunakan satu artefak final: models/final_v4/final_pipeline.joblib.

CLI::

    python pipeline/predict.py                                   # interaktif
    python pipeline/predict.py --text "ulasan satu"              # satu ulasan
    python pipeline/predict.py --file data/02_data_akhir.csv     # file CSV
    python pipeline/predict.py --file data/02_data_akhir.csv --column content \\
        --output output/predictions/hasil_prediksi.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import SENTIMENT_CLASSES  # noqa: E402
from shared.preprocessing import preprocess_text  # noqa: E402
from shared.final_config import FINAL_PIPELINE_PATH  # noqa: E402
from pipeline.aspect_detection import detect_all_aspects  # noqa: E402

# Artefak inference canonical: models/final_v4/final_pipeline.joblib
# (TF-IDF + OneVsOneClassifier + Linear SVM).
FINAL_PIPELINE = Path(FINAL_PIPELINE_PATH)


class SentimentPredictor:
    """
    Prediktor yang HANYA boleh memuat artefak final canonical.

    Pemuatan WAJIB lewat shared.deployment_integrity.load_verified_pipeline()
    supaya metadata, hash artifact, dan runtime fingerprint preprocessing
    diverifikasi. Memuat joblib secara mentah akan melewati gerbang
    integritas dan tidak diperbolehkan.
    """

    def __init__(self, pipeline_path: Path | None = None):
        self.pipeline = None
        self.pipeline_path = Path(pipeline_path) if pipeline_path else FINAL_PIPELINE
        self.model_version = None
        self.load(self.pipeline_path)

    def load(self, pipeline_path: Path | None = None):
        """Muat artefak final lewat deployment integrity gate."""
        from shared.deployment_integrity import load_verified_pipeline

        path = Path(pipeline_path or self.pipeline_path)
        if not path.exists():
            raise FileNotFoundError(f"Final pipeline tidak ditemukan: {path}")

        metadata_path = path.parent / "metadata.json"
        self.pipeline, metadata = load_verified_pipeline(path, metadata_path)
        self.pipeline_path = path
        self.model_version = metadata.get("model_version")
        return self.pipeline

    def _predict_text(self, text: str):
        processed=preprocess_text(text)
        label=str(self.pipeline.predict([processed])[0])
        result={"predicted_label":label,"processed_text":processed}
        # decision score hanya sebagai margin, bukan probabilitas/confidence.
        if hasattr(self.pipeline,"decision_function"):
            scores=np.asarray(self.pipeline.decision_function([processed]))
            if scores.ndim==2 and scores.shape[1]>1:
                row=scores[0]
                order=np.sort(row)[::-1]
                margin=float(order[0]-order[1])
                result["decision_margin"]=round(margin,6)
            else:
                result["decision_score"]=float(np.ravel(scores)[0])
        result["confidence"]=None
        return result

    def predict_single(self,text:str,include_aspects=True,include_processing=False)->Dict:
        if not isinstance(text,str) or not text.strip():
            raise ValueError("Teks ulasan tidak boleh kosong.")
        r=self._predict_text(text)
        result={"original_text":text,"predicted_label":r["predicted_label"]}
        if include_processing:
            result["processed_text"]=r["processed_text"]
        if "decision_margin" in r:
            result["decision_margin"]=r["decision_margin"]
        elif "decision_score" in r:
            result["decision_score"]=r["decision_score"]
        result["confidence"]=None

        if include_aspects:
            detected=detect_all_aspects(text)
            aspect_results={}
            for aspect,sentences in detected.items():
                if aspect=="Umum" or not sentences:
                    continue
                aspect_text=" ".join(sentences)
                ar=self._predict_text(aspect_text)
                aspect_results[aspect]={
                    "label":ar["predicted_label"],
                    "decision_margin":ar.get("decision_margin"),
                    "sample_sentences":sentences[:3],
                }
            if aspect_results:
                result["aspects"]=aspect_results
        return result

    def predict_batch(self,texts:List[str],include_aspects=False)->List[Dict]:
        return [self.predict_single(t,include_aspects=include_aspects) for t in texts]

    def predict_dataframe(self,df:pd.DataFrame,text_column="content")->pd.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Kolom '{text_column}' tidak ditemukan.")
        result=df.copy()
        preds=[]
        margins=[]
        for text in result[text_column].fillna("").astype(str):
            r=self.predict_single(text,include_aspects=False)
            preds.append(r["predicted_label"]); margins.append(r.get("decision_margin"))
        result["predicted_label"]=preds
        result["decision_margin"]=margins
        result["prediction_confidence"]=pd.NA
        return result

def interactive_predict():
    predictor=SentimentPredictor()
    print("Model:",predictor.pipeline_path)
    print("Masukkan 'quit' untuk keluar.")
    while True:
        text=input("\nUlasan: ").strip()
        if text.lower()=="quit": break
        if not text: continue
        print(predictor.predict_single(text,include_aspects=True))


# =============================================================================
# CLI
# =============================================================================

def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Baca CSV/TSV dengan deteksi delimiter otomatis."""
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


def _display(value) -> str:
    """Label sentimen siap tampil (Positif/Netral/Negatif)."""
    raw = str(value).strip().lower()
    mapping = {
        "positif": "Positif",
        "positive": "Positif",
        "pos": "Positif",
        "netral": "Netral",
        "neutral": "Netral",
        "negatif": "Negatif",
        "negative": "Negatif",
        "neg": "Negatif",
    }
    return mapping.get(raw, str(value))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prediksi sentimen memakai artefak model final.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Teks ulasan yang akan diprediksi. Bisa diulang.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="File CSV berisi kolom ulasan.",
    )
    parser.add_argument(
        "--column",
        default="content",
        help="Nama kolom ulasan pada --file. Default: content",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Simpan hasil prediksi --file ke CSV ini.",
    )
    parser.add_argument(
        "--aspects",
        action="store_true",
        help="Sertakan deteksi aspek pada hasil.",
    )
    args = parser.parse_args()

    predictor = SentimentPredictor()
    print("Model   :", predictor.pipeline_path)
    print("Kelas   :", ", ".join(SENTIMENT_CLASSES))
    print("-" * 72)

    if args.text:
        for text in args.text:
            result = predictor.predict_single(
                text, include_aspects=args.aspects, include_processing=True
            )
            print(f"Ulasan  : {text}")
            print(f"Label   : {_display(result['predicted_label'])}")
            print(f"Teks    : {result.get('processed_text', '')}")
            if "decision_margin" in result:
                print(f"Margin  : {result['decision_margin']}")
            if args.aspects and result.get("aspects"):
                for aspect, detail in result["aspects"].items():
                    print(f"  - {aspect}: {_display(detail['label'])}")
            print("-" * 72)
        return 0

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"ERROR: File tidak ditemukan: {path}")
            return 1

        df = _read_csv_auto(path)
        if args.column not in df.columns:
            print(
                f"ERROR: Kolom '{args.column}' tidak ditemukan. "
                f"Kolom tersedia: {list(df.columns)}"
            )
            return 1

        result = predictor.predict_dataframe(df, text_column=args.column)
        result["sentiment_label"] = result["predicted_label"].map(_display)
        print(f"Baris diprediksi : {len(result):,}")
        print(result["sentiment_label"].value_counts().to_string())

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n[SAVED] {out}")
        return 0

    interactive_predict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
