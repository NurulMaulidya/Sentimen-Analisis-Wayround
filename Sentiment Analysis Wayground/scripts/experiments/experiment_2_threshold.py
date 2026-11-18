import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import EXPERIMENTS_DIR  # noqa: E402
from scripts.experiments.common import load_dataset, save_json, print_header
from shared.constants import SENTIMENT_CLASSES
from shared.labeling import label_text as classify_sentiment, lexicon_info
from shared.research_config import BASELINE_NEUTRAL_THRESHOLD
from shared.splitting import split_parts_for_dataset
from shared.experiment_artifacts import EXPERIMENT_JSON_FILES
OUTPUT_DIR = EXPERIMENTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / EXPERIMENT_JSON_FILES[2]
DEFAULT_EXPERT_FILE = (
    PROJECT_ROOT / "dataset" / "Hasil_Validasi_Pakar.tsv"
)
CANDIDATE_THRESHOLDS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
)

# Baseline only. This value is NOT read from shared/final_config.py.
# The experiment still evaluates every candidate threshold below.
DEFAULT_CONFIGURED_THRESHOLD = BASELINE_NEUTRAL_THRESHOLD

CANONICAL_RAW_TEXT_COLUMN_CANDIDATES = (
    "content",
    "review",
    "text",
    "ulasan",
)

EXPERT_RAW_TEXT_COLUMN_CANDIDATES = (
    "Ulasan Asli",
    "content",
    "review",
    "text",
    "ulasan",
    "komentar",
    "review_text",
)
EXPERT_PREPROCESSED_TEXT_COLUMN_CANDIDATES = (
    "Teks Preprocessing",
    "processed_text",
    "preprocessed_text",
)
TEXT_COLUMN_CANDIDATES = (
    "processed_text",
    "preprocessed_text",
    "clean_text",
    "content",
    "review",
    "text",
    "ulasan",
)
EXPERT_TEXT_COLUMN_CANDIDATES = (
    "Teks Preprocessing",
    "processed_text",
    "preprocessed_text",
    "content",
    "Ulasan Asli",
    "review",
    "text",
    "ulasan",
    "komentar",
    "review_text",
)
EXPERT_LABEL_COLUMN_CANDIDATES = (
    "Label",
    "sentiment_label_expert",
    "label_pakar",
    "label_expert",
    "sentiment_expert",
    "expert_label",
    "predicted_label",
    "label",
    "sentiment",
    "sentimen",
)
EXPERT_HEADER_MARKERS = (
    "no",
    "ulasan asli",
    "teks preprocessing",
    "label",
    "kesesuaian label awal",
    "catatan pakar",
    "content",
    "processed_text",
    "sentiment_label_expert",
    "label_pakar",
    "label_expert",
    "ulasan",
    "catatan pakar",
)
def normalize_label(label):
    if label is None or pd.isna(label):
        return ""
    value = str(label).strip().lower()
    mapping = {
        "positif": "Positif",
        "positive": "Positif",
        "pos": "Positif",
        "1": "Positif",
        "netral": "Netral",
        "neutral": "Netral",
        "neu": "Netral",
        "0": "Netral",
        "negatif": "Negatif",
        "negative": "Negatif",
        "neg": "Negatif",
        "-1": "Negatif",
    }
    return mapping.get(value, str(label).strip())
def normalize_match_text(value):
    if value is None or pd.isna(value):
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("\r", " ")
        .replace("\n", " ")
    )
def compact_match_text(value):
    return " ".join(normalize_match_text(value).split())
def detect_column(df, candidates, description):
    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        key = str(candidate).strip().lower()
        if key in normalized_columns:
            return normalized_columns[key]
    raise ValueError(
        f"Kolom {description} tidak ditemukan.\n"
        f"Kolom tersedia: {list(df.columns)}\n"
        f"Kolom kandidat: {list(candidates)}"
    )
def prepare_canonical_text(df, text_column):
    if text_column not in df.columns:
        raise ValueError(
            f"Kolom teks '{text_column}' tidak ditemukan."
        )
    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    texts = texts[texts != ""]
    if texts.empty:
        raise ValueError(
            "Tidak ditemukan teks valid pada dataset."
        )
    return texts
def _read_expert_raw(expert_path):
    """
    Membaca file pakar tanpa menganggap baris pertama sebagai header.
    Ini penting karena Hasil_Validasi_Pakar.tsv adalah file formulir,
    bukan TSV tabel murni. Baris pertama dapat berisi judul formulir.
    """
    attempts = (
        ("\t", "tab"),
        (";", "semicolon"),
        (",", "comma"),
    )
    errors = []
    for delimiter, name in attempts:
        try:
            raw = pd.read_csv(
                expert_path,
                sep=delimiter,
                header=None,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
                engine="python",
                quotechar='"',
                on_bad_lines="warn",
            )
            if raw.shape[1] >= 2:
                print(
                    f"[EXPERT] Parser delimiter : {name}"
                )
                print(
                    f"[EXPERT] Raw shape         : "
                    f"{raw.shape[0]:,} x {raw.shape[1]}"
                )
                return raw
            errors.append(
                f"{name}: hanya {raw.shape[1]} kolom"
            )
        except Exception as exc:
            errors.append(
                f"{name}: {type(exc).__name__}: {exc}"
            )
    raise ValueError(
        "File validasi pakar tidak dapat dibaca.\n"
        + "\n".join(errors)
    )
def _find_expert_header_row(raw):
    """
    Mencari baris header tabel secara otomatis.
    Tidak mengubah file sumber. Hanya menentukan dari baris mana
    pandas harus membangun DataFrame.
    """
    best_row = None
    best_score = -1
    for row_index in range(len(raw)):
        values = [
            str(value).strip().lower()
            for value in raw.iloc[row_index].tolist()
        ]
        non_empty = [value for value in values if value]
        if not non_empty:
            continue
        score = 0
        for value in non_empty:
            if value in EXPERT_HEADER_MARKERS:
                score += 3
            if "sentiment" in value:
                score += 2
            if "label" in value:
                score += 2
            if "processed" in value:
                score += 2
            if "content" in value:
                score += 2
            if "catatan" in value:
                score += 1
        if score > best_score:
            best_score = score
            best_row = row_index
    if best_row is None or best_score < 3:
        raise ValueError(
            "Header tabel validasi pakar tidak berhasil ditemukan.\n"
            "Kolom/struktur file tidak dikenali oleh script."
        )
    print(
        f"[EXPERT] Header row ditemukan : {best_row + 1}"
    )
    print(
        f"[EXPERT] Header score         : {best_score}"
    )
    return best_row

def load_expert_validation(expert_path):
    if not expert_path.exists():
        raise FileNotFoundError(
            f"File validasi pakar tidak ditemukan: {expert_path}"
        )
    raw = _read_expert_raw(expert_path)
    header_row = _find_expert_header_row(raw)
    header = [str(value).strip() for value in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    valid_columns = []
    seen = set()
    for column in data.columns:
        column_name = str(column).strip()
        if not column_name:
            continue
        if column_name in seen:
            suffix = 2
            candidate = f"{column_name}_{suffix}"
            while candidate in seen:
                suffix += 1
                candidate = f"{column_name}_{suffix}"
            column_name = candidate
        seen.add(column_name)
        valid_columns.append(column_name)
    data = data.iloc[:, :len(valid_columns)].copy()
    data.columns = valid_columns
    for column in data.columns:
        data[column] = data[column].astype(str).str.strip()
    raw_text_col = detect_column(
        data,
        EXPERT_RAW_TEXT_COLUMN_CANDIDATES,
        "ulasan asli pada file validasi pakar",
    )
    preprocessed_text_col = detect_column(
        data,
        EXPERT_PREPROCESSED_TEXT_COLUMN_CANDIDATES,
        "teks preprocessing pada file validasi pakar",
    )
    label_col = detect_column(
        data,
        EXPERT_LABEL_COLUMN_CANDIDATES,
        "label pakar",
    )
    data["_match_raw_text"] = data[raw_text_col].map(compact_match_text)
    data["_expert_preprocessed_text"] = data[preprocessed_text_col].map(
        compact_match_text
    )
    data["_expert_label"] = data[label_col].map(normalize_label)
    data = data[
        (data["_match_raw_text"] != "")
        & data["_expert_label"].isin(SENTIMENT_CLASSES)
    ].copy()
    if data.empty:
        raise ValueError(
            "Tidak ditemukan baris valid yang memiliki "
            "ulasan asli dan label pakar."
        )
    duplicate_mask = data["_match_raw_text"].duplicated(keep=False)
    if duplicate_mask.any():
        conflicts = (
            data.loc[duplicate_mask]
            .groupby("_match_raw_text")["_expert_label"]
            .nunique()
        )
        conflict_keys = set(conflicts[conflicts > 1].index)
        if conflict_keys:
            raise ValueError(
                "Ditemukan ulasan asli validasi pakar dengan label berbeda. "
                f"Jumlah konflik: {len(conflict_keys)}"
            )
        data = data.drop_duplicates(
            subset="_match_raw_text",
            keep="first",
        )
    print(f"[EXPERT] Rows valid : {len(data):,}")
    print("[EXPERT] Distribusi label:")
    print(
        data["_expert_label"]
        .value_counts()
        .reindex(SENTIMENT_CLASSES, fill_value=0)
        .to_string()
    )
    return data, raw_text_col, preprocessed_text_col, label_col

def build_expert_validation_set(
    canonical_df,
    text_column,
    expert_df,
):
    work = canonical_df.copy()
    canonical_raw_col = detect_column(
        work,
        CANONICAL_RAW_TEXT_COLUMN_CANDIDATES,
        "teks asli canonical dataset",
    )
    work["_match_raw_text"] = work[canonical_raw_col].map(
        compact_match_text
    )
    expert_map = (
        expert_df.set_index("_match_raw_text")["_expert_label"].to_dict()
    )
    work["_expert_label"] = work["_match_raw_text"].map(expert_map)
    matched = work[
        work["_expert_label"].isin(SENTIMENT_CLASSES)
        & work[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()
    if matched.empty:
        raise ValueError(
            "Tidak ada data canonical yang berhasil dicocokkan "
            "dengan validasi pakar berdasarkan ulasan asli."
        )
    matched = matched.drop_duplicates(
        subset=["_match_raw_text"],
        keep="first",
    ).copy()
    print(f"[MATCH] Canonical raw text column : {canonical_raw_col}")
    print(f"[MATCH] Expert rows available     : {len(expert_df):,}")
    print(f"[MATCH] Expert matched unique     : {len(matched):,}")
    print(
        f"[MATCH] Coverage                  : "
        f"{len(matched) / len(expert_df) * 100:.2f}%"
    )
    return matched
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(SENTIMENT_CLASSES),
    )
    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            6,
        ),
        "precision_macro": round(
            float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=list(SENTIMENT_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
            6,
        ),
        "recall_macro": round(
            float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=list(SENTIMENT_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
            6,
        ),
        "f1_macro": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(SENTIMENT_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
            6,
        ),
        "confusion_matrix": cm.tolist(),
    }
def calculate_distribution(labels):
    counts = (
        pd.Series(labels)
        .value_counts()
        .reindex(
            SENTIMENT_CLASSES,
            fill_value=0,
        )
    )
    total = int(len(labels))
    distribution = {}
    for label in SENTIMENT_CLASSES:
        count = int(counts[label])
        distribution[label] = {
            "count": count,
            "percentage": round(
                (count / total * 100)
                if total
                else 0.0,
                4,
            ),
        }
    return distribution
def run_threshold_experiment(
    canonical_df,
    text_column,
    expert_df,
    configured_threshold=DEFAULT_CONFIGURED_THRESHOLD,
):
    print_header(
        "EXPERIMENT 2: NEUTRAL THRESHOLD "
        "SELECTION WITH EXPERT VALIDATION"
    )

    if float(configured_threshold) < 0:
        raise ValueError(
            "configured_threshold tidak boleh negatif."
        )

    if float(configured_threshold) not in [
        float(value) for value in CANDIDATE_THRESHOLDS
    ]:
        raise ValueError(
            "configured_threshold harus berada dalam "
            f"CANDIDATE_THRESHOLDS: {list(CANDIDATE_THRESHOLDS)}"
        )
    texts = prepare_canonical_text(
        canonical_df,
        text_column,
    )
    development, _, split_info = split_parts_for_dataset(canonical_df)
    validation_df = build_expert_validation_set(
        canonical_df=development,
        text_column=text_column,
        expert_df=expert_df,
    )
    validation_texts = (
        validation_df[text_column]
        .astype(str)
    )
    y_true = validation_df[
        "_expert_label"
    ].tolist()
    print(
        f"[DATA] Total canonical texts : "
        f"{len(texts):,}"
    )
    print(
        f"[DATA] Expert rows available : "
        f"{len(expert_df):,}"
    )
    print(
        f"[DATA] Expert matched unique : "
        f"{len(validation_df):,}"
    )
    print(
        f"[DATA] Text column           : "
        f"{text_column}"
    )
    print(
        f"[CONFIG] Configured threshold : "
        f"{float(configured_threshold)}"
    )
    print(
        f"[CONFIG] Candidate thresholds : "
        f"{list(CANDIDATE_THRESHOLDS)}"
    )
    print()
    print(
        "[RULE] Seleksi threshold menggunakan "
        "Expert Validation."
    )
    print(
        "[RULE] Metric utama: Macro F1."
    )
    print(
        "[RULE] Threshold tidak dipilih berdasarkan "
        "class balance."
    )
    print(
        "[RULE] Threshold tidak dipilih berdasarkan "
        "performa SVM."
    )
    print()
    results = []
    for threshold in CANDIDATE_THRESHOLDS:
        expert_predictions = [
            normalize_label(
                classify_sentiment(
                    text,
                    threshold=float(threshold),
                )["label"]
            )
            for text in validation_texts
        ]
        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=expert_predictions,
        )
        # Distribusi full dataset hanya untuk audit.
        # Tidak digunakan untuk seleksi threshold.
        full_predictions = [
            normalize_label(
                classify_sentiment(
                    text,
                    threshold=float(threshold),
                )["label"]
            )
            for text in texts
        ]
        result = {
            "threshold": float(threshold),
            "expert_validation": {
                "n": int(len(validation_df)),
                **metrics,
            },
            "full_dataset_distribution": (
                calculate_distribution(
                    full_predictions
                )
            ),
        }
        results.append(result)
        print(
            f"T={threshold:<3} | "
            f"Expert Macro F1="
            f"{metrics['f1_macro']:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Prec={metrics['precision_macro']:.4f} | "
            f"Recall={metrics['recall_macro']:.4f}"
        )
    # Macro F1 adalah kriteria utama.
    # Jika seri, gunakan Accuracy lalu threshold lebih rendah.
    selected = max(
        results,
        key=lambda item: (
            item["expert_validation"]["f1_macro"],
            item["expert_validation"]["accuracy"],
            -item["threshold"],
        ),
    )
    selected_threshold = float(
        selected["threshold"]
    )
    for item in results:
        item["selected"] = (
            float(item["threshold"])
            == selected_threshold
        )
    configured_result = next(
        (
            item
            for item in results
            if float(item["threshold"])
            == float(configured_threshold)
        ),
        None,
    )
    print()
    print(
        f"[SELECTED] Threshold terbaik : "
        f"{selected_threshold}"
    )
    print(
        f"[SELECTED] Expert Macro F1   : "
        f"{selected['expert_validation']['f1_macro']:.4f}"
    )
    if configured_result is not None:
        delta = (
            selected["expert_validation"]["f1_macro"]
            - configured_result[
                "expert_validation"
            ]["f1_macro"]
        )
        print(
            f"[COMPARE] Delta Macro F1 vs configured "
            f"({float(configured_threshold)}): "
            f"{delta:+.4f}"
        )
    print()
    print(
        "[IMPORTANT] Canonical dataset TIDAK diubah."
    )
    print(
        "[NEXT] Jika threshold berubah, lakukan "
        "relabeling, integrasi validasi pakar, "
        "validasi dataset, dan eksperimen model ulang."
    )
    metadata = {
        "experiment": "experiment_2_threshold",
        "description": (
            "Seleksi neutral threshold berdasarkan "
            "agreement dengan label validasi pakar."
        ),
        "methodology": {
            "purpose": (
                "Menguji kandidat threshold dan memilih "
                "threshold dengan Macro F1 tertinggi "
                "terhadap label pakar."
            ),
            "selection_method": (
                "highest_expert_validation_macro_f1"
            ),
            "primary_metric": "f1_macro",
            "selection_not_based_on": [
                "class_balance",
                "SVM_performance",
            ],
            "tie_breaker": [
                "accuracy",
                "lower_threshold",
            ],
            "requires_relabeling_if_changed": True,
            "requires_revalidation_if_changed": True,
        },
        "dataset": {
            "canonical_rows": int(len(canonical_df)),
            "total_valid_texts": int(len(texts)),
            "expert_rows_after_label_validation": int(
                len(expert_df)
            ),
            "expert_matched_unique_rows": int(
                len(validation_df)
            ),
            "text_column": text_column,
        },
        "expert_validation": {
            "selection_partition": "development_only",
            "holdout_used_for_selection": False,
            "split_source": split_info["source"],
            "lexicon": lexicon_info(),
            "file": str(DEFAULT_EXPERT_FILE),
            "n": int(len(validation_df)),
            "class_distribution": (
                calculate_distribution(y_true)
            ),
        },
        "configuration": {
            "configured_threshold": float(
                configured_threshold
            ),
            "configured_threshold_is_baseline_only": True,
            "final_config_dependency": False,
            "source_of_truth": (
                "experiment_result_until_final_config_is_locked"
            ),
            "selected_threshold": selected_threshold,
            "candidate_thresholds": [
                float(value)
                for value in CANDIDATE_THRESHOLDS
            ],
            "sentiment_classes": list(
                SENTIMENT_CLASSES
            ),
        },
        "results": results,
        "selected_result": selected,
        "configured_result": configured_result,
    }
    save_json(
        metadata,
        OUTPUT_FILE,
    )
    print(
        f"[OUTPUT] {OUTPUT_FILE}"
    )
    return metadata
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2 - Neutral threshold "
            "selection with expert validation."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path canonical dataset CSV. "
            "Default menggunakan "
            "stage3_final_labeled.csv."
        ),
    )
    parser.add_argument(
        "--expert",
        type=str,
        default=str(DEFAULT_EXPERT_FILE),
        help=(
            "Path file validasi pakar TSV/CSV. "
            "File tidak dimodifikasi."
        ),
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help=(
            "Nama kolom teks canonical dataset. "
            "Default dideteksi otomatis."
        ),
    )
    parser.add_argument(
        "--configured-threshold",
        type=float,
        default=DEFAULT_CONFIGURED_THRESHOLD,
        choices=CANDIDATE_THRESHOLDS,
        help=(
            "Threshold baseline untuk pembanding saja. "
            "Tidak dibaca dari shared/final_config.py dan "
            "tidak menentukan threshold terbaik."
        ),
    )
    return parser
def main():
    parser = build_parser()
    args = parser.parse_args()
    print_header("EXPERIMENT 2")
    df, dataset_path = load_dataset(
        args.data
    )
    print(
        f"[DATA] File : {dataset_path}"
    )
    print(
        f"[DATA] Rows : {len(df):,}"
    )
    text_column = (
        args.text_column
        if args.text_column
        else detect_column(
            df,
            TEXT_COLUMN_CANDIDATES,
            "teks canonical dataset",
        )
    )
    expert_path = Path(args.expert)
    if not expert_path.is_absolute():
        expert_path = (
            PROJECT_ROOT / expert_path
        )
    print(
        f"[DATA] Expert file : {expert_path}"
    )
    print(
        f"[DATA] Text column  : {text_column}"
    )
    print()
    (
        expert_df,
        expert_raw_text_col,
        expert_preprocessed_text_col,
        expert_label_col,
    ) = load_expert_validation(expert_path)
    print(
        f"[EXPERT] Text column  : "
        f"{expert_raw_text_col}"
    )
    print(
        f"[EXPERT] Label column : "
        f"{expert_label_col}"
    )
    print(
        f"[EXPERT] Valid labels : "
        f"{len(expert_df):,}"
    )
    print()
    return run_threshold_experiment(
        canonical_df=df,
        text_column=text_column,
        expert_df=expert_df,
        configured_threshold=args.configured_threshold,
    )
if __name__ == "__main__":
    main()
