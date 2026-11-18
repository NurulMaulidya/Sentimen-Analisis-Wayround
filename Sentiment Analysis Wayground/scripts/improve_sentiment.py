"""Prepare blind annotation sheets and compare research-only sentiment candidates."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from pipeline.aspect_detection import detect_all_aspects
from scripts.train_final_model import build_classifier, build_final_pipeline, build_vectorizer
from shared.candidate_sentiment import aspect_clauses, candidate_weak_label, preprocess_candidate
from shared.constants import ASPECTS
from shared.deployment_integrity import load_verified_pipeline
from shared.environment import require_locked_environment
from shared.labels import CANONICAL_CLASSES, normalize_label_or_raise
from shared.preprocessing import preprocess_text
from shared.research_config import RESEARCH_END_EXCLUSIVE
from shared.splitting import build_groups

SNAPSHOT = ROOT / "data/pipeline_output_v4"
ASPECT_COLUMNS = dict(zip(ASPECTS, ["akses", "gamifikasi", "materi", "teknis"]))
CANDIDATES = ["baseline", "label_pakar", "normalisasi", "leksikon", "karakter", "karakter_saja"]


def read_csv(path):
    return pd.read_csv(path, dtype={"reviewId": str}, keep_default_na=False)


def save_json(path, report):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def read_development():
    return read_csv(SNAPSHOT / "stage5_split_v2/development.csv")


def text_key(series):
    return series.str.casefold().str.replace(r"[^\w]+", " ", regex=True).str.strip()


def fresh_reviews(raw_path, canonical):
    raw = read_csv(raw_path)
    dates = pd.to_datetime(raw["at"], format="mixed", utc=True, errors="coerce")
    fresh = raw.loc[dates >= pd.Timestamp(RESEARCH_END_EXCLUSIVE, tz="UTC"), ["reviewId", "at", "content"]].copy()
    if "language_keep" in raw:
        fresh = fresh.loc[raw.loc[fresh.index, "language_keep"].astype(str).str.lower().isin(["true", "1"])]
    fresh["key"] = text_key(fresh["content"])
    known_keys = text_key(canonical["content"])
    fresh = fresh.loc[fresh["key"].ne("") & ~fresh["reviewId"].isin(canonical["reviewId"])
                      & ~fresh["key"].isin(known_keys)]
    fresh = fresh.drop_duplicates("reviewId").drop_duplicates("key")
    if fresh.empty:
        return fresh.drop(columns="key")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), norm="l2")
    known_matrix = vectorizer.fit_transform(known_keys)
    fresh_matrix = vectorizer.transform(fresh["key"])
    similarity = (fresh_matrix @ known_matrix.T).max(axis=1).toarray().ravel()
    # Similarity is a conservative screen for text reuse across periods.
    similar_fresh = (fresh_matrix @ fresh_matrix.T).toarray()
    retained = []
    for index, score in enumerate(similarity):
        if score < 0.90 and not (similar_fresh[index, retained] >= 0.90).any():
            retained.append(index)
    return fresh.iloc[retained].drop(columns="key")


def prepare(run, raw):
    if run.exists():
        raise FileExistsError(f"Run sudah ada: {run}")
    development = read_development()
    canonical = read_csv(SNAPSHOT / "stage3_final_labeled.csv")
    fresh = fresh_reviews(raw, canonical)
    if fresh.empty:
        raise ValueError("Tidak ada ulasan periode baru yang lolos pemeriksaan duplikat.")
    rating_label = development["rating"].astype(int).map(
        {1: "negatif", 2: "negatif", 3: "netral", 4: "positif", 5: "positif"}
    )
    priority = development.assign(
        priority=2 * development["sentiment_score"].astype(float).eq(0).astype(int)
        + rating_label.ne(development["sentiment_label_final"].str.lower()).astype(int)
    )
    sample = (priority.sample(frac=1, random_state=42)
              .sort_values("priority", ascending=False, kind="stable")
              .groupby("sentiment_label_final").head(50)
              .sample(frac=1, random_state=42))
    run.mkdir(parents=True)
    sample[["reviewId", "content"]].assign(label="", catatan="").to_csv(
        run / "anotasi_development.csv", index=False, encoding="utf-8-sig")
    fresh.to_csv(run / "sumber_uji_baru.csv", index=False, encoding="utf-8-sig")
    fresh.assign(label="", catatan="", **dict.fromkeys(ASPECT_COLUMNS.values(), "")).to_csv(
        run / "anotasi_uji_baru.csv", index=False, encoding="utf-8-sig")
    print(f"Anotasi development: {len(sample)}. Uji baru: {len(fresh)}. Folder: {run}")


def annotations(path, allowed, require_all=False):
    frame = read_csv(path)
    if frame["reviewId"].duplicated().any() or not set(frame["reviewId"]).issubset(allowed):
        raise ValueError("reviewId anotasi duplikat atau di luar partisi yang diizinkan.")
    filled = frame.loc[frame["label"].str.strip().ne("")].copy()
    filled["label"] = filled["label"].map(normalize_label_or_raise)
    if require_all and (set(filled["reviewId"]) != allowed or set(filled["label"]) != set(CANONICAL_CLASSES)):
        raise ValueError("Lengkapi semua label uji dan pastikan ketiga kelas terwakili.")
    return filled.set_index("reviewId")


def metrics(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=CANONICAL_CLASSES, average="macro", zero_division=0),
        "per_class": classification_report(actual, predicted, labels=CANONICAL_CLASSES,
                                           output_dict=True, zero_division=0),
        "confusion_matrix_labels": CANONICAL_CLASSES,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=CANONICAL_CLASSES).tolist(),
    }


def paired_groups(original, candidate):
    # Join duplicate groups from both text recipes to prevent fold leakage.
    parents = list(range(len(original)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for keys in (original, candidate):
        first = {}
        for index, key in enumerate(keys):
            parents[root(index)] = root(first.setdefault(key, index))
    return [root(index) for index in range(len(original))]


def model_for(name):
    if name in {"karakter", "karakter_saja"}:
        return Pipeline([
            ("features", FeatureUnion([
                ("word", build_vectorizer()),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         min_df=2, max_features=10000)),
            ])),
            ("classifier", build_classifier()),
        ])
    return build_final_pipeline()


def benchmark(run, annotation_path):
    require_locked_environment()
    output = run / ("benchmark_anotasi" if annotation_path else "benchmark_awal")
    if output.exists():
        raise FileExistsError(f"Folder hasil sudah ada: {output}")
    development = read_development()
    original = development["processed_text"]
    revised = pd.Series([preprocess_candidate(text, stored)
                         for text, stored in zip(development["content"], original)],
                        index=development.index)
    old_labels = development["sentiment_label_final"].str.lower()
    human_labels = old_labels.copy()
    annotated = pd.Series(False, index=development.index)
    if annotation_path:
        audit_sheet = read_csv(run / "anotasi_development.csv")
        allowed = set(audit_sheet["reviewId"]) & set(development["reviewId"])
        corrections = annotations(annotation_path, allowed)
        if corrections.empty:
            raise ValueError("Isi minimal satu label pada lembar development.")
        mapped = development["reviewId"].map(corrections["label"])
        annotated = mapped.notna()
        human_labels.loc[annotated] = mapped.loc[annotated]
    weak_labels = human_labels.copy()
    weak_mask = development["label_source"].eq("Lexicon") & ~annotated
    weak_labels.loc[weak_mask] = revised.loc[weak_mask].map(candidate_weak_label)
    groups = paired_groups(build_groups(development), build_groups(development.assign(processed_text=revised)))
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                 .split(original, old_labels, groups))
    # Common validation labels and identical folds for each ablation.
    variants = {
        "baseline": (original, old_labels),
        "normalisasi": (revised, human_labels),
        "leksikon": (revised, weak_labels),
        "karakter": (revised, weak_labels),
        "karakter_saja": (original, old_labels),
    }
    if annotated.any():
        variants["label_pakar"] = (original, human_labels)
    output.mkdir()
    scores = {}
    oof = pd.DataFrame({"reviewId": development["reviewId"], "label_evaluasi": human_labels,
                        "anotasi_baru": annotated})
    for name, (texts, labels) in variants.items():
        predicted = np.empty(len(development), dtype=object)
        fold_scores = []
        for train, valid in folds:
            model = model_for(name).fit(texts.iloc[train], labels.iloc[train])
            predicted[valid] = model.predict(texts.iloc[valid])
            fold_scores.append(f1_score(human_labels.iloc[valid], predicted[valid],
                                        labels=CANONICAL_CLASSES, average="macro", zero_division=0))
        oof[name] = predicted
        scores[name] = {"cv_macro_f1_mean": float(np.mean(fold_scores)), "fold_macro_f1": fold_scores,
                        "oof": metrics(human_labels, predicted)}
        if annotated.any():
            scores[name]["oof_annotated_subset"] = metrics(human_labels[annotated], predicted[annotated])
        joblib.dump(model_for(name).fit(texts, labels), output / f"{name}.joblib")
        print(f"{name}: macro F1 CV {np.mean(fold_scores):.2%}", flush=True)
    oof.to_csv(output / "prediksi_oof.csv", index=False)
    save_json(output / "hasil_cv.json", {
        "scope": "development_only_mixed_weak_and_human_labels",
        "human_annotations": int(annotated.sum()), "human_changes": int(human_labels.ne(old_labels).sum()),
        "normalized_text_changes": int(revised.ne(original).sum()),
        "weak_label_changes": int(weak_labels.ne(human_labels).sum()),
        "scores": scores,
    })


def aspect_metrics(frame, model, preprocess, detector):
    actual, predicted, detected = [], [], []
    for row in frame.itertuples(index=False):
        aspects = detector(row.content)
        for aspect, column in ASPECT_COLUMNS.items():
            target = getattr(row, column).strip().lower()
            actual.append("absen" if target == "absen" else normalize_label_or_raise(target))
            detected.append(bool(aspects[aspect]))
            text = preprocess(" ".join(aspects[aspect]))
            predicted.append(str(model.predict([text])[0]) if text else "absen")
    present = np.array(actual) != "absen"
    overlap = (np.array(predicted) != "absen") & present
    return {
        "reviews": len(frame),
        "detection_f1": f1_score(present, detected, zero_division=0),
        "joint_macro_f1": f1_score(actual, predicted, labels=["absen", *CANONICAL_CLASSES],
                                   average="macro", zero_division=0),
        "matched_aspects": int(overlap.sum()),
        "matched_sentiment_macro_f1": (
            f1_score(np.array(actual)[overlap], np.array(predicted)[overlap],
                     labels=CANONICAL_CLASSES, average="macro", zero_division=0) if overlap.any() else None
        ),
    }


def evaluate(run, annotation_path, candidate, benchmark_name):
    output = run / "evaluasi_uji_baru.json"
    if output.exists():
        raise FileExistsError("Evaluasi akhir sudah tercatat untuk run ini.")
    source = read_csv(run / "sumber_uji_baru.csv").set_index("reviewId")
    labels = annotations(annotation_path, set(source.index), require_all=True)
    frame = source.join(labels[["label", *ASPECT_COLUMNS.values()]])
    filled = frame[list(ASPECT_COLUMNS.values())].apply(lambda col: col.str.strip().ne(""))
    if (filled.any(axis=1) & ~filled.all(axis=1)).any():
        raise ValueError("Isi semua kolom aspek untuk baris yang dinilai, atau kosongkan semuanya.")
    v4, metadata = load_verified_pipeline()
    challenger = joblib.load(run / benchmark_name / f"{candidate}.joblib")
    transform = preprocess_text if candidate in {"label_pakar", "karakter_saja"} else preprocess_candidate
    results = {"v4_version": metadata["model_version"], "candidate": candidate,
               "benchmark": benchmark_name, "rows": len(frame),
               "v4": metrics(frame["label"], v4.predict(frame["content"].map(preprocess_text))),
               "candidate_metrics": metrics(frame["label"], challenger.predict(frame["content"].map(transform)))}
    aspect_frame = frame.loc[filled.all(axis=1)]
    if len(aspect_frame):
        results["aspects"] = {
            "v4": aspect_metrics(aspect_frame, v4, preprocess_text, detect_all_aspects),
            "split_clauses_same_v4": aspect_metrics(aspect_frame, v4, preprocess_text, aspect_clauses),
        }
    save_json(output, results)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "benchmark", "evaluate"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/01_semua_data.csv")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--candidate", choices=CANDIDATES[1:])
    parser.add_argument("--benchmark", choices=["benchmark_awal", "benchmark_anotasi"], default="benchmark_anotasi")
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "runs") or run == ROOT / "runs":
        parser.error("--run harus merupakan subfolder runs/")
    if args.stage == "prepare":
        prepare(run, args.raw)
    elif args.stage == "benchmark":
        benchmark(run, args.annotations)
    elif args.annotations and args.candidate:
        evaluate(run, args.annotations, args.candidate, args.benchmark)
    else:
        parser.error("evaluate memerlukan --annotations dan --candidate")


if __name__ == "__main__":
    main()

