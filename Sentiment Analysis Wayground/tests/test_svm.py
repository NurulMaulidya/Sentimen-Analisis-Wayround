"""
Test SVM Training - Pipeline Lengkap
====================================
Test training model SVM dengan data sample.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.preprocessing import preprocess_text
from shared.lexicon import classify_sentiment
from pipeline.aspect_detection import detect_all_aspects

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)


def create_sample_dataset():
    """Buat dataset sample untuk training."""
    samples = [
        # Positif samples
        ("Aplikasi sangat bagus membantu belajar", "Positif"),
        ("Seru banget dapat poin banyak dari game", "Positif"),
        ("Soalnya lengkap dan cocok untuk latihan", "Positif"),
        ("Bagus untuk mengerjakan tugas sekolah", "Positif"),
        ("Mudah diakses dan tidak lemot", "Positif"),
        ("Konten lengkap pembahasan jelas", "Positif"),
        ("Game kompetitif seru dapat ranking", "Posatif"),
        ("Sangat membantu guru dalam mengajarkan", "Positif"),
        ("Aplikasi pembelajaran terbaik", "Positif"),
        ("Recommended untuk belajar", "Positif"),

        # Negatif samples
        ("Sangat kecewa error terus crash", "Negatif"),
        ("Login sulit tidak bisa akses", "Negatif"),
        ("Aplikasi sering hang dan lag", "Negatif"),
        ("Sulit koneksi timeout terus", "Negatif"),
        ("Bug banyak данные hilang", "Negatif"),
        ("Iklan terlalu banyak mengganggu", "Negatif"),
        ("Sering force close saat ujian", "Negatif"),
        ("Tidak bisa digunakan offline", "Negatif"),
        ("Sangat lambat loading lama", "Negatif"),
        ("Koneksi error tidak stabil", "Negatif"),

        # Netral samples
        ("Aplikasi untuk mengerjakan tugas", "Netral"),
        ("Ini aplikasi pembelajaran", "Netral"),
        ("Ada soal pilihan ganda", "Netral"),
        ("Bisa untuk latihan", "Netral"),
        ("Nomor urut dari satu sampai sepuluh", "Netral"),
        ("Tersedia dalam bahasa Indonesia", "Netral"),
        ("Developed by Quizizz Inc", "Netral"),
        ("Platform untuk kuis online", "Netral"),
        ("Memiliki fitur tertentu", "Netral"),
        ("Cocok untuk berbagai mata pelajaran", "Netral"),
    ]

    return samples


def test_svm_pipeline():
    """Test pipeline SVM lengkap."""
    print("\n" + "="*70)
    print("TEST: SVM PIPELINE LENGKAP")
    print("="*70)

    # Create dataset
    samples = create_sample_dataset()
    texts = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    print(f"\nTotal samples: {len(samples)}")
    print(f"Labels distribution:")
    for label in ['Positif', 'Netral', 'Negatif']:
        count = labels.count(label)
        print(f"  {label}: {count}")

    # Preprocess
    print("\n[1/5] Preprocessing texts...")
    processed_texts = [preprocess_text(t) for t in texts]
    print(f"  Sample: '{texts[0]}' -> '{processed_texts[0]}'")

    # TF-IDF
    print("\n[2/5] TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(processed_texts)
    y = np.array(labels)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Split
    print("\n[3/5] Splitting data (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
        # Removed stratify because class sizes are too small
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Train SVM
    print("\n[4/5] Training SVM (kernel=linear, C=10)...")
    model = SVC(kernel='linear', C=10, random_state=42)
    model.fit(X_train, y_train)
    print("  Training complete!")

    # Evaluate
    print("\n[5/5] Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['Positif', 'Netral', 'Negatif'])
    print(f"            Predicted")
    print(f"            Pos    Net    Neg")
    print(f"Actual Pos   {cm[0,0]:3d}   {cm[0,1]:3d}   {cm[0,2]:3d}")
    print(f"       Net   {cm[1,0]:3d}   {cm[1,1]:3d}   {cm[1,2]:3d}")
    print(f"       Neg   {cm[2,0]:3d}   {cm[2,1]:3d}   {cm[2,2]:3d}")

    # Test prediction
    print("\n" + "="*50)
    print("TEST PREDICTIONS")
    print("="*50)

    test_samples = [
        "Aplikasi sangat bagus membantu belajar",
        "Error terus crash tidak bisa digunakan",
        "Soalnya untuk latihan saja"
    ]

    for text in test_samples:
        processed = preprocess_text(text)
        X_new = vectorizer.transform([processed])
        prediction = model.predict(X_new)[0]

        # Get confidence (distance from hyperplane)
        distances = model.decision_function(X_new)[0]
        confidence = max(abs(d) for d in distances)

        print(f"\nText: {text}")
        print(f"Processed: {processed}")
        print(f"Prediction: {prediction} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    test_svm_pipeline()
