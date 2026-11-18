"""
Test Script untuk Sentiment Analysis Model
==========================================
Jalankan script ini untuk testing pipeline secara langsung.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.preprocessing import preprocess_text
from shared.lexicon import classify_sentiment_tuple, get_label_distribution
from pipeline.aspect_detection import detect_all_aspects


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n" + "="*60)
    print("TEST 1: PREPROCESSING")
    print("="*60)

    test_texts = [
        "Aplikasinya seru banget buat kuis tapi ga bisa dibuka pas offline error mulu",
        "Sangat membantu untuk belajar, soalnya lengkap dan bagus",
        "Login sulit dan aplikasi sering crash saat ujian",
        "Dapat poin banyak dari game kompetisinya, seru banget"
    ]

    for text in test_texts:
        result = preprocess_text(text)
        print(f"\nInput:  {text[:60]}...")
        print(f"Output: {result}")


def test_lexicon_labeling():
    """Test lexicon-based sentiment labeling."""
    print("\n" + "="*60)
    print("TEST 2: LEXICON LABELING")
    print("="*60)

    test_texts = [
        ("Aplikasi sangat bagus dan membantu belajar", "should be positive"),
        ("Sangat kecewa, error terus dan sulit login", "should be negative"),
        ("Aplikasi untuk mengerjakan tugas sekolah", "should be neutral"),
        ("Gamnya seru dapat poin banyak tapi sering lag", "mixed/mild negative"),
        ("Tidak bisa digunakan saat ujian, sangat merugikan", "should be negative"),
    ]

    for text, expected in test_texts:
        processed = preprocess_text(text)
        label, score, norm_score = classify_sentiment_tuple(processed)
        print(f"\nText: {text[:50]}...")
        print(f"  Processed: {processed[:50]}...")
        print(f"  Label: {label} (score: {score:.3f}, normalized: {norm_score:.3f}) - {expected}")


def test_aspect_detection():
    """Test aspect detection."""
    print("\n" + "="*60)
    print("TEST 3: ASPECT DETECTION")
    print("="*60)

    test_texts = [
        "Login sulit dan koneksi sering timeout saat loading",
        "Soal-soalnya lengkap sesuai kurikulum, cocok untuk latihan ujian",
        "Gamnya seru dapat poin banyak dari kompetisi ranking",
        "Aplikasi sering crash dan error saat sedang serius, sangat mengganggu",
        "Mudah diakses dan tidak lemot, bagus untuk belajar offline"
    ]

    for text in test_texts:
        aspects = detect_all_aspects(text)
        print(f"\nText: {text}")
        print("  Detected aspects:")
        for aspect, sentences in aspects.items():
            if aspect != 'Umum' and sentences:
                print(f"    [{aspect}]: {len(sentences)} kalimat")


def test_combined_pipeline():
    """Test combined pipeline: preprocessing -> aspect -> sentiment."""
    print("\n" + "="*60)
    print("TEST 4: COMBINED PIPELINE")
    print("="*60)

    text = "Aplikasi ini seru untuk belajar tapi sering error saat ujian dan sulit login"

    # Step 1: Preprocess
    processed = preprocess_text(text)

    # Step 2: Detect aspects
    aspects = detect_all_aspects(text)

    # Step 3: Label overall sentiment
    label, score, norm_score = classify_sentiment_tuple(processed)

    print(f"\nOriginal: {text}")
    print(f"Processed: {processed}")
    print(f"Overall Sentiment: {label} (score: {score:.3f})")
    print("\nAspects detected:")

    for aspect, sentences in aspects.items():
        if aspect != 'Umum' and sentences:
            # Label aspect sentiment
            aspect_text = ' '.join(sentences)
            aspect_processed = preprocess_text(aspect_text)
            aspect_label, _, _ = classify_sentiment_tuple(aspect_processed)
            print(f"  [{aspect}]: {aspect_label}")
            print(f"    Sentences: {sentences[:2]}")


def create_sample_data():
    """Create sample data for testing."""
    print("\n" + "="*60)
    print("CREATING SAMPLE DATA")
    print("="*60)

    import pandas as pd

    sample_reviews = [
        {"content": "Aplikasi sangat bagus membantu belajar", "rating": 5},
        {"content": "Seru banget dapat poin banyak", "rating": 5},
        {"content": "Soalnya lengkap cocok untuk latihan", "rating": 4},
        {"content": "Login sulit dan error terus", "rating": 1},
        {"content": "Sering crash saat ujian", "rating": 1},
        {"content": "Bagus untuk mengerjakan tugas sekolah", "rating": 4},
        {"content": "Gamnya asyik tapi sering lag", "rating": 3},
        {"content": "Koneksi sering timeout saat loading", "rating": 2},
        {"content": "Tidak bisa digunakan offline", "rating": 2},
        {"content": "Mudah diakses dan cepat", "rating": 5},
    ]

    df = pd.DataFrame(sample_reviews)

    # Preprocess
    df['processed_text'] = df['content'].apply(preprocess_text)

    # Label sentiment
    results = df['processed_text'].apply(classify_sentiment_tuple)
    df['sentiment_label'] = [r[0] for r in results]
    df['sentiment_score'] = [r[1] for r in results]

    # Detect aspects
    df['aspects'] = df['content'].apply(lambda x: list(detect_all_aspects(x).keys()))

    print("\nSample DataFrame:")
    print(df.to_string())

    # Distribution
    dist = get_label_distribution(df)
    print("\nSentiment Distribution:")
    for label, stats in dist.items():
        print(f"  {label}: {stats['count']} ({stats['percentage']}%)")

    return df


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS MODEL - TESTING SCRIPT")
    print("="*70)

    try:
        test_preprocessing()
        test_lexicon_labeling()
        test_aspect_detection()
        test_combined_pipeline()
        create_sample_data()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
