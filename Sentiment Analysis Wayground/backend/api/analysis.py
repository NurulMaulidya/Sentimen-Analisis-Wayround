"""
Analysis API Blueprint
======================
All API endpoints for the sentiment analysis website.

This blueprint implements the "pengujian model" flow described in
metodologi_final.md Bab 15:

    upload -> preprocess -> predict -> evaluate -> aspect analysis

The model is NEVER retrained here. The deployed artifact
``models/final_v4/final_pipeline.joblib`` is loaded and only ``predict()`` is
called. TF-IDF is only ever used through ``transform()`` inside the
pre-fitted pipeline.
"""
from flask import Blueprint, request, jsonify, send_file
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import os
import csv

from werkzeug.utils import secure_filename

from backend.database_models import (
    db, AnalysisSession, AnalysisUploadContext, ReviewData, AspectResult, EvaluationResult,
    AnalysisExecution, UploadRow, SinglePrediction, ReviewFeatures, utcnow,
)
from backend.services.preprocessing_service import (
    preprocess_full, STAGE_ORDER, STAGE_DISPLAY_NAMES, STAGE_TO_FIELD
)
from backend.services.prediction_service import predict_batch, predict_with_labels
from backend.services.inference_models import CANDIDATE_KEY, load_inference_model, model_key_for_version
from backend.services.aspect_service import (
    ASPECTS,
    ASPECT_SENTIMENT_SCOPE,
    aspect_sentiment_labels,
    detect_aspects_for_review,
    aggregate_aspect_distribution,
    get_aspect_summary,
)
from backend.services.analysis_state import invalidate_results, update_status, fingerprint_signature
from shared.deployment_integrity import (
    effective_fingerprint,
    runtime_fingerprint,
)
from backend.services.upload_service import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_ROWS,
    NEW_DATA,
    POSSIBLE_CONTENT_COLUMNS,
    POSSIBLE_LABEL_COLUMNS,
    UploadValidationError,
    allowed_file,
    count_rows,
    find_content_column,
    find_label_column,
    read_uploaded_dataframe,
    validate_upload,
)
from backend.services.web_flow import transform_reviews, feature_details, fitted_features, source_row_dict, session_summary, review_upload_details
from shared.labeling import label_text
from shared.labels import (
    CANONICAL_CLASSES,
    normalize_label,
    normalize_label_or_raise,
    to_display,
)

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/v1')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_uploaded_file(file):
    """Save uploaded file and return path."""
    from backend.config import UPLOAD_DIR

    if not allowed_file(file.filename):
        return None

    # secure_filename mencegah path traversal pada nama file.
    safe_name = secure_filename(file.filename) or "upload.csv"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4
    filename = f'{uuid4().hex}_{safe_name}'
    filepath = UPLOAD_DIR / filename
    file.save(filepath)
    return filepath


# =============================================================================
# ANALYSIS MANAGEMENT
# =============================================================================

@analysis_bp.route('/analysis', methods=['GET'])
def get_analyses():
    """Get all analysis sessions."""
    analyses = AnalysisSession.query.order_by(AnalysisSession.created_at.desc()).all()
    return jsonify({
        'success': True,
        'analyses': [a.to_dict() for a in analyses]
    })


@analysis_bp.route('/analysis/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get single analysis session."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)

    # Get counts
    total_reviews = ReviewData.query.filter_by(analysis_id=analysis_id).count()
    duplicates = ReviewData.query.filter_by(analysis_id=analysis_id, is_duplicate=True).count()
    predicted = ReviewData.query.filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.predicted_label.isnot(None)
    ).count()

    result = analysis.to_dict()
    result['total_reviews'] = total_reviews
    result['duplicates'] = duplicates
    result['predicted'] = result['prediction_count']
    result['flow'] = session_summary(analysis)

    return jsonify({
        'success': True,
        'analysis': result
    })


@analysis_bp.route('/analysis/<int:analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete analysis session and all related data."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)

    # Delete file if exists
    if analysis.filepath and os.path.exists(analysis.filepath):
        try:
            os.remove(analysis.filepath)
        except:
            pass

    db.session.delete(analysis)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'Analysis {analysis_id} deleted'
    })


# =============================================================================
# FILE UPLOAD
# =============================================================================

@analysis_bp.route('/analysis/upload', methods=['POST'])
def upload_analysis():
    """
    Upload a test file and create a new analysis session.

    Validation performed here (metodologi_final.md Bab 15 & 19):
      * extension must be .csv / .tsv / .txt
      * file must be parseable and non-empty
      * a review/content column must exist
      * row count limit
      * actual labels, when present, must map to a known class
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': (
                'Format file tidak didukung. Gunakan file .csv atau .tsv '
                'yang berisi kolom ulasan.'
            )
        }), 400

    filepath = save_uploaded_file(file)
    if filepath is None:
        return jsonify({'error': 'Failed to save file'}), 500

    mode = request.form.get('mode', NEW_DATA)
    origins = []
    committed = False
    try:
        try:
            df = read_uploaded_dataframe(filepath)
        except Exception as exc:
            return jsonify({
                'error': f'File tidak dapat dibaca: {exc}. '
                         'Pastikan file CSV/TSV yang valid dan tidak rusak.'
            }), 400

        # Validasi struktur file: kolom ulasan wajib, label opsional tetapi
        # nilai yang tidak dikenal harus ditolak.
        try:
            content_col, label_col, has_labels = validate_upload(
                df, request.form.get('content_column'), request.form.get('label_column'),
                mode=mode, origins_out=origins,
            )
        except UploadValidationError as exc:
            payload = {'error': exc.message}
            payload.update(exc.extra)
            return jsonify(payload), 400

        # Create analysis session
        analysis = AnalysisSession(
            name=request.form.get('name', f'Analysis {datetime.now().strftime("%Y-%m-%d %H:%M")}'),
            filename=file.filename,
            filepath=str(filepath),
            original_count=len(df),
            has_actual_labels=has_labels,
            label_column=label_col,
            status='uploaded'
        )
        db.session.add(analysis)
        db.session.flush()

        # Save reviews. Penghitungan baris memakai count_rows() supaya
        # angka yang dilaporkan sama dengan angka yang benar-benar disimpan.
        counts = count_rows(df, content_col, label_col, has_labels)
        review_origins = {}
        for row_number, (_, row) in enumerate(df.iterrows(), start=1):
            content = str(row[content_col]) if pd.notna(row.get(content_col)) else ''
            source = UploadRow(analysis_id=analysis.id, row_number=row_number, content=content,
                               raw_label=str(row[label_col]) if label_col and pd.notna(row[label_col]) else '' if label_col else None,
                               raw_data=json.dumps({str(k): str(v) if pd.notna(v) else '' for k, v in row.items()}))
            db.session.add(source)
            if not content.strip():
                continue

            # Baris tanpa label aktual tetap diprediksi; labelnya hanya
            # tidak bisa dipakai sebagai pembanding saat evaluasi.
            actual_label = normalize_label(row.get(label_col)) if has_labels else None

            review = ReviewData(
                analysis_id=analysis.id,
                original_content=content,
                actual_label=actual_label
            )
            db.session.add(review)
            db.session.flush()
            source.review_id = review.id
            review_origins[str(review.id)] = origins[row_number - 1]

        analysis.upload_context = AnalysisUploadContext(
            mode=mode, origins=origins, review_origins=review_origins,
        )
        db.session.commit()
        committed = True

        return jsonify({
            'success': True,
            'analysis': analysis.to_dict(),
            'columns_detected': list(df.columns),
            'content_column': content_col,
            'label_column': label_col,
            'has_actual_labels': has_labels,
            'rows_uploaded': len(df),
            'rows_saved': counts['rows_saved'],
            'rows_skipped_empty': counts['rows_skipped_empty'],
            # Tidak ada baris yang "dilewati" karena label kosong: baris
            # tersebut tetap diprediksi, hanya tidak masuk hitungan evaluasi.
            'rows_skipped_without_label': 0,
            'rows_without_actual_label': counts['rows_without_actual_label'],
            'rows_preserved': len(df),
        })

    except Exception as e:
        db.session.rollback()
        # Clean up file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500
    finally:
        if not committed and filepath and os.path.exists(filepath):
            os.remove(filepath)


# =============================================================================
# PIPELINE OPERATIONS
# =============================================================================

@analysis_bp.route('/analysis/<int:analysis_id>/deduplicate', methods=['POST'])
def deduplicate_analysis(analysis_id):
    """Remove duplicate reviews based on content."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    invalidate_results(analysis, clear_preprocessing=True)

    reviews = ReviewData.query.filter_by(analysis_id=analysis_id).all()

    seen_content = {}
    duplicates_count = 0

    for review in reviews:
        normalized = review.original_content.strip().lower()
        if not normalized:
            review.is_duplicate = True
            duplicates_count += 1
            continue

        if normalized in seen_content:
            review.is_duplicate = True
            review.duplicate_of = seen_content[normalized]
            duplicates_count += 1
        else:
            seen_content[normalized] = review.id
            review.is_duplicate = False
            review.duplicate_of = None

    after_dedup = len(reviews) - duplicates_count

    analysis.dedup_count = duplicates_count
    analysis.after_dedup_count = after_dedup
    analysis.status = 'deduplicated'

    db.session.commit()

    return jsonify({
        'success': True,
        'analysis_id': analysis_id,
        'total_reviews': len(reviews),
        'duplicates_found': duplicates_count,
        'after_dedup': after_dedup
    })


@analysis_bp.route('/analysis/<int:analysis_id>/preprocess', methods=['POST'])
def preprocess_analysis(analysis_id):
    """Run 7-step preprocessing on all reviews."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    if analysis.status == 'uploaded':
        return jsonify({'error': 'Jalankan deduplikasi terlebih dahulu'}), 409
    invalidate_results(analysis, clear_preprocessing=True)

    reviews = ReviewData.query.filter_by(
        analysis_id=analysis_id,
        is_duplicate=False
    ).all()

    lexicon_labels = {}
    processed = 0
    if not reviews:
        return jsonify(error='Tidak ada ulasan valid setelah validasi dan deduplikasi. Unggah file dengan teks ulasan.'), 400
    for review in reviews:
        result = preprocess_full(review.original_content)

        review.cleaned_text = result['cleaned']
        review.case_folded_text = result['case_folded']
        review.tokenized_tokens = json.dumps(result['tokenized'])
        review.normalized_tokens = json.dumps(result['normalized'])
        review.negation_tokens = json.dumps(result['negation'])
        review.stopwords_tokens = json.dumps(result['stopwords'])
        review.stemmed_tokens = json.dumps(result['stemmed'])
        review.final_processed_text = result['final']
        if result['final']:
            lexicon_labels[str(review.id)] = normalize_label(label_text(result['final'])['label'])

        processed += 1

    if analysis.upload_context:
        analysis.upload_context.lexicon_labels = lexicon_labels
    analysis.status = 'preprocessed'
    analysis.execution = AnalysisExecution(preprocessing_signature=fingerprint_signature(runtime_fingerprint()))
    db.session.commit()

    return jsonify({
        'success': True,
        'analysis_id': analysis_id,
        'processed': processed
    })


@analysis_bp.route('/analysis/<int:analysis_id>/aspects', methods=['POST'])
def analyze_aspects(analysis_id):
    """Detect aspects for all reviews."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    if not analysis.execution or not analysis.execution.model_version:
        return jsonify({'error': 'Jalankan prediksi dengan versi model tercatat terlebih dahulu'}), 409
    model_key = model_key_for_version(analysis.execution.model_version)
    pipeline, metadata, digest = load_inference_model(model_key)
    if (metadata['model_version'] != analysis.execution.model_version
            or digest != analysis.execution.pipeline_sha256):
        return jsonify(error='Model hasil tersimpan berbeda. Jalankan ulang prediksi.'), 409

    # Delete old aspect results
    AspectResult.query.filter_by(analysis_id=analysis_id).delete()

    reviews = ReviewData.query.filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.is_duplicate == False,
        ReviewData.predicted_label.isnot(None)
    ).all()

    detected_count = 0

    for review in reviews:
        result = detect_aspects_for_review(
            review.original_content,
            review.predicted_label,
            use_clauses=model_key == CANDIDATE_KEY,
        )

        # Sentimen per aspek dihitung dari kalimat-kalimat aspek itu sendiri
        # (definisi ABSA), bukan disalin dari label tingkat review.
        if result['aspect_details']:
            aspect_labels = aspect_sentiment_labels(
                result['aspect_details'], pipeline=pipeline
            )
            for aspect, label in aspect_labels.items():
                result['aspect_details'][aspect]['sentiment'] = label

        # Store detected aspects in review
        review.detected_aspects = json.dumps(result['detected_aspects'])

        # Store aspect results
        for aspect, details in result['aspect_details'].items():
            aspect_result = AspectResult(
                analysis_id=analysis_id,
                review_id=review.id,
                aspect_name=aspect,
                sentences=json.dumps(details['sentences']),
                sentiment=details.get('sentiment')
            )
            db.session.add(aspect_result)

        if result['has_aspects']:
            detected_count += 1

    analysis.execution.aspects_ready = True
    update_status(analysis)
    db.session.commit()

    return jsonify({
        'success': True,
        'analysis_id': analysis_id,
        'reviews_with_aspects': detected_count
    })


@analysis_bp.route('/analysis/<int:analysis_id>/predict', methods=['POST'])
def predict_analysis(analysis_id):
    """Run prediction on all preprocessed reviews."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)

    reviews = ReviewData.query.filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.is_duplicate == False,
        ReviewData.final_processed_text.isnot(None),
        ReviewData.final_processed_text != ''
    ).all()
    if not reviews:
        return jsonify({'success': False, 'error': 'Tidak ada teks hasil preprocessing yang valid'}), 400

    # Load pipeline once
    data = request.get_json(silent=True) or {}
    inspected = ReviewFeatures.query.filter_by(analysis_id=analysis_id).first()
    inspected_version = inspected.model_version if inspected else None
    pipeline, metadata, digest = load_inference_model(data.get('model', model_key_for_version(inspected_version)))
    if 'model' not in data and inspected_version and inspected_version != metadata['model_version']:
        return jsonify(error='Versi TF-IDF tersimpan berbeda. Ulangi transformasi TF-IDF.'), 409
    if not analysis.execution or analysis.execution.preprocessing_signature != fingerprint_signature(effective_fingerprint(metadata)):
        return jsonify({'error': 'Preprocessing perlu dijalankan ulang agar sesuai dengan model aktif'}), 409
    invalidate_results(analysis, keep_signature=True)

    reviews, matrix = transform_reviews(analysis, pipeline, metadata)
    predicted = 0
    valid_indices = [i for i, review in enumerate(reviews) if (review.final_processed_text or '').strip() and matrix.getrow(i).nnz]
    if not valid_indices:
        analysis.status = 'preprocessed'
        db.session.commit()
        return jsonify(error='Tidak ada fitur TF-IDF yang dikenali. Ulasan tidak diprediksi; periksa hasil TF-IDF.'), 400
    if valid_indices:
        predictions = pipeline.named_steps['classifier'].predict(matrix[valid_indices])
        for index, pred in zip(valid_indices, predictions):
            review = reviews[index]
            review.predicted_label = normalize_label(pred)
            actual = normalize_label(review.actual_label)
            if actual is not None:
                review.is_correct = review.predicted_label == actual
            predicted += 1

    analysis.status = 'predicted'
    analysis.execution.model_version = metadata['model_version']
    analysis.execution.pipeline_sha256 = digest
    analysis.execution.predicted_at = utcnow()
    db.session.commit()

    # Get distribution
    distribution = {
        'positif': 0,
        'netral': 0,
        'negatif': 0
    }
    for label, count in db.session.query(
        ReviewData.predicted_label,
        db.func.count(ReviewData.id)
    ).filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.predicted_label.isnot(None)
    ).group_by(ReviewData.predicted_label).all():
        if label in distribution:
            distribution[label] = count

    return jsonify({
        'success': True,
        'analysis_id': analysis_id,
        'predicted': predicted,
        'model_version': metadata['model_version'],
        'distribution': distribution
    })


@analysis_bp.route('/analysis/<int:analysis_id>/evaluate', methods=['POST'])
def evaluate_analysis(analysis_id):
    """Evaluate predictions against actual labels."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    if not analysis.execution or not analysis.execution.model_version:
        return jsonify({'error': 'Jalankan prediksi dengan versi model tercatat terlebih dahulu'}), 409

    if not analysis.has_actual_labels:
        return jsonify({
            'success': False,
            'error': 'No actual labels available for evaluation'
        }), 400

    reviews = ReviewData.query.filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.is_duplicate == False,
        ReviewData.actual_label.isnot(None),
        ReviewData.predicted_label.isnot(None)
    ).all()

    if not reviews:
        return jsonify({
            'success': False,
            'error': 'No reviews with both actual and predicted labels'
        }), 400

    # Get predictions. Only rows that have BOTH an actual and a predicted
    # label take part in the evaluation.
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score
    )

    actual = [normalize_label(r.actual_label) for r in reviews]
    predicted = [normalize_label(r.predicted_label) for r in reviews]
    if any(label not in CANONICAL_CLASSES for label in actual):
        return jsonify({'error': 'Label aktual tidak dikenal'}), 400

    labels = list(CANONICAL_CLASSES)

    metrics = {
        'accuracy': float(accuracy_score(actual, predicted)),
        'precision_macro': float(precision_score(actual, predicted, labels=labels, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(actual, predicted, labels=labels, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(actual, predicted, labels=labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(actual, predicted, labels=labels, average='weighted', zero_division=0))
    }

    cm = confusion_matrix(actual, predicted, labels=labels).tolist()
    report = classification_report(actual, predicted, labels=labels, zero_division=0, output_dict=True)

    actual_vs_predicted = [
        {'id': r.id, 'original_content': r.original_content, 'actual': a, 'predicted': p, 'correct': a == p}
        for r, a, p in zip(reviews, actual, predicted)
    ]

    correct = sum(1 for avp in actual_vs_predicted if avp['correct'])
    incorrect = len(actual_vs_predicted) - correct

    # Save evaluation
    analysis.evaluation = None
    db.session.flush()
    evaluation = EvaluationResult(
        analysis_id=analysis_id,
        metrics=json.dumps(metrics),
        classification_report=json.dumps(report),
        confusion_matrix=json.dumps(cm),
        actual_vs_predicted=json.dumps(actual_vs_predicted),
        correct_count=correct,
        incorrect_count=incorrect
    )
    db.session.add(evaluation)

    analysis.execution.evaluation_ready = True
    update_status(analysis)
    db.session.commit()

    return jsonify({
        'success': True,
        'analysis_id': analysis_id,
        'metrics': metrics,
        'evaluation_scope': analysis.evaluation_scope,
        'correct_count': correct,
        'incorrect_count': incorrect,
        'total': len(reviews)
    })


# =============================================================================
# DATA RETRIEVAL
# =============================================================================

@analysis_bp.route('/analysis/<int:analysis_id>/original', methods=['GET'])
def get_original_data(analysis_id):
    """Get original dataset with pagination."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    page = max(1, request.args.get('page', 1, type=int))
    limit = max(1, min(100, request.args.get('limit', 50, type=int)))
    search = request.args.get('search', '')

    sources = UploadRow.query.filter_by(analysis_id=analysis_id)
    if sources.count():
        if search:
            sources = sources.filter(UploadRow.content.ilike(f'%{search}%'))
        status = request.args.get('status')
        rows = [source_row_dict(r, context=analysis.upload_context) for r in sources.order_by(UploadRow.row_number).all()]
        if status:
            rows = [r for r in rows if r['validation_status'] == status]
        total = len(rows)
        return jsonify(success=True, data=rows[(page - 1) * limit:page * limit], source_preserved=True,
                       pagination={'page': page, 'limit': limit, 'total': total, 'total_pages': (total + limit - 1) // limit})

    query = ReviewData.query.filter_by(analysis_id=analysis_id)

    if search:
        query = query.filter(ReviewData.original_content.ilike(f'%{search}%'))

    total = query.count()
    reviews = query.order_by(ReviewData.id).offset((page - 1) * limit).limit(limit).all()

    return jsonify({
        'success': True,
        'data': [r.to_dict() for r in reviews],
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit if limit else 0
        }
    })


@analysis_bp.route('/analysis/<int:analysis_id>/deduplicated', methods=['GET'])
def get_deduplicated_data(analysis_id):
    """Get deduplicated dataset."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    page = max(1, request.args.get('page', 1, type=int))
    limit = max(1, min(100, request.args.get('limit', 50, type=int)))
    show_duplicates = request.args.get('show_duplicates', 'false').lower() == 'true'

    query = ReviewData.query.filter_by(analysis_id=analysis_id)

    if not show_duplicates:
        query = query.filter_by(is_duplicate=False)

    total = query.count()
    reviews = query.order_by(ReviewData.id).offset((page - 1) * limit).limit(limit).all()

    # Stats
    total_reviews = ReviewData.query.filter_by(analysis_id=analysis_id).count()
    duplicates = ReviewData.query.filter_by(analysis_id=analysis_id, is_duplicate=True).count()

    return jsonify({
        'success': True,
        'data': [{**r.to_dict(), **review_upload_details(r, analysis.upload_context)} for r in reviews],
        'stats': {
            'total': total_reviews,
            'duplicates': duplicates,
            'after_dedup': total_reviews - duplicates
        },
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit if limit else 0
        }
    })


@analysis_bp.route('/analysis/<int:analysis_id>/preprocessing/<stage>', methods=['GET'])
def get_preprocessing_stage(analysis_id, stage):
    """Get data at specific preprocessing stage."""
    db.get_or_404(AnalysisSession, analysis_id)
    if stage not in STAGE_TO_FIELD and stage != 'all':
        return jsonify({'error': f'Invalid stage: {stage}'}), 400

    page = max(1, request.args.get('page', 1, type=int))
    limit = max(1, min(100, request.args.get('limit', 50, type=int)))
    search = request.args.get('search', '')

    query = ReviewData.query.filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.is_duplicate == False
    )

    if search:
        query = query.filter(ReviewData.original_content.ilike(f'%{search}%'))

    total = query.count()
    reviews = query.order_by(ReviewData.id).offset((page - 1) * limit).limit(limit).all()

    # Format data based on stage
    data = []
    field_map = {
        'original': 'original_content',
        'cleaning': 'cleaned_text',
        'case_folding': 'case_folded_text',
        'tokenizing': 'tokenized_tokens',
        'normalization': 'normalized_tokens',
        'negation': 'negation_tokens',
        'stopwords': 'stopwords_tokens',
        'stemming': 'stemmed_tokens',
        'final': 'final_processed_text'
    }

    for r in reviews:
        item = {
            'id': r.id,
            'original_content': r.original_content
        }

        if stage == 'all':
            for s, field in field_map.items():
                if 'tokens' in field:
                    val = getattr(r, field)
                    item[s] = json.loads(val) if val else []
                else:
                    item[s] = getattr(r, field, '')
        else:
            field = field_map.get(stage)
            if field:
                if 'tokens' in field:
                    val = getattr(r, field)
                    item[stage] = json.loads(val) if val else []
                else:
                    item[stage] = getattr(r, field, '')

        data.append(item)

    return jsonify({
        'success': True,
        'stage': stage,
        'data': data,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit if limit else 0
        }
    })


@analysis_bp.route('/analysis/<int:analysis_id>/compare/<int:review_id>', methods=['GET'])
def get_review_comparison(analysis_id, review_id):
    """Get all preprocessing stages for a single review."""
    review = ReviewData.query.filter_by(
        id=review_id,
        analysis_id=analysis_id
    ).first_or_404()

    return jsonify({
        'success': True,
        'data': review.to_comparison_dict()
    })


@analysis_bp.route('/analysis/<int:analysis_id>/aspects', methods=['GET'])
def get_aspects_data(analysis_id):
    """Get aspect analysis results."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    page = max(1, request.args.get('page', 1, type=int))
    limit = max(1, min(100, request.args.get('limit', 50, type=int)))
    aspect_filter = request.args.get('aspect')

    # Get aspect results
    query = AspectResult.query.filter_by(analysis_id=analysis_id)

    if aspect_filter:
        query = query.filter_by(aspect_name=aspect_filter)

    total = query.count()
    aspects = query.order_by(AspectResult.id).offset((page - 1) * limit).limit(limit).all()

    # Get distribution
    distribution = aggregate_aspect_distribution([
        {'detected_aspects': [a.aspect_name], 'aspect_sentiment': a.sentiment}
        for a in AspectResult.query.filter_by(analysis_id=analysis_id).all()
    ])

    # Jumlah ulasan unik yang memiliki minimal satu aspek. Karena satu
    # ulasan dapat masuk ke beberapa aspek, nilai ini bisa lebih kecil
    # daripada total kemunculan aspek.
    reviews_with_aspects = (
        db.session.query(AspectResult.review_id)
        .filter_by(analysis_id=analysis_id)
        .distinct()
        .count()
    )

    summary = get_aspect_summary(distribution, reviews_with_aspects=reviews_with_aspects)

    return jsonify({
        'success': True,
        'data': [a.to_dict() for a in aspects],
        'distribution': distribution,
        'summary': summary,
        'ready': analysis.to_dict()['aspects_ready'],
        'model_version': analysis.execution.model_version if analysis.execution else None,
        'sentiment_scope': ASPECT_SENTIMENT_SCOPE,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit if limit else 0
        }
    })


@analysis_bp.route('/analysis/<int:analysis_id>/predictions', methods=['GET'])
def get_predictions_data(analysis_id):
    """Get prediction results."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    page = max(1, request.args.get('page', 1, type=int))
    limit = max(1, min(100, request.args.get('limit', 50, type=int)))
    sentiment = request.args.get('sentiment')
    search = request.args.get('search', '')

    query = ReviewData.query.filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.is_duplicate == False,
        ReviewData.predicted_label.isnot(None)
    )

    if sentiment:
        query = query.filter(ReviewData.predicted_label == sentiment.lower())

    if search:
        query = query.filter(
            db.or_(
                ReviewData.original_content.ilike(f'%{search}%'),
                ReviewData.final_processed_text.ilike(f'%{search}%')
            )
        )

    total = query.count()
    reviews = query.order_by(ReviewData.id).offset((page - 1) * limit).limit(limit).all()

    # Distribution
    distribution = {'positif': 0, 'netral': 0, 'negatif': 0}
    for label, count in db.session.query(
        ReviewData.predicted_label,
        db.func.count(ReviewData.id)
    ).filter(
        ReviewData.analysis_id == analysis_id,
        ReviewData.is_duplicate == False,
        ReviewData.predicted_label.isnot(None)
    ).group_by(ReviewData.predicted_label).all():
        if label in distribution:
            distribution[label] = count

    data = []
    for r in reviews:
        item = {
            'id': r.id,
            'original_content': r.original_content,
            'final_processed_text': r.final_processed_text,
            'predicted_label': r.predicted_label,
            'actual_label': r.actual_label,
            'is_correct': r.is_correct,
            **review_upload_details(r, analysis.upload_context),
            'detected_aspects': json.loads(r.detected_aspects) if r.detected_aspects else []
        }
        data.append(item)

    return jsonify({
        'success': True,
        'data': data,
        'distribution': distribution,
        'mode': analysis.mode,
        'model_version': analysis.execution.model_version if analysis.execution else None,
        'excluded_empty_processed': ReviewData.query.filter_by(analysis_id=analysis_id, is_duplicate=False, final_processed_text='').count(),
        'excluded_zero_vectors': session_summary(analysis)['zero_vectors'],
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'total_pages': (total + limit - 1) // limit if limit else 0
        }
    })


@analysis_bp.route('/analysis/<int:analysis_id>/evaluation', methods=['GET'])
def get_evaluation_data(analysis_id):
    """Get evaluation results."""
    analysis = db.get_or_404(AnalysisSession, analysis_id)
    evaluation = EvaluationResult.query.filter_by(analysis_id=analysis_id).first()

    if not evaluation:
        return jsonify({
            'success': True,
            'has_labels': analysis.has_actual_labels,
            'evaluation': None
        })

    return jsonify({
        'success': True,
        'has_labels': True,
        'evaluation': evaluation.to_dict()
    })


# =============================================================================
# SINGLE REVIEW PREDICTION
# =============================================================================

@analysis_bp.route('/predict-single', methods=['POST'])
def predict_single_review():
    """Preprocess and predict a single review."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Teks harus berupa string yang tidak kosong'}), 400

    # Preprocess
    preprocessed = preprocess_full(text)
    if not preprocessed['final'].strip():
        return jsonify({'error': 'Teks kosong setelah preprocessing'}), 400

    # Predict
    model_key = data.get('model', 'v4')
    pipeline, metadata, digest = load_inference_model(model_key)
    vectorizer = fitted_features(pipeline)
    matrix = vectorizer.transform([preprocessed['final']])
    if not matrix.nnz:
        return jsonify({'error': 'Tidak ada fitur yang dikenali model (vektor TF-IDF nol). Ulasan tidak diprediksi.'}), 400
    features = feature_details(vectorizer, matrix.getrow(0))
    pred = pipeline.named_steps['classifier'].predict(matrix)[0]
    label = _normalize_label(pred)

    # Detect aspects
    aspects = detect_aspects_for_review(text, label, use_clauses=model_key == CANDIDATE_KEY)

    # Sentimen per aspek dihitung dari kalimat aspek itu sendiri memakai
    # artefak SVM canonical yang sama (lihat backend/services/aspect_service.py).
    # Ini konsisten dengan alur analisis batch (POST /analysis/<id>/aspects).
    if aspects['aspect_details']:
        aspect_labels = aspect_sentiment_labels(
            aspects['aspect_details'], pipeline=pipeline
        )
        for aspect, aspect_label in aspect_labels.items():
            aspects['aspect_details'][aspect]['sentiment'] = aspect_label

    payload = {
        'success': True,
        'original': text,
        'preprocessing_steps': {
            'cleaning': preprocessed['cleaned'],
            'case_folding': preprocessed['case_folded'],
            'tokenizing': preprocessed['tokenized'],
            'normalization': preprocessed['normalized'],
            'negation': preprocessed['negation'],
            'stopwords': preprocessed['stopwords'],
            'stemming': preprocessed['stemmed'],
            'final': preprocessed['final']
        },
        'predicted_label': label,
        'model_version': metadata['model_version'],
        'model': model_key,
        'detected_aspects': aspects['detected_aspects'],
        'aspect_sentiment_scope': aspects['aspect_sentiment_scope'],
    }
    payload['tfidf'] = features
    payload['pipeline_sha256'] = digest
    payload['preprocessing_signature'] = fingerprint_signature(effective_fingerprint(metadata))
    payload['aspect_details'] = aspects['aspect_details']
    record = SinglePrediction(payload=json.dumps(payload))
    db.session.add(record)
    db.session.commit()
    return jsonify(record.to_dict())


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_label(label) -> str:
    """Normalize label to the canonical lowercase class name."""
    return normalize_label(label)


# =============================================================================
# DASHBOARD STATS
# =============================================================================

@analysis_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics."""
    total_analyses = AnalysisSession.query.count()
    total_reviews = ReviewData.query.count()
    completed = AnalysisSession.query.filter_by(status='completed').count()

    # Distribution from latest completed analysis
    latest = AnalysisSession.query.filter_by(status='completed').order_by(
        AnalysisSession.created_at.desc(), AnalysisSession.id.desc()
    ).first()

    distribution = {'positif': 0, 'netral': 0, 'negatif': 0}
    aspect_counts = dict.fromkeys(ASPECTS, 0)
    aspect_sentiment = {aspect: dict.fromkeys(CANONICAL_CLASSES, 0) for aspect in ASPECTS}
    if latest:
        for label, count in db.session.query(
            ReviewData.predicted_label,
            db.func.count(ReviewData.id)
        ).filter(
            ReviewData.analysis_id == latest.id,
            ReviewData.is_duplicate.is_(False),
            ReviewData.predicted_label.isnot(None)
        ).group_by(ReviewData.predicted_label).all():
            if label in distribution:
                distribution[label] = count
        rows = db.session.query(
            AspectResult.aspect_name, AspectResult.sentiment, db.func.count(AspectResult.id)
        ).join(ReviewData, ReviewData.id == AspectResult.review_id).filter(
            AspectResult.analysis_id == latest.id,
            ReviewData.analysis_id == latest.id,
            ReviewData.is_duplicate.is_(False),
            ReviewData.predicted_label.isnot(None),
        ).group_by(AspectResult.aspect_name, AspectResult.sentiment).all()
        for aspect, label, count in rows:
            if aspect in aspect_counts:
                aspect_counts[aspect] += count
                label = normalize_label(label)
                if label in aspect_sentiment[aspect]:
                    aspect_sentiment[aspect][label] += count

    return jsonify({
        'success': True,
        'stats': {
            'total_analyses': total_analyses,
            'total_reviews': total_reviews,
            'completed_analyses': completed,
            'total_predicted': ReviewData.query.filter(ReviewData.is_duplicate == False, ReviewData.predicted_label.isnot(None)).count(),
            'single_predictions': SinglePrediction.query.count(),
            'latest_analysis': latest.to_dict() if latest else None,
            'distribution_scope': 'latest_completed_analysis',
            'distribution': distribution,
            'aspect_counts': aspect_counts,
            'aspect_sentiment': aspect_sentiment,
        }
    })


@analysis_bp.route('/stages', methods=['GET'])
def get_stages():
    """Get preprocessing stages info."""
    return jsonify({
        'success': True,
        'stages': [
            {'id': stage, 'name': STAGE_DISPLAY_NAMES.get(stage, stage)}
            for stage in STAGE_ORDER
        ]
    })
