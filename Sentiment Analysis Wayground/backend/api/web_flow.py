"""Inspection, history and exports for the web inference workflow."""
import csv
import io
import json

from flask import Blueprint, jsonify, request, Response

from backend.database_models import (db, AnalysisSession, ReviewData, ReviewFeatures, SinglePrediction,
                                     UploadRow, AnalysisRunSnapshot, AspectResult)
from backend.services.web_flow import feature_details, fitted_features, transform_reviews, session_summary, source_row_dict, review_upload_details
from backend.services.analysis_state import fingerprint_signature
from backend.services.inference_models import load_inference_model, model_key_for_version
from backend.services.upload_service import read_uploaded_dataframe, allowed_file, MAX_UPLOAD_ROWS, find_content_column, label_candidates, label_issue_rows
from shared.deployment_integrity import (
    effective_fingerprint,
    load_verified_pipeline,
    verify_selection_contract,
)
from shared.labels import CANONICAL_CLASSES
from shared.final_config import PROJECT_ROOT, FINAL_EVALUATION_PATH, FINAL_PREDICTIONS_PATH
from backend.services.preprocessing_service import preprocess_full

web_flow_bp = Blueprint('web_flow', __name__, url_prefix='/api/v1')


def pagination(items):
    page = max(1, request.args.get('page', 1, type=int))
    limit = max(1, min(100, request.args.get('limit', 20, type=int)))
    return {'data': items[(page - 1) * limit:page * limit], 'pagination': {
        'page': page, 'limit': limit, 'total': len(items), 'total_pages': (len(items) + limit - 1) // limit}}


@web_flow_bp.post('/analysis/inspect')
def inspect_upload():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify(error='Gunakan file CSV/TSV/TXT.'), 400
    try:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='wayground_preview_') as directory:
            path = Path(directory) / 'preview.csv'
            file.save(path)
            frame = read_uploaded_dataframe(path)
        if frame.empty or len(frame) > MAX_UPLOAD_ROWS:
            return jsonify(error=f'File harus berisi 1 sampai {MAX_UPLOAD_ROWS} baris.'), 400
        candidates = label_candidates(frame)
        return jsonify(success=True, columns=list(frame.columns), total=len(frame),
                       content_column=find_content_column(frame), label_candidates=candidates,
                       label_issues={col: label_issue_rows(frame, col) for col in candidates},
                       preview=frame.head(5).fillna('').astype(str).to_dict(orient='records'))
    except Exception:
        return jsonify(error='File tidak dapat dibaca. Periksa delimiter dan encoding.'), 400


@web_flow_bp.route('/analysis/<int:sid>/tfidf', methods=['GET', 'POST'])
def tfidf(sid):
    analysis = db.get_or_404(AnalysisSession, sid)
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        version = analysis.execution.model_version if analysis.execution else None
        pipeline, metadata, digest = load_inference_model(data.get('model', model_key_for_version(version)))
        if not analysis.execution or analysis.execution.preprocessing_signature != fingerprint_signature(effective_fingerprint(metadata)):
            return jsonify(error='Jalankan preprocessing yang sesuai model terlebih dahulu.'), 409
        if analysis.execution.model_version:
            # Inspect existing prediction features rather than changing its representation/version.
            if version != metadata['model_version'] or analysis.execution.pipeline_sha256 != digest:
                return jsonify(error='Versi hasil berbeda. Buat analisis baru atau ulangi preprocessing.'), 409
        transform_reviews(analysis, pipeline, metadata)
        db.session.commit()
    query = ReviewFeatures.query.filter_by(analysis_id=sid).order_by(ReviewFeatures.review_id)
    items = []
    for f in query.all():
        review = db.session.get(ReviewData, f.review_id)
        detail = json.loads(f.payload)
        items.append({'id': f.review_id, 'original_content': review.original_content,
                      'final_processed_text': review.final_processed_text, 'model_version': f.model_version,
                      **{k: v for k, v in detail.items() if k != 'features'}})
    return jsonify(success=True, **pagination(items), flow=session_summary(analysis))


@web_flow_bp.get('/analysis/<int:sid>/features/<int:rid>')
def features(sid, rid):
    review = ReviewData.query.filter_by(analysis_id=sid, id=rid).first_or_404()
    row = ReviewFeatures.query.filter_by(analysis_id=sid, review_id=rid).first()
    return jsonify(success=True, review=review.to_comparison_dict(),
                   tfidf=dict(json.loads(row.payload), model_version=row.model_version) if row else None)


@web_flow_bp.get('/analysis/<int:sid>/aspect-reviews')
def aspect_reviews(sid):
    analysis = db.get_or_404(AnalysisSession, sid)
    reviews = ReviewData.query.filter_by(analysis_id=sid, is_duplicate=False).filter(ReviewData.predicted_label.isnot(None)).order_by(ReviewData.id).all()
    aspect_filter = request.args.get('aspect', '')
    sentiment = request.args.get('sentiment', '')
    search = request.args.get('search', '').strip().casefold()
    details_by_review = {}
    for detail in AspectResult.query.filter_by(analysis_id=sid).order_by(AspectResult.id):
        details_by_review.setdefault(detail.review_id, []).append(detail)
    items = []
    for r in reviews:
        if search and search not in (r.original_content or '').casefold():
            continue
        aspects = json.loads(r.detected_aspects or '[]')
        if aspect_filter == '__none__' and aspects:
            continue
        if aspect_filter and aspect_filter != '__none__' and aspect_filter not in aspects:
            continue
        details = details_by_review.get(r.id, [])
        if sentiment and not any(
            a.sentiment == sentiment and (not aspect_filter or a.aspect_name == aspect_filter)
            for a in details
        ):
            continue
        items.append({**r.to_dict(), 'aspect_details': [a.to_dict() for a in details]})
    return jsonify(success=True, **pagination(items), flow=session_summary(analysis))


@web_flow_bp.route('/single-history', methods=['GET'])
def single_history():
    search = request.args.get('search', '').strip().casefold()
    order = SinglePrediction.id.asc() if request.args.get('order') == 'oldest' else SinglePrediction.id.desc()
    items = [r.to_dict() for r in SinglePrediction.query.order_by(order)]
    if search:
        items = [item for item in items if search in item.get('original', '').casefold()]
    return jsonify(success=True, **pagination(items))


@web_flow_bp.route('/single-history/<int:pid>', methods=['GET', 'DELETE'])
def single_detail(pid):
    row = db.get_or_404(SinglePrediction, pid)
    payload = row.to_dict()
    if request.method == 'DELETE':
        db.session.delete(row)
        db.session.commit()
    return jsonify(success=True, prediction=payload)


@web_flow_bp.get('/analysis/<int:sid>/runs')
def stored_runs(sid):
    db.get_or_404(AnalysisSession, sid)
    rows = AnalysisRunSnapshot.query.filter_by(analysis_id=sid).order_by(AnalysisRunSnapshot.id.desc()).all()
    return jsonify(success=True, runs=[{'id': r.id, 'created_at': r.created_at.isoformat() + 'Z',
                                       'analysis': json.loads(r.payload)['analysis']} for r in rows])


@web_flow_bp.get('/analysis/<int:sid>/runs/<int:rid>')
def stored_run(sid, rid):
    row = AnalysisRunSnapshot.query.filter_by(analysis_id=sid, id=rid).first_or_404()
    return jsonify(success=True, snapshot=json.loads(row.payload))


@web_flow_bp.get('/research')
def research():
    _, metadata = load_verified_pipeline()
    evaluation = json.loads(FINAL_EVALUATION_PATH.read_text(encoding='utf-8-sig'))
    from shared.constants import EXPERIMENTS_DIR
    selected, _ = verify_selection_contract()
    from shared.experiment_artifacts import EXPERIMENT_JSON_FILES
    experiments = []
    for number, filename in EXPERIMENT_JSON_FILES.items():
        artifact = json.loads((EXPERIMENTS_DIR / filename).read_text(encoding='utf-8-sig'))
        # Show compact scientific summaries, not OOF rows or fold matrices.
        best = artifact.get('best', artifact.get('selected_result', {}))
        metric = best.get('cv_f1_mean', best.get('cv_macro_f1_mean', best.get('cv_f1_macro_mean')))
        metric_label = 'Mean grouped CV macro F1'
        note = ''
        if number == 1:
            result = artifact['configuration']['final_split']
            note = 'Split canonical ditetapkan sebelumnya; rasio terbaik sensitivitas tidak mengganti holdout.'
        elif number == 2:
            result = 'Threshold lexicon ' + str(artifact['configuration']['selected_threshold'])
            metric = best['expert_validation']['f1_macro']
            metric_label = 'Macro F1 terhadap label pakar pada development persiapan'
        elif number == 3:
            result = f"Maks. fitur {best['max_features']}; n-gram {best['ngram_range']}; sublinear TF {best['sublinear_tf']}"
        elif number == 4:
            result = f"Kernel {best['kernel']}; C={best['C']}; gamma={best['gamma']}"
        elif number == 5:
            result = 'Tanpa resampling' if best['method'] == 'none' else best['method']
        elif number == 6:
            result = artifact['winner_by_cv_f1_macro'].upper()
            winner = next(s for s in artifact['strategies'] if s['strategy'] == artifact['winner_by_cv_f1_macro'])
            metric = winner['mean_f1_macro']
        else:
            result = f"C={best['C']}; gap train/CV {best['gap_percentage_points']:.2f} poin persentase"
            note = 'Diagnostik regularisasi; tidak otomatis mengganti konfigurasi final. Batas gap belum terpenuhi.' if not best['gap_pass'] else 'Diagnostik regularisasi.'
        experiments.append({'number': number, 'artifact': filename, 'result': result,
                            'metric': metric, 'metric_label': metric_label, 'note': note})
    return jsonify(success=True, model_version=metadata['model_version'], evaluation=evaluation,
                   configuration=metadata['model'], provenance=metadata.get('evaluation_provenance', evaluation.get('provenance')),
                   selection=selected, experiments=experiments)


@web_flow_bp.get('/research/examples')
def research_examples():
    pipeline, metadata = load_verified_pipeline()
    vectorizer = fitted_features(pipeline)
    examples = []
    seen = set()
    with FINAL_PREDICTIONS_PATH.open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            label = row['actual_sentiment']
            if label in seen:
                continue
            steps = preprocess_full(row['content'])
            if steps['final'] != row['processed_text']:
                continue
            features = feature_details(vectorizer, vectorizer.transform([steps['final']]).getrow(0))
            if not features['nonzero']:
                continue
            features['model_version'] = metadata['model_version']
            examples.append({'id': row['dataset_index'], 'original': row['content'],
                             'reference': label, 'prediction': row['predicted_sentiment'],
                             'steps': {'cleaning': steps['cleaned'], 'case_folding': steps['case_folded'],
                                       'tokenizing': steps['tokenized'], 'normalization': steps['normalized'],
                                       'negation': steps['negation'], 'stopwords': steps['stopwords'],
                                       'stemming': steps['stemmed'], 'final': steps['final']},
                             'tfidf': features})
            seen.add(label)
            if len(seen) == 3:
                break
    return jsonify(success=True, model_version=metadata['model_version'], examples=examples)


@web_flow_bp.get('/analysis/<int:sid>/export/<kind>')
def export(sid, kind):
    analysis = db.get_or_404(AnalysisSession, sid)
    reviews = ReviewData.query.filter_by(analysis_id=sid).order_by(ReviewData.id).all()
    if kind == 'validation':
        sources = UploadRow.query.filter_by(analysis_id=sid).order_by(UploadRow.row_number).all()
        rows = [source_row_dict(r, context=analysis.upload_context) for r in sources] if sources else [r.to_dict() for r in reviews]
    elif kind == 'preprocessing':
        rows = [r.to_comparison_dict() for r in reviews]
    elif kind == 'tfidf':
        rows = [{'review_id': f.review_id, 'model_version': f.model_version, **json.loads(f.payload)}
                for f in ReviewFeatures.query.filter_by(analysis_id=sid).order_by(ReviewFeatures.review_id).all()]
    elif kind in ['predictions', 'aspects']:
        rows = [{**r.to_dict(), **review_upload_details(r, analysis.upload_context), 'mode': analysis.mode}
                for r in reviews if r.predicted_label]
        if kind == 'aspects':
            for row in rows:
                row['aspect_details'] = [a.to_dict() for a in AspectResult.query.filter_by(analysis_id=sid, review_id=row['id']).all()]
    elif kind in ('evaluation', 'confusion-matrix'):
        if not analysis.evaluation:
            return jsonify(error='Evaluasi belum tersedia.'), 409
        if kind == 'confusion-matrix':
            matrix = json.loads(analysis.evaluation.confusion_matrix)
            rows = [{'actual_label': actual, **dict(zip(CANONICAL_CLASSES, counts))}
                    for actual, counts in zip(CANONICAL_CLASSES, matrix)]
        else:
            rows = json.loads(analysis.evaluation.actual_vs_predicted)
        rows = [{**row, 'evaluation_scope': analysis.evaluation_scope} for row in rows]
    else:
        return jsonify(error='Jenis ekspor tidak dikenal.'), 400
    stream = io.StringIO()
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['id']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        clean = {}
        for key, value in row.items():
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value) if value is not None else ''
            clean[key] = "'" + text if text.startswith(('=', '+', '-', '@')) else text
        writer.writerow(clean)
    return Response('\ufeff' + stream.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=analysis_{sid}_{kind}.csv'})
