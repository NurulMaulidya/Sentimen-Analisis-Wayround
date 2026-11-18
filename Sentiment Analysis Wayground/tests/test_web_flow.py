"""Integration coverage for uploaded rows, fitted feature inspection and stored history."""
import csv
import io
from contextlib import closing
from pathlib import Path

import pytest

from tests.conftest import PRODUCTION_DB


@pytest.fixture
def api():
    """
    Backend yang terisolasi ke database TEST.

    Database sudah diarahkan lewat WAYGROUND_DATABASE_URI oleh
    tests/conftest.py; assertion di bawah membuktikan engine tidak menunjuk
    ke database produksi.
    """
    from backend.app import app
    from backend.database_models import db
    app.config['TESTING'] = True
    with app.app_context():
        assert Path(db.engine.url.database).resolve() != PRODUCTION_DB.resolve(), \
            'Test tidak boleh memakai database produksi.'
        db.drop_all()
        db.create_all()
    return app.test_client()


def file_payload(rows, headers=('content', 'label')):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return io.BytesIO(stream.getvalue().encode('utf-8'))


def upload(api, rows=None, **kwargs):
    rows = rows or [('Aplikasi error tidak bisa login', 'negatif'),
                    ('Sangat membantu belajar qxujiparsial', ''), ('zzqxv hjkkqx', ''),
                    ('123 !!!', ''), ('Aplikasi error tidak bisa login', 'negatif'), ('', '')]
    response = api.post('/api/v1/analysis/upload', data={'file': (file_payload(rows), 'sample.csv'), **kwargs})
    assert response.status_code == 200, response.json
    return response.json['analysis']['id']


def step(api, sid, stage):
    response = api.post(f'/api/v1/analysis/{sid}/{stage}')
    assert response.status_code == 200, response.json
    return response.json


def test_aspect_review_filters_require_matching_aspect_and_sentiment(api):
    from backend.app import app
    from backend.database_models import db, AspectResult, ReviewData

    sid = upload(api, [('Aplikasi membantu belajar tetapi login error', 'netral')])
    with app.app_context():
        review = ReviewData.query.filter_by(analysis_id=sid).one()
        review.predicted_label = 'netral'
        review.detected_aspects = '["Kemudahan Akses", "Materi/Konten"]'
        db.session.add_all([
            AspectResult(analysis_id=sid, review_id=review.id, aspect_name='Kemudahan Akses', sentiment='negatif'),
            AspectResult(analysis_id=sid, review_id=review.id, aspect_name='Materi/Konten', sentiment='positif'),
        ])
        db.session.commit()

    endpoint = f'/api/v1/analysis/{sid}/aspect-reviews'
    matching = api.get(endpoint, query_string={'aspect': 'Kemudahan Akses', 'sentiment': 'negatif', 'search': 'LOGIN'})
    assert matching.json['pagination']['total'] == 1
    mismatched = api.get(endpoint, query_string={'aspect': 'Materi/Konten', 'sentiment': 'negatif'})
    assert mismatched.json['data'] == []


def test_character_candidate_single_and_batch_use_same_features_and_clauses(api):
    text = 'Soalnya bagus, tetapi sering error.'
    single = api.post('/api/v1/predict-single', json={'text': text, 'model': 'karakter'})
    assert single.status_code == 200, single.json
    version = single.json['model_version']
    assert version.startswith('kandidat-karakter-')
    assert any(feature['feature'].startswith('char__') for feature in single.json['tfidf']['features'])
    assert single.json['aspect_details']['Materi/Konten']['sentences'] == ['Soalnya bagus']

    sid = upload(api, [(text, 'netral')])
    step(api, sid, 'deduplicate')
    step(api, sid, 'preprocess')
    transform = api.post(f'/api/v1/analysis/{sid}/tfidf', json={'model': 'karakter'})
    assert transform.status_code == 200, transform.json
    assert transform.json['data'][0]['model_version'] == version
    step(api, sid, 'predict')  # The staged workflow reuses the selected TF-IDF model.
    step(api, sid, 'aspects')
    step(api, sid, 'evaluate')
    result = api.get(f'/api/v1/analysis/{sid}').json['analysis']
    assert result['model_version'] == version
    assert result['pipeline_sha256'] == single.json['pipeline_sha256']
    rows = api.get(f'/api/v1/analysis/{sid}/aspect-reviews').json['data']
    assert rows[0]['predicted_label'] == single.json['predicted_label']
    assert rows[0]['aspect_details']
    assert not api.get('/api/v1/research').json['model_version'].startswith('kandidat-karakter-')


def test_preview_column_mapping_and_preservation_of_empty_source_rows(api):
    rows = [('Bagus membantu belajar qxujikolom', 'positif'), ('', ''), ('Aplikasi error qxujikolom', '')]
    preview = api.post('/api/v1/analysis/inspect', data={'file': (file_payload(rows, ('komentar', 'nilai')), 'custom.csv')})
    assert preview.status_code == 200 and preview.json['columns'] == ['komentar', 'nilai']
    response = api.post('/api/v1/analysis/upload', data={'file': (file_payload(rows, ('komentar', 'nilai')), 'custom.csv'),
                                                       'content_column': 'komentar', 'label_column': 'nilai'})
    assert response.status_code == 200
    sid = response.json['analysis']['id']
    original = api.get(f'/api/v1/analysis/{sid}/original').json
    assert original['pagination']['total'] == 3
    assert [r['row_number'] for r in original['data']] == [1, 2, 3]
    assert original['data'][1]['validation_status'] == 'empty'
    assert original['data'][1]['raw_data'] == {'komentar': '', 'nilai': ''}
    assert api.get(f'/api/v1/analysis/{sid}/original?status=empty').json['pagination']['total'] == 1
    flow = api.get(f'/api/v1/analysis/{sid}').json['analysis']['flow']
    assert flow['uploaded'] == flow['source_rows_preserved'] == 3
    assert flow['empty_source'] == 1 and flow['valid_source'] == 2 and flow['valid_actual_labels'] == 1


def test_dashboard_separates_research_and_latest_upload_aspects(api):
    research = api.get('/api/v1/model/stats').json['stats']
    assert research['labeled_count'] == research['train_samples'] + research['test_samples']
    assert research['aspect_counts']['Performa Teknis'] == 195
    assert research['aspect_sentiment']['Performa Teknis'] == {'negatif': 157, 'netral': 18, 'positif': 27}
    assert api.get('/api/v1/stats').json['stats']['latest_analysis'] is None

    sid = upload(api, [('Aplikasi sering error saat login', 'negatif')])
    for stage in ['deduplicate', 'preprocess', 'predict', 'aspects', 'evaluate']:
        step(api, sid, stage)
    uploaded = api.get('/api/v1/stats').json['stats']
    assert uploaded['latest_analysis']['id'] == sid
    assert sum(uploaded['distribution'].values()) == 1
    assert uploaded['aspect_counts']['Performa Teknis'] == 1
    assert sum(uploaded['aspect_sentiment']['Performa Teknis'].values()) == 1
    assert api.get('/api/v1/model/stats').json['stats']['aspect_sentiment'] == research['aspect_sentiment']


def test_tfidf_prerequisite_zero_vectors_and_partial_label_evaluation(api):
    sid = upload(api)
    assert api.post(f'/api/v1/analysis/{sid}/tfidf').status_code == 409
    for stage in ['deduplicate', 'preprocess', 'tfidf', 'predict', 'aspects', 'evaluate']:
        step(api, sid, stage)
    features = api.get(f'/api/v1/analysis/{sid}/tfidf').json
    assert features['flow']['duplicates'] == 1
    assert features['flow']['empty_processed'] == features['flow']['zero_vectors'] == 1
    assert features['flow']['predicted'] == 2 and features['flow']['evaluable'] == 1
    assert features['flow']['predicted_without_label'] == 1
    oov = next(r for r in features['data'] if r['original_content'] == 'zzqxv hjkkqx')
    assert oov['nonzero'] == 0 and oov['status'] == 'zero_vector'
    predictions = api.get(f'/api/v1/analysis/{sid}/predictions').json
    assert predictions['pagination']['total'] == sum(predictions['distribution'].values()) == 2
    assert predictions['excluded_zero_vectors'] == 1
    evaluation = api.get(f'/api/v1/analysis/{sid}/evaluation').json['evaluation']
    assert sum(map(sum, evaluation['confusion_matrix'])) == 1
    assert evaluation['actual_vs_predicted'][0]['original_content']
    export = api.get(f'/api/v1/analysis/{sid}/export/validation')
    assert export.status_code == 200
    assert len(list(csv.DictReader(io.StringIO(export.data.decode('utf-8-sig'))))) == 6


def test_feature_weights_match_fitted_transform_and_predictions(api):
    from shared.deployment_integrity import load_verified_pipeline
    pipeline, _ = load_verified_pipeline()
    sid = upload(api, [('Aplikasi error tidak bisa login', 'negatif')])
    for stage in ['deduplicate', 'preprocess', 'tfidf', 'predict']:
        step(api, sid, stage)
    row = api.get(f'/api/v1/analysis/{sid}/tfidf').json['data'][0]
    detail = api.get(f'/api/v1/analysis/{sid}/features/{row["id"]}').json
    vectorizer = pipeline.named_steps['tfidf']
    vector = vectorizer.transform([detail['review']['final']]).getrow(0)
    actual = {f['feature']: f['weight'] for f in detail['tfidf']['features']}
    expected = {str(vectorizer.get_feature_names_out()[i]): float(v) for i, v in zip(vector.indices, vector.data)}
    assert actual == expected
    prediction = api.get(f'/api/v1/analysis/{sid}/predictions').json['data'][0]
    assert prediction['predicted_label'] == pipeline.predict([detail['review']['final']])[0]
    assert api.get(f'/api/v1/analysis/{sid+1}/features/{row["id"]}').status_code == 404


def test_rerun_preserves_snapshot_and_invalidates_features(api):
    sid = upload(api, [('Aplikasi error tidak bisa login', 'negatif')])
    for stage in ['deduplicate', 'preprocess', 'predict', 'aspects', 'evaluate']:
        step(api, sid, stage)
    step(api, sid, 'preprocess')
    assert not api.get(f'/api/v1/analysis/{sid}/tfidf').json['data']
    runs = api.get(f'/api/v1/analysis/{sid}/runs').json['runs']
    assert len(runs) == 1
    snapshot = api.get(f'/api/v1/analysis/{sid}/runs/{runs[0]["id"]}').json['snapshot']
    assert snapshot['analysis']['model_version'] == '20261029_223454'
    assert snapshot['reviews'][0]['predicted_label']
    assert snapshot['features'][0]['data']['features']
    assert snapshot['evaluation']['correct_count'] + snapshot['evaluation']['incorrect_count'] == 1
    assert api.delete(f'/api/v1/analysis/{sid}').status_code == 200
    assert api.get(f'/api/v1/analysis/{sid}/runs').status_code == 404


def test_single_history_is_stored_and_can_be_read_without_model_inference(api, monkeypatch):
    response = api.post('/api/v1/predict-single', json={'text': 'Aplikasi error tidak bisa login'})
    assert response.status_code == 200
    result = response.json
    assert result['tfidf']['nonzero'] > 0 and result['preprocessing_steps']['negation'] is not None
    from backend.api import analysis
    monkeypatch.setattr(analysis, 'load_inference_model', lambda *_: pytest.fail('History must not load or run model'))
    stored = api.get(f'/api/v1/single-history/{result["id"]}').json['prediction']
    assert stored == result
    assert api.get('/api/v1/single-history?page=0&limit=0').json['pagination']['limit'] == 1
    assert api.delete(f'/api/v1/single-history/{result["id"]}').status_code == 200
    assert api.get(f'/api/v1/single-history/{result["id"]}').status_code == 404


def test_aspects_include_unmatched_reviews_and_counts_ignore_pagination(api):
    sid = upload(api, [('Aplikasi error tidak bisa login. Login error terus.', ''), ('zzqxv bagus', '')])
    for stage in ['deduplicate', 'preprocess', 'predict', 'aspects']:
        step(api, sid, stage)
    reviews = api.get(f'/api/v1/analysis/{sid}/aspect-reviews?limit=1').json
    assert reviews['pagination']['total'] == 2 and len(reviews['data']) == 1
    assert reviews['flow']['with_aspects'] + reviews['flow']['without_aspects'] == 2
    no_aspect = api.get(f'/api/v1/analysis/{sid}/aspect-reviews?aspect=__none__').json
    assert no_aspect['pagination']['total'] == 1 and no_aspect['data'][0]['detected_aspects'] == []
    distribution = api.get(f'/api/v1/analysis/{sid}/aspects?limit=1').json
    assert distribution['summary']['total_reviews_with_aspects'] == 1


def test_research_metrics_are_separate_from_uploaded_data_and_unknown_labels_fail(api):
    before = api.get('/api/v1/research')
    assert before.status_code == 200
    assert len(before.json['experiments']) == 7
    assert before.json['evaluation']['dataset']['holdout_samples'] == 604
    upload(api, [('Sangat membantu belajar qxujiterpisah', 'negatif')])
    assert api.get('/api/v1/research').json['evaluation'] == before.json['evaluation']
    invalid = api.post('/api/v1/analysis/upload', data={'file': (file_payload([('Bagus', 'aneh')]), 'bad.csv')})
    assert invalid.status_code == 400
    from backend import config
    assert not list(config.UPLOAD_DIR.glob('*bad.csv'))


def test_zero_features_single_prediction_does_not_create_history(api):
    response = api.post('/api/v1/predict-single', json={'text': 'zzqxv hjkkqx'})
    assert response.status_code == 400 and 'TF-IDF' in response.json['error']
    assert api.get('/api/v1/single-history').json['pagination']['total'] == 0


def test_zero_features_batch_is_not_labeled_neutral(api):
    sid = upload(api, [('zzqxv hjkkqx', '')])
    for stage in ['deduplicate', 'preprocess', 'tfidf']:
        step(api, sid, stage)
    response = api.post(f'/api/v1/analysis/{sid}/predict')
    assert response.status_code == 400
    analysis = api.get(f'/api/v1/analysis/{sid}').json['analysis']
    assert analysis['flow']['predicted'] == 0 and analysis['flow']['zero_vectors'] == 1
    assert analysis['model_version'] is None and not analysis['predictions_ready']


def test_schema_upgrade_from_stage5_adds_tables_without_rewriting_old_execution(tmp_path):
    import sqlite3
    from flask import Flask
    from backend.database_models import db, AnalysisSession, AnalysisExecution, ensure_analysis_schema
    app = Flask('web-flow-migration-test')
    path = tmp_path / 'stage5.sqlite'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{path}'
    db.init_app(app)
    new_tables = {'upload_rows', 'review_features', 'single_predictions', 'analysis_run_snapshots',
                  'analysis_upload_contexts'}
    with app.app_context():
        for table in db.metadata.sorted_tables:
            if table.name not in new_tables:
                table.create(bind=db.engine, checkfirst=True)
        session = AnalysisSession(name='stored stage5', filename='old.csv', filepath='old', original_count=9)
        db.session.add(session)
        db.session.flush()
        db.session.add(AnalysisExecution(analysis_id=session.id, model_version='old-preserved-version'))
        db.session.commit()
        ensure_analysis_schema()
        ensure_analysis_schema()
        assert AnalysisSession.query.one().original_count == 9
        assert AnalysisExecution.query.one().model_version == 'old-preserved-version'
    backups = list((tmp_path / 'backups').glob('*.sqlite'))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as connection:
        names = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
        assert not names.intersection(new_tables)
        assert connection.execute('select model_version from analysis_executions').fetchone()[0] == 'old-preserved-version'


def test_research_demo_evaluates_canonical_rows_separately_from_holdout(api):
    import pandas as pd

    from shared.holdout_guard import CANONICAL_HOLDOUT_PATH

    development = pd.read_csv(CANONICAL_HOLDOUT_PATH.with_name('development.csv'), dtype=str).iloc[0]
    holdout = pd.read_csv(CANONICAL_HOLDOUT_PATH, dtype=str).iloc[0]
    rows = [
        (development['reviewId'], 'Aplikasi error tidak bisa login', 'negatif'),
        ('', holdout['content'].upper(), 'netral'),
        ('', '', ''),
    ]
    headers = ('review_id', 'Ulasan Asli', 'label_pakar')
    processed_only = [('', holdout['content'] + ' 987654321 !!!', 'netral')]
    derived = api.post('/api/v1/analysis/upload', data={
        'file': (file_payload(processed_only, headers), 'derived.csv'),
    })
    assert derived.status_code == 400
    assert 'holdout: 1' in derived.json['error']
    response = api.post('/api/v1/analysis/upload', data={
        'file': (file_payload(rows, headers), 'expert.csv'),
    })
    assert response.status_code == 400
    assert 'development: 1' in response.json['error']
    assert 'holdout: 1' in response.json['error']

    response = api.post('/api/v1/analysis/upload', data={
        'file': (file_payload(rows, headers), 'expert.csv'),
        'mode': 'research_demo',
    })
    assert response.status_code == 200, response.json
    analysis = response.json['analysis']
    assert analysis['mode'] == 'research_demo'
    sid = analysis['id']
    endpoint = f'/api/v1/analysis/{sid}'
    source = api.get(f'{endpoint}/original').json['data']
    assert [r['data_origin'] for r in source] == [
        'development', 'holdout', 'Tidak cocok dengan split canonical',
    ]
    for stage in ('deduplicate', 'preprocess', 'tfidf', 'predict', 'aspects'):
        step(api, sid, stage)
    predictions = api.get(f'{endpoint}/predictions').json
    assert predictions['mode'] == 'research_demo'
    row = next(r for r in predictions['data'] if r['data_origin'] == 'development')
    assert row['actual_label'] == 'negatif'
    assert row['lexicon_label']
    assert row['is_correct'] == (row['predicted_label'] == row['actual_label'])
    before = api.get('/api/v1/research').json['evaluation']
    assert step(api, sid, 'evaluate')['evaluation_scope'] == 'research_descriptive'
    evaluation = api.get(f'{endpoint}/evaluation').json['evaluation']
    assert evaluation['scope'] == 'research_descriptive'
    assert sum(map(sum, evaluation['confusion_matrix'])) == len(predictions['data'])
    exported = api.get(f'{endpoint}/export/confusion-matrix')
    assert exported.status_code == 200
    matrix_rows = list(csv.DictReader(io.StringIO(exported.data.decode('utf-8-sig'))))
    assert [r['actual_label'] for r in matrix_rows] == ['negatif', 'netral', 'positif']
    assert all(r['evaluation_scope'] == 'research_descriptive' for r in matrix_rows)
    assert api.get('/api/v1/research').json['evaluation'] == before
    assert api.get(endpoint).json['analysis']['status'] == 'completed'


def test_expert_form_preamble_is_skipped_for_preview_and_upload(api):
    form = (
        'FORM VALIDASI PAKAR 300 ULASAN WAYGROUND\tUnnamed: 1\tUnnamed: 2\n'
        'Nama Pakar\tContoh\t\n'
        'No\tUlasan Asli\tLabel \n'
        '1\tContoh ulasan baru untuk demonstrasi\tPositif\n'
    ).encode()
    preview = api.post('/api/v1/analysis/inspect', data={'file': (io.BytesIO(form), 'form.tsv')})
    assert preview.status_code == 200, preview.json
    assert preview.json['total'] == 1
    assert preview.json['columns'] == ['No', 'Ulasan Asli', 'Label']
    upload_response = api.post('/api/v1/analysis/upload', data={
        'file': (io.BytesIO(form), 'form.tsv'), 'mode': 'research_demo',
    })
    assert upload_response.status_code == 200, upload_response.json
    assert upload_response.json['analysis']['has_actual_labels']


def test_upload_evaluation_matches_sklearn_and_offline_pipeline_across_files(api):
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_score, recall_score,
    )
    from backend.services.inference_models import load_inference_model
    from shared.labels import CANONICAL_CLASSES, normalize_label
    from shared.preprocessing import preprocess_text

    rows = [
        ('Aplikasi crash gagal masuk ketika kuis qxujimatriks pertama', 'negatif'),
        ('Soal terbuka pada menu kelas qxujimatriks kedua', 'netral'),
        ('Latihan soal sangat menyenangkan qxujimatriks ketiga', 'positif'),
        ('Materi kuis sangat menarik qxujimatriks keempat', ''),
    ]
    pipeline, _, _ = load_inference_model('v4')
    recorded = []
    for file_rows in (rows, rows[:1], [(rows[3][0], '')]):
        sid = upload(api, file_rows)
        endpoint = f'/api/v1/analysis/{sid}'
        for stage in ('deduplicate', 'preprocess', 'tfidf', 'predict', 'aspects'):
            step(api, sid, stage)
        predictions = api.get(f'{endpoint}/predictions').json['data']
        offline = list(map(normalize_label, pipeline.predict([preprocess_text(text) for text, _ in file_rows])))
        assert [r['predicted_label'] for r in predictions] == offline
        pairs = [(r['actual_label'], r['predicted_label']) for r in predictions if r['actual_label']]
        if not pairs:
            assert api.post(f'{endpoint}/evaluate').status_code == 400
            assert api.get(f'{endpoint}/evaluation').json['evaluation'] is None
            assert api.get(f'{endpoint}/export/confusion-matrix').status_code == 409
            continue
        actual, predicted = zip(*pairs)
        args = dict(labels=CANONICAL_CLASSES, average='macro', zero_division=0)
        response = step(api, sid, 'evaluate')
        result = api.get(f'{endpoint}/evaluation').json['evaluation']
        assert result['metrics'] == response['metrics'] == {
            'accuracy': accuracy_score(actual, predicted),
            'precision_macro': precision_score(actual, predicted, **args),
            'recall_macro': recall_score(actual, predicted, **args),
            'f1_macro': f1_score(actual, predicted, **args),
            'f1_weighted': f1_score(actual, predicted, **{**args, 'average': 'weighted'}),
        }
        assert result['confusion_matrix'] == confusion_matrix(actual, predicted, labels=CANONICAL_CLASSES).tolist()
        assert result['classification_report'] == classification_report(
            actual, predicted, labels=CANONICAL_CLASSES, zero_division=0, output_dict=True,
        )
        exported = list(csv.DictReader(io.StringIO(api.get(f'{endpoint}/export/predictions').get_data(as_text=True))))
        assert [r['predicted_label'] for r in exported] == offline
        recorded.append((sid, result))
    assert recorded[0][1]['confusion_matrix'] != recorded[1][1]['confusion_matrix']
    for sid, result in recorded:
        endpoint = f'/api/v1/analysis/{sid}'
        assert api.get(f'{endpoint}/evaluation').json['evaluation'] == result
        assert api.get(endpoint).json['analysis']['evaluated_count'] == sum(map(sum, result['confusion_matrix']))


def test_label_candidates_and_issues_appear_before_upload(api):
    rows = [
        ('Contoh kuis qxanalisislabel', 'neutral', 'Positif'),
        ('Contoh latihan qxanalisislabel', '', 'Positif / Netral'),
    ]
    headers = ('  UlAsAn AsLi ', ' Label_Aktual ', ' Label Pakar Final ')
    preview = api.post('/api/v1/analysis/inspect', data={'file': (file_payload(rows, headers), 'labels.csv')}).json
    assert preview['content_column'] == headers[0]
    assert preview['label_candidates'] == list(headers[1:])
    assert preview['label_issues'][headers[1]]['empty'] == [2]
    assert preview['label_issues'][headers[2]]['invalid'] == [2]

    def send(**fields):
        return api.post('/api/v1/analysis/upload', data={
            'file': (file_payload(rows, headers), 'labels.csv'), 'mode': 'research_demo', **fields,
        })
    assert send().json['label_candidates'] == list(headers[1:])
    invalid = send(label_column=headers[2])
    assert invalid.status_code == 400 and invalid.json['label_issues']['invalid'] == [2]
    uploaded = send(label_column=headers[1])
    assert uploaded.status_code == 200, uploaded.json
    sid = uploaded.json['analysis']['id']
    original = api.get(f'/api/v1/analysis/{sid}/original').json['data']
    assert original[0]['actual_label'] == 'netral'
    assert original[1]['label_status'] == 'empty'
