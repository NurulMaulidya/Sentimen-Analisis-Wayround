"""Isolated browser fixture: real built frontend/API/model, SQLite memory only."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5075)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    from backend import config
    config.DATABASE_URI = 'sqlite:///:memory:'
    config.UPLOAD_DIR = args.artifacts.resolve() / 'uploads'
    from backend.app import app
    from backend.database_models import db, AnalysisSession, ReviewData
    from flask import send_file, jsonify
    ids = {}
    dist = ROOT / 'frontend/dist'
    app.view_functions['index'] = lambda: send_file(dist / 'index.html')

    @app.route('/__verification/fixture')
    def fixture():
        return jsonify({'ids': ids, 'upload_file': str(sample.resolve())})

    @app.route('/<path:path>')
    def spa(path):
        file = (dist / path).resolve()
        if file.is_relative_to(dist.resolve()) and file.is_file():
            return send_file(file)
        if path.startswith('api/'):
            return jsonify({'error': 'Not found'}), 404
        return send_file(dist / 'index.html')

    client = app.test_client()
    ids = {}
    for name, labels, count, legacy in [('Berlabel browser', True, 3, False), ('Tanpa label browser', False, 3, False), ('Riwayat lama browser', False, 1, True), ('Pagination browser', False, 25, False)]:
        with app.app_context():
            session = AnalysisSession(name=name, filename='browser-fixture.csv', filepath='memory', original_count=count,
                                      has_actual_labels=labels, status='completed' if legacy else 'uploaded')
            db.session.add(session)
            db.session.flush()
            sid = session.id
            texts = ['Aplikasi error tidak bisa login', 'Sangat membantu belajar mengerjakan soal', 'Saya menggunakan aplikasi untuk belajar']
            for i in range(count):
                db.session.add(ReviewData(analysis_id=sid, original_content=texts[i % 3] + (f' ulasan {i}' if count > 3 else ''),
                    actual_label=['negatif', 'positif', 'netral'][i % 3] if labels else None,
                    final_processed_text='teks lama' if legacy else None, predicted_label='netral' if legacy else None))
            db.session.commit()
        ids[name] = sid
        if not legacy:
            for stage in ['deduplicate', 'preprocess', 'predict', 'aspects'] + (['evaluate'] if labels else []):
                response = client.post(f'/api/v1/analysis/{sid}/{stage}')
                if response.status_code != 200:
                    raise RuntimeError((stage, response.json))
    sample = args.artifacts / 'upload-sample.csv'
    sample.write_text('content,label\nAplikasi error tidak bisa login,negatif\nSangat membantu belajar dan soal qxujibrowser,\nzzqxv hjkkqx,\n123 !!!,\nAplikasi error tidak bisa login,negatif\n,\n', encoding='utf-8')
    (args.artifacts / 'fixture.json').write_text(json.dumps({'ids': ids, 'upload_file': str(sample.resolve()), 'database': 'sqlite memory'}, indent=2), encoding='utf-8')
    print('Browser fixture ready; production database is not used.', flush=True)
    # In-memory SQLite shares one connection across requests in this fixture.
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False, threaded=False)


if __name__ == '__main__':
    main()
