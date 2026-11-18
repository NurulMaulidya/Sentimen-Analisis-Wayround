"""Render saved diagnostic evaluation without fitting or reconstructing a split."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output_v4/verified_visualizations')
    args = parser.parse_args()
    from shared.deployment_integrity import load_verified_pipeline, sha256
    from shared.final_config import FINAL_EVALUATION_PATH
    _, metadata = load_verified_pipeline()
    evaluation = json.loads(FINAL_EVALUATION_PATH.read_text(encoding='utf-8-sig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = evaluation['confusion_matrix_labels']
    matrix = np.asarray(evaluation['confusion_matrix'])
    assert matrix.shape == (len(labels), len(labels))
    assert int(matrix.sum()) == evaluation['dataset']['holdout_samples']
    output = args.output.resolve()
    # Keep generated plots away from immutable deployment inputs.
    protected = [ROOT / 'models', ROOT / 'data', ROOT / 'output_v4/experiments', ROOT / 'output_v4/research_run']
    if not output.is_relative_to(ROOT) or any(output.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Use a separate visualization directory inside the project')
    output.mkdir(parents=True, exist_ok=True)
    title = f"Model {metadata['model_version']} | diagnostic holdout (prior exposure)"
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    image = ax.imshow(matrix, cmap='Blues', vmin=0)
    fig.colorbar(image, ax=ax, label='Jumlah ulasan')
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel='Prediksi', ylabel='Label aktual', title=title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    positions = np.arange(len(labels))
    for offset, metric in enumerate(['precision', 'recall', 'f1-score']):
        values = [evaluation['per_class'][label][metric] for label in labels]
        bars = ax.bar(positions + (offset - 1) * .25, values, width=.25, label=metric)
        ax.bar_label(bars, fmt='%.3f', fontsize=8)
    ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1.15), ylabel='Skor (0–1)', title=title)
    ax.legend(loc='upper left', ncols=3)
    fig.savefig(output / 'per_class_metrics.png', dpi=160)
    plt.close(fig)
    report = {'model_version': metadata['model_version'], 'evaluation_sha256': sha256(FINAL_EVALUATION_PATH),
              'evaluation_scope': evaluation['methodology']['final_evaluation_set'],
              'labels': labels, 'confusion_matrix': matrix.tolist(), 'samples': int(matrix.sum()),
              'metrics': evaluation['holdout_metrics'], 'refitted': False, 'split_reconstructed': False,
              'files': ['confusion_matrix.png', 'per_class_metrics.png']}
    (output / 'visualization_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
